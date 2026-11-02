"""The simplified mantle model: slab tracers, CMB heat and a synthetic dVs field.

Used by PlateDriver (run_SDP.py) and by real reconstructions (slablet.gplates).
Never imports either of them, nor pygplates, geopandas or shapely.

Inputs from a caller are trench samples, a dict of 1-D arrays:
    lon, lat                  degrees, in the mantle frame (slablets sink vertically there)
    convergence_km_per_myr    > 0 where the plates converge (km/Myr = mm/yr)
and optionally, for dipping slabs (p.slab_dip_deg < 90 or p.slab_trench_follow):
    downdip_azimuth_deg       direction from the trench towards the overriding plate (NaN: unknown)
    trench_east_km_per_myr, trench_north_km_per_myr   the trench's own velocity in the mantle frame
The caller builds them from its own boundaries and is responsible for the frame.

The state is the slab tracer DataFrame (columns SLAB_COLUMNS) and the 2-D LLSVP state
(new_llsvp_state / update_llsvp): the CMB heat for llsvp: heat, the pile thickness (km)
for llsvp: pile, None for llsvp: none.
"""
import numpy as np
import pandas as pd
import xarray as xr

from .sphere import lonlat_to_xyz, move_along_azimuth, spherical_gaussian_smooth
from . import pile as _pile

# Slablets from trenches converging at least this fast (km/Myr = mm/yr) get full weight;
# slower (near-stalled) boundaries contribute proportionally less.
SLAB_FULL_WEIGHT_CONVERGENCE_KM_PER_MYR = 10.0
# with p.resolution_independent, slab amounts are per REF_SAMPLE_DEG of trench and per REF_CELL_DEG x REF_LAYER_KM cell
REF_SAMPLE_DEG, REF_CELL_DEG, REF_LAYER_KM = 1.0, 5.0, 100.0

# slab_at_cmb: rest removes a resting slablet once exp(-age / tau) falls below this.
SLAB_REST_MIN_FRACTION = 1e-3

SLAB_COLUMNS = ['lon', 'lat', 'depth_km', 'age', 'active', 'sink_rate_km_per_myr', 'weight', 'attached']
# Extra columns, present only when dipping slabs are on and the trench samples carry the
# optional keys: the down-dip azimuth and the trench velocity at seeding.
DIP_COLUMNS = {'downdip_azimuth_deg': 'dip_azimuth_deg', 'trench_east_km_per_myr': 'trench_east',
               'trench_north_km_per_myr': 'trench_north'}
SURFACE_RADIUS_KM = 6371.0


def _dipping(p):
    return p.slab_dip_deg < 90.0 or p.slab_trench_follow or (p.slab_dip_shallow_km > 0
                                                             and p.slab_dip_shallow_deg < 90.0)


def _cot_dip(depth, p):
    """1 / tan(dip) at each depth (0 where vertical)."""
    def cot(d):
        return 0.0 if d >= 90.0 else 1.0 / np.tan(np.radians(d))
    return np.where(np.asarray(depth) < p.slab_dip_shallow_km, cot(p.slab_dip_shallow_deg), cot(p.slab_dip_deg))


def _down_dip_km(depth, p):
    """Horizontal distance from the trench of a slab reaching depth (capped at upper_mantle_km)."""
    d = np.minimum(np.asarray(depth, float), p.upper_mantle_km)
    s = np.minimum(d, p.slab_dip_shallow_km)
    return s * _cot_dip(0.0, p) * (p.slab_dip_shallow_km > 0) + (d - s) * _cot_dip(p.upper_mantle_km, p)


def _horizontal_step(lon, lat, depth, azimuth, east, north, vertical_km, delta_time, p):
    """New lon, lat of upper-mantle slablets after one step of dip and trench-following
    motion (vertical_km: how far each sinks this step).  NaN azimuth: no dip motion."""
    de = np.zeros(len(lon))
    dn = np.zeros(len(lon))
    cot = _cot_dip(depth, p)
    if np.any(cot > 0):
        along = np.where(np.isfinite(azimuth), vertical_km * cot, 0.0)
        az = np.radians(np.nan_to_num(azimuth))
        de += along * np.sin(az)
        dn += along * np.cos(az)
    if p.slab_trench_follow:
        de += np.nan_to_num(east) * delta_time
        dn += np.nan_to_num(north) * delta_time
    return move_along_azimuth(lon, lat, np.degrees(np.arctan2(de, dn)), np.hypot(de, dn),
                              SURFACE_RADIUS_KM - depth)


def empty_samples():
    return {'lon': np.zeros(0), 'lat': np.zeros(0), 'convergence_km_per_myr': np.zeros(0)}


def sink_slabs(slab_tracers, delta_time, p, flow=None):
    """Sink existing slablets vertically (lon/lat fixed) for delta_time Myr.

    With dipping slabs on (p.slab_dip_deg < 90 or p.slab_trench_follow) slablets above
    p.upper_mantle_km also move horizontally: towards the overriding plate by sink / tan(dip)
    and with their trench's velocity at seeding; below it they sink vertically.

    Rate: the slablet's own sink_rate_km_per_myr above p.upper_mantle_km, that rate
    divided by p.slab_slowing_factor below (times 1 + p.slab_lower_rate_gradient x the fraction
    of the way to the CMB).  Initial slab columns ignore the gradient.  Slablets that reach p.cmb_depth_km are removed
    (p.slab_at_cmb = 'remove'), or stop there and stay until their thermal anomaly is below
    SLAB_REST_MIN_FRACTION of its start ('rest').  With p.slab_cmb_tau_myr > 0 a resting
    slablet's weight also decays as exp(-time resting / p.slab_cmb_tau_myr).
    Returns a new DataFrame.
    """
    if p.slab_at_cmb not in ('remove', 'rest'):
        raise ValueError('slab_at_cmb must be remove or rest, not {!r}'.format(p.slab_at_cmb))
    slab_tracers = slab_tracers.copy()
    if len(slab_tracers) > 0:
        active = slab_tracers['active'].astype(bool)
        upper = active & (slab_tracers['depth_km'] < p.upper_mantle_km)
        lower = active & (slab_tracers['depth_km'] >= p.upper_mantle_km)
        if _dipping(p) and 'dip_azimuth_deg' in slab_tracers and upper.any():
            u = slab_tracers.loc[upper]
            lon, lat = _horizontal_step(
                u['lon'].values.astype(float), u['lat'].values.astype(float), u['depth_km'].values.astype(float),
                u['dip_azimuth_deg'].values.astype(float), u['trench_east'].values.astype(float),
                u['trench_north'].values.astype(float),
                u['sink_rate_km_per_myr'].values.astype(float) * delta_time, delta_time, p)
            slab_tracers.loc[upper, 'lon'] = lon
            slab_tracers.loc[upper, 'lat'] = lat
        slab_tracers.loc[upper, 'depth_km'] += (
            slab_tracers.loc[upper, 'sink_rate_km_per_myr'] * delta_time)
        if p.slab_lower_rate_gradient:
            frac = ((slab_tracers.loc[lower, 'depth_km'].astype(float) - p.upper_mantle_km)
                    / (p.cmb_depth_km - p.upper_mantle_km)).clip(0.0, 1.0)
            if p.slab_lower_rate_km_per_myr > 0 or _uses_age(slab_tracers, p):
                slab_tracers.loc[lower, 'depth_km'] += (_lower_rate(slab_tracers.loc[lower], p)
                                                        * (1.0 + p.slab_lower_rate_gradient * frac.values) * delta_time)
            else:
                slab_tracers.loc[lower, 'depth_km'] += (
                    slab_tracers.loc[lower, 'sink_rate_km_per_myr'] / p.slab_slowing_factor
                    * (1.0 + p.slab_lower_rate_gradient * frac) * delta_time)
        elif p.slab_lower_rate_km_per_myr > 0 or _uses_age(slab_tracers, p):
            slab_tracers.loc[lower, 'depth_km'] += _lower_rate(slab_tracers.loc[lower], p) * delta_time
        else:
            slab_tracers.loc[lower, 'depth_km'] += (
                slab_tracers.loc[lower, 'sink_rate_km_per_myr'] / p.slab_slowing_factor * delta_time)
        slab_tracers.loc[active, 'age'] += delta_time
        if p.slab_at_cmb == 'rest':
            slab_tracers['depth_km'] = np.minimum(slab_tracers['depth_km'].astype(float), p.cmb_depth_km)
            faded = pd.Series(thermal_fraction(slab_tracers, p) < SLAB_REST_MIN_FRACTION, index=slab_tracers.index)
            if p.slab_cmb_tau_myr > 0:
                resting = slab_tracers['depth_km'].astype(float) >= p.cmb_depth_km
                slab_tracers.loc[resting, 'weight'] = (slab_tracers.loc[resting, 'weight'].astype(float)
                                                       * np.exp(-delta_time / p.slab_cmb_tau_myr))
                faded |= slab_tracers['weight'].astype(float) < SLAB_REST_MIN_FRACTION * delta_time
            slab_tracers = slab_tracers[~faded].reset_index(drop=True)
        else:
            slab_tracers = slab_tracers[slab_tracers['depth_km'] < p.cmb_depth_km].reset_index(drop=True)
        if p.slab_cmb_spread_km_per_myr > 0:
            slab_tracers = spread_cmb_slabs(slab_tracers, delta_time, p)
        slab_tracers = flow_lower_slabs(slab_tracers, delta_time, p, flow)
    return slab_tracers


def spread_cmb_slabs(slab_tracers, delta_time, p):
    """Move slablets within p.slab_cmb_spread_layer_km of the CMB sideways, down the gradient of
    the basal flow potential ψ (slablet/pile.py flow_potential, from the slab load on the CMB): slab
    material arriving at the base spreads away from where slabs pile up, as in convection models.
    The velocity is -∇ψ scaled so that its rms over the sphere is p.slab_cmb_spread_km_per_myr; each
    slablet takes that velocity interpolated (bilinear between cell centres) to its position and
    moves along a great circle on the CMB.
    Returns the DataFrame with lon and lat updated."""
    deep = (slab_tracers['active'].astype(bool)
            & (slab_tracers['depth_km'].astype(float) >= p.cmb_depth_km - p.slab_cmb_spread_layer_km)).values
    if not deep.any():
        return slab_tracers
    lons_g, lats_g = mantle_grid(p.mantle_grid_spacing_deg)
    psi = _pile.flow_potential(_pile.slab_load(slab_tracers, p), p)
    d = np.radians(p.mantle_grid_spacing_deg)
    v_north = -np.gradient(psi, d, axis=0)
    v_east = -(np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2 * d * np.cos(np.radians(lats_g))[:, None])
    w = np.cos(np.radians(lats_g))[:, None] * np.ones(len(lons_g))
    rms = np.sqrt(((v_north ** 2 + v_east ** 2) * w).sum() / w.sum())
    if rms <= 0:
        return slab_tracers
    scale = p.slab_cmb_spread_km_per_myr / rms
    s = slab_tracers.loc[deep]
    fi = np.clip((s['lat'].values.astype(float) + 90) / p.mantle_grid_spacing_deg - 0.5, 0.0, len(lats_g) - 1.0)
    fj = (s['lon'].values.astype(float) + 180) / p.mantle_grid_spacing_deg - 0.5
    i0 = np.minimum(np.floor(fi).astype(int), len(lats_g) - 2)
    j0 = np.floor(fj).astype(int)
    ti, tj = fi - i0, fj - j0
    j0, j1 = j0 % len(lons_g), (j0 + 1) % len(lons_g)

    def bilinear(f):                                     # between cell centres, wrapping in longitude
        return ((1 - ti) * ((1 - tj) * f[i0, j0] + tj * f[i0, j1])
                + ti * ((1 - tj) * f[i0 + 1, j0] + tj * f[i0 + 1, j1]))

    ve, vn = scale * bilinear(v_east), scale * bilinear(v_north)
    lon, lat = move_along_azimuth(s['lon'].values.astype(float), s['lat'].values.astype(float),
                                  np.degrees(np.arctan2(ve, vn)), np.hypot(ve, vn) * delta_time, _pile.CMB_RADIUS_KM)
    slab_tracers.loc[deep, 'lon'] = lon
    slab_tracers.loc[deep, 'lat'] = lat
    return slab_tracers


def _age_ratio(plate_age, p):
    """r = clipped seafloor age at subduction / p.slab_age_ref_myr (unknown ages: the default)."""
    a = np.asarray(plate_age, float)
    a = np.where(np.isfinite(a), a, p.slab_age_default_myr)
    return np.clip(a, p.slab_age_min_myr, p.slab_age_max_myr) / p.slab_age_ref_myr


def _uses_age(slabs, p):
    return p.slab_age_ref_myr > 0 and 'plate_age' in slabs


def thermal_tau(slabs, p):
    """Thermal e-folding time of each slablet: p.slab_thermal_tau_myr, times r^slab_age_tau_exponent with the
    seafloor-age options (a scalar without them)."""
    if _uses_age(slabs, p) and p.slab_age_tau_exponent:
        return p.slab_thermal_tau_myr * _age_ratio(slabs['plate_age'].values, p) ** p.slab_age_tau_exponent
    return p.slab_thermal_tau_myr


def thermal_fraction(slabs, p):
    """exp(-age / tau): the fraction of its thermal anomaly each slablet still has."""
    return np.exp(-slabs['age'].values.astype(float) / thermal_tau(slabs, p))


def _lower_rate(slabs, p):
    """Lower-mantle sinking rate of each slablet before the depth gradient."""
    if p.slab_lower_rate_km_per_myr > 0:
        rate = np.full(len(slabs), float(p.slab_lower_rate_km_per_myr))
        if p.slab_lower_rate_exponent:
            rate = rate * (slabs['sink_rate_km_per_myr'].values.astype(float)
                           / p.slab_lower_rate_ref_km_per_myr) ** p.slab_lower_rate_exponent
    else:
        rate = slabs['sink_rate_km_per_myr'].values.astype(float) / p.slab_slowing_factor
    if _uses_age(slabs, p) and p.slab_age_rate_exponent:
        rate = rate * _age_ratio(slabs['plate_age'].values, p) ** p.slab_age_rate_exponent
    return rate


def _grid_bilinear(f, lats_g, lons_g, lat, lon):
    """f (nlat, nlon) on cell centres, bilinear at (lat, lon), wrapping in longitude; NaN counts as 0."""
    f = np.nan_to_num(f)
    dlat, dlon = lats_g[1] - lats_g[0], lons_g[1] - lons_g[0]
    fi = np.clip((lat - lats_g[0]) / dlat, 0.0, len(lats_g) - 1.0)
    fj = (lon - lons_g[0]) / dlon
    i0 = np.minimum(np.floor(fi).astype(int), len(lats_g) - 2)
    j0 = np.floor(fj).astype(int)
    ti, tj = fi - i0, fj - j0
    j0, j1 = j0 % len(lons_g), (j0 + 1) % len(lons_g)
    return ((1 - ti) * ((1 - tj) * f[i0, j0] + tj * f[i0, j1])
            + ti * ((1 - tj) * f[i0 + 1, j0] + tj * f[i0 + 1, j1]))


def flow_lower_slabs(slab_tracers, delta_time, p, flow):
    """Move lower-mantle slablets sideways with a fraction of the plate velocity above them: p.slab_flow_top at
    upper_mantle_km to p.slab_flow_bottom at the CMB, linear in depth (the same angular rate as that fraction of
    the plate's).  flow: {'lats', 'lons' (cell centres), 'vel_e', 'vel_n' (km/Myr at the surface; NaN = 0)}."""
    if flow is None or (p.slab_flow_top == 0 and p.slab_flow_bottom == 0):
        return slab_tracers
    lower = (slab_tracers['active'].astype(bool) & (slab_tracers['depth_km'].astype(float) >= p.upper_mantle_km)).values
    if not lower.any():
        return slab_tracers
    s = slab_tracers.loc[lower]
    lon, lat = s['lon'].values.astype(float), s['lat'].values.astype(float)
    frac = np.clip((s['depth_km'].values.astype(float) - p.upper_mantle_km) / (p.cmb_depth_km - p.upper_mantle_km), 0, 1)
    k = p.slab_flow_top + (p.slab_flow_bottom - p.slab_flow_top) * frac
    ve = k * _grid_bilinear(flow['vel_e'], flow['lats'], flow['lons'], lat, lon)
    vn = k * _grid_bilinear(flow['vel_n'], flow['lats'], flow['lons'], lat, lon)
    lon, lat = move_along_azimuth(lon, lat, np.degrees(np.arctan2(ve, vn)), np.hypot(ve, vn) * delta_time,
                                  SURFACE_RADIUS_KM)
    slab_tracers.loc[lower, 'lon'] = lon
    slab_tracers.loc[lower, 'lat'] = lat
    return slab_tracers


def _sample_scale(p):
    """Slablet weight factor: the trench length each slablet stands for, in REF_SAMPLE_DEG (1 without
    p.resolution_independent)."""
    return p.slab_sample_deg / REF_SAMPLE_DEG if p.resolution_independent else 1.0


def _cell_scale(spacing_deg, p, depth_interval_km=None):
    """Factor turning a sum binned per cell into an amount per REF_CELL_DEG (x REF_LAYER_KM) cell
    (1 without p.resolution_independent)."""
    if not p.resolution_independent:
        return 1.0
    f = (REF_CELL_DEG / spacing_deg) ** 2
    return f * REF_LAYER_KM / depth_interval_km if depth_interval_km else f


def seed_slabs(slab_tracers, samples, delta_time, p):
    """Add one slablet at the surface for each trench sample.

    Sink rate: p.slab_sink_rate_factor × convergence, capped at p.slab_max_sink_rate.
    Weight: delta_time × min(1, convergence / 10 km/Myr), so the mantle signal per Myr
    of subduction does not depend on the time step and near-stalled trenches add little;
    with p.resolution_independent also × slab_sample_deg (per 1 deg of trench).
    """
    if samples is None or not len(samples['lon']):
        return slab_tracers
    convergence = np.asarray(samples['convergence_km_per_myr'], dtype=float)
    new = pd.DataFrame({
        'lon': samples['lon'], 'lat': samples['lat'], 'depth_km': 0.0, 'age': 0.0, 'active': True,
        'sink_rate_km_per_myr': np.minimum(convergence * p.slab_sink_rate_factor, p.slab_max_sink_rate),
        'weight': delta_time * np.minimum(1.0, convergence / SLAB_FULL_WEIGHT_CONVERGENCE_KM_PER_MYR)
                  * _sample_scale(p),
        'attached': True})
    if _dipping(p) and 'downdip_azimuth_deg' in samples:
        for key, col in DIP_COLUMNS.items():
            new[col] = np.asarray(samples[key], dtype=float)
    if p.slab_age_ref_myr > 0:
        _add_plate_age(new, samples.get('seafloor_age_myr', np.full(len(new), np.nan)), p)
    return new if len(slab_tracers) == 0 else pd.concat([slab_tracers, new], ignore_index=True)


def _add_plate_age(slabs, plate_age, p):
    """The seafloor age at subduction as column plate_age, and the weight times r^slab_age_weight_exponent."""
    slabs['plate_age'] = np.asarray(plate_age, float)
    if p.slab_age_weight_exponent:
        slabs['weight'] = slabs['weight'].values * _age_ratio(slabs['plate_age'].values, p) ** p.slab_age_weight_exponent


def initial_slab_columns(samples, delta_time, p, depth_km=None):
    """Slab tracers for the trenches of a starting configuration.

    Each trench sample gets the column of slablets the model itself would have built
    had it been subducting at its present rate until its slab reached depth_km
    (default p.initial_slab_depth_km): one slablet per delta_time of that history,
    sinking at the rates of sink_slabs.  The sink rate is floored at
    SLAB_FULL_WEIGHT_CONVERGENCE_KM_PER_MYR so a near-stalled trench does not need
    thousands of Myr of history.  Columns are vertical (lon/lat fixed, mantle frame).
    Returns a DataFrame with SLAB_COLUMNS (empty if there are no samples).
    """
    if depth_km is None:
        depth_km = p.initial_slab_depth_km
    if depth_km <= 0 or samples is None or not len(samples['lon']):
        return pd.DataFrame(columns=SLAB_COLUMNS)
    s_lon, s_lat = np.asarray(samples['lon']), np.asarray(samples['lat'])
    convergence = np.asarray(samples['convergence_km_per_myr'], dtype=float)
    upper_mantle_km, slowing_factor = p.upper_mantle_km, p.slab_slowing_factor
    rate = np.clip(convergence * p.slab_sink_rate_factor, SLAB_FULL_WEIGHT_CONVERGENCE_KM_PER_MYR,
                   p.slab_max_sink_rate)
    # lower-mantle rate: rate / slowing_factor, or the fixed rate (times the seafloor-age factor)
    if p.slab_lower_rate_km_per_myr > 0 or p.slab_age_ref_myr > 0:
        probe = pd.DataFrame({'sink_rate_km_per_myr': rate})
        if p.slab_age_ref_myr > 0:
            probe['plate_age'] = np.asarray(samples.get('seafloor_age_myr', np.full(len(rate), np.nan)), float)
        lower_rate = _lower_rate(probe, p)
    else:
        lower_rate = rate / slowing_factor
    # time for the slab to reach depth_km: upper mantle at rate, lower at lower_rate
    upper = min(depth_km, upper_mantle_km)
    t_end = upper / rate + max(0.0, depth_km - upper_mantle_km) / lower_rate
    n = np.ceil(t_end / delta_time).astype(int) + 1          # ages 0, dt, ..., >= t_end
    point = np.repeat(np.arange(len(s_lat)), n)
    age = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) * delta_time
    r = rate[point]
    t_um = upper_mantle_km / r
    depth = np.where(age <= t_um, r * age, upper_mantle_km + (age - t_um) * lower_rate[point])
    out = pd.DataFrame({
        'lon': s_lon[point], 'lat': s_lat[point],
        'depth_km': np.minimum(depth, depth_km), 'age': age, 'active': True,
        'sink_rate_km_per_myr': r,
        'weight': delta_time * np.minimum(1.0, convergence[point] / SLAB_FULL_WEIGHT_CONVERGENCE_KM_PER_MYR)
                  * _sample_scale(p),
        'attached': True})
    if p.slab_age_ref_myr > 0:
        _add_plate_age(out, np.asarray(samples.get('seafloor_age_myr', np.full(len(s_lon), np.nan)), float)[point], p)
    if _dipping(p) and 'downdip_azimuth_deg' in samples:
        # where the slablet would be had the trench kept its present velocity and dip: moved
        # down-dip while in the upper mantle; seeded where the trench was age Myr ago and then
        # carried with it while in the upper mantle, so offset by -v x (age - time in the upper mantle)
        for key, col in DIP_COLUMNS.items():
            out[col] = np.asarray(samples[key], dtype=float)[point]
        az, east, north = (out[c].values for c in DIP_COLUMNS.values())
        de = np.zeros(len(out))
        dn = np.zeros(len(out))
        if _dipping(p):
            along = np.where(np.isfinite(az), _down_dip_km(out['depth_km'].values, p), 0.0)
            de += along * np.sin(np.radians(np.nan_to_num(az)))
            dn += along * np.cos(np.radians(np.nan_to_num(az)))
        if p.slab_trench_follow:
            back = np.maximum(0.0, age - t_um)
            de -= np.nan_to_num(east) * back
            dn -= np.nan_to_num(north) * back
        lon, lat = move_along_azimuth(out['lon'].values, out['lat'].values, np.degrees(np.arctan2(de, dn)),
                                      np.hypot(de, dn), SURFACE_RADIUS_KM - out['depth_km'].values)
        out['lon'], out['lat'] = lon, lat
    return out


def slab_depth_near(mid_lon, mid_lat, side_xyz, slab_lon, slab_lat, slab_depth,
                    radius_deg=5.0, line_tol_deg=1.0, depth_cap_km=None, cell_deg=0.25):
    """Depth (km) of the deepest slab tracer near each boundary segment, on one side of it.

    side_xyz (n, 3) is the segment's great-circle normal, signed to point to the side
    where the slab lies (the overriding plate).  A tracer counts if it is within
    radius_deg (great-circle) of the segment midpoint and not more than line_tol_deg on
    the other side of the segment's great circle (slablets are seeded on the trench
    line, and the line moves).  Slablets sink vertically where they were seeded, so the
    deep part of a retreating trench's slab lies under the overriding plate, as a
    dipping slab does.  0 where there is no slab.

    For speed, depths are capped at depth_cap_km (the depth beyond which no force grows)
    and the slab is reduced to its deepest tracer in each cell_deg lon/lat cell, keeping
    that tracer's own position.  Only a cell whose deepest tracer is on the wrong side of
    a segment while a shallower one is on the right side can differ, which cell_deg well
    below line_tol_deg makes rare.
    """
    z = np.zeros(len(mid_lon))
    if len(slab_depth) == 0 or len(mid_lon) == 0:
        return z
    slab_lon, slab_lat = np.asarray(slab_lon, float), np.asarray(slab_lat, float)
    slab_depth = np.asarray(slab_depth, float)
    if depth_cap_km is not None:
        slab_depth = np.minimum(slab_depth, depth_cap_km)
    if cell_deg:
        cell = (np.floor((slab_lat + 90.0) / cell_deg).astype(np.int64) * 100000
                + np.floor(((slab_lon + 180.0) % 360.0) / cell_deg).astype(np.int64))
        order = np.lexsort((-slab_depth, cell))                # by cell, deepest first
        first = order[np.r_[True, cell[order][1:] != cell[order][:-1]]]
        slab_lon, slab_lat, slab_depth = slab_lon[first], slab_lat[first], slab_depth[first]
    from scipy.spatial import cKDTree
    s_xyz = lonlat_to_xyz(np.asarray(slab_lon, float), np.asarray(slab_lat, float))
    m_xyz = lonlat_to_xyz(np.asarray(mid_lon, float), np.asarray(mid_lat, float))
    chord = 2.0 * np.sin(np.radians(radius_deg) / 2.0)
    pairs = cKDTree(m_xyz).sparse_distance_matrix(cKDTree(s_xyz), chord, output_type='ndarray')
    if len(pairs) == 0:
        return z
    seg, idx = pairs['i'], pairs['j']
    side = np.einsum('ij,ij->i', s_xyz[idx], np.asarray(side_xyz, float)[seg])   # sin(distance from the great circle), + on the slab side
    ok = side > -np.sin(np.radians(line_tol_deg))
    np.maximum.at(z, seg[ok], np.asarray(slab_depth, float)[idx[ok]])
    return z


def mantle_grid(spacing_deg):
    lons_g = np.arange(-180 + spacing_deg / 2, 180, spacing_deg)
    lats_g = np.arange( -90 + spacing_deg / 2,  90, spacing_deg)
    return lons_g, lats_g


def new_cmb_heat(p):
    """A cold start for the CMB heat field (nlat × nlon on the mantle grid)."""
    lons_g, lats_g = mantle_grid(p.mantle_grid_spacing_deg)
    return np.zeros((len(lats_g), len(lons_g)))


LLSVP_MODES = ('heat', 'pile', 'none')


def _check_llsvp(p):
    if p.llsvp not in LLSVP_MODES:
        raise ValueError('llsvp must be one of {}, not {!r}'.format(LLSVP_MODES, p.llsvp))


def _tbl_params(p):
    """The pile parameters for the thin boundary layer (p.tbl_*): a local-drive pile."""
    import dataclasses
    return dataclasses.replace(p, pile_mean_thickness_km=p.tbl_thickness_km, pile_drive='local',
                               pile_mobility_km_per_myr=p.tbl_mobility_km_per_myr,
                               pile_slab_head_km=p.tbl_slab_head_km, pile_slab_layer_km=p.tbl_slab_layer_km)


def _with_tbl(p):
    return p.llsvp == 'pile' and p.tbl_thickness_km > 0


def new_llsvp_state(p):
    """Start state of the LLSVP model chosen by p.llsvp ({'pile', 'tbl'} with the boundary layer)."""
    _check_llsvp(p)
    if p.llsvp == 'heat':
        return new_cmb_heat(p)
    if _with_tbl(p):
        return {'pile': _pile.new_pile(p), 'tbl': _pile.new_pile(_tbl_params(p))}
    if p.llsvp == 'pile':
        return _pile.new_pile(p)
    return None


def update_llsvp(state, slab_tracers, delta_time, p):
    """One step of the LLSVP model chosen by p.llsvp."""
    _check_llsvp(p)
    if p.llsvp == 'heat':
        return update_cmb_heat(state, slab_tracers, delta_time, p)
    if _with_tbl(p):
        return {'pile': _pile.update_pile(state['pile'], slab_tracers, delta_time, p),
                'tbl': _pile.update_pile(state['tbl'], slab_tracers, delta_time, _tbl_params(p))}
    if p.llsvp == 'pile':
        return _pile.update_pile(state, slab_tracers, delta_time, p)
    return None


def _slab_weights(slab_tracers):
    if 'weight' in slab_tracers.columns:
        return slab_tracers['weight'].values.astype(float)
    return np.ones(len(slab_tracers))


def _per_area(binned, lats_g, p):
    """With p.mantle_bin_area_normalise, a lon/lat-binned sum (..., nlat, nlon) divided by
    each cell's area relative to an equatorial cell (cos of the cell-centre latitude), so the
    field is an amount per unit area, which is what the area-weighted smoothing assumes."""
    if not p.mantle_bin_area_normalise:
        return binned
    return binned / np.cos(np.radians(lats_g))[:, None]


def update_cmb_heat(cmb_heat, slab_tracers, delta_time, p):
    """Accumulate CMB heat each timestep, building LLSVP-like warm regions.

    Core heating increments the 2D field (nlat × nlon) everywhere each step;
    slablets suppress it with depth-weighted coupling.  A wide spherical
    Gaussian keeps the pattern large-scale.  Positive values = warm regions
    (slow dVs); passed to dvs_field as the upwelling anomaly.

    Two physical mechanisms are combined:
      - Thermal (material): slablets near the CMB suppress with full weight (1.0).
      - Dynamic pressure (incompressibility): surface slablets suppress with
        p.cmb_surface_coupling weight, representing instantaneous Stokes-flow coupling.
    Coupling varies linearly with depth from p.cmb_surface_coupling at 0 km to 1.0 at
    p.cmb_depth_km, so the LLSVP anti-correlation with subduction appears from the
    first timestep rather than waiting ~50 Myr for material to arrive physically.

    p.cmb_heating_rate is added per Myr everywhere; p.cmb_slab_suppression is removed per
    unit-weight slablet per Myr at full coupling; p.cmb_sigma_deg is the smoothing per Myr
    (for other time steps the width scales with sqrt(delta_time), so it acts like
    diffusion); p.cmb_max_heat caps the field.
    """
    lons_g, lats_g = mantle_grid(p.mantle_grid_spacing_deg)
    nlat, nlon = cmb_heat.shape
    spacing_deg = p.mantle_grid_spacing_deg

    # Core heating everywhere
    cmb_heat = cmb_heat + p.cmb_heating_rate * delta_time

    if len(slab_tracers) > 0:
        all_active = slab_tracers[slab_tracers['active'].astype(bool)]
        if len(all_active) > 0:
            lons_f   = all_active['lon'].values.astype(float)
            lats_f   = all_active['lat'].values.astype(float)
            depths_f = all_active['depth_km'].values.astype(float)

            coupling = p.cmb_surface_coupling + (1.0 - p.cmb_surface_coupling) * (depths_f / p.cmb_depth_km)
            weights  = p.cmb_slab_suppression * delta_time * coupling * _slab_weights(all_active)
            if p.cmb_suppression_thermal_decay:
                weights = weights * thermal_fraction(all_active, p)

            lon_idx = np.clip(np.floor((lons_f + 180) / spacing_deg).astype(int), 0, nlon - 1)
            lat_idx = np.clip(np.floor((lats_f + 90)  / spacing_deg).astype(int), 0, nlat - 1)
            suppress = np.zeros_like(cmb_heat)
            np.add.at(suppress, (lat_idx, lon_idx), weights)
            cmb_heat = np.maximum(cmb_heat - _per_area(suppress, lats_g, p) * _cell_scale(spacing_deg, p), 0.0)

    cmb_heat = spherical_gaussian_smooth(cmb_heat, lats_g, lons_g, p.cmb_sigma_deg * np.sqrt(delta_time))

    return np.minimum(cmb_heat, p.cmb_max_heat)


def _remove_area_weighted_mean(field, lats):
    w = np.cos(np.radians(lats))[:, None] * np.ones(field.shape[-1])
    means = (field * w).sum(axis=(-2, -1)) / w.sum()
    return field - means[:, None, None]


# On a fine field grid, horizontal kernels at least this fraction of the coarse grid spacing are
# applied on the coarse grid and interpolated up (dvs_field): a fine-grid kernel that wide is huge.
_COARSE_PATH_FRACTION = 0.4


def slab_sigma_profile(depths, p):
    """Horizontal slab smoothing (great-circle degrees) at each depth for the slab width profile
    (p.slab_sigma_cmb_deg > 0): p.slab_sigma_deg above p.upper_mantle_km; below it from
    p.slab_sigma_lower_deg (or p.slab_sigma_deg) growing linearly to p.slab_sigma_cmb_deg at the
    CMB.  Rounded to 0.05 degrees so that layers share smoothing kernels."""
    lower0 = p.slab_sigma_lower_deg if p.slab_sigma_lower_deg > 0 else p.slab_sigma_deg
    frac = np.clip((np.asarray(depths, float) - p.upper_mantle_km) / (p.cmb_depth_km - p.upper_mantle_km), 0.0, 1.0)
    sig = np.where(np.asarray(depths) < p.upper_mantle_km, p.slab_sigma_deg,
                   lower0 + (p.slab_sigma_cmb_deg - lower0) * frac)
    return np.round(sig / 0.05) * 0.05


def _block_mean(f, lats_f, k):
    """(..., nlat, nlon) on a fine grid -> the coarse grid k times wider (cos-latitude weights)."""
    k = int(round(k))
    w = np.cos(np.radians(lats_f))[:, None] * np.ones(f.shape[-1])
    sh = f.shape[:-2] + (f.shape[-2] // k, k, f.shape[-1] // k, k)
    num = (f * w).reshape(sh).sum(axis=(-3, -1))
    return num / w.reshape(sh[-4:]).sum(axis=(-3, -1))


def _upsample(f, lats_c, lons_c, lats_f, lons_f):
    """Bilinear interpolation between coarse cell centres onto fine ones, wrapping in longitude
    and holding the outermost coarse row towards the poles.  f is (..., nlat_c, nlon_c)."""
    dc = lats_c[1] - lats_c[0]
    fi = np.clip((lats_f - lats_c[0]) / dc, 0.0, len(lats_c) - 1.0)
    i0 = np.minimum(np.floor(fi).astype(int), len(lats_c) - 2)
    ti = (fi - i0)[:, None]
    fj = (lons_f - lons_c[0]) / dc
    j0 = np.floor(fj).astype(int)
    tj = (fj - j0)[None, :]
    j0, j1 = j0 % len(lons_c), (j0 + 1) % len(lons_c)
    a = f[..., i0, :][..., j0] * (1 - tj) + f[..., i0, :][..., j1] * tj
    b = f[..., i0 + 1, :][..., j0] * (1 - tj) + f[..., i0 + 1, :][..., j1] * tj
    return a * (1 - ti) + b * ti


def lithosphere_field(thickness_km, lats_g, lons_g, depths_g, depth_interval_km, p):
    """The lid: +p.lithosphere_dVs % at the surface falling linearly to 0 at the base, averaged over each layer.
    thickness_km is on any regular cell-centre grid that is a multiple or a fraction of the field grid."""
    th = np.asarray(thickness_km, float)
    k = (lats_g[1] - lats_g[0]) * th.shape[0] / 180.0
    if k > 1.001:                                         # field grid coarser: block mean
        th = _block_mean(th, np.arange(-90 + 90.0 / th.shape[0], 90, 180.0 / th.shape[0]), k)
    elif k < 0.999:                                       # finer: repeat
        r = int(round(1 / k))
        th = np.repeat(np.repeat(th, r, axis=0), r, axis=1)
    sub_z = (depths_g[:, None] + depth_interval_km * (np.arange(10) + 0.5)[None, :] / 10 - depth_interval_km / 2)
    frac = np.clip(1.0 - sub_z[:, :, None, None] / np.maximum(th, 1.0)[None, None], 0.0, 1.0).mean(axis=1)
    return p.lithosphere_dVs * frac


def plume_sources(h, p):
    """Plume sources: lon, lat and strength (thickness / maximum) of the local maxima of the hot layer h (model
    grid) within p.plume_spacing_deg, where it is thicker than p.plume_min_fraction of its maximum."""
    from scipy.ndimage import maximum_filter
    lons_c, lats_c = mantle_grid(p.mantle_grid_spacing_deg)
    hmax = float(h.max())
    if hmax <= 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    if p.plume_source == 'sites':
        lon, lat = plume_sites(p.plume_spacing_deg, p.plume_seed)
        hs = _grid_bilinear(h, lats_c, lons_c, lat, lon)
        on = hs > p.plume_min_fraction * hmax
        return lon[on], lat[on], hs[on] / hmax
    if p.plume_source != 'maxima':
        raise ValueError('plume_source must be maxima or sites, not {!r}'.format(p.plume_source))
    size = max(3, int(round(p.plume_spacing_deg / p.mantle_grid_spacing_deg)) | 1)
    peak = (h == maximum_filter(h, size=size, mode=('nearest', 'wrap'))) & (h > p.plume_min_fraction * hmax)
    i, j = np.nonzero(peak)
    return lons_c[j], lats_c[i], h[i, j] / hmax


def plume_sites(spacing_deg, seed=0):
    """lon, lat of fixed random sites, uniform on the sphere, one per spacing_deg x spacing_deg on average."""
    n = max(1, int(round(4 * np.pi / np.radians(spacing_deg) ** 2)))
    rng = np.random.default_rng(seed)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    lon = rng.uniform(-180.0, 180.0, n)
    return lon, lat


def plume_field(plumes, lats_g, lons_g, depths_g, p):
    """Vertical conduits: p.plume_dVs x strength x exp(-d^2 / 2 sigma^2) (d: great-circle distance from the axis;
    the strongest conduit where they overlap) at depths from p.plume_top_km down."""
    lon, lat, strength = plumes
    lo, la = np.meshgrid(lons_g, lats_g)
    cell = lonlat_to_xyz(lo.ravel(), la.ravel())
    axis = lonlat_to_xyz(np.asarray(lon, float), np.asarray(lat, float))
    d = np.degrees(np.arccos(np.clip(cell @ axis.T, -1.0, 1.0)))
    amp = (p.plume_dVs * strength[None, :] * np.exp(-0.5 * (d / p.plume_sigma_deg) ** 2)).min(axis=1)
    layer = (depths_g >= p.plume_top_km).astype(float)
    return layer[:, None, None] * amp.reshape(la.shape)[None]


def dvs_field(slab_tracers, llsvp_state, p, lithosphere_km=None):
    """Bin slab tracers onto a 3D grid and return a dVs/Vs anomaly Dataset.

    Each slablet contributes a cold temperature anomaly (p.slab_temp_anomaly_K × its
    weight) that decays exponentially with age (p.slab_thermal_tau_myr).  The summed ΔT
    per grid cell is smoothed (spherical Gaussian horizontally, Gaussian in depth) and
    converted to dVs/Vs % (p.slab_dVs_per_K, default -0.001 %/K, so a fresh unit-weight
    slablet gives -0.5% per cell).

    llsvp_state is the state of the LLSVP model (p.llsvp):
      heat  the CMB heat (2D nlat×nlon from update_cmb_heat): a warm negative-dVs anomaly is
            added in the deepest layers, decaying upward as exp(-(distance from CMB / p.cmb_decay_km)²);
      pile  the pile thickness (km): p.pile_dVs is added to the part of each layer inside the
            pile, which fills each column from the CMB up to its thickness; with p.pile_halo_km > 0,
            p.pile_halo_dVs (scaled down for piles thinner than p.pile_halo_h_km) is added above
            the pile top, decaying upward as exp(-height / p.pile_halo_km);
      none  nothing is added (llsvp_state is None).

    Slab width: one horizontal smoothing (p.slab_sigma_deg), or p.slab_sigma_lower_deg below the
    upper mantle, or with p.slab_sigma_cmb_deg > 0 a profile (slab_sigma_profile): narrow slabs above
    p.upper_mantle_km, widening with depth below it.  With p.mantle_field_spacing_deg the field is drawn
    on that finer grid (float32); the LLSVP state stays on the model grid and is interpolated up, and
    kernels wider than _COARSE_PATH_FRACTION of the model grid are applied on the model grid.  With
    p.dvs_save_native the field before the tomography blur is also returned, as dVs_native.

    Each depth layer has zero area-weighted mean, like tomography models.
    Returns xr.Dataset with variables 'dVs' and 'cmb_heat' (heat) or 'pile_thickness_km'
    (pile) (dims: depth, lat, lon).
    """
    _check_llsvp(p)
    cmb_heat = llsvp_state if p.llsvp == 'heat' else None
    from scipy.ndimage import gaussian_filter1d

    coarse_deg, depth_interval_km = p.mantle_grid_spacing_deg, p.mantle_depth_interval_km
    fine = 0 < p.mantle_field_spacing_deg < coarse_deg
    spacing_deg = p.mantle_field_spacing_deg if fine else coarse_deg
    max_depth_km = p.cmb_depth_km
    lons_g, lats_g = mantle_grid(spacing_deg)
    lons_c, lats_c = mantle_grid(coarse_deg)
    depths_g = np.arange(depth_interval_km / 2, max_depth_km, depth_interval_km)
    nlon, nlat, ndepth = len(lons_g), len(lats_g), len(depths_g)
    field = np.zeros((ndepth, nlat, nlon), dtype=float)

    def _hsmooth(f, sig_deg):
        """Horizontal smoothing on the field grid; on a fine field grid, a wide kernel is applied
        on the coarse grid and interpolated back up (the result is smooth at that scale anyway)."""
        if fine and sig_deg >= _COARSE_PATH_FRACTION * coarse_deg:
            return _upsample(spherical_gaussian_smooth(_block_mean(f, lats_g, coarse_deg / spacing_deg),
                                                       lats_c, lons_c, sig_deg), lats_c, lons_c, lats_g, lons_g)
        return spherical_gaussian_smooth(f, lats_g, lons_g, sig_deg)

    def _smooth(f, sig_h_deg, sig_d_km):
        f = _hsmooth(f, sig_h_deg)
        if sig_d_km > 0:
            f = gaussian_filter1d(f, sig_d_km / depth_interval_km, axis=0, mode='nearest')
        return f

    if len(slab_tracers) > 0:
        active = slab_tracers[slab_tracers['active'].astype(bool)]
        if len(active) > 0:
            ages_f   = active['age'].values.astype(float)
            lons_f   = active['lon'].values.astype(float)
            lats_f   = active['lat'].values.astype(float)
            depths_f = active['depth_km'].values.astype(float)

            dT  = p.slab_temp_anomaly_K * thermal_fraction(active, p) * _slab_weights(active)
            dVs = dT * p.slab_dVs_per_K

            lon_idx = np.clip(
                np.floor((lons_f + 180) / spacing_deg).astype(int), 0, nlon - 1)
            lat_idx = np.clip(
                np.floor((lats_f + 90)  / spacing_deg).astype(int), 0, nlat - 1)
            dep_idx = np.clip(
                np.floor(depths_f / depth_interval_km).astype(int), 0, ndepth - 1)
            np.add.at(field, (dep_idx, lat_idx, lon_idx), dVs)
            field = _per_area(field, lats_g, p)
            if p.resolution_independent:              # a sum per cell: per 5 deg x 100 km cell
                field *= _cell_scale(spacing_deg, p, depth_interval_km)
            elif fine:                                # a sum per cell: per coarse-cell area, as on the model grid
                field *= (coarse_deg / spacing_deg) ** 2

            if p.slab_sigma_cmb_deg > 0:              # slab width profile: narrow above, widening below
                sig = slab_sigma_profile(depths_g, p)
                upper_g = depths_g < p.upper_mantle_km
                out = np.zeros_like(field)
                for part, sig_d in ((upper_g, p.slab_sigma_depth_upper_km if p.slab_sigma_depth_upper_km >= 0
                                     else p.slab_sigma_depth_km), (~upper_g, p.slab_sigma_depth_km)):
                    f = np.where(part[:, None, None], field, 0.0)
                    for s_k in np.unique(sig[part]):
                        k = part & (sig == s_k)
                        f[k] = _hsmooth(f[k], s_k)
                    if sig_d > 0:
                        f = gaussian_filter1d(f, sig_d / depth_interval_km, axis=0, mode='nearest')
                    out += f
                field = out
            elif p.slab_sigma_lower_deg > 0:            # wider slabs below the upper mantle
                lower_g = depths_g >= p.upper_mantle_km
                upper_f = np.where(lower_g[:, None, None], 0.0, field)
                lower_f = np.where(lower_g[:, None, None], field, 0.0)
                field = (_smooth(upper_f, p.slab_sigma_deg, p.slab_sigma_depth_km)
                         + _smooth(lower_f, p.slab_sigma_lower_deg, p.slab_sigma_depth_km))
            else:
                field = _smooth(field, p.slab_sigma_deg, p.slab_sigma_depth_km)

    if p.slab_dVs_max > 0:                     # slabs saturate at their coldest core's anomaly
        capped = depths_g < p.slab_dVs_max_depth_km
        field[capped] = p.slab_dVs_max * np.tanh(field[capped] / p.slab_dVs_max)

    if p.wedge_dVs < 0 and len(slab_tracers) > 0 and 'dip_azimuth_deg' in slab_tracers:
        w = slab_tracers[slab_tracers['active'].astype(bool).values
                         & (slab_tracers['depth_km'].astype(float) >= p.wedge_top_km).values
                         & (slab_tracers['depth_km'].astype(float) < min(p.wedge_bottom_km, p.upper_mantle_km)).values
                         & np.isfinite(slab_tracers['dip_azimuth_deg'].astype(float)).values]
        if len(w):
            dep = w['depth_km'].values.astype(float)
            lon_w, lat_w = move_along_azimuth(w['lon'].values.astype(float), w['lat'].values.astype(float),
                                              w['dip_azimuth_deg'].values.astype(float), p.wedge_offset_km,
                                              SURFACE_RADIUS_KM - dep)
            amount = np.abs(p.slab_temp_anomaly_K * p.slab_dVs_per_K) * thermal_fraction(w, p) * _slab_weights(w)
            wf = np.zeros_like(field)
            np.add.at(wf, (np.clip(np.floor(dep / depth_interval_km).astype(int), 0, ndepth - 1),
                           np.clip(np.floor((lat_w + 90) / spacing_deg).astype(int), 0, nlat - 1),
                           np.clip(np.floor((lon_w + 180) / spacing_deg).astype(int), 0, nlon - 1)), amount)
            wf = _per_area(wf, lats_g, p)
            if p.resolution_independent:
                wf *= _cell_scale(spacing_deg, p, depth_interval_km)
            elif fine:
                wf *= (coarse_deg / spacing_deg) ** 2
            upper = depths_g < p.upper_mantle_km
            wf[upper] = _hsmooth(wf[upper], p.wedge_sigma_deg)
            field += p.wedge_dVs * np.tanh(wf / p.wedge_saturation)

    if fine and cmb_heat is not None:
        cmb_heat = _upsample(cmb_heat, lats_c, lons_c, lats_g, lons_g)
    tbl_h = None
    if isinstance(llsvp_state, dict):
        llsvp_state, tbl_h = llsvp_state['pile'], llsvp_state['tbl']
    pile_h = llsvp_state if p.llsvp == 'pile' else None
    plumes = plume_sources(tbl_h if tbl_h is not None else pile_h, p) if p.plume_dVs < 0 and pile_h is not None else None
    if fine and pile_h is not None:
        pile_h = _upsample(pile_h, lats_c, lons_c, lats_g, lons_g)
    if fine and tbl_h is not None:
        tbl_h = _upsample(tbl_h, lats_c, lons_c, lats_g, lons_g)

    # Inject LLSVP-like warm anomaly from CMB heat accumulation.
    # Iterate deepest→shallowest; break once decay becomes negligible.
    if cmb_heat is not None and cmb_heat.shape == (nlat, nlon):
        _n_injected = 0
        for dep_i in range(ndepth - 1, -1, -1):
            dist_from_cmb = max_depth_km - depths_g[dep_i]
            decay = np.exp(-(dist_from_cmb / p.cmb_decay_km) ** 2)
            if decay < 1e-4:
                break
            field[dep_i] += cmb_heat * p.cmb_dVs_scale * decay
            _n_injected += 1
        print(f'  [mantle] LLSVP injected into {_n_injected} depth layers; '
              f'cmb_heat range {cmb_heat.min():.3f}–{cmb_heat.max():.3f}; '
              f'dVs at CMB: {field[-1].min():.4f}–{field[-1].max():.4f} %')
    elif cmb_heat is not None:
        print(f'  [mantle] WARNING: cmb_heat shape {cmb_heat.shape} != expected ({nlat},{nlon}) — LLSVP skipped')
    if pile_h is not None:
        field += p.pile_dVs * _pile.pile_layer_fractions(pile_h, depths_g, depth_interval_km, max_depth_km)
        if p.pile_halo_km > 0:                 # warm layer above the pile top
            top = max_depth_km - pile_h
            above = np.clip(top[None, :, :] - depths_g[:, None, None], 0.0, None)
            strength = np.clip(pile_h / p.pile_halo_h_km, 0.0, 1.0)
            field += np.where(depths_g[:, None, None] < top[None, :, :],
                              p.pile_halo_dVs * strength[None, :, :] * np.exp(-above / p.pile_halo_km), 0.0)
    if tbl_h is not None:                      # the boundary layer, on top of the pile
        both = pile_h + tbl_h
        field += p.tbl_dVs * (_pile.pile_layer_fractions(both, depths_g, depth_interval_km, max_depth_km)
                              - _pile.pile_layer_fractions(pile_h, depths_g, depth_interval_km, max_depth_km))
        if p.tbl_halo_km > 0:
            top = max_depth_km - both
            above = np.clip(top[None, :, :] - depths_g[:, None, None], 0.0, None)
            strength = np.clip(tbl_h / max(p.tbl_thickness_km, 1e-9), 0.0, 1.0)
            field += np.where(depths_g[:, None, None] < top[None, :, :],
                              p.tbl_dVs * strength[None, :, :] * np.exp(-above / p.tbl_halo_km), 0.0)
    if plumes is not None and len(plumes[0]):
        field += plume_field(plumes, lats_g, lons_g, depths_g, p)

    if p.lithosphere_dVs > 0 and lithosphere_km is not None:
        field += lithosphere_field(lithosphere_km, lats_g, lons_g, depths_g, depth_interval_km, p)

    field = _remove_area_weighted_mean(field, lats_g)
    native = None

    if p.tomo_smooth_horiz_deg > 0 or p.tomo_smooth_depth_km > 0:
        if p.dvs_save_native:
            native = field.astype(np.float32)
        field = _smooth(field, p.tomo_smooth_horiz_deg, p.tomo_smooth_depth_km)
        field = _remove_area_weighted_mean(field, lats_g)

    data_vars = {
        'dVs': xr.DataArray(
            field.astype(np.float32) if fine else field, dims=['depth', 'lat', 'lon'],
            coords={'depth': depths_g, 'lat': lats_g, 'lon': lons_g},
            attrs={'units': '%',
                   'long_name': 'dVs/Vs anomaly (positive = cold fast slab; negative = warm slow region)',
                   'temp_anomaly_K': p.slab_temp_anomaly_K,
                   'thermal_tau_myr': p.slab_thermal_tau_myr,
                   'dVs_per_K': p.slab_dVs_per_K}),
    }
    if native is not None:
        data_vars['dVs_native'] = xr.DataArray(
            native, dims=['depth', 'lat', 'lon'], coords={'depth': depths_g, 'lat': lats_g, 'lon': lons_g},
            attrs={'units': '%', 'long_name': 'dVs/Vs before the tomography blur (tomo_smooth_*)'})
    if cmb_heat is not None and cmb_heat.shape == (nlat, nlon):
        data_vars['cmb_heat'] = xr.DataArray(
            cmb_heat, dims=['lat', 'lon'],
            coords={'lat': lats_g, 'lon': lons_g},
            attrs={'units': 'heat units',
                   'long_name': 'CMB heat accumulation (proxy for LLSVP intensity)'})
    if tbl_h is not None:
        data_vars['tbl_thickness_km'] = xr.DataArray(
            tbl_h, dims=['lat', 'lon'], coords={'lat': lats_g, 'lon': lons_g},
            attrs={'units': 'km', 'long_name': 'thickness of the thin hot boundary layer on top of the pile'})
    if plumes is not None:
        data_vars['plume_lon'] = xr.DataArray(plumes[0], dims=['plume'])
        data_vars['plume_lat'] = xr.DataArray(plumes[1], dims=['plume'])
    if pile_h is not None:
        data_vars['pile_thickness_km'] = xr.DataArray(
            pile_h, dims=['lat', 'lon'], coords={'lat': lats_g, 'lon': lons_g},
            attrs={'units': 'km', 'long_name': 'thickness of the conserved pile on the CMB (llsvp: pile)'})
    return xr.Dataset(data_vars)
