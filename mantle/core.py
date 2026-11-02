"""The simplified mantle model: slab tracers, CMB heat and a synthetic dVs field.

Shared by PlateDriver (run_SDP.py) and the real-reconstruction stream (reconstructions/).
This package never imports either of them, nor pygplates, geopandas or shapely.

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


def sink_slabs(slab_tracers, delta_time, p):
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
            slab_tracers.loc[lower, 'depth_km'] += (
                slab_tracers.loc[lower, 'sink_rate_km_per_myr'] / p.slab_slowing_factor
                * (1.0 + p.slab_lower_rate_gradient * frac) * delta_time)
        else:
            slab_tracers.loc[lower, 'depth_km'] += (
                slab_tracers.loc[lower, 'sink_rate_km_per_myr'] / p.slab_slowing_factor * delta_time)
        slab_tracers.loc[active, 'age'] += delta_time
        if p.slab_at_cmb == 'rest':
            slab_tracers['depth_km'] = np.minimum(slab_tracers['depth_km'].astype(float), p.cmb_depth_km)
            faded = np.exp(-slab_tracers['age'].astype(float) / p.slab_thermal_tau_myr) < SLAB_REST_MIN_FRACTION
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
    return slab_tracers


def spread_cmb_slabs(slab_tracers, delta_time, p):
    """Move slablets within p.slab_cmb_spread_layer_km of the CMB sideways, down the gradient of
    the basal flow potential ψ (mantle/pile.py flow_potential, from the slab load on the CMB): slab
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


def seed_slabs(slab_tracers, samples, delta_time, p):
    """Add one slablet at the surface for each trench sample.

    Sink rate: p.slab_sink_rate_factor × convergence, capped at p.slab_max_sink_rate.
    Weight: delta_time × min(1, convergence / 10 km/Myr), so the mantle signal per Myr
    of subduction does not depend on the time step and near-stalled trenches add little.
    """
    if samples is None or not len(samples['lon']):
        return slab_tracers
    convergence = np.asarray(samples['convergence_km_per_myr'], dtype=float)
    new = pd.DataFrame({
        'lon': samples['lon'], 'lat': samples['lat'], 'depth_km': 0.0, 'age': 0.0, 'active': True,
        'sink_rate_km_per_myr': np.minimum(convergence * p.slab_sink_rate_factor, p.slab_max_sink_rate),
        'weight': delta_time * np.minimum(1.0, convergence / SLAB_FULL_WEIGHT_CONVERGENCE_KM_PER_MYR),
        'attached': True})
    if _dipping(p) and 'downdip_azimuth_deg' in samples:
        for key, col in DIP_COLUMNS.items():
            new[col] = np.asarray(samples[key], dtype=float)
    return new if len(slab_tracers) == 0 else pd.concat([slab_tracers, new], ignore_index=True)


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
    # time for the slab to reach depth_km: upper mantle at rate, lower at rate / slowing_factor
    upper = min(depth_km, upper_mantle_km)
    t_end = upper / rate + max(0.0, depth_km - upper_mantle_km) * slowing_factor / rate
    n = np.ceil(t_end / delta_time).astype(int) + 1          # ages 0, dt, ..., >= t_end
    point = np.repeat(np.arange(len(s_lat)), n)
    age = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) * delta_time
    r = rate[point]
    t_um = upper_mantle_km / r
    depth = np.where(age <= t_um, r * age, upper_mantle_km + (age - t_um) * r / slowing_factor)
    out = pd.DataFrame({
        'lon': s_lon[point], 'lat': s_lat[point],
        'depth_km': np.minimum(depth, depth_km), 'age': age, 'active': True,
        'sink_rate_km_per_myr': r,
        'weight': delta_time * np.minimum(1.0, convergence[point] / SLAB_FULL_WEIGHT_CONVERGENCE_KM_PER_MYR),
        'attached': True})
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


def new_llsvp_state(p):
    """Start state of the LLSVP model chosen by p.llsvp."""
    _check_llsvp(p)
    if p.llsvp == 'heat':
        return new_cmb_heat(p)
    if p.llsvp == 'pile':
        return _pile.new_pile(p)
    return None


def update_llsvp(state, slab_tracers, delta_time, p):
    """One step of the LLSVP model chosen by p.llsvp."""
    _check_llsvp(p)
    if p.llsvp == 'heat':
        return update_cmb_heat(state, slab_tracers, delta_time, p)
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
                weights = weights * np.exp(-all_active['age'].values.astype(float) / p.slab_thermal_tau_myr)

            lon_idx = np.clip(np.floor((lons_f + 180) / spacing_deg).astype(int), 0, nlon - 1)
            lat_idx = np.clip(np.floor((lats_f + 90)  / spacing_deg).astype(int), 0, nlat - 1)
            suppress = np.zeros_like(cmb_heat)
            np.add.at(suppress, (lat_idx, lon_idx), weights)
            cmb_heat = np.maximum(cmb_heat - _per_area(suppress, lats_g, p), 0.0)

    cmb_heat = spherical_gaussian_smooth(cmb_heat, lats_g, lons_g, p.cmb_sigma_deg * np.sqrt(delta_time))

    return np.minimum(cmb_heat, p.cmb_max_heat)


def _remove_area_weighted_mean(field, lats):
    w = np.cos(np.radians(lats))[:, None] * np.ones(field.shape[-1])
    means = (field * w).sum(axis=(-2, -1)) / w.sum()
    return field - means[:, None, None]


def dvs_field(slab_tracers, llsvp_state, p):
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

    Each depth layer has zero area-weighted mean, like tomography models.
    Returns xr.Dataset with variables 'dVs' and 'cmb_heat' (heat) or 'pile_thickness_km'
    (pile) (dims: depth, lat, lon).
    """
    _check_llsvp(p)
    cmb_heat = llsvp_state if p.llsvp == 'heat' else None
    from scipy.ndimage import gaussian_filter1d

    spacing_deg, depth_interval_km = p.mantle_grid_spacing_deg, p.mantle_depth_interval_km
    max_depth_km = p.cmb_depth_km
    lons_g, lats_g = mantle_grid(spacing_deg)
    depths_g = np.arange(depth_interval_km / 2, max_depth_km, depth_interval_km)
    nlon, nlat, ndepth = len(lons_g), len(lats_g), len(depths_g)
    field = np.zeros((ndepth, nlat, nlon), dtype=float)

    def _smooth(f, sig_h_deg, sig_d_km):
        f = spherical_gaussian_smooth(f, lats_g, lons_g, sig_h_deg)
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

            dT  = p.slab_temp_anomaly_K * np.exp(-ages_f / p.slab_thermal_tau_myr) * _slab_weights(active)
            dVs = dT * p.slab_dVs_per_K

            lon_idx = np.clip(
                np.floor((lons_f + 180) / spacing_deg).astype(int), 0, nlon - 1)
            lat_idx = np.clip(
                np.floor((lats_f + 90)  / spacing_deg).astype(int), 0, nlat - 1)
            dep_idx = np.clip(
                np.floor(depths_f / depth_interval_km).astype(int), 0, ndepth - 1)
            np.add.at(field, (dep_idx, lat_idx, lon_idx), dVs)
            field = _per_area(field, lats_g, p)

            if p.slab_sigma_lower_deg > 0:            # wider slabs below the upper mantle
                lower_g = depths_g >= p.upper_mantle_km
                upper_f = np.where(lower_g[:, None, None], 0.0, field)
                lower_f = np.where(lower_g[:, None, None], field, 0.0)
                field = (_smooth(upper_f, p.slab_sigma_deg, p.slab_sigma_depth_km)
                         + _smooth(lower_f, p.slab_sigma_lower_deg, p.slab_sigma_depth_km))
            else:
                field = _smooth(field, p.slab_sigma_deg, p.slab_sigma_depth_km)

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
    if p.llsvp == 'pile' and llsvp_state is not None:
        field += p.pile_dVs * _pile.pile_layer_fractions(llsvp_state, depths_g, depth_interval_km, max_depth_km)
        if p.pile_halo_km > 0:                 # warm layer above the pile top
            top = max_depth_km - llsvp_state
            above = np.clip(top[None, :, :] - depths_g[:, None, None], 0.0, None)
            strength = np.clip(llsvp_state / p.pile_halo_h_km, 0.0, 1.0)
            field += np.where(depths_g[:, None, None] < top[None, :, :],
                              p.pile_halo_dVs * strength[None, :, :] * np.exp(-above / p.pile_halo_km), 0.0)

    field = _remove_area_weighted_mean(field, lats_g)

    if p.tomo_smooth_horiz_deg > 0 or p.tomo_smooth_depth_km > 0:
        field = _smooth(field, p.tomo_smooth_horiz_deg, p.tomo_smooth_depth_km)
        field = _remove_area_weighted_mean(field, lats_g)

    data_vars = {
        'dVs': xr.DataArray(
            field, dims=['depth', 'lat', 'lon'],
            coords={'depth': depths_g, 'lat': lats_g, 'lon': lons_g},
            attrs={'units': '%',
                   'long_name': 'dVs/Vs anomaly (positive = cold fast slab; negative = warm slow region)',
                   'temp_anomaly_K': p.slab_temp_anomaly_K,
                   'thermal_tau_myr': p.slab_thermal_tau_myr,
                   'dVs_per_K': p.slab_dVs_per_K}),
    }
    if cmb_heat is not None and cmb_heat.shape == (nlat, nlon):
        data_vars['cmb_heat'] = xr.DataArray(
            cmb_heat, dims=['lat', 'lon'],
            coords={'lat': lats_g, 'lon': lons_g},
            attrs={'units': 'heat units',
                   'long_name': 'CMB heat accumulation (proxy for LLSVP intensity)'})
    if p.llsvp == 'pile' and llsvp_state is not None:
        data_vars['pile_thickness_km'] = xr.DataArray(
            llsvp_state, dims=['lat', 'lon'], coords={'lat': lats_g, 'lon': lons_g},
            attrs={'units': 'km', 'long_name': 'thickness of the conserved pile on the CMB (llsvp: pile)'})
    return xr.Dataset(data_vars)
