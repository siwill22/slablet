"""The simplified mantle model: slab tracers, CMB heat and a synthetic dVs field.

Shared by PlateDriver (run_SDP.py) and the real-reconstruction stream (reconstructions/).
This package never imports either of them, nor pygplates, geopandas or shapely.

Inputs from a caller are trench samples, a dict of 1-D arrays:
    lon, lat                  degrees, in the mantle frame (slablets sink vertically there)
    convergence_km_per_myr    > 0 where the plates converge (km/Myr = mm/yr)
The caller builds them from its own boundaries and is responsible for the frame.

The state is the slab tracer DataFrame (columns SLAB_COLUMNS) and the 2-D cmb_heat array.
"""
import numpy as np
import pandas as pd
import xarray as xr

from .sphere import lonlat_to_xyz, spherical_gaussian_smooth

# Slablets from trenches converging at least this fast (km/Myr = mm/yr) get full weight;
# slower (near-stalled) boundaries contribute proportionally less.
SLAB_FULL_WEIGHT_CONVERGENCE_KM_PER_MYR = 10.0

SLAB_COLUMNS = ['lon', 'lat', 'depth_km', 'age', 'active', 'sink_rate_km_per_myr', 'weight', 'attached']


def empty_samples():
    return {'lon': np.zeros(0), 'lat': np.zeros(0), 'convergence_km_per_myr': np.zeros(0)}


def sink_slabs(slab_tracers, delta_time, p):
    """Sink existing slablets vertically (lon/lat fixed) for delta_time Myr.

    Rate: the slablet's own sink_rate_km_per_myr above p.upper_mantle_km, that rate
    divided by p.slab_slowing_factor below.  Slablets that reach p.cmb_depth_km are removed.
    Returns a new DataFrame.
    """
    slab_tracers = slab_tracers.copy()
    if len(slab_tracers) > 0:
        active = slab_tracers['active'].astype(bool)
        upper = active & (slab_tracers['depth_km'] < p.upper_mantle_km)
        lower = active & (slab_tracers['depth_km'] >= p.upper_mantle_km)
        slab_tracers.loc[upper, 'depth_km'] += (
            slab_tracers.loc[upper, 'sink_rate_km_per_myr'] * delta_time)
        slab_tracers.loc[lower, 'depth_km'] += (
            slab_tracers.loc[lower, 'sink_rate_km_per_myr'] / p.slab_slowing_factor * delta_time)
        slab_tracers.loc[active, 'age'] += delta_time
        slab_tracers = slab_tracers[slab_tracers['depth_km'] < p.cmb_depth_km].reset_index(drop=True)
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
    return pd.DataFrame({
        'lon': s_lon[point], 'lat': s_lat[point],
        'depth_km': np.minimum(depth, depth_km), 'age': age, 'active': True,
        'sink_rate_km_per_myr': r,
        'weight': delta_time * np.minimum(1.0, convergence[point] / SLAB_FULL_WEIGHT_CONVERGENCE_KM_PER_MYR),
        'attached': True})


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


def _slab_weights(slab_tracers):
    if 'weight' in slab_tracers.columns:
        return slab_tracers['weight'].values.astype(float)
    return np.ones(len(slab_tracers))


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

            lon_idx = np.clip(np.floor((lons_f + 180) / spacing_deg).astype(int), 0, nlon - 1)
            lat_idx = np.clip(np.floor((lats_f + 90)  / spacing_deg).astype(int), 0, nlat - 1)
            suppress = np.zeros_like(cmb_heat)
            np.add.at(suppress, (lat_idx, lon_idx), weights)
            cmb_heat = np.maximum(cmb_heat - suppress, 0.0)

    cmb_heat = spherical_gaussian_smooth(cmb_heat, lats_g, lons_g, p.cmb_sigma_deg * np.sqrt(delta_time))

    return np.minimum(cmb_heat, p.cmb_max_heat)


def _remove_area_weighted_mean(field, lats):
    w = np.cos(np.radians(lats))[:, None] * np.ones(field.shape[-1])
    means = (field * w).sum(axis=(-2, -1)) / w.sum()
    return field - means[:, None, None]


def dvs_field(slab_tracers, cmb_heat, p):
    """Bin slab tracers onto a 3D grid and return a dVs/Vs anomaly Dataset.

    Each slablet contributes a cold temperature anomaly (p.slab_temp_anomaly_K × its
    weight) that decays exponentially with age (p.slab_thermal_tau_myr).  The summed ΔT
    per grid cell is smoothed (spherical Gaussian horizontally, Gaussian in depth) and
    converted to dVs/Vs % (p.slab_dVs_per_K, default -0.001 %/K, so a fresh unit-weight
    slablet gives -0.5% per cell).

    If cmb_heat (a 2D nlat×nlon array from update_cmb_heat) is given, a warm
    negative-dVs anomaly is added in the deepest layers, decaying upward as
    exp(-(distance from CMB / p.cmb_decay_km)²).

    Each depth layer has zero area-weighted mean, like tomography models.
    Returns xr.Dataset with variables 'dVs' and optionally 'cmb_heat'
    (dims: depth, lat, lon).
    """
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
    return xr.Dataset(data_vars)
