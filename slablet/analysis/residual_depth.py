"""Residual depth: seafloor depth corrected for sediment load, against the depth a cooling law predicts from age.

    ds = residual_depth(AAD_REGION, law='gdh1')     # Dataset on the 0.1 deg nodes of the age grid
    cells = cell_means(ds['residual_m'], 1.0)      # cos(lat)-weighted means on 1 deg cell-centred cells
    pts = spot_measurements(AAD_REGION)            # Holdt et al. (2022) and Dunn (2026), crust-corrected points

residual = the law's depth at the seafloor age - the sediment-corrected depth, so a positive residual is seafloor
shallower than the law predicts.  The residual is made with the same cooling law as the model it is compared with
(decision 11 of docs/plans/2026-11-08-cross-repo-reconciliation.md), and from observed present-day ages, so that it
stays independent of the model (decision 13).  Inputs, all under $SLABLET_DATA (default ~/Data):

    age       Seton et al. (2020), AgeGrids/2020/age.2020.1.GeeK2007.6m.nc (0.1 deg, gridline registered)
    depth     SRTM15+ v2.7 at 6 arc minutes, GMT's earth_relief_06m (gridline: the same nodes as the age grid)
    sediment  GlobSed (Straume et al. 2019), SedThickness/GlobSed/GlobSed.nc (5 arc minutes), bilinear to the nodes

The sediment correction unloads the column isostatically: the corrected depth is water depth + s (rho_m - rho_s) /
(rho_m - rho_w), with rho_s the mean density of a column of thickness s whose porosity decays exponentially with
depth.  The lithology is pyBacktrack's 'average ocean floor sediment' (Kominz et al. 2011: grain density 2647 kg/m^3,
surface porosity 0.66, decay length 1333 m), with mantle 3330 and seawater 1030 kg/m^3, the values pyBacktrack's
backstripping uses.  There is no correction for crustal thickness, so seafloor on thin crust (the AAD's, for one)
reads deeper than a crust-corrected spot measurement there.  Plateaus, seamounts and margins are not masked; only
nodes with an age, a sediment thickness and water above them are kept.

All operations are pointwise on the age grid's nodes, except cell_means, which weights each node by cos(latitude)
so that a cell's mean is an area mean on the sphere.
"""
import os

import numpy as np
import pandas as pd
import xarray as xr

from slablet import cooling

AAD_REGION = (90.0, 160.0, -65.0, -35.0)     # west, east, south, north: the SEIR from Kerguelen to Macquarie
GRAIN_DENSITY = 2647.0
SURFACE_POROSITY = 0.66
POROSITY_DECAY_M = 1333.0
MANTLE_DENSITY = 3330.0
WATER_DENSITY = 1030.0


def data_root():
    from slablet.paths import DATA
    return DATA


def _paths():
    root = data_root()
    return {'age': os.path.join(root, 'AgeGrids', '2020', 'age.2020.1.GeeK2007.6m.nc'),
            'sediment': os.path.join(root, 'SedThickness', 'GlobSed', 'GlobSed.nc'),
            'spreading_rate': os.path.join(root, 'AgeGrids', '2020', 'full_rate.2020.1.GeeK2007.6m.nc'),
            'holdt2022': os.path.join(root, 'residual_topography', 'holdt2022',
                                      'Residual depthtopographic measurements',
                                      'Holdt_et_al_2022_SupportingInformation', 'oceanic_measurements_circles_1_deg.xyz'),
            'dunn2026': os.path.join(root, 'residual_topography', 'dunn2026', 'ResidualDepths_SouthernOcean.csv')}


def _check_region(region):
    w, e, s, n = region
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError('region {} must be west < east within -180..180 and south < north (one that crosses '
                         'the dateline is not supported yet)'.format(region))


def _grid(path, region):
    da = xr.open_dataarray(path) if len(xr.open_dataset(path).data_vars) == 1 else xr.open_dataset(path)['z']
    da = da.rename({d: {'x': 'lon', 'y': 'lat'}.get(d, d) for d in da.dims})
    w, e, s, n = region
    return da.sel(lon=slice(w, e), lat=slice(s, n)).astype(float)


def sediment_correction_m(sediment_m):
    """How much deeper the basement would sit with the sediment replaced by water (m, >= 0)."""
    s = np.asarray(sediment_m, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_porosity = np.where(s > 0, SURFACE_POROSITY * POROSITY_DECAY_M / s * (1 - np.exp(-s / POROSITY_DECAY_M)),
                                 SURFACE_POROSITY)
    rho_s = mean_porosity * WATER_DENSITY + (1 - mean_porosity) * GRAIN_DENSITY
    return s * (MANTLE_DENSITY - rho_s) / (MANTLE_DENSITY - WATER_DENSITY)


def observations(region=AAD_REGION):
    """Seton (2020) age, SRTM15+ water depth and GlobSed sediment on the age grid's 0.1 deg nodes in region."""
    import pygmt
    _check_region(region)
    p = _paths()
    age = _grid(p['age'], region)
    relief = pygmt.datasets.load_earth_relief(resolution='06m', region=list(region), registration='gridline')
    relief = relief.rename({'lon': 'lon', 'lat': 'lat'}).astype(float)
    if relief.shape != age.shape or not (np.allclose(relief.lon, age.lon) and np.allclose(relief.lat, age.lat)):
        raise ValueError('earth_relief_06m and the age grid do not share nodes in {}'.format(region))
    sed = _grid(p['sediment'], (region[0] - 0.2, region[1] + 0.2, region[2] - 0.2, region[3] + 0.2))
    sed = sed.interp(lon=age.lon, lat=age.lat)
    ds = xr.Dataset({'age_myr': age, 'water_depth_m': -relief.assign_coords(lon=age.lon, lat=age.lat),
                     'sediment_m': sed})
    ds.attrs.update(age='Seton et al. (2020)', depth=relief.attrs.get('title', 'earth_relief_06m'),
                    sediment='GlobSed (Straume et al. 2019)')
    return ds


def residual_depth(region=AAD_REGION, law=None, obs=None):
    """Residual depth (m) on the age grid's nodes in region, made with one cooling law (default GDH1).
    Pass obs (from observations()) to make several laws' residuals from one read of the inputs."""
    law = cooling.get(law) if law is None or isinstance(law, str) else law
    ds = observations(region) if obs is None else obs.copy()
    ok = np.isfinite(ds.age_myr) & np.isfinite(ds.sediment_m) & (ds.water_depth_m > 0)
    corrected = ds.water_depth_m + sediment_correction_m(ds.sediment_m.values)
    predicted = xr.zeros_like(ds.age_myr)
    predicted.values[ok.values] = cooling.depth_m(law, ds.age_myr.values[ok.values])
    ds['corrected_depth_m'] = corrected.where(ok)
    ds['predicted_depth_m'] = predicted.where(ok)
    ds['residual_m'] = (predicted - corrected).where(ok)
    ds.attrs['cooling_law'] = law.name
    return ds


def cell_means(field, spacing_deg=1.0, min_fraction=0.5):
    """cos(lat)-weighted means of a node field on cell-centred cells (centres at half-spacings, as the plate-history
    contract's grids).  A cell is NaN unless at least min_fraction of its nodes have values."""
    lon, lat = field.lon.values, field.lat.values
    lo0 = np.floor(lon.min() / spacing_deg) * spacing_deg
    la0 = np.floor(lat.min() / spacing_deg) * spacing_deg
    ix = np.floor((lon - lo0) / spacing_deg + 1e-9).astype(int)
    iy = np.floor((lat - la0) / spacing_deg + 1e-9).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1
    w = np.cos(np.radians(lat))[:, None] * np.ones(len(lon))[None, :]
    v = field.values
    have = np.isfinite(v)
    flat = (iy[:, None] * nx + ix[None, :]).ravel()
    num = np.bincount(flat, weights=np.where(have, v * w, 0).ravel(), minlength=nx * ny)
    den = np.bincount(flat, weights=np.where(have, w, 0).ravel(), minlength=nx * ny)
    cnt = np.bincount(flat, weights=have.ravel().astype(float), minlength=nx * ny)
    tot = np.bincount(flat, minlength=nx * ny)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(cnt >= min_fraction * tot, num / den, np.nan).reshape(ny, nx)
    out = xr.DataArray(mean, dims=('lat', 'lon'), coords={'lat': la0 + spacing_deg * (np.arange(ny) + 0.5),
                                                          'lon': lo0 + spacing_deg * (np.arange(nx) + 0.5)})
    # drop the partial cells beyond the field's edges (the last gridline node starts a cell of its own)
    return out.sel(lon=slice(lon.min(), lon.max()), lat=slice(lat.min(), lat.max()))


def spot_measurements(region=AAD_REGION):
    """Crust-corrected residual-depth points (m): Holdt et al. (2022) circles and Dunn (2026) points marked 'c'.
    Both are measured against the plate model of Richards et al. (2020), i.e. cooling law 'rhcw18'."""
    p = _paths()
    h = pd.read_csv(p['holdt2022'], sep=r'\s+', header=None, names=['lon', 'lat', 'residual_km', 'err_lo', 'err_hi'])
    h = h.assign(residual_m=1e3 * h.residual_km, error_m=1e3 * h.err_hi, source='Holdt et al. (2022)')
    d = pd.read_csv(p['dunn2026'])
    d = d[d.Plotting_Symbol == 'c']
    d = pd.DataFrame({'lon': d.Longitude, 'lat': d.Latitude, 'residual_m': 1e3 * d['Residual_Depth_(km)'],
                      'error_m': 500.0 * (d['Maximum_Residual_Depth_(km)'] - d['Minimum_Residual_Depth_(km)']),
                      'source': 'Dunn (2026)'})
    pts = pd.concat([h[['lon', 'lat', 'residual_m', 'error_m', 'source']], d], ignore_index=True)
    pts['lon'] = (pts.lon + 180.0) % 360.0 - 180.0
    w, e, s, n = region
    return pts[(pts.lon >= w) & (pts.lon <= e) & (pts.lat >= s) & (pts.lat <= n)].reset_index(drop=True)


def sample(field, lon, lat):
    """Bilinear values of a node field at points (NaN where any surrounding node is NaN)."""
    return field.interp(lon=xr.DataArray(np.asarray(lon), dims='p'), lat=xr.DataArray(np.asarray(lat), dims='p')).values
