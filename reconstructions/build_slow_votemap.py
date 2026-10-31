"""A slow-anomaly vote map of the lower mantle from several S-wave tomography models (LLSVP target).

    conda run -n pygmt17 python reconstructions/build_slow_votemap.py

Like the cluster/vote analyses of Lekic et al. (2012) and Shephard et al. (2017): each model, at each
depth, votes for the cells in its own slowest SLOW_FRACTION of the sphere (area-weighted), after block-
averaging onto the 5 deg mantle grid.  The count of votes (0..number of models) marks where the models
agree on slow material.  Models (GLAD-M25 is deliberately left out):
    REVEAL, SEMUCB-WM1      (Schouten et al. supplementary NetCDF)
    S40RTS, SMEAN, GypsumS  (GMT grids per depth; SMEAN is itself a composite of older models)
    S362ANI                 (text, absolute Vsv and Vsh; Voigt-average-like mean of the two, minus the layer mean)
Each model uses its depth nearest to each target depth (GypsumS stops at 2650 km, S362ANI is every 250 km).
--exclude REVEAL SEMUCB-WM1 --name slow_votes_S4 builds the map from the four models the tuning never saw.
Output: runs_real/votemaps/<name>.nc (default slow_votes_S), variables votes (depth, lat, lon) and minus_votes
(= -votes, so that "slow" is negative like dVs and the tomography scorer can read it), and n_models.
"""
import glob
import os
import re
import sys

import numpy as np
import xarray as xr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mantle.core import mantle_grid                          # noqa: E402
from reconstructions import tomography_score as ts            # noqa: E402

SRC = os.path.expanduser('~/Data/SeismicTomography')
DEPTHS = np.arange(1000.0, 2801.0, 100.0)
SLOW_FRACTION = 0.3
SPACING = 5.0


def block_average(values, lon, lat, lons_g, lats_g):
    """values (nlat, nlon) on a regular lon/lat grid -> mantle cells (cos-lat weights, NaN-aware)."""
    lon = ((np.asarray(lon, float) + 180.0) % 360.0) - 180.0
    _, first = np.unique(np.round(lon, 6), return_index=True)        # drop a duplicated 180 / -180 column
    vals, lon = values[:, first], lon[first]
    i = np.clip(np.floor((np.asarray(lat) + 90.0) / SPACING).astype(int), 0, len(lats_g) - 1)
    j = np.clip(np.floor((lon + 180.0) / SPACING).astype(int), 0, len(lons_g) - 1)
    w = np.cos(np.radians(lat))[:, None] * np.ones(len(lon))
    ok = np.isfinite(vals)
    acc, wsum = np.zeros((len(lats_g), len(lons_g))), np.zeros((len(lats_g), len(lons_g)))
    ii, jj = np.broadcast_to(i[:, None], w.shape).ravel(), np.broadcast_to(j[None, :], w.shape).ravel()
    np.add.at(acc, (ii, jj), np.where(ok, vals * w, 0.0).ravel())
    np.add.at(wsum, (ii, jj), np.where(ok, w, 0.0).ravel())
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(wsum > 0, acc / wsum, np.nan)


def _nearest(available, z):
    return min(available, key=lambda a: abs(a - z))


def per_depth_grids(pattern, regex):
    """{depth: path} for files matching a glob pattern, depth from the regex group."""
    out = {}
    for f in glob.glob(pattern):
        m = re.search(regex, os.path.basename(f))
        if m:
            out[float(m.group(1))] = f
    return out


def read_gmt_grid(path):
    ds = xr.open_dataset(path, engine='netcdf4') if path.endswith('.nc') else _open_grd(path)
    v = ds[list(ds.data_vars)[0]]
    lo, la = [c for c in v.dims if c.startswith(('lon', 'x'))][0], [c for c in v.dims if c.startswith(('lat', 'y'))][0]
    v = v.transpose(la, lo)
    return v.values.astype(float), v[lo].values, v[la].values


def _open_grd(path):
    for engine in (None, 'netcdf4', 'scipy', 'h5netcdf'):
        try:
            return xr.open_dataset(path, engine=engine) if engine else xr.open_dataset(path)
        except Exception:
            continue
    import pygmt
    g = pygmt.load_dataarray(path)
    return g.to_dataset(name='z')


def model_gridded(name, lons_g, lats_g):
    """(ndepth, nlat, nlon) on the mantle grid for DEPTHS."""
    out = np.full((len(DEPTHS), len(lats_g), len(lons_g)), np.nan)
    if name in ('REVEAL', 'SEMUCB-WM1'):
        f = {'REVEAL': 'REVEAL_downsampled_anomaly.nc', 'SEMUCB-WM1': 'SEMUCB_WM1_anomaly.nc'}[name]
        t, _ = ts.tomography_on_grid(os.path.join(SRC, 'Schouten_Supplementary_material', 'Models', f),
                                     'vs_anomaly', lons_g, lats_g, list(DEPTHS))
        return t
    if name == 'S362ANI':
        files = {}
        for f in glob.glob(os.path.join(SRC, 's362ani', 'grids', 'S362ANI.vsv.*.txt')):
            files[float(re.search(r'\.(\d+)\.txt$', f).group(1))] = f
        for k, z in enumerate(DEPTHS):
            d = _nearest(files, z)
            a = np.loadtxt(files[d])
            b = np.loadtxt(files[d].replace('.vsv.', '.vsh.'))
            v = 0.5 * (a[:, 2] + b[:, 2])
            lon_u, lat_u = np.unique(a[:, 0]), np.unique(a[:, 1])
            grid = np.full((len(lat_u), len(lon_u)), np.nan)
            grid[np.searchsorted(lat_u, a[:, 1]), np.searchsorted(lon_u, a[:, 0])] = v
            out[k] = block_average(grid, lon_u, lat_u, lons_g, lats_g)
        return out
    pattern, regex = {'S40RTS': (os.path.join(SRC, 's40rts', 's40rts_*.grd'), r'_(\d+)\.grd$'),
                      'SMEAN': (os.path.join(SRC, 'SMEAN', 'grd-files', 'smean_*.grd'), r'_(\d+)\.grd$'),
                      'GypsumS': (os.path.join(SRC, 'GypsumS', 'grd-files', 'GypsumS_*.grd'), r'_(\d+)\.grd$')}[name]
    files = per_depth_grids(pattern, regex)
    for k, z in enumerate(DEPTHS):
        v, lo, la = read_gmt_grid(files[_nearest(files, z)])
        out[k] = block_average(v, lo, la, lons_g, lats_g)
    return out


MODELS = ['REVEAL', 'SEMUCB-WM1', 'S40RTS', 'SMEAN', 'GypsumS', 'S362ANI']


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--exclude', nargs='*', default=[], help='models to leave out (e.g. the ones tuned against)')
    ap.add_argument('--name', default='slow_votes_S', help='output file name (without .nc)')
    args = ap.parse_args()
    models = [m for m in MODELS if m not in args.exclude]
    lons_g, lats_g = mantle_grid(SPACING)
    lon2, lat2 = np.meshgrid(lons_g, lats_g)
    w = np.cos(np.radians(lat2))
    votes = np.zeros((len(DEPTHS), len(lats_g), len(lons_g)))
    for name in models:
        f = model_gridded(name, lons_g, lats_g)
        for k in range(len(DEPTHS)):
            valid = np.isfinite(f[k])
            wk = w * valid
            votes[k] += ts._quantile_mask(np.where(valid, f[k] - (f[k][valid] * wk[valid]).sum() / wk.sum(), np.inf),
                                          wk, SLOW_FRACTION, True)
        print(name, 'done')
    ds = xr.Dataset({'votes': (('depth', 'lat', 'lon'), votes), 'minus_votes': (('depth', 'lat', 'lon'), -votes)},
                    coords={'depth': DEPTHS, 'lat': lats_g, 'lon': lons_g},
                    attrs={'models': ', '.join(models), 'n_models': len(models), 'slow_fraction': SLOW_FRACTION})
    out = os.path.join(REPO, 'runs_real', 'votemaps', args.name + '.nc')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.to_netcdf(out)
    for z in (1000, 2000, 2800):
        v = ds.votes.sel(depth=z).values
        k = len(models) // 2 + 1
        print(z, 'km: area with >= {} of {} votes: {:.2f}'.format(k, len(models), (w * (v >= k)).sum() / w.sum()))
    print('written', out)


if __name__ == '__main__':
    main()
