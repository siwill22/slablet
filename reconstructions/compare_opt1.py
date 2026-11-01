"""Compare mantle runs with the Müller et al. (2022) OPT1 convection model at low spherical-harmonic degree.

    conda run -n pygmt17 python reconstructions/compare_opt1.py \\
        runs_real/muller2022_dip_pile_1000 runs_real/muller2022_dip_slabs_1000 [--workers 10]

OPT1 (Müller et al. 2022, Solid Earth 13, 1127-1159; Zenodo 6622194): temperature anomaly grids, 0.25 deg,
0-1000 Ma every 20 Myr, 65 depths, mean removed at each depth, in K (non-dimensional x 3100).  Same Müller
2022 optimised mantle frame as the runs, so positions compare directly.

Step 1 (cached in runs_real/opt1/opt1_grid5.nc): OPT1 block-averaged (cos-lat weights) onto the 5 deg mantle
grid at each run depth (the nearest OPT1 depth, which is within 50 km), for every 20 Myr.
Step 2, for each run, at every common age and depth, comparing dVs with -T (hot = slow; scale-free):
    r_<l>      correlation of degree l alone (l = 1..LMAX), and r_1_8 over degrees 1-8
    pow_<l>    power of degree l (orthonormal harmonics), of the run (run_pow_) and OPT1 (opt1_pow_)
    slow_iou   IoU of the run's slowest 30 % and OPT1's hottest 30 % (area-weighted); fast_iou: 20 %, cold
    d1_angle   angle (deg) between the degree-1 hot directions
    d2_angle   angle (deg, 0-90) between the degree-2 hot axes: the eigenvector with the largest eigenvalue
               of the degree-2 tensor of the slow field (-dVs) and of T; d2_prolate = (l1 - l2) / (l1 - l3) of
               each (1 = two antipodal hot regions, 0 = a hot ring), so the axis is meaningful when that is high
Step 3 (spectra.csv): full-resolution OPT1 power spectra to degree 60 (Driscoll-Healy expansion of the
0.25 deg grid) at every 100 Myr and four depths, next to OPT1 on the 5 deg grid and each run, to show
which degrees the 5 deg grid and the run's smoothing can carry.
Outputs in runs_real/opt1/: opt1_grid5.nc, compare_<run>.csv, spectra.csv.  All on a sphere.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyshtools as pysh
import xarray as xr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mantle.core import mantle_grid                                 # noqa: E402
from reconstructions import tomography_score as ts                  # noqa: E402
from reconstructions.build_slow_votemap import block_average        # noqa: E402

SRC = os.path.expanduser('~/Data/zenodo/OPT1_temperature_anomaly_grids_dimensional')
OUT = os.path.join(REPO, 'runs_real', 'opt1')
AGES = np.arange(0, 1001, 20)
DEPTHS = np.arange(50.0, 2851.0, 100.0)                            # the run depths
LMAX = 12
SPECTRA_AGES = np.arange(0, 1001, 100)
SPECTRA_DEPTHS = [650.0, 1450.0, 2250.0, 2750.0]
LONS, LATS = mantle_grid(5.0)
LON2, LAT2 = np.meshgrid(LONS, LATS)
W = np.cos(np.radians(LAT2))


def opt1_depths():
    names = os.listdir(SRC)
    return sorted({int(n.split('Ma-')[1][:4]) for n in names if n.startswith('OPT1-temp-000Ma-') and n.endswith('Dim.grd')})


def opt1_file(age, depth_km):
    return os.path.join(SRC, 'OPT1-temp-{:03d}Ma-{:04d}km_mean_removed_Dim.grd'.format(int(age), int(depth_km)))


def read_opt1(age, depth_km):
    ds = xr.open_dataset(opt1_file(age, depth_km))
    return ds.z.values.astype(float), ds.lon.values, ds.lat.values


def _grid_age(args):
    age, pairs = args
    out = np.zeros((len(pairs), len(LATS), len(LONS)))
    for k, (_, d) in enumerate(pairs):
        v, lo, la = read_opt1(age, d)
        out[k] = block_average(v, lo, la, LONS, LATS)
    return age, out


def build_grid(workers):
    path = os.path.join(OUT, 'opt1_grid5.nc')
    if os.path.exists(path):
        return xr.open_dataset(path)
    avail = np.array(opt1_depths(), float)
    pairs = [(z, int(avail[np.argmin(np.abs(avail - z))])) for z in DEPTHS]
    T = np.zeros((len(AGES), len(DEPTHS), len(LATS), len(LONS)))
    with ProcessPoolExecutor(workers) as ex:
        for age, f in ex.map(_grid_age, [(a, pairs) for a in AGES]):
            T[list(AGES).index(age)] = f
            print('OPT1 gridded at', age, 'Ma', flush=True)
    ds = xr.Dataset({'T': (('age', 'depth', 'lat', 'lon'), T)},
                    coords={'age': AGES, 'depth': DEPTHS, 'lat': LATS, 'lon': LONS,
                            'opt1_depth': ('depth', [p[1] for p in pairs])},
                    attrs={'source': SRC, 'units': 'K (mean removed per depth)',
                           'method': 'cos-lat weighted block average of the 0.25 deg grids onto 5 deg cells'})
    os.makedirs(OUT, exist_ok=True)
    ds.to_netcdf(path)
    return ds


# ---------- low-degree measures ----------
_XYZ = np.stack([np.cos(np.radians(LAT2)) * np.cos(np.radians(LON2)),
                 np.cos(np.radians(LAT2)) * np.sin(np.radians(LON2)), np.sin(np.radians(LAT2))], -1)
_X, _Y, _Z = _XYZ[..., 0], _XYZ[..., 1], _XYZ[..., 2]
_BASIS = np.stack([np.ones_like(_X), _X, _Y, _Z, _X * _Y, _X * _Z, _Y * _Z, _X ** 2 - _Y ** 2, 3 * _Z ** 2 - 1], -1)


def low_degree(f):
    """Degree-1 vector and degree-2 tensor of f (weighted least squares on the grid)."""
    a = _BASIS.reshape(-1, 9) * np.sqrt(W.ravel())[:, None]
    c, *_ = np.linalg.lstsq(a, f.ravel() * np.sqrt(W.ravel()), rcond=None)
    d1 = c[1:4]
    cxy, cxz, cyz, cd, cz = c[4:]
    q = np.array([[cd - cz, cxy / 2, cxz / 2], [cxy / 2, -cd - cz, cyz / 2], [cxz / 2, cyz / 2, 2 * cz]])
    lam, vec = np.linalg.eigh(q)                                    # ascending
    return d1, lam, vec[:, -1]


def _angle(a, b, axis=False):
    c = np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(abs(c) if axis else c, -1, 1))))


def _lonlat(v):
    v = v if v[0] >= 0 else -v                                      # report the end facing 0 deg E
    return float(np.degrees(np.arctan2(v[1], v[0]))), float(np.degrees(np.arcsin(v[2] / np.linalg.norm(v))))


def degree_power(c):
    return np.array([(c[:, l, :l + 1] ** 2).sum() for l in range(c.shape[1])])


def compare(run_dir, grid):
    rows = []
    for age in grid.age.values:
        f = os.path.join(run_dir, 'mantle_temp_{:04d}Ma.nc'.format(int(age)))
        if not os.path.exists(f):
            continue
        run = xr.open_dataset(f).dVs.transpose('depth', 'lat', 'lon')
        for z in DEPTHS:
            slow = -run.sel(depth=z).values                         # hot / slow positive
            t = grid.T.sel(age=age, depth=z).values
            cr = ts._coeffs(slow, LAT2, LON2, W, LMAX)
            ct = ts._coeffs(t, LAT2, LON2, W, LMAX)
            row = {'age_ma': int(age), 'depth_km': z, 'opt1_depth_km': int(grid.opt1_depth.sel(depth=z))}
            for l in range(1, LMAX + 1):
                row['r_{}'.format(l)] = ts._corr(cr, ct, l, l)
            row['r_1_8'] = ts._corr(cr, ct, 1, 8)
            row['r_2_3'] = ts._corr(cr, ct, 2, 3)
            pr, pt = degree_power(cr), degree_power(ct)
            for l in range(1, LMAX + 1):
                row['run_pow_{}'.format(l)], row['opt1_pow_{}'.format(l)] = pr[l], pt[l]
            row['slow_iou'] = ts._iou(ts._quantile_mask(-slow, W, 0.3, True), ts._quantile_mask(-t, W, 0.3, True), W)
            row['fast_iou'] = ts._iou(ts._quantile_mask(-slow, W, 0.2, False), ts._quantile_mask(-t, W, 0.2, False), W)
            d1r, lr, ar = low_degree(slow)
            d1t, lt, at = low_degree(t)
            row['d1_angle'] = _angle(d1r, d1t)
            row['d2_angle'] = _angle(ar, at, axis=True)
            row['run_d2_lon'], row['run_d2_lat'] = _lonlat(ar)
            row['opt1_d2_lon'], row['opt1_d2_lat'] = _lonlat(at)
            row['run_d2_prolate'] = (lr[2] - lr[1]) / (lr[2] - lr[0])
            row['opt1_d2_prolate'] = (lt[2] - lt[1]) / (lt[2] - lt[0])
            rows.append(row)
    return pd.DataFrame(rows)


def _native(args):
    age, d, z = args
    v, lo, la = read_opt1(age, d)
    g = v[::-1][:-1, :-1]                                           # Driscoll-Healy: 90 .. -89.75, 0 .. 359.75
    c = pysh.expand.SHExpandDH(g, sampling=2, lmax_calc=60, norm=4)
    return age, z, degree_power(c)


def spectra(grid, runs, workers):
    avail = np.array(opt1_depths(), float)
    jobs = [(a, int(avail[np.argmin(np.abs(avail - z))]), z) for a in SPECTRA_AGES for z in SPECTRA_DEPTHS]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for age, z, p in ex.map(_native, jobs):
            for l in range(1, 61):
                rows.append({'age_ma': age, 'depth_km': z, 'source': 'opt1_native', 'l': l, 'power': p[l]})
            p5 = degree_power(ts._coeffs(grid.T.sel(age=age, depth=z).values, LAT2, LON2, W, LMAX))
            rows += [{'age_ma': age, 'depth_km': z, 'source': 'opt1_grid5', 'l': l, 'power': p5[l]}
                     for l in range(1, LMAX + 1)]
            for name, d in runs.items():
                f = os.path.join(d, 'mantle_temp_{:04d}Ma.nc'.format(int(age)))
                if os.path.exists(f):
                    m = xr.open_dataset(f).dVs.sel(depth=z).values
                    pr = degree_power(ts._coeffs(-m, LAT2, LON2, W, LMAX))
                    rows += [{'age_ma': age, 'depth_km': z, 'source': name, 'l': l, 'power': pr[l]}
                             for l in range(1, LMAX + 1)]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    grid = build_grid(args.workers)
    runs = {os.path.basename(os.path.normpath(r)): os.path.join(REPO, r) for r in args.runs}
    for name, d in runs.items():
        df = compare(d, grid)
        df.to_csv(os.path.join(OUT, 'compare_{}.csv'.format(name)), index=False, float_format='%.5g')
        deep = df[df.depth_km >= 2000]
        print('{}: {} ages; deep band (2050-2850 km) mean r_1 {:.2f}, r_2 {:.2f}, r_1_8 {:.2f}, slow IoU {:.2f}'.format(
            name, df.age_ma.nunique(), deep.r_1.mean(), deep.r_2.mean(), deep.r_1_8.mean(), deep.slow_iou.mean()))
    spectra(grid, runs, args.workers).to_csv(os.path.join(OUT, 'spectra.csv'), index=False, float_format='%.5g')
    print('written', OUT)


if __name__ == '__main__':
    main()
