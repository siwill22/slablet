"""Diagnostic: which fixed rotation of a run's present-day dVs fits the tomography best, per depth band?

    conda run -n pygmt17 python -m slablet.frame.best_rotation configs/muller2022_dip_pile.yaml [--out f.csv]

A fixed rotation of the whole model would also move today's trenches, so it is not a candidate
frame correction (see frame_rotation in run_mantle.py for age-dependent ones).  It shows which way,
and about which axis, the model's deep structure wants to move to match the tomography: a rotation
about the spin axis (longitude), about an equatorial axis (latitude, like true polar wander), or none.

For each depth, the model and each tomography model are expanded in spherical harmonics up to
degree 8 (weighted least squares, orthonormal, as in tomography_score.py).  A rotation R moves the
model to g(x) = f(R^T x), evaluated from its expansion at the rotated grid points.  Scores are
area-weighted correlations on the mantle grid of the degree 2-3 parts (r_2_3) and the degree 1-8
parts (r_1_L) of g and the tomography.  Rotations: axes on a Fibonacci sphere (N_AXES, covering both
senses) x ANGLES_DEG, plus no rotation.  Output: one row per rotation, tomography model and band
(mid: < 1975 km, deep: >= 1975 km), mean over the band's depths.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyshtools as pysh
import xarray as xr
import yaml

from slablet.paths import WORK

from slablet.sphere import lonlat_to_xyz, xyz_to_lonlat           # noqa: E402
from slablet.analysis import tomography_score as ts                # noqa: E402
from slablet.gplates.run_mantle import rotation_matrix            # noqa: E402

N_AXES = 400
ANGLES_DEG = [5, 10, 15, 20, 30]
LMAX = 8
MID_DEEP_KM = 1975.0


def fibonacci_axes(n):
    k = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1 - 2 * k / n))
    lon = (np.degrees(np.pi * (1 + 5 ** 0.5) * k) + 180.0) % 360.0 - 180.0
    return lat, lon


def _band(c, lo, hi):
    out = np.zeros_like(c)
    out[:, lo:hi + 1, :] = c[:, lo:hi + 1, :]
    return out


def _corr(a, b, w):
    a = a - (a * w).sum() / w.sum()
    b = b - (b * w).sum() / w.sum()
    return (a * b * w).sum() / np.sqrt((a * a * w).sum() * (b * b * w).sum())


_STATE = {}


def _init(state):
    _STATE.update(state)


def _score_rotations(rots):
    s = _STATE
    rows = []
    for axis_lat, axis_lon, angle in rots:
        R = rotation_matrix(axis_lat, axis_lon, angle)
        lon_r, lat_r = xyz_to_lonlat(s['xyz'] @ R)          # R^T x for each grid point (row vectors)
        for k, z in enumerate(s['depths']):
            g23 = pysh.expand.MakeGridPoint(s['m23'][k], lat_r, lon_r, norm=4)
            g18 = pysh.expand.MakeGridPoint(s['m18'][k], lat_r, lon_r, norm=4)
            for name in s['tomo_names']:
                rows.append({'axis_lat': axis_lat, 'axis_lon': axis_lon, 'angle_deg': angle, 'tomo': name,
                             'depth_km': z, 'r_2_3': _corr(g23, s['t23'][name][k], s['w']),
                             'r_1_L': _corr(g18, s['t18'][name][k], s['w'])})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    ap.add_argument('--out')
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run = xr.open_dataset(os.path.join(WORK, cfg['output_dir'], 'mantle_temp_0000Ma.nc'))
    lons_g, lats_g = run.lon.values, run.lat.values
    lon2, lat2 = np.meshgrid(lons_g, lats_g)
    w = np.cos(np.radians(lat2)).ravel()
    depths = ts.tomography_depths(cfg['tomography'][0])
    model = run.dVs.interp(depth=depths).transpose('depth', 'lat', 'lon').values
    ones = np.cos(np.radians(lat2))
    mc = [ts._coeffs(model[k], lat2, lon2, ones, LMAX) for k in range(len(depths))]
    state = {'xyz': lonlat_to_xyz(lon2.ravel(), lat2.ravel()), 'depths': depths, 'w': w,
             'm23': [_band(c, 2, 3) for c in mc], 'm18': [_band(c, 1, LMAX) for c in mc],
             't23': {}, 't18': {}, 'tomo_names': []}
    for tc in cfg['tomography']:
        t, _ = ts.tomography_on_grid(tc['file'], tc['variable'], lons_g, lats_g, depths)
        tcs = []
        for k in range(len(depths)):
            valid = np.isfinite(t[k])
            tcs.append(ts._coeffs(np.where(valid, t[k], 0.0), lat2, lon2, ones * valid, LMAX))
        name = tc['name']
        state['tomo_names'].append(name)
        state['t23'][name] = [pysh.expand.MakeGridPoint(_band(c, 2, 3), lat2.ravel(), lon2.ravel(), norm=4) for c in tcs]
        state['t18'][name] = [pysh.expand.MakeGridPoint(_band(c, 1, LMAX), lat2.ravel(), lon2.ravel(), norm=4) for c in tcs]
    a_lat, a_lon = fibonacci_axes(N_AXES)
    rots = [(90.0, 0.0, 0.0)] + [(la, lo, an) for la, lo in zip(a_lat, a_lon) for an in ANGLES_DEG]
    chunks = [rots[i::args.workers * 4] for i in range(args.workers * 4)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(state,)) as ex:
        rows = [r for part in ex.map(_score_rotations, chunks) for r in part]
    df = pd.DataFrame(rows)
    df['band'] = np.where(df.depth_km < MID_DEEP_KM, 'mid', 'deep')
    out = df.groupby(['axis_lat', 'axis_lon', 'angle_deg', 'tomo', 'band'], as_index=False)[['r_2_3', 'r_1_L']].mean()
    path = args.out or os.path.join(WORK, cfg['output_dir'], 'best_rotation.csv')
    out.to_csv(path, index=False, float_format='%.4f')
    for band in ('mid', 'deep'):
        b = out[out.band == band].groupby(['axis_lat', 'axis_lon', 'angle_deg'], as_index=False)[['r_2_3', 'r_1_L']].mean()
        none = b[b.angle_deg == 0].iloc[0]
        best = b.sort_values('r_2_3', ascending=False).head(5)
        print('{}: no rotation r_2_3 {:.3f}, r_1_L {:.3f}; best by r_2_3:'.format(band, none.r_2_3, none.r_1_L))
        print(best.round(3).to_string(index=False))
    print('written', path)


if __name__ == '__main__':
    main()
