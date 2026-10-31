"""Score a real-reconstruction mantle run's present-day dVs against seismic tomography (shape only).

    conda run -n pygmt17 python reconstructions/tomography_score.py configs/real/muller2022.yaml

Reads <output_dir>/mantle_temp_0000Ma.nc of the run and the config's `tomography:` block
(one model, or a list of them: name, file, variable, and either depths_km, a list matched to
the nearest depth in the file, or min_depth_km / max_depth_km, all file depths in between).
--tomo NAME scores one of them only.  Both fields are compared on the run's lon/lat mantle grid:
  - the tomography is block-averaged into the mantle cells with cos(lat) weights (it must be
    on a regular lon/lat grid finer than the mantle grid);
  - the model is interpolated linearly in depth.
Amplitudes and units never enter: the model's dVs scaling is not calibrated.

Scores per depth (cells weighted by area, cos lat):
  slow_iou   intersection over union of the slowest slow_fraction of the sphere in each field
             (each field thresholded at its own area quantile, so the two regions have equal area)
  fast_iou   the same for the fastest fast_fraction
  r_2_3      correlation of spherical-harmonic degrees 2-3 (weighted least-squares fit, orthonormal)
  r_1_L      correlation of degrees 1..lmax
Two equal-area random regions of fraction q have an expected IoU of q / (2 - q).

Null: the model rotated about the spin axis in steps of one grid cell (all rotations).  A
rotation by whole cells moves the field exactly, as if every trench sample had been rotated.
For each score, `*_pct` is the share of rotations that score below the unrotated model (1.0 = the
real longitude beats every rotation); the rotation-averaged score is in `*_null_mean`.
Output: <output_dir>/tomography_score_<name>.csv (one row per depth) and a summary on screen.
This tests the method on real reconstructions; it is not a target for PlateDriver's synthetic worlds.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyshtools as pysh
import xarray as xr
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _coord(ds, *names):
    for n in names:
        if n in ds.coords:
            return n
    raise SystemExit('none of {} in {}'.format(names, list(ds.coords)))


def tomography_on_grid(path, variable, lons_g, lats_g, depths=None):
    """Tomography block-averaged (cos-lat weights) into the cells centred on lons_g x lats_g.
    NaN nodes are skipped; a cell with no finite node is NaN.
    Returns (field[depth, lat, lon], depths)."""
    ds = xr.open_dataset(path)
    la, lo, de = _coord(ds, 'latitude', 'lat'), _coord(ds, 'longitude', 'lon'), _coord(ds, 'depth')
    da = ds[variable]
    if depths is not None:
        da = da.sel({de: depths}, method='nearest')
    da = da.transpose(de, la, lo)
    lon_in = da[lo].values.astype(float)
    keep = np.ones(len(lon_in), bool)
    if np.isclose(lon_in, -180.0).any():
        keep &= ~np.isclose(lon_in, 180.0)          # 180 duplicates -180
    lon = ((lon_in + 180.0) % 360.0) - 180.0
    vals = da.values[:, :, keep].astype(float)
    lon, lat = lon[keep], da[la].values.astype(float)
    spacing = lons_g[1] - lons_g[0]
    i = np.clip(np.floor((lat + 90.0) / spacing).astype(int), 0, len(lats_g) - 1)
    j = np.clip(np.floor((lon + 180.0) / spacing).astype(int), 0, len(lons_g) - 1)
    w = np.cos(np.radians(lat))[:, None] * np.ones(len(lon))
    ii, jj = np.broadcast_to(i[:, None], w.shape).ravel(), np.broadcast_to(j[None, :], w.shape).ravel()
    out = np.zeros((vals.shape[0], len(lats_g), len(lons_g)))
    for k in range(vals.shape[0]):
        ok = np.isfinite(vals[k])
        acc, wsum = np.zeros(out.shape[1:]), np.zeros(out.shape[1:])
        np.add.at(acc, (ii, jj), np.where(ok, vals[k] * w, 0.0).ravel())
        np.add.at(wsum, (ii, jj), np.where(ok, w, 0.0).ravel())
        with np.errstate(invalid='ignore', divide='ignore'):
            out[k] = np.where(wsum > 0, acc / wsum, np.nan)
    return out, da[de].values.astype(float)


def _quantile_mask(f, w, q, slow):
    """Cells holding the lowest (slow) or highest area fraction q of f."""
    order = np.argsort(f.ravel() if slow else -f.ravel())
    cum = np.cumsum(w.ravel()[order]) / w.sum()
    mask = np.zeros(f.size, bool)
    mask[order[cum <= q]] = True
    return mask.reshape(f.shape)


def _iou(a, b, w):
    return (w * (a & b)).sum() / (w * (a | b)).sum()


def _coeffs(f, lat2, lon2, w, lmax):
    cilm, _ = pysh.expand.SHExpandWLSQ(f.ravel(), w.ravel(), lat2.ravel(), lon2.ravel(), lmax, norm=4)
    return cilm


def _corr(ca, cb, l0, l1):
    a, b = ca[:, l0:l1 + 1, :], cb[:, l0:l1 + 1, :]
    return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())


def score_one(m, t_slow, t_fast, ct, w, lat2, lon2, slow_fraction, fast_fraction, lmax):
    cm = _coeffs(m, lat2, lon2, w, lmax)
    return {'slow_iou': _iou(_quantile_mask(m, w, slow_fraction, True), t_slow, w),
            'fast_iou': _iou(_quantile_mask(m, w, fast_fraction, False), t_fast, w),
            'r_2_3': _corr(cm, ct, 2, 3),
            'r_1_L': _corr(cm, ct, 1, lmax)}


def tomography_depths(tc):
    """The file depths to score: nearest to each of tc['depths_km'], or all between
    min_depth_km and max_depth_km."""
    ds = xr.open_dataset(tc['file'])
    d = ds[_coord(ds, 'depth')].values.astype(float)
    if 'depths_km' in tc:
        return np.array([d[np.argmin(np.abs(d - z))] for z in tc['depths_km']])
    return d[(d >= float(tc.get('min_depth_km', 1000))) & (d <= float(tc.get('max_depth_km', 2800)))]


def score_depths(model, tomo, depths, lons_g, lats_g, slow_q, fast_q, lmax):
    """Scores (one row per depth) of model[depth, lat, lon] against tomo[depth, lat, lon], with the
    longitude-rotation null.  Either field can be a tomography model."""
    lon2, lat2 = np.meshgrid(lons_g, lats_g)
    w = np.cos(np.radians(lat2))
    rows = []
    for k, z in enumerate(depths):
        valid = np.isfinite(tomo[k]) & np.isfinite(model[k])   # cells without data carry no weight
        t, wk = np.where(valid, tomo[k], 0.0), w * valid
        m = np.where(valid, model[k], 0.0)
        t_slow, t_fast = _quantile_mask(t, wk, slow_q, True), _quantile_mask(t, wk, fast_q, False)
        ct = _coeffs(t, lat2, lon2, wk, lmax)
        null = pd.DataFrame([score_one(np.roll(m, sh, axis=1), t_slow, t_fast, ct, wk, lat2, lon2,
                                       slow_q, fast_q, lmax) for sh in range(len(lons_g))])
        row = {'depth_km': z, 'empty_cell_area': 1.0 - wk.sum() / w.sum()}
        for c in null.columns:
            row[c] = null[c].iloc[0]
            row[c + '_pct'] = (null[c].iloc[1:] < null[c].iloc[0]).mean()
            row[c + '_null_mean'] = null[c].iloc[1:].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df, label, slow_q, fast_q):
    depths = df.depth_km.values
    print('{}, {:.0f}-{:.0f} km ({} depths); random-region IoU: slow {:.3f}, fast {:.3f}'.format(
        label, depths.min(), depths.max(), len(depths), slow_q / (2 - slow_q), fast_q / (2 - fast_q)))
    cols = ['slow_iou', 'fast_iou', 'r_2_3', 'r_1_L']
    print('{:>12s} '.format('depths km') + ' '.join('{:>22s}'.format(c + ' (null, pct)') for c in cols))
    for lo, hi in [(depths.min(), 1950), (2000, depths.max())]:
        b = df[(df.depth_km >= lo) & (df.depth_km <= hi)]
        print('{:>5.0f}-{:<6.0f} '.format(lo, hi) + ' '.join(
            '{:>8.3f} ({:6.3f}, {:4.2f})'.format(b[c].mean(), b[c + '_null_mean'].mean(), b[c + '_pct'].mean())
            for c in cols))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    ap.add_argument('--age', type=float, default=0.0, help='model age to score (Ma; tomography is present day)')
    ap.add_argument('--tomo', help='score only the tomography model with this name')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    models = cfg['tomography'] if isinstance(cfg['tomography'], list) else [cfg['tomography']]
    if args.tomo:
        models = [tc for tc in models if tc.get('name') == args.tomo]
        if not models:
            raise SystemExit('no tomography model named ' + args.tomo)
    out_dir = os.path.join(REPO, cfg['output_dir'])
    run = xr.open_dataset(os.path.join(out_dir, 'mantle_temp_{:04.0f}Ma.nc'.format(args.age)))
    lons_g, lats_g = run.lon.values, run.lat.values

    for tc in models:
        slow_q, fast_q = float(tc.get('slow_fraction', 0.3)), float(tc.get('fast_fraction', 0.2))
        lmax = int(tc.get('lmax', 8))
        tomo, depths = tomography_on_grid(tc['file'], tc['variable'], lons_g, lats_g, tomography_depths(tc))
        model = run.dVs.interp(depth=depths).transpose('depth', 'lat', 'lon').values
        df = score_depths(model, tomo, depths, lons_g, lats_g, slow_q, fast_q, lmax)
        name = tc.get('name', os.path.splitext(os.path.basename(tc['file']))[0])
        path = os.path.join(out_dir, 'tomography_score_{}.csv'.format(name))
        df.to_csv(path, index=False, float_format='%.4f')
        print_summary(df, '{} vs {} at {:.0f} Ma'.format(cfg['name'], name, args.age), slow_q, fast_q)
        print('written', os.path.relpath(path, REPO))


if __name__ == '__main__':
    sys.exit(main())
