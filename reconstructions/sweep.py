"""Run many variants of a real-reconstruction mantle config in parallel and score each one.

    conda run -n pygmt17 python reconstructions/sweep.py sweeps/s01.yaml [--workers 10]

A sweep file:
    name: s01_one_at_a_time
    base: configs/real/muller2022.yaml        # the config every variant starts from
    common: {output_every_myr: 100000}         # merged into every variant (dicts merge key by key)
    variants:
      base: {}
      tau_50: {mantle: {slab_thermal_tau_myr: 50}}
      start_1000: {start_ma: 1000}

Each variant runs into runs_real/sweeps/<name>/<variant>/ and is scored against every
tomography model of the config (reconstructions/tomography_score.py).  One row per variant
goes to runs_real/sweeps/<name>/results.csv and is appended to runs_real/sweeps/all_results.csv:
the overrides, run time, and for each tomography model and depth band (mid: < 1975 km,
deep: >= 1975 km) the mean slow_iou, fast_iou, r_2_3, r_1_L and the rotation-null percentiles
of slow_iou and r_2_3.  The shape scores ignore amplitude, so a row also has the model's
deep / mid ratio of the area-weighted standard deviation of dVs (amp_deep_mid; REVEAL and
SEMUCB-WM1 both have about 1.5): a nearly empty deep mantle can still correlate well.  With
llsvp: pile it also has the pile's size: the fraction of the CMB where it is over 100 km thick
(pile_area_100km), its maximum and 95th-percentile thickness (km).

`skill` summarises a row in one number: the mean, over models, bands and the four scores, of the
score divided by the same score of the second tomography model against the first (how well two
published models agree; computed once, runs_real/sweeps/ceiling.csv).  1.0 = the model agrees with
the tomography as well as the two tomography models agree with each other.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')       # one thread per worker; set before numpy loads
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import contextlib
import copy
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from reconstructions import tomography_score as ts   # noqa: E402
from reconstructions.run_mantle import run            # noqa: E402

SWEEPS = os.path.join(REPO, 'runs_real', 'sweeps')
METRICS = ['slow_iou', 'fast_iou', 'r_2_3', 'r_1_L']
MID_DEEP_KM = 1975.0


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def band_means(df):
    row = {}
    for band, sel in (('mid', df.depth_km < MID_DEEP_KM), ('deep', df.depth_km >= MID_DEEP_KM)):
        b = df[sel]
        for m in METRICS:
            row['{}_{}'.format(band, m)] = b[m].mean()
        for m in ('slow_iou', 'r_2_3'):
            row['{}_{}_pct'.format(band, m)] = b[m + '_pct'].mean()
    return row


def ceiling(cfg):
    """Scores of the second tomography model against the first, by band (cached)."""
    path = os.path.join(SWEEPS, 'ceiling.csv')
    models = cfg['tomography']
    key = '{} vs {}'.format(models[1]['name'], models[0]['name'])
    if os.path.exists(path):
        c = pd.read_csv(path)
        if (c['pair'] == key).any():
            return c[c['pair'] == key].iloc[0].to_dict()
    run_ds = xr.open_dataset(os.path.join(REPO, 'runs_real', 'muller2022', 'mantle_temp_0000Ma.nc'))
    lons_g, lats_g = run_ds.lon.values, run_ds.lat.values
    a, da = ts.tomography_on_grid(models[0]['file'], models[0]['variable'], lons_g, lats_g, ts.tomography_depths(models[0]))
    b, _ = ts.tomography_on_grid(models[1]['file'], models[1]['variable'], lons_g, lats_g, ts.tomography_depths(models[1]))
    df = ts.score_depths(b, a, da, lons_g, lats_g, models[0].get('slow_fraction', 0.3),
                         models[0].get('fast_fraction', 0.2), models[0].get('lmax', 8))
    row = {'pair': key, **band_means(df)}
    os.makedirs(SWEEPS, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False, mode='a', header=not os.path.exists(path))
    return row


def amplitude(cfg, depths=np.arange(1000.0, 2801.0, 100.0)):
    """Model dVs standard deviation (area-weighted) averaged over the mid and deep bands."""
    ds = xr.open_dataset(os.path.join(REPO, cfg['output_dir'], 'mantle_temp_0000Ma.nc'))
    f = ds.dVs.interp(depth=depths).transpose('depth', 'lat', 'lon').values
    w = np.cos(np.radians(ds.lat.values))[:, None] * np.ones(len(ds.lon))
    mean = (f * w).sum(axis=(1, 2)) / w.sum()
    sd = np.sqrt((((f - mean[:, None, None]) ** 2) * w).sum(axis=(1, 2)) / w.sum())
    mid, deep = sd[depths < MID_DEEP_KM].mean(), sd[depths >= MID_DEEP_KM].mean()
    row = {'amp_mid': mid, 'amp_deep': deep, 'amp_deep_mid': deep / mid}
    if 'pile_thickness_km' in ds:                 # llsvp: pile; size of the piles at the CMB
        h = ds.pile_thickness_km.values
        wh = np.cos(np.radians(ds.lat.values))[:, None] * np.ones(h.shape[1])
        row.update({'pile_area_100km': (wh * (h > 100)).sum() / wh.sum(), 'pile_h_max': h.max(),
                    'pile_h_p95': np.percentile(h, 95)})
    return row


def run_variant(sweep, name, over, cfg):
    t0 = time.perf_counter()
    row = {'sweep': sweep, 'variant': name, 'overrides': json.dumps(over, sort_keys=True)}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            summ = run(cfg, verbose=False)
            scores = ts.score_run(cfg)
        row.update({'run_seconds': summ['seconds'], 'n_slablets': summ['n_slablets']})
        for tname, (df, _, _) in scores.items():
            for k, v in band_means(df).items():
                row['{}_{}'.format(tname, k)] = v
        row.update(amplitude(cfg))
    except Exception as e:                      # keep the sweep going; the error is in the table
        row['error'] = '{}: {}'.format(type(e).__name__, e)
    row['seconds'] = time.perf_counter() - t0
    return row


def add_skill(rows, ceil, tomo_names):
    for row in rows:
        vals = []
        for t in tomo_names:
            for band in ('mid', 'deep'):
                for m in METRICS:
                    k = '{}_{}_{}'.format(t, band, m)
                    c = ceil.get('{}_{}'.format(band, m))
                    if k in row and c:
                        vals.append(row[k] / c)
        row['skill'] = float(np.mean(vals)) if vals else np.nan
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('sweep')
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    with open(args.sweep) as f:
        spec = yaml.safe_load(f)
    with open(os.path.join(REPO, spec['base'])) as f:
        base = merge(yaml.safe_load(f), spec.get('common'))
    out = os.path.join(SWEEPS, spec['name'])
    os.makedirs(out, exist_ok=True)
    ceil = ceiling(base)
    tomo_names = [t['name'] for t in base['tomography']]

    jobs = {}
    for name, over in spec['variants'].items():
        cfg = merge(base, over)
        cfg['output_dir'] = os.path.relpath(os.path.join(out, name), REPO)
        jobs[name] = (over, cfg)
    t0 = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_variant, spec['name'], n, o, c): n for n, (o, c) in jobs.items()}
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            print('[{}/{}] {} {:.0f} s{}'.format(k, len(jobs), row['variant'], row['seconds'],
                                               ' ERROR ' + row['error'] if 'error' in row else ''), flush=True)
    rows = add_skill(rows, ceil, tomo_names)
    df = pd.DataFrame(rows).sort_values('skill', ascending=False)
    df.to_csv(os.path.join(out, 'results.csv'), index=False, float_format='%.4f')
    allp = os.path.join(SWEEPS, 'all_results.csv')
    if os.path.exists(allp):
        df = pd.concat([pd.read_csv(allp), df], ignore_index=True)
    df.to_csv(allp, index=False, float_format='%.4f')
    print('{} variants in {:.0f} s; results in {}'.format(len(rows), time.perf_counter() - t0,
                                                         os.path.relpath(os.path.join(out, 'results.csv'), REPO)))


if __name__ == '__main__':
    main()
