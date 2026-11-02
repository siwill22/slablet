"""Run many variants of a real-reconstruction mantle config in parallel and score each one.

    conda run -n pygmt17 python -m slablet.analysis.sweep sweeps/s01.yaml [--workers 10]

A sweep file:
    name: s01_one_at_a_time
    base: configs/muller2022.yaml        # the config every variant starts from
    common: {output_every_myr: 100000}         # merged into every variant (dicts merge key by key)
    variants:
      base: {}
      tau_50: {mantle: {slab_thermal_tau_myr: 50}}
      start_1000: {start_ma: 1000}

Each variant runs into runs_real/sweeps/<name>/<variant>/ and is scored against every
tomography model of the config (slablet/analysis/tomography_score.py).  One row per variant
goes to runs_real/sweeps/<name>/results.csv and is appended to runs_real/sweeps/all_results.csv:
the overrides, run time, and for each tomography model and depth band (mid: < 1975 km,
deep: >= 1975 km) the mean slow_iou, fast_iou, r_2_3, r_1_L and the rotation-null percentiles
of slow_iou and r_2_3.  The shape scores ignore amplitude, so a row also has the model's
deep / mid ratio of the area-weighted standard deviation of dVs (amp_deep_mid; REVEAL and
SEMUCB-WM1 both have about 1.5): a nearly empty deep mantle can still correlate well.  With
llsvp: pile it also has the pile's size: the fraction of the CMB where it is over 100 km thick
(pile_area_100km), its maximum and 95th-percentile thickness (km).  And the upper-mantle slabs
against Slab1.0 (slablet/analysis/slab_check.py): slab1_slab1_to_model_km, slab1_model_to_slab1_km,
slab1_x_misfit_km and, per depth band, the median distance from the trench (x_model_<top km>,
x_slab1_<top km>).  And the lower-mantle slabs against two fast-anomaly vote maps (VOTE_MAPS):
votes_*_fast_iou, *_fast_iou_pct and *_r_1_L, means over depth; and the slow regions against a slow
vote map of six S models (slablet/analysis/build_slow_votemap.py): slowvotes_<band>_slow_iou, _r_2_3 and
_maj_iou (IoU with the cells where at least 4 of the 6 models are slow).

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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from slablet.paths import WORK

from slablet.analysis import slab_check               # noqa: E402
from slablet.analysis import tomography_score as ts   # noqa: E402
from slablet.gplates.run_mantle import run            # noqa: E402

SWEEPS = os.path.join(WORK, 'runs_real', 'sweeps')
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
    run_ds = xr.open_dataset(os.path.join(WORK, 'runs_real', 'muller2022', 'mantle_temp_0000Ma.nc'))
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
    ds = xr.open_dataset(os.path.join(WORK, cfg['output_dir'], 'mantle_temp_0000Ma.nc'))
    f = ds.dVs.interp(depth=depths).transpose('depth', 'lat', 'lon').values
    w = np.cos(np.radians(ds.lat.values))[:, None] * np.ones(len(ds.lon))
    mean = (f * w).sum(axis=(1, 2)) / w.sum()
    sd = np.sqrt((((f - mean[:, None, None]) ** 2) * w).sum(axis=(1, 2)) / w.sum())
    mid, deep = sd[depths < MID_DEEP_KM].mean(), sd[depths >= MID_DEEP_KM].mean()
    row = {'amp_mid': mid, 'amp_deep': deep, 'amp_deep_mid': deep / mid}
    for z in (2000, 2500, 2750):                  # slow tail / fast tail (REVEAL ~1.1, SEMUCB ~1.2 near the CMB)
        k = int(np.argmin(np.abs(depths - z)))
        row['tail_ratio_{}'.format(z)] = -np.percentile(f[k], 1) / np.percentile(f[k], 99)
    if 'pile_thickness_km' in ds:                 # llsvp: pile; size of the piles at the CMB
        h = ds.pile_thickness_km.values
        wh = np.cos(np.radians(ds.lat.values))[:, None] * np.ones(h.shape[1])
        row.update({'pile_area_100km': (wh * (h > 100)).sum() / wh.sum(), 'pile_h_max': h.max(),
                    'pile_h_p95': np.percentile(h, 95)})
    return row


# Fast-anomaly vote maps (slablet/analysis/build_votemaps.py): they score the model's slabs only.
VOTE_MAPS = [
    {'name': 'votes_mixed', 'file': os.path.join(WORK, 'runs_real', 'votemaps', 'votemaps.nc'), 'variable': 'votes',
     'depths_km': [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 2000], 'fast_fraction': 0.2, 'lmax': 8},
    {'name': 'votes_allP', 'file': os.path.join(WORK, 'runs_real', 'votemaps', 'votemaps_allP_20220809.nc'),
     'variable': 'votes', 'depths_km': [1000, 1250, 1500, 1750, 2000, 2250, 2500], 'fast_fraction': 0.2, 'lmax': 8},
]


SLOW_VOTES = {'name': 'slow_votes', 'file': os.path.join(WORK, 'runs_real', 'votemaps', 'slow_votes_S.nc'),
              'variable': 'minus_votes', 'depths_km': list(range(1000, 2801, 100)), 'slow_fraction': 0.3, 'lmax': 8}
SLOW_VOTE_MAJORITY = 4                            # of 6 models (slablet/analysis/build_slow_votemap.py)


def slow_vote_scores(cfg):
    """The model's slow regions against the S-model slow vote map, mid and deep bands: slow_iou (equal
    areas), r_2_3, and maj_iou = IoU of the model's slowest 30 % with the cells where at least
    SLOW_VOTE_MAJORITY models are slow."""
    if not os.path.exists(SLOW_VOTES['file']):
        return {}
    (df, _, _), = ts.score_run({**cfg, 'tomography': [SLOW_VOTES]}, write=False).values()
    run_ds = xr.open_dataset(os.path.join(WORK, cfg['output_dir'], 'mantle_temp_0000Ma.nc'))
    votes = xr.open_dataset(SLOW_VOTES['file']).votes
    w = np.cos(np.radians(run_ds.lat.values))[:, None] * np.ones(len(run_ds.lon))
    maj = []
    for z in SLOW_VOTES['depths_km']:
        m = run_ds.dVs.interp(depth=float(z)).values
        a = ts._quantile_mask(m, w, 0.3, True)
        b = votes.sel(depth=float(z)).values >= SLOW_VOTE_MAJORITY
        maj.append((w * (a & b)).sum() / (w * (a | b)).sum())
    df = df.assign(maj_iou=maj)
    row = {}
    for band, sel in (('mid', df.depth_km < MID_DEEP_KM), ('deep', df.depth_km >= MID_DEEP_KM)):
        for c in ('slow_iou', 'r_2_3', 'maj_iou'):
            row['slowvotes_{}_{}'.format(band, c)] = df.loc[sel, c].mean()
    return row


def vote_scores(cfg):
    """Mean over depths of fast_iou (and its rotation percentile) and r_1_L against each vote map."""
    vm = [v for v in VOTE_MAPS if os.path.exists(v['file'])]
    if not vm:
        return {}
    row = {}
    for vname, (df, _, _) in ts.score_run({**cfg, 'tomography': vm}, write=False).items():
        row.update({vname + '_fast_iou': df.fast_iou.mean(), vname + '_fast_iou_pct': df.fast_iou_pct.mean(),
                    vname + '_r_1_L': df.r_1_L.mean()})
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
        row.update(vote_scores(cfg))                 # lower-mantle slabs vs fast vote maps
        row.update(slow_vote_scores(cfg))            # slow regions vs the S-model slow vote map
        sc = slab_check.check_run(cfg)               # upper-mantle slabs vs Slab1.0
        row.update({'slab1_' + k: v for k, v in slab_check.summary(sc).items()})
        for _, b in sc.iterrows():
            row['x_model_{:.0f}'.format(b.depth_lo)] = b.model_x_km
            row['x_slab1_{:.0f}'.format(b.depth_lo)] = b.slab1_x_km
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
    with open(os.path.join(WORK, spec['base'])) as f:
        base = merge(yaml.safe_load(f), spec.get('common'))
    out = os.path.join(SWEEPS, spec['name'])
    os.makedirs(out, exist_ok=True)
    ceil = ceiling(base)
    tomo_names = [t['name'] for t in base['tomography']]

    jobs = {}
    for name, over in spec['variants'].items():
        cfg = merge(base, over)
        cfg['output_dir'] = os.path.relpath(os.path.join(out, name), WORK)
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
                                                         os.path.relpath(os.path.join(out, 'results.csv'), WORK)))


if __name__ == '__main__':
    main()
