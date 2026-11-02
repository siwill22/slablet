"""Sweep mantle options against the Müller et al. (2022) OPT1 convection model, weighted to 0-240 Ma.

    conda run -n pygmt17 python -m slablet.analysis.sweep_opt1 configs/sweeps/o1_spread_screen.yaml [--workers 10]

Sweep file: name, base (config path), common (overrides for every variant), variants ({name: overrides}),
overrides as in sweep.py (mantle.KEY or top-level keys).  Every variant runs from the base start (1000 Ma)
to 0 Ma with output every 20 Myr (OPT1's ages) and is compared with OPT1 by compare_opt1.compare.
Scores, means over ages 0-240 Ma (RECENT) and over the depths of each band (mid 1050-1950, deep 2050-2850 km):
    <band>_r_1_8, <band>_r_2      correlation with OPT1 (-dVs vs T) of degrees 1-8 and of degree 2
    <band>_slow_iou, _fast_iou   overlap of the slowest 30 % / fastest 20 % with OPT1's hottest / coldest
    <band>_spec_misfit           mean over l = 3-12 of |log10 P(l)/P(2) - the same for OPT1|: how far the
                                 spectrum's shape is from OPT1's (0 = same shape)
    opt1_score                   mean of r_1_8, slow_iou and fast_iou over both bands
and the same with an _all suffix over 0-980 Ma.  skill_tomo: the present-day skill against REVEAL and
SEMUCB-WM1 (sweep.py), so that a better OPT1 match is not bought with a worse tomography match.
Output: runs_real/sweeps_opt1/<name>/results.csv; run folders kept for figures.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import contextlib
import copy
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

from slablet.paths import WORK
from slablet.analysis import compare_opt1 as co                     # noqa: E402
from slablet.analysis import sweep                                  # noqa: E402
from slablet.analysis import tomography_score as ts                 # noqa: E402
from slablet.frame.fit_frame import set_value                    # noqa: E402
from slablet.gplates.run_mantle import run                         # noqa: E402

RECENT = 240
OUT = os.path.join(WORK, 'runs_real', 'sweeps_opt1')
BANDS = {'mid': (1000, 1999), 'deep': (2000, 2900)}


def scores(df, max_age):
    q = df[df.age_ma <= max_age]
    row = {}
    for band, (lo, hi) in BANDS.items():
        b = q[(q.depth_km >= lo) & (q.depth_km <= hi)]
        for c in ('r_1_8', 'r_2', 'slow_iou', 'fast_iou'):
            row['{}_{}'.format(band, c)] = b[c].mean()
        ls = range(3, co.LMAX + 1)
        pr = np.log10(b[['run_pow_{}'.format(l) for l in ls]].values / b[['run_pow_2']].values)
        pt = np.log10(b[['opt1_pow_{}'.format(l) for l in ls]].values / b[['opt1_pow_2']].values)
        row['{}_spec_misfit'.format(band)] = float(np.nanmean(np.abs(pr - pt)))
    row['opt1_score'] = float(np.mean([row['{}_{}'.format(b, c)] for b in BANDS for c in ('r_1_8', 'slow_iou', 'fast_iou')]))
    return row


_G = {}


def _init(ceil):
    _G['ceil'] = ceil
    _G['grid'] = co.build_grid(1)


def run_variant(job):
    name, over, cfg = job
    t0 = time.perf_counter()
    row = {'variant': name, 'overrides': json.dumps(over, sort_keys=True)}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run(cfg, verbose=False)
            tomo = ts.score_run(cfg, write=False, null=False)
        trow = {}
        for tname, (df, _, _) in tomo.items():
            for k, v in sweep.band_means(df).items():
                if not k.endswith('_pct'):
                    trow['{}_{}'.format(tname, k)] = v
        row['skill_tomo'] = sweep.add_skill([trow], _G['ceil'], [t['name'] for t in cfg['tomography']])[0]['skill']
        df = co.compare(os.path.join(WORK, cfg['output_dir']), _G['grid'])
        df.to_csv(os.path.join(WORK, cfg['output_dir'], 'compare_opt1.csv'), index=False, float_format='%.5g')
        row.update(scores(df, RECENT))
        row.update({k + '_all': v for k, v in scores(df, 1000).items()})
        if cfg.get('mantle', {}).get('dvs_save_native'):       # the field before the tomography blur
            dfn = co.compare(os.path.join(WORK, cfg['output_dir']), _G['grid'], var='dVs_native')
            dfn.to_csv(os.path.join(WORK, cfg['output_dir'], 'compare_opt1_native.csv'), index=False, float_format='%.5g')
            row.update({k + '_nat': v for k, v in scores(dfn, RECENT).items()})
            row.update({k + '_nat_all': v for k, v in scores(dfn, 1000).items()})
    except Exception as e:
        row['error'] = '{}: {}'.format(type(e).__name__, e)
    row['seconds'] = time.perf_counter() - t0
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('sweep')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.sweep))
    base = yaml.safe_load(open(os.path.join(WORK, spec['base'])))
    for k, v in (spec.get('common') or {}).items():
        set_value(base, '{}={}'.format(k, json.dumps(v)))
    base['output_every_myr'] = 20
    out = os.path.join(OUT, spec['name'])
    os.makedirs(out, exist_ok=True)
    jobs = []
    for name, over in spec['variants'].items():
        cfg = copy.deepcopy(base)
        for k, v in (over or {}).items():
            set_value(cfg, '{}={}'.format(k, json.dumps(v)))
        cfg['output_dir'] = os.path.relpath(os.path.join(out, name), WORK)
        jobs.append((name, over or {}, cfg))
    ceil = sweep.ceiling(base)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(ceil,)) as ex:
        rows = list(ex.map(run_variant, jobs))
    df = pd.DataFrame(rows).sort_values('opt1_score', ascending=False)
    df.to_csv(os.path.join(out, 'results.csv'), index=False, float_format='%.4f')
    cols = ['variant', 'opt1_score', 'mid_r_1_8', 'deep_r_1_8', 'deep_r_2', 'deep_slow_iou', 'deep_fast_iou',
            'mid_spec_misfit', 'deep_spec_misfit', 'opt1_score_all', 'skill_tomo', 'seconds']
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))
    if 'error' in df and df.error.notna().any():
        print(df.loc[df.error.notna(), ['variant', 'error']].to_string(index=False))


if __name__ == '__main__':
    main()
