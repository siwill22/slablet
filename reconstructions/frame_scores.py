"""Score absolute frames against six S-wave tomography models (frame uncertainty, reconstructions/fit_frame.py).

    conda run -n pygmt17 python reconstructions/frame_scores.py configs/real/muller2022_dip_slabs.yaml \\
        --frames runs_real/frame_uq/*/best.yaml runs_real/frame_fit/*/best.yaml --native opt pmag \\
        --out runs_real/frame_uq/scores.csv [--workers 10]

Every frame is run with the same mantle (the config, with --set overrides) and scored, without the rotation
null, against REVEAL, SEMUCB-WM1 and the four models the tuning never saw (S40RTS, SMEAN, GypsumS, S362ANI;
build_tomo_grids.py), and against the four-model slow vote map (slow_votes_S4.nc, majority = 3 of 4).
skill_<model> = the mean over mid and deep bands of slow and fast IoU and r of degrees 2-3 and 1-8, each
divided by the REVEAL-vs-SEMUCB score (as `skill` in sweep.py, but for one model at a time).

Frames: each best.yaml gives its last stage (named by its folder); --native runs the reconstruction's own
rotation file (opt = Müller 2022 optimised mantle frame, pmag = palaeomagnetic) without frame_africa.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import contextlib
import copy
import io
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from reconstructions import sweep                                  # noqa: E402
from reconstructions import tomography_score as ts                 # noqa: E402
from reconstructions.fit_frame import START_FILES, S4, set_value, tomo_set   # noqa: E402
from reconstructions.run_mantle import run                         # noqa: E402

SLOW_S4 = {'name': 'slowvotes_S4', 'file': os.path.join(REPO, 'runs_real', 'votemaps', 'slow_votes_S4.nc'),
           'variable': 'minus_votes', 'depths_km': list(range(1000, 2801, 100)), 'slow_fraction': 0.3, 'lmax': 8}
S4_MAJORITY = 3


def frames_from(paths, native):
    out = []
    for p in paths:
        b = yaml.safe_load(open(p))
        stage = [k for k in b if k.startswith('stage')][-1]
        out.append({'frame': os.path.basename(os.path.dirname(p)), 'source': p, 'stage': stage,
                    'fit_skill': b[stage]['skill'], 'nodes': b[stage]['nodes']})
    for n in native:
        out.append({'frame': 'native_' + n, 'source': START_FILES[n], 'stage': '', 'fit_skill': np.nan, 'nodes': None})
    return out


_CFG = {}


def _init(cfg, ceil):
    _CFG.update({'cfg': cfg, 'ceil': ceil})


def score(fr):
    cfg = copy.deepcopy(_CFG['cfg'])
    if fr['nodes'] is None:
        cfg['rotation_files'] = [fr['source']]
    else:
        cfg['frame_africa'] = {'nodes': fr['nodes']}
    cfg['output_dir'] = os.path.join(_CFG['cfg']['output_dir'], 'tmp', fr['frame'])
    row = {k: v for k, v in fr.items() if k != 'nodes'}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run(cfg, verbose=False)
            res = ts.score_run({**cfg, 'tomography': cfg['tomography'] + [SLOW_S4]}, write=False, null=False)
        for tname, (df, _, _) in res.items():
            vals = []
            for k, v in sweep.band_means(df).items():
                if not k.endswith('_pct'):
                    row['{}_{}'.format(tname, k)] = v
                    if tname != 'slowvotes_S4' and _CFG['ceil'].get(k):
                        vals.append(v / _CFG['ceil'][k])
            if tname != 'slowvotes_S4':
                row['skill_' + tname] = float(np.mean(vals))
        run_ds = xr.open_dataset(os.path.join(REPO, cfg['output_dir'], 'mantle_temp_0000Ma.nc'))
        votes = xr.open_dataset(SLOW_S4['file']).votes
        w = np.cos(np.radians(run_ds.lat.values))[:, None] * np.ones(len(run_ds.lon))
        for band, zs in (('mid', range(1000, 2000, 100)), ('deep', range(2000, 2801, 100))):
            maj = []
            for z in zs:
                a = ts._quantile_mask(run_ds.dVs.interp(depth=float(z)).values, w, 0.3, True)
                b = votes.sel(depth=float(z)).values >= S4_MAJORITY
                maj.append((w * (a & b)).sum() / (w * (a | b)).sum())
            row['slowvotes_S4_{}_maj_iou'.format(band)] = float(np.mean(maj))
    except Exception as e:
        row['error'] = '{}: {}'.format(type(e).__name__, e)
    shutil.rmtree(os.path.join(REPO, cfg['output_dir']), ignore_errors=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    ap.add_argument('--frames', nargs='*', default=[])
    ap.add_argument('--native', nargs='*', default=[])
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ceil = sweep.ceiling(cfg)
    for item in args.set:
        set_value(cfg, item)
    cfg['tomography'] = tomo_set(cfg, 'default') + tomo_set(cfg, 'S4')
    cfg['output_every_myr'] = 100000
    cfg['output_dir'] = os.path.relpath(os.path.join(REPO, os.path.dirname(args.out), 'scoring'), REPO)
    frames = frames_from(args.frames, args.native)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(cfg, ceil)) as ex:
        rows = list(ex.map(score, frames))
    df = pd.DataFrame(rows)
    names = [t['name'] for t in cfg['tomography']]
    df['skill_S4mean'] = df[['skill_' + n for n in S4]].mean(axis=1)
    df['skill_fitset'] = df[['skill_REVEAL', 'skill_SEMUCB-WM1']].mean(axis=1)
    df.to_csv(os.path.join(REPO, args.out), index=False, float_format='%.5f')
    shutil.rmtree(os.path.join(REPO, cfg['output_dir']), ignore_errors=True)
    cols = ['frame', 'fit_skill', 'skill_fitset'] + ['skill_' + n for n in names[2:]] + ['skill_S4mean',
            'slowvotes_S4_deep_slow_iou', 'slowvotes_S4_deep_maj_iou']
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
