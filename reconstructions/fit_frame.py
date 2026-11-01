"""Fit the absolute reference frame to tomography, coarse to fine, without relying on the model's own frame.

    conda run -n pygmt17 python reconstructions/fit_frame.py configs/real/muller2022_dip_slabs.yaml \\
        --start opt --out runs_real/frame_fit/opt [--stages 300 100,200,300] [--workers 10]

The unknowns are Africa's (701) absolute finite rotations at node ages (run_mantle.py frame_africa: a
rotation vector in degrees per node, identity at 0 Ma, interpolated in between); the relative plate
motions come from the reconstruction and every trench sample is moved by the resulting global frame change.
Stages: each stage's nodes start from the previous stage's frame evaluated at the new node ages, so the
first stage (one node at 300 Ma: Africa rotating about one fixed pole since 300 Ma) gives the starting
guess for the next (100, 200, 300 Ma), and so on.  --start sets the guess for the first stage:
    opt    Africa's rotations in the Müller 2022 optimised mantle frame
    pmag   Africa's rotations in the Müller 2022 palaeomagnetic frame
    fixed  Africa at its present position at all ages (all rotation vectors zero)

Objective: `skill` as in sweep.py (the mean over REVEAL and SEMUCB-WM1, mid and deep bands, of slow and
fast IoU and r of degrees 2-3 and 1-8, each divided by the inter-model score), from the present-day field,
without the rotation null.  Search: a simple parallel evolution strategy, one candidate per worker per
generation: the current mean plus mirrored Gaussian steps of size sigma (degrees, per rotation-vector
component); the new mean is the weighted mean of the best three; sigma grows by 1.2 after a generation
that improves on the best so far and shrinks by 0.75 otherwise; a stage stops when sigma < SIGMA_MIN or
after its generation limit.

Outputs in --out: log.csv (every evaluation: stage, generation, sigma, node vectors, skill and band
scores), best.yaml (the best nodes of each stage and their skill), and the config each stage's best used.
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
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pygplates
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import mantle                                                      # noqa: E402
from reconstructions import sweep                                  # noqa: E402
from reconstructions import tomography_score as ts                 # noqa: E402
from reconstructions.run_mantle import africa_frame, run           # noqa: E402

SIGMA0 = [10.0, 5.0, 3.0]          # initial step (deg) of stage 1, 2, 3+
GENERATIONS = [25, 40, 40]
SIGMA_MIN = 0.5
WEIGHTS = np.array([0.5, 0.3, 0.2])
MODEL_DIR = os.path.expanduser('~/Library/Caches/gprm/Muller2022')
START_FILES = {'opt': 'optimisation/1000_0_rotfile_MantleOpt.rot', 'pmag': 'Rotations/1000_0_rotfile.rot'}


def rotation_vector(r):
    """Rotation vector (deg) of a pygplates FiniteRotation."""
    if r.represents_identity_rotation():
        return np.zeros(3)
    lat, lon, angle = r.get_lat_lon_euler_pole_and_angle_degrees()
    return mantle.lonlat_to_xyz(lon, lat) * angle


def start_nodes(kind, ages):
    if kind == 'fixed':
        return {a: np.zeros(3) for a in ages}
    model = pygplates.RotationModel(os.path.join(MODEL_DIR, START_FILES[kind]))
    return {a: rotation_vector(model.get_rotation(float(a), 701, anchor_plate_id=0)) for a in ages}


def nodes_from_frame(nodes, ages):
    """Evaluate the frame defined by nodes at new node ages (the next stage's starting guess)."""
    at = africa_frame({float(k): v for k, v in nodes.items()})
    return {a: rotation_vector(at(float(a))) for a in ages}


_CFG = {}


def _init(cfg, ceil, tomo_names):
    _CFG.update({'cfg': cfg, 'ceil': ceil, 'tomo_names': tomo_names})


def evaluate(args):
    """skill and band scores of one candidate (nodes: {age: vector})."""
    tag, nodes = args
    cfg = copy.deepcopy(_CFG['cfg'])
    cfg['frame_africa'] = {'nodes': {float(a): [float(x) for x in v] for a, v in nodes.items()}}
    cfg['output_dir'] = os.path.join(_CFG['cfg']['output_dir'], 'tmp', tag)
    row = {'tag': tag}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run(cfg, verbose=False)
            scores = ts.score_run(cfg, write=False, null=False)
        for tname, (df, _, _) in scores.items():
            for k, v in sweep.band_means(df).items():
                if not k.endswith('_pct'):
                    row['{}_{}'.format(tname, k)] = v
        row = sweep.add_skill([row], _CFG['ceil'], _CFG['tomo_names'])[0]
    except Exception as e:
        row.update({'skill': np.nan, 'error': '{}: {}'.format(type(e).__name__, e)})
    shutil.rmtree(os.path.join(REPO, cfg['output_dir']), ignore_errors=True)
    return row


def fit_stage(ex, k, ages, x_nodes, sigma, gens, workers, rng, log, t0):
    d = 3 * len(ages)
    m = np.concatenate([x_nodes[a] for a in ages])
    best_x, best_f = m.copy(), -np.inf
    npairs = (workers - 1) // 2
    for g in range(gens):
        z = rng.standard_normal((npairs, d))
        cands = [m] + [m + sigma * zi for zi in z] + [m - sigma * zi for zi in z]
        while len(cands) < workers:
            cands.append(m + sigma * rng.standard_normal(d))
        jobs = [('s{}g{}c{}'.format(k, g, i), {a: c[3 * j:3 * j + 3] for j, a in enumerate(ages)})
                for i, c in enumerate(cands)]
        rows = list(ex.map(evaluate, jobs))
        f = np.array([r['skill'] for r in rows], float)
        for i, (r, c) in enumerate(zip(rows, cands)):
            log.append({'stage': k, 'generation': g, 'candidate': i, 'sigma': sigma,
                        **{'n{:g}_{}'.format(a, 'xyz'[q]): c[3 * j + q] for j, a in enumerate(ages) for q in range(3)},
                        **{kk: vv for kk, vv in r.items() if kk != 'tag'}})
        order = np.argsort(-np.nan_to_num(f, nan=-np.inf))
        improved = f[order[0]] > best_f + 1e-4
        if improved:
            best_f, best_x = f[order[0]], cands[order[0]].copy()
        m = sum(w * cands[i] for w, i in zip(WEIGHTS, order[:3]))
        sigma *= 1.2 if improved else 0.75
        print('stage {} gen {:2d}: best {:.4f}  gen best {:.4f}  sigma {:.2f}  ({:.0f} s)'.format(
            k, g, best_f, f[order[0]], sigma, time.perf_counter() - t0), flush=True)
        if sigma < SIGMA_MIN:
            break
    return {a: best_x[3 * j:3 * j + 3] for j, a in enumerate(ages)}, best_f


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    ap.add_argument('--start', choices=['opt', 'pmag', 'fixed'], default='opt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stages', nargs='+', default=['300', '100,200,300'])
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    cfg['output_every_myr'] = 100000
    cfg['output_dir'] = os.path.relpath(os.path.join(REPO, args.out), REPO)
    os.makedirs(os.path.join(REPO, args.out), exist_ok=True)
    ceil = sweep.ceiling(cfg)
    tomo_names = [t['name'] for t in cfg['tomography']]
    rng = np.random.default_rng(args.seed)
    stages = [[float(a) for a in s.split(',')] for s in args.stages]
    nodes = start_nodes(args.start, stages[0])
    log, best = [], {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(cfg, ceil, tomo_names)) as ex:
        start_row = ex.submit(evaluate, ('start', nodes)).result()
        print('start ({}): skill {:.4f}'.format(args.start, start_row['skill']), flush=True)
        best['start'] = {'skill': float(start_row['skill']), 'nodes': {a: [float(x) for x in v] for a, v in nodes.items()}}
        for k, ages in enumerate(stages):
            if k:
                nodes = nodes_from_frame(nodes, ages)
            nodes, f = fit_stage(ex, k, ages, nodes, SIGMA0[min(k, 2)], GENERATIONS[min(k, 2)], args.workers,
                                 rng, log, t0)
            best['stage{}'.format(k)] = {'ages': ages, 'skill': float(f),
                                         'nodes': {a: [float(x) for x in v] for a, v in nodes.items()}}
            pd.DataFrame(log).to_csv(os.path.join(REPO, args.out, 'log.csv'), index=False, float_format='%.5f')
            with open(os.path.join(REPO, args.out, 'best.yaml'), 'w') as fh:
                yaml.safe_dump(best, fh, sort_keys=False)
            c = copy.deepcopy(cfg)
            c['frame_africa'] = {'nodes': best['stage{}'.format(k)]['nodes']}
            c['output_dir'] = os.path.relpath(os.path.join(REPO, args.out, 'stage{}'.format(k)), REPO)
            c['output_every_myr'] = 50
            with open(os.path.join(REPO, args.out, 'stage{}.yaml'.format(k)), 'w') as fh:
                yaml.safe_dump(c, fh, sort_keys=False)
    shutil.rmtree(os.path.join(REPO, args.out, 'tmp'), ignore_errors=True)
    print('done in {:.0f} s; best.yaml in {}'.format(time.perf_counter() - t0, args.out))


if __name__ == '__main__':
    main()
