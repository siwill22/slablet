"""Resolution tests: how the field changes with the grids, the time step and the slab spacing, and at what cost.

    conda run -n pygmt17 python -m slablet.analysis.sweep_resolution configs/sweeps/r01_resolution.yaml [--workers 4]

Sweep file as in sweep_opt1.py, plus `reference: <variant>`.  Every variant runs from the base start to 0 Ma with
output every 20 Myr.  Each output is then put on a common grid in <run>/grid5/ (cos-lat block mean onto the
5 deg cells, linear in depth onto 50, 150, ... 2850 km), so that variants drawn at different resolutions are
scored alike:
    OPT1 scores (sweep_opt1.scores: 0-240 Ma, and _all for 0-980 Ma), skill_tomo (present day)
    r_ref_<band>     correlation with the reference variant, mean over ages 0-240 Ma and the depths of the band
                     (upper 50-650 km, mid 1050-1950, deep 2050-2850): near 1 = the same large-scale field
    rms_ref_<band>   rms of the difference from the reference over the reference's rms, the same means
On the field as drawn (dVs_native when saved, else dVs), at present day:
    slab1_frac_<z>   of the fast (dVs > 0) mass within 1500 km of a Slab1.0 node with depth z +- 50 km, the
                     share within 150 km of one, at z = 200, 400, 600 km (sharper and better placed = higher)
Cost: seconds for the run, peak memory of its process (MB), MB per saved output.
To save disk, native outputs are deleted after regridding, except at KEEP_AGES.
Output: runs_real/sweeps_resolution/<name>/results.csv; run folders kept for figures.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import contextlib
import copy
import glob
import io
import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.spatial import cKDTree

from slablet.paths import WORK
import slablet                                                      # noqa: E402
from slablet.core import _block_mean, mantle_grid                   # noqa: E402
from slablet.analysis import compare_opt1 as co                     # noqa: E402
from slablet.analysis import slab_check                             # noqa: E402
from slablet.analysis import sweep                                  # noqa: E402
from slablet.analysis import tomography_score as ts                 # noqa: E402
from slablet.frame.fit_frame import set_value                    # noqa: E402
from slablet.gplates.run_mantle import run                         # noqa: E402
from slablet.analysis.sweep_opt1 import RECENT, scores              # noqa: E402

OUT = os.path.join(WORK, 'runs_real', 'sweeps_resolution')
DEPTHS = np.arange(50.0, 2851.0, 100.0)
LON5, LAT5 = mantle_grid(5.0)
W5 = np.cos(np.radians(LAT5))[:, None] * np.ones(len(LON5))
REF_BANDS = {'upper': (0, 699), 'mid': (1000, 1999), 'deep': (2000, 2900)}
SLAB1_DEPTHS = (200, 400, 600)
KEEP_AGES = (0, 100, 200)

_G = {}


def _init(ceil):
    _G['ceil'] = ceil
    _G['grid'] = co.build_grid(1)


def to_grid5(run_dir):
    """Every mantle_temp_*.nc of the run onto the common grid, in run_dir/grid5/; returns MB per native output."""
    out = os.path.join(run_dir, 'grid5')
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(run_dir, 'mantle_temp_*Ma.nc')))
    mb = np.mean([os.path.getsize(f) for f in files]) / 1e6
    for f in files:
        with xr.open_dataset(f) as ds:
            v = ds.dVs.transpose('depth', 'lat', 'lon').astype(float)
            if not np.array_equal(v.depth.values, DEPTHS):
                v = v.interp(depth=DEPTHS)
            lats, arr = v.lat.values, v.values
            age = ds.attrs['age_ma']
        k = 5.0 / (lats[1] - lats[0])
        if k > 1.001:
            arr = _block_mean(arr, lats, k)
        xr.Dataset({'dVs': (('depth', 'lat', 'lon'), arr.astype(np.float32))},
                   coords={'depth': DEPTHS, 'lat': LAT5, 'lon': LON5},
                   attrs={'age_ma': age}).to_netcdf(os.path.join(out, os.path.basename(f)))
        if int(round(age)) not in KEEP_AGES:
            os.remove(f)
    return mb


def slab1_fractions(run_dir):
    with xr.open_dataset(os.path.join(run_dir, 'mantle_temp_0000Ma.nc')) as ds:
        da = (ds.dVs_native if 'dVs_native' in ds else ds.dVs).load()
    lon2, lat2 = np.meshgrid(da.lon.values, da.lat.values)
    w = np.cos(np.radians(lat2)).ravel()
    cell_xyz = slablet.lonlat_to_xyz(lon2.ravel(), lat2.ravel())
    s1lon, s1lat, s1dep = _G.setdefault('slab1', slab_check.load_slab1())
    row = {}
    for z in SLAB1_DEPTHS:
        band = np.abs(s1dep - z) <= 50
        d, _ = cKDTree(slablet.lonlat_to_xyz(s1lon[band], s1lat[band])).query(cell_xyz)
        km = 2 * np.arcsin(np.clip(d / 2, 0, 1)) * 6371.0
        pos = np.clip(da.interp(depth=float(z)).values.astype(float), 0, None).ravel() * w
        row['slab1_frac_{}'.format(z)] = float(pos[km < 150].sum() / pos[km < 1500].sum())
    return row


def run_variant(job):
    name, over, cfg = job
    t0 = time.perf_counter()
    row = {'variant': name, 'overrides': json.dumps(over, sort_keys=True)}
    run_dir = os.path.join(WORK, cfg['output_dir'])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run(cfg, verbose=False)
        row['seconds'] = time.perf_counter() - t0
        row['peak_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # bytes on macOS
        row.update(slab1_fractions(run_dir))
        row['mb_per_output'] = to_grid5(run_dir)
        g5 = os.path.join(run_dir, 'grid5')
        with contextlib.redirect_stdout(io.StringIO()):
            tomo = ts.score_run({**cfg, 'output_dir': os.path.relpath(g5, WORK)}, write=False, null=False)
        trow = {}
        for tname, (df, _, _) in tomo.items():
            for k, v in sweep.band_means(df).items():
                if not k.endswith('_pct'):
                    trow['{}_{}'.format(tname, k)] = v
        row['skill_tomo'] = sweep.add_skill([trow], _G['ceil'], [t['name'] for t in cfg['tomography']])[0]['skill']
        df = co.compare(g5, _G['grid'])
        df.to_csv(os.path.join(run_dir, 'compare_opt1.csv'), index=False, float_format='%.5g')
        row.update(scores(df, RECENT))
        row.update({k + '_all': v for k, v in scores(df, 1000).items()})
    except Exception as e:
        row['error'] = '{}: {}'.format(type(e).__name__, e)
    row['seconds_total'] = time.perf_counter() - t0
    return row


def _wcorr(a, b):
    a, b = a - (a * W5).sum() / W5.sum(), b - (b * W5).sum() / W5.sum()
    return float((a * b * W5).sum() / np.sqrt((a * a * W5).sum() * (b * b * W5).sum()))


def against_reference(out, names, ref):
    """r_ref_<band>, rms_ref_<band> of every variant against the reference, over 0-RECENT Ma."""
    ages = range(0, RECENT + 1, 20)
    load = lambda n, a: xr.open_dataset(os.path.join(out, n, 'grid5', 'mantle_temp_{:04d}Ma.nc'.format(a))).dVs.values
    refs = {a: load(ref, a) for a in ages}
    rows = []
    for n in names:
        row = {'variant': n}
        for band, (lo, hi) in REF_BANDS.items():
            k = np.where((DEPTHS >= lo) & (DEPTHS <= hi))[0]
            r, e = [], []
            for a in ages:
                f, g = load(n, a), refs[a]
                for i in k:
                    r.append(_wcorr(f[i], g[i]))
                    e.append(np.sqrt(((f[i] - g[i]) ** 2 * W5).sum() / ((g[i] ** 2) * W5).sum()))
            row['r_ref_' + band], row['rms_ref_' + band] = np.mean(r), np.mean(e)
        rows.append(row)
    return pd.DataFrame(rows)


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
    # one run per process, so that each process's peak memory is that run's
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(ceil,), max_tasks_per_child=1) as ex:
        rows = list(ex.map(run_variant, jobs))
    df = pd.DataFrame(rows)
    ok = [n for n in df.variant if 'error' not in df or pd.isna(df.set_index('variant').loc[n].get('error'))]
    if spec['reference'] in ok:
        df = df.merge(against_reference(out, ok, spec['reference']), on='variant', how='left')
    df.to_csv(os.path.join(out, 'results.csv'), index=False, float_format='%.4f')
    cols = ['variant', 'seconds', 'peak_mb', 'mb_per_output', 'opt1_score', 'deep_spec_misfit', 'skill_tomo',
            'r_ref_upper', 'r_ref_mid', 'r_ref_deep', 'slab1_frac_200', 'slab1_frac_400', 'slab1_frac_600']
    pd.set_option('display.width', 250)
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))
    if 'error' in df and df.error.notna().any():
        print(df.loc[df.error.notna(), ['variant', 'error']].to_string(index=False))


if __name__ == '__main__':
    main()
