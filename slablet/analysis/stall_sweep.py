"""Does stalling slab at the base of the upper mantle improve the fit to tomography?  The global sweep.

    cd ~/GIT/PlateDriver && conda run -n pygmt17 python -m slablet.analysis.stall_sweep configs/muller2022_refined.yaml \
        [--name stall_global] [--members 10] [--workers 4]

Runs the base config (from its start to 0 Ma, present day only) with stalls off (f = 0, one run) and with:
    exponential stall times: f in 0.25, 0.5, 0.75, 1 x mean tau in 10, 25, 50, 100 Myr
    fixed stall times:       f = 1 x tau in 10, 25, 50 Myr
    10 Myr windows:          f = 0.5, tau = 25 Myr (exponential), against the default 25 Myr windows
each with `--members` seeds (slab_stall_seed 0, 1, ...).  Runs go to runs_real/sweeps/<name>/<setting>_s<seed>/.

Each run gets one row in runs_real/sweeps/<name>/results.csv (rewritten as runs finish; finished runs are
skipped when the sweep is started again, failed ones redone):
  - per tomography model of the config (shape only, slablet/analysis/tomography_score.py, no rotation null):
      tz_fast_iou            fast-region overlap, mean over 600, 700, 800 and 900 km (the transition zone and
                             just below it; fast only, since slow regions there come from what slablet lacks)
      lm_<score>             slow_iou, fast_iou, r_2_3 and r_1_L, means over the config's depths (1000-2800 km)
  - present-day dynamic topography (slablet.dyntopo, the config's block; reported, never used to judge):
      dt_r13_holdt, dt_r13_vd        degree 1-3 correlation with Holdt et al. (2022)'s and Valentine & Davies
                                     (2020)'s fits
      dt_antarctica                  mean south of 50 S of the degree 1-8 field / its global rms
      dt_r_dunn                      correlation with Dunn et al. (2026)'s complete residual depths (degrees 1-8)
  - per tomography model and region of REGIONS: <model>_<region>_fast_iou, the fast-region overlap inside the box
    (600-900 km)

--design joint runs instead (joint_settings) stalls off and eleven stall variants, each at lower-mantle rates of 15,
20 and 25 km/Myr: held in place (f 0.5), lying flat (f 0.5 and 1), and only from trenches retreating faster than
10 or 20 km/Myr (in place or flat); mean stall times 25 and 50 Myr (exponential).  Use --members 5.
--design dip runs stalls off with the dip below 100 km at 40, 50, 60, 75 and 90 deg (dip_settings), and adds per
box of SLAB1_REGIONS the dip-profile misfit and bias (model minus Slab1.0 distance from the trench) and the Slab1.0-
to-model distance.
--design depth: stall depth on the partly trench-following geometry (depth_settings).
Every run also gets depth profiles of fast-region overlap against two S and two P models (profile_scores:
p_<model>_<depth>, p_<model>_<region>_<depth>); --rescore adds them to the finished runs of a sweep.
Judging (slablet/analysis/stall_summary.py): a setting helps in a depth range when its members' mean beats f = 0 by
more than twice the standard error of that mean, for both tomography models.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')       # one thread per worker; set before numpy loads
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse                                     # noqa: E402
import contextlib                                   # noqa: E402
import io                                           # noqa: E402
import time                                         # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed   # noqa: E402

import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
import xarray as xr                                 # noqa: E402
import yaml                                         # noqa: E402

from slablet.paths import WORK                      # noqa: E402
from slablet.analysis.sweep import merge            # noqa: E402

TZ_DEPTHS_KM = [600, 700, 800, 900]
LM_SCORES = ['slow_iou', 'fast_iou', 'r_2_3', 'r_1_L']


# Regions fixed before the joint sweep was run (lon_min, lon_max, lat_min, lat_max; lon_min > lon_max wraps the
# date line), where tomography is usually read as showing slab stalled above or near 660 km, or slab that went
# straight through (Fukao & Obayashi 2013, Goes et al. 2017).  Scored by the overlap of the global fast regions
# inside each box, 600-900 km.
REGIONS = {'stagnant_nw_pacific': (115, 150, 25, 50), 'stagnant_tonga': (172, -172, -30, -14),
           'through_marianas': (138, 150, 10, 24), 'through_java': (100, 125, -15, -3),
           'through_central_america': (-100, -75, 0, 22)}


# Boxes with Slab1.0 coverage, for the per-region dip-profile check (dip design).
SLAB1_REGIONS = {'izu_bonin_marianas': (135, 150, 10, 35), 'japan_kuriles': (138, 160, 35, 55),
                 'tonga_kermadec': (172, -172, -40, -14), 'sumatra_java': (95, 125, -12, 5),
                 'central_america': (-105, -80, 5, 20), 'south_america': (-80, -60, -45, 5)}


# Tomography for the depth profiles (profile_scores): two S and two P models, fast regions every PROFILE_DEPTHS.
_TOMO_DIR = os.path.join(os.environ.get('SLABLET_DATA', os.path.expanduser('~/Data')), 'SeismicTomography',
                         'Schouten_Supplementary_material', 'Models')
PROFILE_MODELS = {'REVEAL': ('REVEAL_anomaly.nc', 'vs_anomaly'), 'SEMUCB': ('SEMUCB_WM1_anomaly.nc', 'vs_anomaly'),
                  'GAPP4': ('GAP_P4_anomaly.nc', 'vp_anomaly'), 'UUP07': ('UU-P07_anomaly.nc', 'vp_anomaly')}
PROFILE_DEPTHS = [400, 500, 600, 700, 800, 900, 1000, 1200, 1500]
_PROFILE = {}


def profile_scores(output_dir, fast_fraction=0.2):
    """Fast-region overlap (fastest fast_fraction of the sphere's area at each depth, cos-lat weights) of a run's
    present-day field with each of PROFILE_MODELS at each of PROFILE_DEPTHS, globally (p_<model>_<depth>) and inside
    each box of REGIONS (p_<model>_<region>_<depth>)."""
    from slablet.analysis import tomography_score as ts
    run = xr.open_dataset(os.path.join(WORK, output_dir, 'mantle_temp_0000Ma.nc'))
    lons, lats = run.lon.values, run.lat.values
    lon2, lat2 = np.meshgrid(lons, lats)
    w = np.cos(np.radians(lat2))
    model = run.dVs.interp(depth=PROFILE_DEPTHS).transpose('depth', 'lat', 'lon').values
    boxes = {}
    for r, (lo0, lo1, la0, la1) in REGIONS.items():
        inlon = (lon2 >= lo0) & (lon2 <= lo1) if lo0 <= lo1 else (lon2 >= lo0) | (lon2 <= lo1)
        boxes[r] = inlon & (lat2 >= la0) & (lat2 <= la1)
    row = {}
    for name, (fname, var) in PROFILE_MODELS.items():
        key = (name, len(lons))
        if key not in _PROFILE:
            _PROFILE[key] = ts.tomography_on_grid(os.path.join(_TOMO_DIR, fname), var, lons, lats, PROFILE_DEPTHS)[0]
        tomo = _PROFILE[key]
        for k, z in enumerate(PROFILE_DEPTHS):
            valid = np.isfinite(tomo[k])
            wk = w * valid
            m = ts._quantile_mask(np.where(valid, model[k], 0.0), wk, fast_fraction, False)
            t = ts._quantile_mask(np.where(valid, tomo[k], 0.0), wk, fast_fraction, False)
            row['p_{}_{}'.format(name, z)] = ts._iou(m, t, wk)
            for r, box in boxes.items():
                union = (wk * box * (m | t)).sum()
                row['p_{}_{}_{}'.format(name, r, z)] = (wk * box * (m & t)).sum() / union if union > 0 else np.nan
    return row


def depth_settings():
    """Stall depth: on two partly trench-following geometries (a quarter or half of the trench velocity in the upper
    mantle, 40 deg; prefixes g25_ and g50_), stalls off, and stalls held in place or lying flat (f 0.5, mean 25 Myr)
    at 450, 550, 640, 669 (the default), 800 and 1000 km.  name -> mantle overrides."""
    out = {}
    for tag, k in (('g25', 0.25), ('g50', 0.5)):
        geo = {'slab_trench_follow': True, 'slab_trench_follow_fraction': k}
        out['off_{}'.format(tag)] = dict(geo)
        for d in (450, 550, 640, 0, 800, 1000):
            for mode, extra in (('place', {}), ('flat', {'slab_stall_flat_factor': 1.0})):
                out['{}_{}_d{}'.format(tag, mode, d or 669)] = {**geo, **extra, 'slab_stall_fraction': 0.5,
                                                                'slab_stall_time_myr': 25, 'slab_stall_depth_km': d}
    return out


def dip_settings():
    """The dip check: stalls off, upper-mantle dip below 100 km of 40-90 deg (30 deg above), and 40-70 deg with
    slablets carried with their trench's velocity at seeding in the upper mantle (slab_trench_follow).
    name -> overrides."""
    out = {'off_dip{}'.format(d): {'slab_dip_deg': d} for d in (40, 50, 60, 75, 90)}
    for d in (40, 50, 55, 60, 70):                # slablets carried with their trench in the upper mantle
        out['off_follow_dip{}'.format(d)] = {'slab_dip_deg': d, 'slab_trench_follow': True}
    for k in (0.25, 0.5, 0.75):                   # ... with a fraction of its velocity
        for d in (40, 55):
            out['off_follow{:g}_dip{}'.format(k, d)] = {'slab_dip_deg': d, 'slab_trench_follow': True,
                                                         'slab_trench_follow_fraction': k}
    return out


def slab1_region_scores(cfg):
    """Per box of SLAB1_REGIONS: the dip-profile misfit (mean over depth bands of |model - Slab1.0| median distance
    from the trench, km) and the median distance from Slab1.0 nodes to the nearest model slablet (km)."""
    from slablet.analysis import slab_check as sc
    path = os.path.join(WORK, cfg['output_dir'], 'slabs_upper_mantle_0000Ma.npz')
    with np.load(path) as f:
        slabs = {k: f[k] for k in f.files}
    t_lon, t_lat = sc.present_trenches(cfg)
    s_lon, s_lat, s_z = sc.load_slab1(cfg.get('slab1_file', sc.SLAB1_FILE))
    row = {}
    for r, (lo0, lo1, la0, la1) in SLAB1_REGIONS.items():
        def inside(lon, lat):
            lon = (np.asarray(lon) + 180) % 360 - 180
            inlon = (lon >= lo0) & (lon <= lo1) if lo0 <= lo1 else (lon >= lo0) | (lon <= lo1)
            return inlon & (np.asarray(lat) >= la0) & (np.asarray(lat) <= la1)
        m = inside(slabs['lon'], slabs['lat'])
        k = inside(s_lon, s_lat)
        df = sc.check({c: v[m] for c, v in slabs.items()}, t_lon, t_lat, (s_lon[k], s_lat[k], s_z[k]))
        row['slab1_{}_x_misfit_km'.format(r)] = float((df.model_x_km - df.slab1_x_km).abs().mean())
        row['slab1_{}_x_bias_km'.format(r)] = float((df.model_x_km - df.slab1_x_km).mean())
        row['slab1_{}_to_model_km'.format(r)] = float(df.slab1_to_model_km.mean())
    return row


def joint_settings():
    """The joint sweep: stall variants x lower-mantle rate, stalls off at each rate.  name -> mantle overrides."""
    variants = {'off': {}}
    for f in (0.5,):
        for tau in (25, 50):
            variants['place_f{:g}_t{}'.format(f, tau)] = {'slab_stall_fraction': f, 'slab_stall_time_myr': tau}
    for f in (0.5, 1.0):
        for tau in (25, 50):
            variants['flat_f{:g}_t{}'.format(f, tau)] = {'slab_stall_fraction': f, 'slab_stall_time_myr': tau,
                                                         'slab_stall_flat_factor': 1.0}
    for tau in (25, 50):
        variants['retreat10_t{}'.format(tau)] = {'slab_stall_fraction': 1.0, 'slab_stall_time_myr': tau,
                                                'slab_stall_retreat_km_per_myr': 10.0}
        variants['retreat10_flat_t{}'.format(tau)] = {'slab_stall_fraction': 1.0, 'slab_stall_time_myr': tau,
                                                     'slab_stall_retreat_km_per_myr': 10.0,
                                                     'slab_stall_flat_factor': 1.0}
    variants['retreat20_flat_t25'] = {'slab_stall_fraction': 1.0, 'slab_stall_time_myr': 25,
                                      'slab_stall_retreat_km_per_myr': 20.0, 'slab_stall_flat_factor': 1.0}
    out = {}
    for name, over in variants.items():
        for lr in (15, 20, 25):
            out['{}_lr{}'.format(name, lr)] = {**over, 'slab_lower_rate_km_per_myr': lr}
    return out


def settings():
    """name -> mantle overrides (without the seed)."""
    out = {'f0': {}}
    for f in (0.25, 0.5, 0.75, 1.0):
        for tau in (10, 25, 50, 100):
            out['f{:g}_t{}_exp'.format(f, tau)] = {'slab_stall_fraction': f, 'slab_stall_time_myr': tau}
    for tau in (10, 25, 50):
        out['f1_t{}_fixed'.format(tau)] = {'slab_stall_fraction': 1.0, 'slab_stall_time_myr': tau,
                                          'slab_stall_form': 'fixed'}
    out['f0.5_t25_exp_w10'] = {'slab_stall_fraction': 0.5, 'slab_stall_time_myr': 25, 'slab_stall_window_myr': 10}
    return out


_REFS = {}


def _refs():
    if not _REFS:
        from slablet.analysis import dyntopo_refs as R
        lon, lat, val, _ = R.dunn_points(('c',))
        _REFS.update(holdt=R.holdt_fit(40), vd=R.vd2020(), dunn=(lon, lat, val), grid=R.grid())
    return _REFS


def dyntopo_scores(path):
    from slablet.analysis import dyntopo_refs as R
    from slablet.sphharm import band_correlation, truncate
    refs = _refs()
    with xr.open_dataset(path) as ds:
        c = ds['dyn_topo_coeffs'].values
    row = {}
    for name in ('holdt', 'vd'):
        k = min(c.shape[1], refs[name].shape[1]) - 1
        row['dt_r13_' + name] = float(band_correlation(c[:, :k + 1, :k + 1], refs[name][:, :k + 1, :k + 1], 1, 3))
    c8 = truncate(c, 8).copy()
    c8[:, 0] = 0.0
    f = refs['grid'].synth(c8)
    lat2 = R.LATS[:, None] * np.ones((1, len(R.LONS)))
    w = np.cos(np.radians(lat2)) * (lat2 < -50)
    row['dt_antarctica'] = float((f * w).sum() / w.sum() / R.rms(c8, 1, 8))
    row['dt_r_dunn'] = R.point_correlation(c, *refs['dunn'], 8)
    return row


_TZ = {}


def region_scores(cfg):
    """Overlap of the fast regions (fastest fast_fraction of the sphere at each depth, as tomography_score) inside
    each of REGIONS, mean over TZ_DEPTHS_KM, per tomography model: <model>_<region>_fast_iou."""
    from slablet.analysis import tomography_score as ts
    run = xr.open_dataset(os.path.join(WORK, cfg['output_dir'], 'mantle_temp_0000Ma.nc'))
    lons, lats = run.lon.values, run.lat.values
    lon2, lat2 = np.meshgrid(lons, lats)
    w = np.cos(np.radians(lat2))
    row = {}
    for tc in cfg['tomography']:
        key = (tc['file'], tc['variable'], len(lons))
        if key not in _TZ:
            _TZ[key] = ts.tomography_on_grid(tc['file'], tc['variable'], lons, lats, TZ_DEPTHS_KM)
        tomo, depths = _TZ[key]
        q = float(tc.get('fast_fraction', 0.2))
        model = run.dVs.interp(depth=depths).transpose('depth', 'lat', 'lon').values
        scores = {r: [] for r in REGIONS}
        for k in range(len(depths)):
            valid = np.isfinite(tomo[k])
            wk = w * valid
            m = ts._quantile_mask(np.where(valid, model[k], 0.0), wk, q, False)
            t = ts._quantile_mask(np.where(valid, tomo[k], 0.0), wk, q, False)
            for r, (lo0, lo1, la0, la1) in REGIONS.items():
                inlon = (lon2 >= lo0) & (lon2 <= lo1) if lo0 <= lo1 else (lon2 >= lo0) | (lon2 <= lo1)
                box = wk * (inlon & (lat2 >= la0) & (lat2 <= la1))
                union = (box * (m | t)).sum()
                scores[r].append((box * (m & t)).sum() / union if union > 0 else np.nan)
        for r, v in scores.items():
            row['{}_{}_fast_iou'.format(tc.get('name'), r)] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    return row


def run_member(setting, seed, cfg):
    from slablet.analysis import tomography_score as ts
    from slablet.gplates.run_mantle import run
    t0 = time.perf_counter()
    row = {'setting': setting, 'seed': seed}
    row.update({k: cfg['mantle'].get(k) for k in ('slab_stall_fraction', 'slab_stall_time_myr', 'slab_stall_form',
                                                  'slab_stall_window_myr', 'slab_stall_flat_factor',
                                                  'slab_stall_retreat_km_per_myr', 'slab_lower_rate_km_per_myr',
                                                  'slab_dip_deg', 'slab_trench_follow',
                                                  'slab_trench_follow_fraction', 'slab_stall_depth_km')})
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run(cfg, verbose=False)
            lm = ts.score_run(cfg, write=False, null=False)
            tz_cfg = {**cfg, 'tomography': [{k: v for k, v in tc.items() if k not in ('min_depth_km', 'max_depth_km')}
                                            | {'depths_km': TZ_DEPTHS_KM} for tc in cfg['tomography']]}
            tz = ts.score_run(tz_cfg, write=False, null=False)
        for name, (df, _, _) in tz.items():
            row[name + '_tz_fast_iou'] = df.fast_iou.mean()
            for _, r in df.iterrows():
                row['{}_fast_iou_{:.0f}'.format(name, r.depth_km)] = r.fast_iou
        for name, (df, _, _) in lm.items():
            for s in LM_SCORES:
                row['{}_lm_{}'.format(name, s)] = df[s].mean()
        row.update(region_scores(cfg))
        row.update(profile_scores(cfg['output_dir']))
        if setting.startswith(('off_dip', 'off_follow')):
            row.update(slab1_region_scores(cfg))
        row.update(dyntopo_scores(os.path.join(WORK, cfg['output_dir'], 'dyntopo_0000Ma.nc')))
    except Exception as e:                          # keep the sweep going; the error is in the table
        row['error'] = '{}: {}'.format(type(e).__name__, e)
    row['seconds'] = time.perf_counter() - t0
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    ap.add_argument('--name', default='stall_global')
    ap.add_argument('--members', type=int, default=10)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only', nargs='*', help='run only these settings')
    ap.add_argument('--rescore', action='store_true',
                    help='add the depth-profile scores (profile_scores) to every finished run of the sweep and stop')
    ap.add_argument('--design', choices=['global', 'joint', 'dip', 'depth'], default='global',
                    help='global: the stall grid at the config\'s sinking rate; joint: stall variants x lower-mantle rate')
    args = ap.parse_args()
    with open(os.path.join(WORK, args.config)) as f:
        base = yaml.safe_load(f)
    base = merge(base, {'output_every_myr': 1e4, 'mantle': {'dvs_save_native': True},
                        'dynamic_topography': {'after_run': True, 'volume': False}})
    out = os.path.join(WORK, 'runs_real', 'sweeps', args.name)
    os.makedirs(out, exist_ok=True)
    table = os.path.join(out, 'results.csv')
    rows, done = [], set()
    if os.path.exists(table):
        old = pd.read_csv(table)
        if 'error' in old:
            old = old[old['error'].isna()]
        rows = old.to_dict('records')
        done = {(s, int(k)) for s, k in zip(old.setting, old.seed)}
    if args.rescore:
        df = pd.DataFrame(rows)
        for i, r in df.iterrows():
            for k, v in profile_scores(os.path.relpath(os.path.join(out, '{}_s{}'.format(r.setting, int(r.seed))),
                                                       WORK)).items():
                df.loc[i, k] = v
        df.to_csv(table, index=False, float_format='%.5g')
        print('rescored', len(df), 'runs')
        return
    jobs = []
    designs = {'global': settings, 'joint': joint_settings, 'dip': dip_settings, 'depth': depth_settings}
    for setting, over in designs[args.design]().items():
        if args.only and setting not in args.only:
            continue
        for seed in range(1 if setting == 'f0' or setting.startswith('off_') else args.members):
            if (setting, seed) in done:
                continue
            cfg = merge(base, {'mantle': {**over, 'slab_stall_seed': seed}})
            cfg['output_dir'] = os.path.relpath(os.path.join(out, '{}_s{}'.format(setting, seed)), WORK)
            jobs.append((setting, seed, cfg))
    print('{} runs to do ({} done already), {} workers'.format(len(jobs), len(done), args.workers), flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_member, *j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            pd.DataFrame(rows).to_csv(table, index=False, float_format='%.5g')     # all rows so far
            print('[{}/{}] {} s{} {:.0f} s{}  ({:.0f} min so far)'.format(
                k, len(jobs), row['setting'], row['seed'], row['seconds'],
                ' ERROR ' + row['error'] if 'error' in row else '', (time.perf_counter() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
