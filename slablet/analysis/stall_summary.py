"""Judge the global stall sweep: which settings beat stalls off (f = 0), in which depth range.

    cd ~/GIT/PlateDriver && conda run -n pygmt17 python -m slablet.analysis.stall_summary [--name stall_global]

Reads runs_real/sweeps/<name>/results.csv (slablet/analysis/stall_sweep.py).  For each setting and score: the
members' mean, the standard error of that mean (sd / sqrt(n)) and the difference from f = 0.  A score "helps" when
the mean beats f = 0 by more than twice its standard error, and "hurts" when it falls short by as much.  A setting
helps in a depth range only when the score helps for both tomography models:
    tz   fast_iou at 600-900 km
    lm   each of slow_iou, fast_iou, r_2_3 and r_1_L at 1000-2800 km, judged separately
Settings whose members are all identical (f = 1 with fixed times: nothing is random) have no spread; their
verdict is the sign of the difference alone, marked as deterministic.
With depth profiles (stall_sweep profile_scores) the bands 400-600, 700-1000 and 1200-1500 km are judged over
all four profile models (p400_600 ...), and in three regions.  For the depth design each geometry's settings
(g25_, g50_) are compared with stalls off on the same geometry.
For the joint design (stall variants x lower-mantle rate), stalls off is the best of its runs at each rate, per
score, and region scores (fast overlap inside fixed boxes) are judged the same way.
The dynamic-topography columns are summarised the same way but never judged.
Writes runs_real/sweeps/<name>/summary.csv (one row per setting) and prints the verdicts.
"""
import argparse
import os

import numpy as np
import pandas as pd

from slablet.paths import WORK

LM_SCORES = ['slow_iou', 'fast_iou', 'r_2_3', 'r_1_L']
DT_COLS = ['dt_r13_holdt', 'dt_r13_vd', 'dt_antarctica', 'dt_r_dunn']


def judged_columns(df):
    """{range label: [(model, column)]} for the scores that are judged (regions too, when the sweep has them)."""
    models = sorted({c.split('_tz_fast_iou')[0] for c in df.columns if c.endswith('_tz_fast_iou')})
    out = {'tz_fast_iou': [(m, m + '_tz_fast_iou') for m in models]}
    for s in LM_SCORES:
        out['lm_' + s] = [(m, '{}_lm_{}'.format(m, s)) for m in models]
    regions = sorted({c[len(models[0]) + 1:-len('_fast_iou')] for c in df.columns
                      if c.startswith(models[0] + '_') and c.endswith('_fast_iou')
                      and ('stagnant_' in c or 'through_' in c)})
    for r in regions:
        out[r] = [(m, '{}_{}_fast_iou'.format(m, r)) for m in models]
    pmodels = sorted({c.split('_')[1] for c in df.columns if c.startswith('pb_')})
    for band in PROFILE_BANDS:
        if pmodels and 'pb_{}_{}'.format(pmodels[0], band) in df:
            out['p' + band] = [(m, 'pb_{}_{}'.format(m, band)) for m in pmodels]
            for r in ('stagnant_nw_pacific', 'stagnant_tonga', 'through_marianas'):
                if band != '1200_1500' and 'pb_{}_{}_{}'.format(pmodels[0], r, band) in df:
                    out['p{}_{}'.format(band, r)] = [(m, 'pb_{}_{}_{}'.format(m, r, band)) for m in pmodels]
    return models, out


def _is_off(setting):
    return setting == 'f0' or setting.startswith('off_')


PROFILE_BANDS = {'400_600': (400, 500, 600), '700_1000': (700, 800, 900, 1000), '1200_1500': (1200, 1500)}


def add_profile_bands(df):
    """Mean fast-region overlap over each of PROFILE_BANDS from the depth profiles (p_<model>_<depth>, and
    p_<model>_<region>_<depth> for regions), as pb_<model>_<band> and pb_<model>_<region>_<band>."""
    import re
    cols = [c for c in df.columns if re.match(r'^p_.+_\d+$', c)]
    stems = sorted({c.rsplit('_', 1)[0] for c in cols})
    new = {}
    for stem in stems:
        for band, ds in PROFILE_BANDS.items():
            have = ['{}_{}'.format(stem, d) for d in ds if '{}_{}'.format(stem, d) in df]
            if len(have) == len(ds):
                new['pb_{}_{}'.format(stem[2:], band)] = df[have].mean(axis=1)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def summarise(df):
    """Stalls off is one run (f0), or one run per lower-mantle rate (off_lr*: the joint design); then every judged
    score of a stall setting is compared with the best stalls-off run for that score, whatever its rate."""
    import re
    df = df[df['error'].isna()] if 'error' in df else df
    df = add_profile_bands(df)
    off = df[df.setting.map(_is_off)]
    models, judged = judged_columns(df)
    base = off.iloc[0].copy()                     # f0, or the first rate (the config's) for the joint design
    for _, cs in judged.items():
        for _, c in cs:
            base[c] = off[c].max()
    # the depth design: each geometry (g25_, g50_ ...) is compared with stalls off on the same geometry
    geo_base = {m.group(1): off[off.setting == 'off_' + m.group(1)].iloc[0]
                for m in (re.match(r'^(g\d+)_', s) for s in df.setting) if m}
    cols = [c for _, cs in judged.items() for _, c in cs] + [c for c in DT_COLS if c in df]
    rows = []
    for setting, g in df[~df.setting.map(_is_off)].groupby('setting', sort=False):
        m = re.match(r'^(g\d+)_', setting)
        b = geo_base[m.group(1)] if m else base
        row = {'setting': setting, 'n': len(g)}
        for k in ('slab_stall_fraction', 'slab_stall_time_myr', 'slab_stall_form', 'slab_stall_window_myr',
                  'slab_stall_flat_factor', 'slab_stall_retreat_km_per_myr', 'slab_lower_rate_km_per_myr'):
            if k in g:
                row[k] = g[k].iloc[0]
        for c in cols:
            mean, se = g[c].mean(), g[c].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan
            row[c + '_mean'], row[c + '_se'], row[c + '_diff'] = mean, se, mean - b[c]
            row[c + '_z'] = (mean - b[c]) / se if se > 1e-9 * max(abs(mean), 1.0) else np.nan
        deterministic = all(np.isnan(row[c + '_z']) for _, cs in judged.items() for _, c in cs)
        row['deterministic'] = deterministic       # every member identical (f = 1 with fixed times): no spread
        for label, cs in judged.items():
            z = np.array([row[c + ('_diff' if deterministic else '_z')] for _, c in cs])
            lim = 0.0 if deterministic else 2.0
            row[label + '_verdict'] = ('helps' if np.all(z > lim) else 'hurts' if np.all(z < -lim) else
                                       'mixed' if np.any(z > lim) and np.any(z < -lim) else '-')
        rows.append(row)
    return base, pd.DataFrame(rows), models, judged


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--name', default='stall_global')
    ap.add_argument('--only', nargs='*', help='print only these judged labels (all are written)')
    args = ap.parse_args()
    d = os.path.join(WORK, 'runs_real', 'sweeps', args.name)
    base, s, models, judged = summarise(pd.read_csv(os.path.join(d, 'results.csv')))
    s.to_csv(os.path.join(d, 'summary.csv'), index=False, float_format='%.5g')
    labels = [lab for lab in judged if not args.only or lab in args.only]
    print('stalls off (best over rates): ' + '  '.join('{} {}'.format(lab, ' / '.join('{:.3f}'.format(base[c]) for _, c in judged[lab]))
                                for lab in labels))
    print('differences from f = 0 ({} then z = diff / se); verdict (both models): + helps, x hurts, ~ mixed, . '
          'neither'.format(' / '.join(models)))
    print('{:22s} {:>3s} '.format('setting', 'n') + ' '.join('{:>27s}'.format(lab) for lab in labels))
    for _, r in s.iterrows():
        cells = []
        for lab in labels:
            diffs = '/'.join('{:+.3f}'.format(r[c + '_diff']) for _, c in judged[lab])
            zs = '/'.join('{:+.1f}'.format(r[c + '_z']) for _, c in judged[lab])
            mark = {'helps': '+', 'hurts': 'x', 'mixed': '~', '-': '.'}[r[lab + '_verdict']]
            cells.append('{:>27s}'.format('{} {} {}'.format(diffs, 'det' if r.deterministic else zs, mark)))
        print('{:22s} {:>3d} '.format(r.setting, int(r.n)) + ' '.join(cells))
    print('written', os.path.relpath(os.path.join(d, 'summary.csv'), WORK))


if __name__ == '__main__':
    main()
