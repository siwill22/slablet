"""Measure present-day slab dips from Slab1.0 at each trench sample and see what in the reconstruction predicts them.

    cd ~/GIT/PlateDriver && conda run -n pygmt17 python -m slablet.analysis.dip_fit <config> [--deep-km 400]

For each present-day trench sample (the run's trench-sample cache at end_ma, mantle frame) a profile is walked from
the trench along its down-dip azimuth, every 10 km out to 1500 km, reading the Slab1.0 depth at the nearest grid node
(0.25 deg).  The dip below 100 km is atan(dz / dx) between where the profile first reaches 100 km and where it first
reaches --deep-km (or its deepest point if that is at least 250 km), with dx the horizontal distance at the mean depth
of the two points (great-circle distance at the surface x (R - z) / R).  Samples whose profile never reaches 250 km
are left out (no Slab1.0 slab, or a shallow one).

Candidate predictors, all at the present day from the reconstruction:
    retreat_km_per_myr        trench velocity away from the overriding plate (mantle frame; slablet.core.trench_retreat)
    convergence_km_per_myr    the convergence rate the run uses
    seafloor_age_myr          age of the subducting seafloor at the trench (needs seafloor: true and its cache)
Writes <output_dir>/dip_fit.csv (one row per sample) and prints the correlations and a linear fit of dip on
retreat; samples are not independent (1 deg apart along strike), so the correlations are also given per region
(SLAB1_REGIONS of stall_sweep.py, regional medians).
"""
import argparse
import os

import numpy as np
import pandas as pd
import yaml

from slablet.paths import WORK
from slablet import core
from slablet.sphere import move_along_azimuth
from slablet.analysis import slab_check as sc
from slablet.analysis.stall_sweep import SLAB1_REGIONS

R_KM = 6371.0
STEP_KM, MAX_KM = 10.0, 1500.0


def slab1_grid(path=sc.SLAB1_FILE):
    import xarray as xr
    ds = xr.open_dataset(path)
    return ds['lon'].values, ds['lat'].values, -ds['z'].values.astype(float)    # depth positive down, NaN: no slab


def profile_dips(lon, lat, azimuth, grid, deep_km=400.0):
    """Dip below 100 km (deg) for each trench point, NaN where Slab1.0 does not reach 250 km along the profile."""
    g_lon, g_lat, z = grid
    x = np.arange(0.0, MAX_KM + STEP_KM, STEP_KM)
    dips = np.full(len(lon), np.nan)
    for k in range(len(lon)):
        if not np.isfinite(azimuth[k]):
            continue
        p_lon, p_lat = move_along_azimuth(np.full(len(x), lon[k]), np.full(len(x), lat[k]),
                                          np.full(len(x), azimuth[k]), x, R_KM)
        i = np.clip(np.round((p_lat - g_lat[0]) / (g_lat[1] - g_lat[0])).astype(int), 0, len(g_lat) - 1)
        j = np.round(((p_lon + 180.0) % 360.0 - 180.0 - g_lon[0]) / (g_lon[1] - g_lon[0])).astype(int) % len(g_lon)
        d = z[i, j]
        if not np.isfinite(d).any() or np.nanmax(d) < 250.0:
            continue
        a = np.nonzero(d >= 100.0)[0]
        target = min(deep_km, np.nanmax(d))
        b = np.nonzero(d >= target)[0]
        if not len(a) or not len(b) or b[0] <= a[0]:
            continue
        z1, z2 = d[a[0]], d[b[0]]
        dx = (x[b[0]] - x[a[0]]) * (R_KM - 0.5 * (z1 + z2)) / R_KM
        dips[k] = np.degrees(np.arctan2(z2 - z1, dx))
    return dips


def region_of(lon, lat):
    out = np.full(len(lon), '', dtype=object)
    lon = (np.asarray(lon) + 180) % 360 - 180
    for r, (lo0, lo1, la0, la1) in SLAB1_REGIONS.items():
        inlon = (lon >= lo0) & (lon <= lo1) if lo0 <= lo1 else (lon >= lo0) | (lon <= lo1)
        out[inlon & (lat >= la0) & (lat <= la1)] = r
    return out


def measure(cfg, deep_km=400.0):
    from slablet.gplates.trench_samples import CachedTrenchSamples
    p_sample = cfg.get('mantle', {}).get('slab_sample_deg', 1.0)
    s = CachedTrenchSamples.for_config(cfg, p_sample).samples(float(cfg['end_ma']))
    df = pd.DataFrame({'lon': s['lon'], 'lat': s['lat'], 'azimuth': s['downdip_azimuth_deg'],
                       'convergence_km_per_myr': s['convergence_km_per_myr'],
                       'retreat_km_per_myr': core.trench_retreat(s)})
    if cfg.get('seafloor'):
        from slablet.gplates.seafloor_tracker import SeafloorCache
        df['seafloor_age_myr'] = SeafloorCache(cfg, p_sample).at(float(cfg['end_ma']))['trench_age']
    df = df[df.convergence_km_per_myr > 0].reset_index(drop=True)
    df['dip_deg'] = profile_dips(df.lon.values, df.lat.values, df.azimuth.values,
                                 slab1_grid(cfg.get('slab1_file', sc.SLAB1_FILE)), deep_km)
    df['region'] = region_of(df.lon.values, df.lat.values)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    ap.add_argument('--deep-km', type=float, default=400.0)
    args = ap.parse_args()
    with open(os.path.join(WORK, args.config)) as f:
        cfg = yaml.safe_load(f)
    df = measure(cfg, args.deep_km)
    out = os.path.join(WORK, cfg['output_dir'], 'dip_fit.csv')
    df.to_csv(out, index=False, float_format='%.4g')
    ok = df[np.isfinite(df.dip_deg)]
    print('{} trench samples with a Slab1.0 dip (of {}); dip median {:.0f} deg, 10-90 % {:.0f}-{:.0f}'.format(
        len(ok), len(df), ok.dip_deg.median(), *np.percentile(ok.dip_deg, [10, 90])))
    preds = [c for c in ('retreat_km_per_myr', 'convergence_km_per_myr', 'seafloor_age_myr') if c in ok]
    for c in preds:
        v = np.isfinite(ok[c])
        print('  {:24s} Pearson r {:+.2f}  Spearman {:+.2f}  (n {})'.format(
            c, np.corrcoef(ok[c][v], ok.dip_deg[v])[0, 1], ok[c][v].rank().corr(ok.dip_deg[v].rank()), int(v.sum())))
    a, b = np.polyfit(ok.retreat_km_per_myr, ok.dip_deg, 1)
    print('  linear fit: dip = {:.1f} + {:.2f} x retreat (km/Myr); residual sd {:.1f} deg'.format(
        b, a, np.std(ok.dip_deg - (b + a * ok.retreat_km_per_myr))))
    g = ok[ok.region != ''].groupby('region')[['dip_deg'] + preds].median()
    g['n'] = ok[ok.region != ''].groupby('region').size()
    print(g.round(1).to_string())
    print('written', os.path.relpath(out, WORK))


if __name__ == '__main__':
    main()
