"""First-order check of the model's upper-mantle slabs against Slab1.0 (Hayes et al. 2012).

    conda run -n pygmt17 python -m slablet.analysis.slab_check configs/muller2022_tuned.yaml

Reads the present-day slablets a run saved (slabs_upper_mantle_0000Ma.npz in output_dir), the
present-day trench samples (from the cache), and the Slab1.0 depth grid (slab1_file in the
config, default ~/Data/allslabs/Slab1.0_allslabs_0p25deg.nc: depth in km, negative down).

Only where Slab1.0 has a slab: a model slablet counts if it is within REGION_DEG of any Slab1.0
point (Slab1.0 covers 13 subduction zones, not all of them).  For each depth band (BANDS_KM):
  slab1_x_km, model_x_km    median horizontal distance (km at the surface) from the nearest
                            present-day trench: the dip profile (a 45° slab is at x = depth)
  slab1_to_model_km         median distance from each Slab1.0 point to the nearest model slablet
                            in the same band (are Slab1.0's slabs where the model has slab?)
  model_to_slab1_km         median distance from each model slablet to the nearest Slab1.0 point
                            in the same band (is the model's slab where Slab1.0 has slab?)
Distances are great-circle distances on the sphere (km at the surface radius).
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.spatial import cKDTree

from slablet.paths import WORK

from slablet.sphere import lonlat_to_xyz                          # noqa: E402
from slablet.gplates.trench_samples import CachedTrenchSamples   # noqa: E402
from slablet.paths import DATA

SLAB1_FILE = os.path.join(DATA, 'allslabs', 'Slab1.0_allslabs_0p25deg.nc')
BANDS_KM = [(50, 150), (150, 250), (250, 350), (350, 450), (450, 550), (550, 660)]
REGION_DEG = 3.0
R_KM = 6371.0
_SLAB1 = {}


def load_slab1(path=SLAB1_FILE):
    """lon, lat, depth (km, positive down) of every Slab1.0 node."""
    if path not in _SLAB1:
        ds = xr.open_dataset(path)
        z = ds['z'].values
        lat, lon = np.meshgrid(ds['lat'].values, ds['lon'].values, indexing='ij')
        ok = np.isfinite(z)
        _SLAB1[path] = (lon[ok], lat[ok], -z[ok].astype(float))
    return _SLAB1[path]


def _nearest_km(from_lon, from_lat, to_lon, to_lat):
    if len(to_lon) == 0 or len(from_lon) == 0:
        return np.full(len(from_lon), np.nan)
    d, _ = cKDTree(lonlat_to_xyz(to_lon, to_lat)).query(lonlat_to_xyz(from_lon, from_lat))
    return 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0)) * R_KM


def check(slabs, trench_lon, trench_lat, slab1=None):
    """One row per depth band (see the module docstring).  slabs: dict of arrays lon, lat,
    depth_km, weight (slablets with weight 0 are ignored)."""
    s_lon, s_lat, s_z = slab1 if slab1 is not None else load_slab1()
    keep = np.asarray(slabs['weight']) > 0
    m_lon, m_lat, m_z = (np.asarray(slabs[k], float)[keep] for k in ('lon', 'lat', 'depth_km'))
    near = _nearest_km(m_lon, m_lat, s_lon, s_lat) <= np.radians(REGION_DEG) * R_KM
    m_lon, m_lat, m_z = m_lon[near], m_lat[near], m_z[near]
    rows = []
    for lo, hi in BANDS_KM:
        a = (s_z >= lo) & (s_z < hi)
        b = (m_z >= lo) & (m_z < hi)
        rows.append({'depth_lo': lo, 'depth_hi': hi, 'n_slab1': int(a.sum()), 'n_model': int(b.sum()),
                     'slab1_x_km': np.median(_nearest_km(s_lon[a], s_lat[a], trench_lon, trench_lat)),
                     'model_x_km': np.median(_nearest_km(m_lon[b], m_lat[b], trench_lon, trench_lat))
                     if b.any() else np.nan,
                     'slab1_to_model_km': np.median(_nearest_km(s_lon[a], s_lat[a], m_lon[b], m_lat[b])),
                     'model_to_slab1_km': np.median(_nearest_km(m_lon[b], m_lat[b], s_lon[a], s_lat[a]))
                     if b.any() else np.nan})
    return pd.DataFrame(rows)


def present_trenches(cfg):
    p_sample = cfg.get('mantle', {}).get('slab_sample_deg', 1.0)
    t = CachedTrenchSamples.for_config(cfg, p_sample).samples(float(cfg['end_ma']))
    return t['lon'], t['lat']


def check_run(cfg):
    path = os.path.join(WORK, cfg['output_dir'], 'slabs_upper_mantle_{:04.0f}Ma.npz'.format(float(cfg['end_ma'])))
    with np.load(path) as f:
        slabs = {k: f[k] for k in f.files}
    t_lon, t_lat = present_trenches(cfg)
    return check(slabs, t_lon, t_lat, load_slab1(cfg.get('slab1_file', SLAB1_FILE)))


def summary(df):
    """One-number summaries: mean over bands of the two nearest distances, and of the dip-profile misfit."""
    return {'slab1_to_model_km': df.slab1_to_model_km.mean(), 'model_to_slab1_km': df.model_to_slab1_km.mean(),
            'x_misfit_km': (df.model_x_km - df.slab1_x_km).abs().mean()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    df = check_run(cfg)
    print(df.round(0).to_string(index=False))
    print({k: round(v) for k, v in summary(df).items()})


if __name__ == '__main__':
    main()
