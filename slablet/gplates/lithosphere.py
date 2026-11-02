"""Lithosphere thickness through a reconstruction, on the 1 deg cell-centre grid, cached per output time.

    thickness_km(cfg, age)  ->  (180, 360) array, lat ascending from -89.5, lon from -179.5

Continents and cratons are the reconstruction's own polygons (continent_files and craton_files of the run config's
reconstruction, slablet/gplates/models.py; Müller 2022: Continents/shapes_continents.gpml and
Cratons/shapes_cratons_Merdith_etal.gpml), reconstructed to the age and tested cell by cell (pygplates
PlatePartitioner).  Oceans take the tracked seafloor age (slablet/gplates/seafloor_tracker.py, the nearest 10 Myr):
    ocean      OCEAN_KM_PER_SQRT_MYR x sqrt(age), at most OCEAN_MAX_KM (age unknown: DEFAULT_OCEAN_AGE_MYR)
    continent  CONTINENT_KM
    craton     CRATON_KM
These are round Earth-like values for the shape of the field, not fitted.  Cached as
<seafloor cache>/../lithosphere_v1/t<age>.npz.
"""
import os

import numpy as np
import pygplates

from slablet.gplates.models import load as load_reconstruction             # noqa: E402
from slablet.gplates.seafloor_tracker import GRID_EVERY_MYR, SeafloorCache   # noqa: E402

OCEAN_KM_PER_SQRT_MYR = 10.0
OCEAN_MAX_KM = 100.0
DEFAULT_OCEAN_AGE_MYR = 60.0
CONTINENT_KM = 150.0
CRATON_KM = 250.0
LATS = np.arange(-89.5, 90, 1.0)
LONS = np.arange(-179.5, 180, 1.0)
_MODEL = {}


def _masks(cfg, age):
    if 'rm' not in _MODEL:
        recon = load_reconstruction(cfg)
        _MODEL['rm'] = pygplates.RotationModel(sorted(recon.rotation_files),
                                               default_anchor_plate_id=int(cfg.get('anchor_plate_id', 0)))
        _MODEL['polys'] = {}
        for k, files in (('continent', recon.continent_files), ('craton', recon.craton_files)):
            if files is None:
                raise SystemExit('{}: no {} polygons for the lithosphere lid; give {}_files in the reconstruction '
                                 'block ([] for none)'.format(recon.label, k, k))
            fc = pygplates.FeatureCollection()
            for f in files:
                fc.add(pygplates.FeatureCollection(f))
            _MODEL['polys'][k] = fc
        lo, la = np.meshgrid(LONS, LATS)
        _MODEL['points'] = [pygplates.PointOnSphere(float(a), float(b)) for a, b in zip(la.ravel(), lo.ravel())]
    out = {}
    for k, fc in _MODEL['polys'].items():
        part = pygplates.PlatePartitioner(fc, _MODEL['rm'], float(age))
        out[k] = np.array([part.partition_point(pt) is not None for pt in _MODEL['points']]).reshape(len(LATS), len(LONS))
    return out


def thickness_km(cfg, age):
    sea = SeafloorCache(cfg)
    folder = os.path.join(os.path.dirname(sea.folder), 'lithosphere_v1')
    path = os.path.join(folder, 't{:04d}.npz'.format(int(round(age))))
    if os.path.exists(path):
        with np.load(path) as f:
            return f['thickness_km']
    m = _masks(cfg, age)
    t10 = int(GRID_EVERY_MYR * round(age / GRID_EVERY_MYR))
    sea_age = sea.at(t10)['age_grid'].astype(float)
    sea_age = np.where(np.isfinite(sea_age), sea_age, DEFAULT_OCEAN_AGE_MYR)
    th = np.minimum(OCEAN_KM_PER_SQRT_MYR * np.sqrt(np.maximum(sea_age, 0.0)), OCEAN_MAX_KM)
    th = np.where(m['continent'], CONTINENT_KM, th)
    th = np.where(m['craton'], CRATON_KM, th).astype(np.float32)
    os.makedirs(folder, exist_ok=True)
    tmp = '{}.{}.tmp.npz'.format(path[:-4], os.getpid())
    np.savez_compressed(tmp, thickness_km=th, continent=m['continent'], craton=m['craton'])
    os.replace(tmp, path)
    return th
