"""Ridge samples for the asthenosphere layer (slablet/asthenosphere.py) from a GPlates reconstruction, cached per time.

    samples = CachedRidgeSamples.for_config(cfg).samples(time_ma)
      lon, lat (degrees), divergence_km_per_myr (> 0 = spreading: full rate normal to the ridge),
      length_km (ridge length each sample stands for, on the sphere)

Points every sample_deg along the mid-ocean ridges of the resolved topologies, from pygplates' plate boundary
statistics (relative velocity over [t + 1 Myr, t]).  Only diverging points are kept; the normal component means
transform-like stretches carry almost no sink.  Positions are in the frame of anchor_plate_id.  Cached beside the
trench samples (slablet/gplates/trench_samples.py): cache_dir/<model name>-<hash>/ridges_anchor<id>_s<deg>/.
"""
import os

import numpy as np
import pygplates

from slablet.gplates.trench_samples import _model_key, load_model

EARTH_RADIUS_KM = 6371.0
SAMPLE_KEYS = ('lon', 'lat', 'divergence_km_per_myr', 'length_km')


def ridge_samples_at(rotation_model, features, time_ma, sample_deg=0.5, anchor_plate_id=0):
    snap = pygplates.TopologicalSnapshot(features, rotation_model, float(time_ma), anchor_plate_id)
    stats = snap.calculate_plate_boundary_statistics(
        np.radians(sample_deg), boundary_section_filter=pygplates.FeatureType.gpml_mid_ocean_ridge)
    stats = [s for s in stats if np.isfinite(s.convergence_velocity_orthogonal)
             and s.convergence_velocity_orthogonal < 0]
    if not stats:
        return {k: np.zeros(0) for k in SAMPLE_KEYS}
    latlon = np.array([s.boundary_point.to_lat_lon() for s in stats], dtype=float)
    return {'lon': latlon[:, 1], 'lat': latlon[:, 0],
            'divergence_km_per_myr': -np.array([s.convergence_velocity_orthogonal for s in stats], dtype=float),
            'length_km': np.array([s.boundary_length for s in stats], dtype=float) * EARTH_RADIUS_KM}


class CachedRidgeSamples:
    """samples(time_ma) from the cache, resolving the model only for times not cached yet."""

    def __init__(self, recon, name, cache_dir, sample_deg=0.5, anchor_plate_id=0):
        self.files = (recon.rotation_files, recon.topology_files)
        self.sample_deg, self.anchor_plate_id = sample_deg, anchor_plate_id
        top = os.path.join(cache_dir, '{}-{}'.format(name, _model_key(recon.model_dir, sum(self.files, []))))
        self.folder = os.path.join(top, 'ridges_anchor{}_s{:g}'.format(anchor_plate_id, sample_deg))
        os.makedirs(self.folder, exist_ok=True)
        self._model = None

    @classmethod
    def for_config(cls, cfg, sample_deg=0.5):
        from slablet.gplates.models import load
        from slablet.paths import WORK
        return cls(load(cfg), cfg['name'], os.path.join(WORK, cfg['cache_dir']), sample_deg=sample_deg,
                   anchor_plate_id=int(cfg.get('anchor_plate_id', 0)))

    def samples(self, time_ma):
        path = os.path.join(self.folder, 't{:09.3f}.npz'.format(float(time_ma)))
        if os.path.exists(path):
            with np.load(path) as f:
                return {k: f[k] for k in SAMPLE_KEYS}
        if self._model is None:
            self._model = load_model(*self.files)
        s = ridge_samples_at(*self._model, time_ma, self.sample_deg, self.anchor_plate_id)
        tmp = '{}.{}.tmp.npz'.format(path[:-4], os.getpid())
        np.savez(tmp, **s)
        os.replace(tmp, path)
        return s
