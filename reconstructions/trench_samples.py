"""Trench samples for the mantle model from a GPlates reconstruction, cached per time.

The samples follow the contract in mantle/core.py: lon, lat (degrees) and
convergence_km_per_myr (> 0 = converging), at points every sample_deg along the
subduction zones of the resolved topologies.  Convergence is the component of the
relative plate velocity along the boundary normal, over [t + 1 Myr, t]
(pygplates' default velocity interval).

The positions are in the frame of anchor_plate_id.  Slablets sink vertically in that
frame, so it must be a mantle frame (fixed to the deep mantle), not a palaeomagnetic
frame fixed to the spin axis.  Model folders often hold several rotation files for
different frames (Müller et al. 2022: palaeomagnetic, optimised mantle, no-net-rotation),
so the files are always listed explicitly (glob patterns relative to the model folder),
never discovered.

Resolving topologies is most of the cost of a mantle run (about 90 ms per Myr for
Müller et al. 2019), so samples are cached as one .npz per time, under
cache_dir/<model name>-<hash of the files used>/anchor<id>_s<sample_deg>/.
"""
import glob
import hashlib
import os

import numpy as np
import pygplates


def model_files(model_dir, rotation_patterns, topology_patterns):
    """The rotation and topology files matching the patterns (relative to model_dir).
    Every pattern must match at least one file."""
    def _find(patterns):
        found = []
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(model_dir, pat)))
            if not hits:
                raise SystemExit('no file matches {} in {}'.format(pat, model_dir))
            found += hits
        return found
    return _find(rotation_patterns), _find(topology_patterns)


def load_model(rotation_files, topology_files):
    return (pygplates.RotationModel(rotation_files),
            [pygplates.FeatureCollection(f) for f in topology_files])


def trench_samples_at(rotation_model, features, time_ma, sample_deg=1.0, anchor_plate_id=0):
    """Samples along the subduction zones resolved at time_ma (only converging points)."""
    snap = pygplates.TopologicalSnapshot(features, rotation_model, float(time_ma), anchor_plate_id)
    stats = snap.calculate_plate_boundary_statistics(
        np.radians(sample_deg), boundary_section_filter=pygplates.FeatureType.gpml_subduction_zone)
    if not stats:
        return {'lon': np.zeros(0), 'lat': np.zeros(0), 'convergence_km_per_myr': np.zeros(0)}
    conv = np.array([s.convergence_velocity_orthogonal for s in stats], dtype=float)
    latlon = np.array([s.boundary_point.to_lat_lon() for s in stats], dtype=float)
    keep = np.isfinite(conv) & (conv > 0)
    return {'lon': latlon[keep, 1], 'lat': latlon[keep, 0], 'convergence_km_per_myr': conv[keep]}


def _model_key(model_dir, files):
    """Short hash of the files' names, sizes and modification times, so a different file
    choice or an edited model never reads samples cached from the old one."""
    h = hashlib.sha1()
    for f in files:
        st = os.stat(f)
        h.update('{}|{}|{}'.format(os.path.relpath(f, model_dir), st.st_size, int(st.st_mtime)).encode())
    return h.hexdigest()[:10]


class CachedTrenchSamples:
    """samples(time_ma) from the cache, resolving the model (once, on first need) only
    for the times that are not cached yet."""

    def __init__(self, model_dir, rotation_files, topology_files, name, cache_dir,
                 sample_deg=1.0, anchor_plate_id=0):
        self.files = model_files(model_dir, rotation_files, topology_files)
        self.sample_deg, self.anchor_plate_id = sample_deg, anchor_plate_id
        self.folder = os.path.join(cache_dir, '{}-{}'.format(name, _model_key(model_dir, sum(self.files, []))),
                                   'anchor{}_s{:g}'.format(anchor_plate_id, sample_deg))
        os.makedirs(self.folder, exist_ok=True)
        self._model = None
        self.n_resolved = 0

    def samples(self, time_ma):
        path = os.path.join(self.folder, 't{:09.3f}.npz'.format(float(time_ma)))
        if os.path.exists(path):
            with np.load(path) as f:
                return {k: f[k] for k in ('lon', 'lat', 'convergence_km_per_myr')}
        if self._model is None:
            self._model = load_model(*self.files)
        s = trench_samples_at(*self._model, time_ma, self.sample_deg, self.anchor_plate_id)
        np.savez(path, **s)
        self.n_resolved += 1
        return s
