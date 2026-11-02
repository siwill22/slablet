"""Trench samples for the mantle model from a GPlates reconstruction, cached per time.

The samples follow the contract in slablet/core.py: lon, lat (degrees) and
convergence_km_per_myr (> 0 = converging), at points every sample_deg along the
subduction zones of the resolved topologies.  Convergence is the component of the
relative plate velocity along the boundary normal, over [t + 1 Myr, t]
(pygplates' default velocity interval).

The positions are in the frame of anchor_plate_id.  Slablets sink vertically in that
frame, so it must be a mantle frame (fixed to the deep mantle), not a palaeomagnetic
frame fixed to the spin axis.  Model folders often hold several rotation files for
different frames (Müller et al. 2022: palaeomagnetic, optimised mantle, no-net-rotation),
so the files always come from the run config's reconstruction block (slablet.gplates.models),
never discovered.

Resolving topologies is most of the cost of a mantle run (about 90 ms per Myr for
Müller et al. 2019), so samples are cached as one .npz per time, under
cache_dir/<model name>-<hash of the files used>/anchor<id>_s<sample_deg>_v2/ (v2: with the
down-dip azimuth and trench velocity; the v1 folders hold only the three required keys).
"""
import hashlib
import os

import numpy as np
import pygplates


def load_model(rotation_files, topology_files):
    return (pygplates.RotationModel(rotation_files),
            [pygplates.FeatureCollection(f) for f in topology_files])


SAMPLE_KEYS = ('lon', 'lat', 'convergence_km_per_myr', 'downdip_azimuth_deg',
               'trench_east_km_per_myr', 'trench_north_km_per_myr')


def _empty():
    return {k: np.zeros(0) for k in SAMPLE_KEYS}


def trench_samples_at(rotation_model, features, time_ma, sample_deg=1.0, anchor_plate_id=0):
    """Samples along the subduction zones resolved at time_ma (only converging points).

    Besides the required keys, each sample has
      downdip_azimuth_deg   direction from the trench towards the overriding plate (clockwise
                            from north): the side of the boundary whose point 0.5° away lies in
                            the overriding plate's polygon; NaN where neither side or both do,
                            or there is no overriding plate
      trench_east/north_km_per_myr   velocity of the trench itself in the anchor frame
    """
    snap = pygplates.TopologicalSnapshot(features, rotation_model, float(time_ma), anchor_plate_id)
    stats = snap.calculate_plate_boundary_statistics(
        np.radians(sample_deg), boundary_section_filter=pygplates.FeatureType.gpml_subduction_zone)
    stats = [s for s in stats if np.isfinite(s.convergence_velocity_orthogonal)
             and s.convergence_velocity_orthogonal > 0]
    if not stats:
        return _empty()
    conv = np.array([s.convergence_velocity_orthogonal for s in stats], dtype=float)
    latlon = np.array([s.boundary_point.to_lat_lon() for s in stats], dtype=float)
    azimuth = np.full(len(stats), np.nan)
    step = np.radians(0.5)
    for i, s in enumerate(stats):
        over = s.shared_sub_segment.get_overriding_plate()
        if over is None:
            continue
        poly = over.get_resolved_boundary()
        p, n = np.array(s.boundary_point.to_xyz()), np.array(s.boundary_normal.to_xyz())
        inside = [poly.is_point_in_polygon(tuple((p + sg * step * n) / np.linalg.norm(p + sg * step * n)))
                  for sg in (1.0, -1.0)]
        if inside[0] != inside[1]:
            azimuth[i] = np.degrees(s.boundary_normal_azimuth) + (0.0 if inside[0] else 180.0)
    ned = pygplates.LocalCartesian.convert_from_geocentric_to_north_east_down(
        [s.boundary_point for s in stats], [s.boundary_velocity for s in stats])
    return {'lon': latlon[:, 1], 'lat': latlon[:, 0], 'convergence_km_per_myr': conv,
            'downdip_azimuth_deg': azimuth % 360.0,
            'trench_east_km_per_myr': np.array([v.get_y() for v in ned]),
            'trench_north_km_per_myr': np.array([v.get_x() for v in ned])}


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

    def __init__(self, recon, name, cache_dir, sample_deg=1.0, anchor_plate_id=0):
        """recon: a slablet.gplates.models.Reconstruction."""
        self.files = (recon.rotation_files, recon.topology_files)
        self.sample_deg, self.anchor_plate_id = sample_deg, anchor_plate_id
        self.folder = os.path.join(cache_dir, '{}-{}'.format(name, _model_key(recon.model_dir, sum(self.files, []))),
                                   'anchor{}_s{:g}_v2'.format(anchor_plate_id, sample_deg))
        os.makedirs(self.folder, exist_ok=True)
        self._model = None
        self.n_resolved = 0

    @classmethod
    def for_config(cls, cfg, sample_deg=1.0):
        """The cache for a run config (name, cache_dir, anchor_plate_id and its reconstruction)."""
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
        s = trench_samples_at(*self._model, time_ma, self.sample_deg, self.anchor_plate_id)
        tmp = '{}.{}.tmp.npz'.format(path[:-4], os.getpid())   # atomic: parallel runs may share the cache
        np.savez(tmp, **s)
        os.replace(tmp, path)
        self.n_resolved += 1
        return s
