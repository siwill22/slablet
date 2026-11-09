"""A real reconstruction as a plate history (the contract in slablet/plate_history.py).

    history = RealPlateHistory(cfg)          cfg: a run config with a reconstruction block (configs/*.yaml)
    step = history.step(time_ma)             a PlateStep over [t + 1, t] Myr

Plate identity and the continent and craton masks are exact point-in-polygon tests on the sphere
(ptt.utils.points_in_polygons.find_polygons).  On 40,000 points against Müller et al. (2022) at 100 Ma
it agreed everywhere with pygplates' PlatePartitioner and took 0.13 s against 0.59 s.  Plates are the
resolved topological boundaries only: deforming networks count as the rigid plates under them
(Müller et al. 2022 has none).  Rotation vectors are over [t + 1, t] Myr, the interval of the trench and
ridge samples, which come from their per-time caches (trench_samples.py, ridge_samples.py).  The
subducting seafloor age and the seafloor-age grid come from the seafloor tracker's cache
(seafloor_tracker.py; the grid is the nearest saved one, every 10 Myr) when it exists, else NaN.
Everything is in the frame of the config's anchor plate, which must be fixed to the mantle.
"""
import os

import numpy as np
import pygplates
import xarray as xr
from ptt.utils import points_in_polygons

from slablet import plate_history as PH
from slablet.gplates.models import load as load_reconstruction
from slablet.gplates.ridge_samples import CachedRidgeSamples
from slablet.gplates.seafloor_tracker import GRID_EVERY_MYR, SeafloorCache, folder_for
from slablet.gplates.trench_samples import CachedTrenchSamples

TRENCH_SAMPLE_DEG = 1.0
RIDGE_SAMPLE_DEG = 0.5
TRENCH_LINE_STEP_DEG = 0.5
AGE_GRID_DEG = 1.0


def _points(lon, lat):
    return [pygplates.PointOnSphere(float(la), float(lo)) for lo, la in
            zip(np.ravel(np.asarray(lon, float)), np.ravel(np.asarray(lat, float)))]


def _query(polygons, proxies, fill, overlaps='any'):
    """A point query over polygons: the proxy of the polygon holding each point, else fill.  Where polygons overlap,
    overlaps='highest' takes the highest proxy (for plate IDs: pygplates PlatePartitioner's default order, highest
    plate ID first); 'any' takes whichever ptt meets first (enough for a mask, where every proxy is True)."""
    def at(lon, lat):
        shape = np.shape(lon)
        if not polygons:
            return np.full(shape, fill)
        if overlaps == 'highest':
            found = points_in_polygons.find_polygons(_points(lon, lat), polygons, proxies, all_polygons=True)
            return np.array([fill if not f else max(f) for f in found]).reshape(shape)
        found = points_in_polygons.find_polygons(_points(lon, lat), polygons, proxies)
        return np.array([fill if f is None else f for f in found]).reshape(shape)
    return at


class RealPlateHistory:
    def __init__(self, cfg):
        recon = load_reconstruction(cfg)
        if recon.frame == 'spin axis':
            raise SystemExit('{}: the anchor plate is fixed to the spin axis; a plate history needs a mantle frame'
                             .format(recon.label))
        self.recon = recon
        self.anchor = int(cfg.get('anchor_plate_id', 0))
        self.rm = pygplates.RotationModel(recon.rotation_files, default_anchor_plate_id=self.anchor)
        self.topo = [pygplates.FeatureCollection(f) for f in recon.topology_files]
        self.polygons = {k: [pygplates.FeatureCollection(f) for f in (files or [])]
                         for k, files in (('continent', recon.continent_files), ('craton', recon.craton_files))}
        self.trenches = CachedTrenchSamples.for_config(cfg, sample_deg=TRENCH_SAMPLE_DEG)
        self.ridges = CachedRidgeSamples.for_config(cfg, sample_deg=RIDGE_SAMPLE_DEG)
        self.seafloor = (SeafloorCache(cfg, TRENCH_SAMPLE_DEG)
                         if os.path.isdir(folder_for(cfg, TRENCH_SAMPLE_DEG)[1]) else None)

    def _plates(self, t):
        resolved = []
        pygplates.resolve_topologies(self.topo, self.rm, resolved, float(t), anchor_plate_id=self.anchor,
                                     resolve_topology_types=pygplates.ResolveTopologyType.boundary)
        return ([r.get_resolved_boundary() for r in resolved],
                [int(r.get_feature().get_reconstruction_plate_id()) for r in resolved])

    def _omega(self, pids, t):
        om = {}
        for pid in sorted(set(pids)):
            rot = self.rm.get_rotation(float(t), pid, float(t) + 1.0, anchor_plate_id=self.anchor)
            if rot.represents_identity_rotation():
                om[pid] = np.zeros(3)
            else:
                pole, angle = rot.get_euler_pole_and_angle()
                om[pid] = np.array(pole.to_xyz()) * angle
        return om

    def _mask(self, kind, t):
        rfgs = []
        for fc in self.polygons[kind]:
            pygplates.reconstruct(fc, self.rm, rfgs, float(t), anchor_plate_id=self.anchor)
        polys = [r.get_reconstructed_geometry() for r in rfgs
                 if isinstance(r.get_reconstructed_geometry(), pygplates.PolygonOnSphere)]
        return _query(polys, [True] * len(polys), False)

    def _trench_lines(self, t):
        snap = pygplates.TopologicalSnapshot(self.topo, self.rm, float(t), self.anchor)
        lines = []
        for sec in snap.get_resolved_topological_sections():
            if sec.get_topological_section_feature().get_feature_type() != pygplates.FeatureType.gpml_subduction_zone:
                continue
            for sub in sec.get_shared_sub_segments():
                pts = sub.get_resolved_geometry().to_tessellated(np.radians(TRENCH_LINE_STEP_DEG)).to_lat_lon_array()
                if len(pts) > 1:
                    lines.append(pts[:, ::-1].copy())
        return lines

    def _seafloor(self, t, n_trench):
        lons, lats = PH.cell_centred_grid(AGE_GRID_DEG)
        grid = np.full((len(lats), len(lons)), np.nan, np.float32)
        trench_age = np.full(n_trench, np.nan)
        attrs = {'units': 'Myr', 'source': 'none: no seafloor tracker cache'}
        if self.seafloor is not None:
            rec = self.seafloor.at(t)
            if len(rec['trench_age']) == n_trench:
                trench_age = rec['trench_age'].astype(float)
            t_grid = int(round(t / GRID_EVERY_MYR)) * GRID_EVERY_MYR
            grid = self.seafloor.at(t_grid)['age_grid']
            attrs = {'units': 'Myr', 'source': 'slablet seafloor tracker', 'grid_time': t_grid}
        da = xr.DataArray(grid, dims=('lat', 'lon'), coords={'lat': lats, 'lon': lons},
                          name='seafloor_age', attrs=attrs)
        return da, trench_age

    def step(self, time_ma, delta_time=1.0):
        if delta_time != 1.0:
            raise ValueError('the cached samples are rates over [t + 1, t] Myr; only delta_time = 1 is supported')
        t = float(time_ma)
        polys, pids = self._plates(t)
        trench = dict(self.trenches.samples(t))
        seafloor_age, trench['seafloor_age_myr'] = self._seafloor(t, len(trench['lon']))
        trench['downdip_azimuth_deg'] = np.asarray(trench['downdip_azimuth_deg'], float) % 360.0
        return PH.check(PH.PlateStep(
            time=t, delta_time=delta_time,
            plate_id_at=_query(polys, pids, 0, overlaps='highest'),
            omega=self._omega(pids, t),
            trench={k: np.asarray(trench[k], float) for k in PH.TRENCH_KEYS},
            ridge={k: np.asarray(v, float) for k, v in self.ridges.samples(t).items()},
            seafloor_age=seafloor_age,
            continent_at=self._mask('continent', t),
            craton_at=self._mask('craton', t),
            trench_lines=self._trench_lines(t)))
