"""Drive the asthenosphere layer (slablet/asthenosphere.py) with a GPlates reconstruction on the spherical mesh.

    setting = PlateSetting(cfg, mesh)               cfg: a run config (configs/*.yaml), mesh: sphere_mesh.voronoi_mesh
    forcing, info = setting.forcing_at(time_ma, params)
    fields = layer_fields(mesh, h, forcing, params)

At each time, from the reconstruction (in the frame of the config's anchor plate, which must be a mantle frame; the
mesosphere is fixed in it):
  plates     plate ID of each cell centre (resolved topologies) and that plate's rotation over [t + 1, t] Myr, so the
             drag across each edge is exact for rigid plates (sphere_mesh.rigid_edge_flux);
  walls      mesh edges whose centre-to-centre arc crosses a subduction zone line (always: the slab is a curtain);
  trench sink (trench_mode 'sink' only) the layer that the subducting plate's drag brings to a trench leaves with
             the slab instead of piling up: each trench sample (slablet/gplates/trench_samples.py, 1 deg apart) takes
             h x (1/2) x (subducting plate velocity towards the trench, mantle frame) x its length from the cell just
             on the subducting side (60 km from the trench); none where that plate moves away.  The 1/2 is the
             depth-mean of the drag (Couette) profile, so the sink passes on exactly what drag delivers;
  keels      cells whose centre lies in a craton polygon (keel_mode 'craton'), optionally leaving some out by plate ID,
             or in a continent polygon (keel_mode 'continent');
  supply     uniform over open cells, or (supply_where 'ocean') only where the cell is not continent, as Buck;
  ridges     ridge samples (slablet/gplates/ridge_samples.py): sink = normal divergence x lithosphere_km x length,
             into the cell holding the sample; none where that cell is continent (rifts) or keel.
All geometry is on the sphere (great-circle arcs, Voronoi cells = nearest centre).  Driven only by the plate history.
"""
import numpy as np
import pygplates

from slablet import asthenosphere as A
from slablet import sphere_mesh as SM
from slablet.gplates.models import load as load_reconstruction
from slablet.gplates.ridge_samples import CachedRidgeSamples
from slablet.gplates.trench_samples import CachedTrenchSamples
from slablet.sphere import lonlat_to_xyz, move_along_azimuth

TRENCH_SAMPLE_DEG = 1.0
TRENCH_SIDE_KM = 60.0


class PlateSetting:
    def __init__(self, cfg, mesh, exclude_craton_ids=()):
        recon = load_reconstruction(cfg)
        self.cfg, self.mesh = cfg, mesh
        self.anchor = int(cfg.get('anchor_plate_id', 0))
        self.rm = pygplates.RotationModel(recon.rotation_files, default_anchor_plate_id=self.anchor)
        self.topo = [pygplates.FeatureCollection(f) for f in recon.topology_files]
        self.exclude = set(int(x) for x in exclude_craton_ids)
        self.polys = {}
        for k, files in (('craton', recon.craton_files), ('continent', recon.continent_files)):
            fc = pygplates.FeatureCollection()
            for f in files or []:
                for feat in pygplates.FeatureCollection(f):
                    if k == 'craton' and feat.get_reconstruction_plate_id() in self.exclude:
                        continue
                    fc.add(feat)
            self.polys[k] = fc
        self.ridges = CachedRidgeSamples.for_config(cfg)
        self.trenches = CachedTrenchSamples.for_config(cfg, sample_deg=TRENCH_SAMPLE_DEG)
        self.points = [pygplates.PointOnSphere(float(la), float(lo)) for la, lo in zip(mesh.lat, mesh.lon)]

    def _inside(self, features, time_ma):
        part = pygplates.PlatePartitioner(features, self.rm, float(time_ma))
        return part, np.array([part.partition_point(p) is not None for p in self.points])

    def plate_ids(self, time_ma):
        part = pygplates.PlatePartitioner(self.topo, self.rm, float(time_ma))
        ids = np.zeros(self.mesh.n, int)
        for k, p in enumerate(self.points):
            r = part.partition_point(p)
            ids[k] = r.get_feature().get_reconstruction_plate_id() if r is not None else 0
        return ids

    def omegas(self, ids, time_ma):
        """Rotation vector (rad/Myr) of each plate over [t + 1, t] Myr, per cell."""
        om = {}
        for pid in np.unique(ids):
            rot = self.rm.get_rotation(float(time_ma), int(pid), float(time_ma) + 1.0, anchor_plate_id=self.anchor)
            if rot.represents_identity_rotation():
                om[pid] = np.zeros(3)
                continue
            pole, angle = rot.get_euler_pole_and_angle()
            om[pid] = np.array(pole.to_xyz()) * angle
        return np.array([om[i] for i in ids])

    def trench_segments(self, time_ma, step_deg=0.5):
        """Subduction zone lines as short great-circle arcs (a, b), unit vectors."""
        snap = pygplates.TopologicalSnapshot(self.topo, self.rm, float(time_ma), self.anchor)
        a, b = [], []
        for sec in snap.get_resolved_topological_sections():
            if sec.get_topological_section_feature().get_feature_type() != pygplates.FeatureType.gpml_subduction_zone:
                continue
            for sub in sec.get_shared_sub_segments():
                pts = np.array([p.to_xyz() for p in
                                sub.get_resolved_geometry().to_tessellated(np.radians(step_deg)).get_points()])
                if len(pts) > 1:
                    a.append(pts[:-1])
                    b.append(pts[1:])
        if not a:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.concatenate(a), np.concatenate(b)

    def forcing_at(self, time_ma, p):
        """Forcing over [t + 1, t] Myr, and what went into it (for maps and checks)."""
        mesh = self.mesh
        ids = self.plate_ids(time_ma)
        om = self.omegas(ids, time_ma)
        qi, qj = SM.rigid_edge_flux(mesh, om[mesh.edge_i], om[mesh.edge_j])
        _, continent = self._inside(self.polys['continent'], time_ma)
        if p.keel_mode == 'continent':                  # all continental lithosphere as keels
            keel = continent.copy()
        else:
            _, keel = self._inside(self.polys['craton'], time_ma)
        seg_a, seg_b = self.trench_segments(time_ma)
        blocked = SM.edges_crossing(mesh, seg_a, seg_b)
        outflow, n_trench = None, 0
        if p.trench_mode == 'sink':
            ts = self.trenches.samples(time_ma)
            ok = np.isfinite(ts['downdip_azimuth_deg'])
            az = ts['downdip_azimuth_deg'][ok]
            lo, la = move_along_azimuth(ts['lon'][ok], ts['lat'][ok], az + 180.0, TRENCH_SIDE_KM, mesh.radius_km)
            c = SM.nearest_cell(mesh, lo, la)
            r = lonlat_to_xyz(lo, la)
            v = np.cross(om[c], r * mesh.radius_km)                    # subducting plate velocity, km/Myr
            east = np.column_stack([-np.sin(np.radians(lo)), np.cos(np.radians(lo)), np.zeros(len(lo))])
            north = np.column_stack([-np.sin(np.radians(la)) * np.cos(np.radians(lo)),
                                     -np.sin(np.radians(la)) * np.sin(np.radians(lo)), np.cos(np.radians(la))])
            u_in = (np.einsum('ij,ij->i', v, east) * np.sin(np.radians(az))
                    + np.einsum('ij,ij->i', v, north) * np.cos(np.radians(az)))     # towards the trench
            length = np.radians(TRENCH_SAMPLE_DEG) * mesh.radius_km
            outflow = np.bincount(c, 0.5 * np.maximum(u_in, 0.0) * length, minlength=mesh.n)
            n_trench = int(ok.sum())
        s = self.ridges.samples(time_ma)
        cell = SM.nearest_cell(mesh, s['lon'], s['lat'])
        on_land = continent[cell] | keel[cell]
        sink = np.bincount(cell[~on_land], (s['divergence_km_per_myr'] * p.lithosphere_km * s['length_km'])[~on_land],
                           minlength=mesh.n)
        f = A.Forcing(qi, qj, blocked=blocked, closed=keel, sink=sink, outflow=outflow, supply=p.supply_mode,
                      supply_mask=~continent if p.supply_where == 'ocean' else None, balance_outflow=True)
        info = dict(plate_id=ids, omega=om, keel=keel, continent=continent, trench_a=seg_a, trench_b=seg_b,
                    ridge=s, ridge_on_land=on_land, n_trench_sink_samples=n_trench)
        return f, info


def layer_fields(mesh, h, forcing, p, min_h_km=10.0, c=None):
    """What the maps show, per cell: depth anomaly d (km, + = deeper; NaN under keels), and the depth-mean layer
    velocity (km/Myr, east and north) of the total flow and of its drag and pressure-driven parts (NaN where
    h < min_h_km), and the flux divergence (km/Myr, + = layer leaving).  With plume fraction c (plume mode),
    also c_plume."""
    drag, press = A.edge_flux_parts(mesh, h, forcing, p, c)
    keel = np.zeros(mesh.n, bool) if forcing.closed is None else np.asarray(forcing.closed, bool)
    thin = keel | (h < min_h_km)
    out = {'h_km': np.where(keel, np.nan, h), 'd_km': np.where(keel, np.nan, p.depth_anomaly_km(h, c))}
    if c is not None:
        out['c_plume'] = np.where(keel, np.nan, c)
    for name, F in (('total', drag + press), ('drag', drag), ('pressure', press)):
        e, n = SM.east_north(mesh, SM.cell_vectors(mesh, F))
        out[name + '_east'] = np.where(thin, np.nan, e / np.maximum(h, min_h_km))
        out[name + '_north'] = np.where(thin, np.nan, n / np.maximum(h, min_h_km))
    tot = drag + press
    out['divergence_km_per_myr'] = (np.bincount(mesh.edge_i, tot, minlength=mesh.n)
                                    - np.bincount(mesh.edge_j, tot, minlength=mesh.n)) / mesh.area
    return out
