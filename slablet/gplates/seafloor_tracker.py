"""Seafloor age and plate velocities through a topological reconstruction, cached per Myr.

    conda run -n pygmt17 python -m slablet.gplates.seafloor_tracker configs/muller2022_dip_pile_sharp.yaml

Seafloor: every Myr, points are seeded every SEED_DEG along every diverging plate boundary (convergence below
-DIVERGE_KM_PER_MYR; not subduction zones), SEED_OFFSET_DEG to each side.  All points move with the topologies one Myr
at a time (pygplates TopologicalModel.reconstruct_geometry) and are dropped by pygplates' default deactivation
(subducted, or consumed at a ridge).  Before the run's start there is no tracked seafloor, so ages are missing (NaN)
until seafloor made after the start reaches the trenches: allow 200-300 Myr of spin-up.

Written per Myr t to <cache>/<model>/anchor<id>_s<sample_deg>_v2/seafloor_v1/t<t>.npz (beside the trench samples):
    trench_age     seafloor age (Myr) at each trench sample of that time (same order as the trench-sample cache): the
                   nearest tracked point within PROBE_RADIUS_KM of a point PROBE_KM from the trench on the subducting
                   side (away from the overriding plate; the trench point itself where that side is unknown); NaN: none
    vel_e, vel_n   plate velocity (km/Myr, east / north, in the anchor plate's frame) at the centres of the
                   VEL_DEG grid (lat ascending from -90 + VEL_DEG / 2, lon from -180 + VEL_DEG / 2); NaN outside plates
    age_grid       (every GRID_EVERY_MYR, or --grid-every) seafloor age on the 1 deg cell-centre grid: the nearest
                   tracked point within GRID_RADIUS_KM, else NaN (continents, and ocean older than the tracking)
--grids-only <folder> writes just the age grids (every --grid-every Myr) to <folder>/t<t>.npz and leaves the cache alone.
--rift-points <file.npz> writes just the diverging-boundary points and their opening rates every Myr (no tracking).
    n_points       tracked points alive at t
Distances are great-circle distances on a sphere of radius 6371 km.
"""
import argparse
import os
import time

import numpy as np
import pygplates
import yaml
from scipy.spatial import cKDTree

from slablet.gplates.trench_samples import CachedTrenchSamples        # noqa: E402

SEED_DEG = 1.0
SEED_OFFSET_DEG = 0.1
DIVERGE_KM_PER_MYR = 5.0
PROBE_KM = 100.0
PROBE_RADIUS_KM = 300.0
GRID_RADIUS_KM = 150.0
GRID_EVERY_MYR = 10
VEL_DEG = 2.0
R_KM = 6371.0
VERSION = 'seafloor_v1'


def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def chord(km):
    return 2 * np.sin(km / R_KM / 2)


def offset_points(xyz, azimuth_deg, km):
    """Points km along the great circle from xyz (unit vectors) towards azimuth (clockwise from north)."""
    z = np.array([0.0, 0.0, 1.0])
    east = np.cross(z, xyz)
    norm = np.linalg.norm(east, axis=1, keepdims=True)
    east = np.where(norm > 1e-12, east / np.maximum(norm, 1e-12), np.array([0.0, 1.0, 0.0]))
    north = np.cross(xyz, east)
    a = np.radians(azimuth_deg)[:, None]
    d = np.sin(a) * east + np.cos(a) * north
    ang = km / R_KM
    return np.cos(ang) * xyz + np.sin(ang) * d


def seeds(snap):
    """Unit vectors of new seafloor points at the diverging boundaries of a snapshot."""
    stats = snap.calculate_plate_boundary_statistics(np.radians(SEED_DEG))
    out = []
    for s in stats:
        v = s.convergence_velocity_orthogonal
        if not np.isfinite(v) or v > -DIVERGE_KM_PER_MYR:
            continue
        if s.boundary_feature.get_feature_type() == pygplates.FeatureType.gpml_subduction_zone:
            continue
        p, n = np.array(s.boundary_point.to_xyz()), np.array(s.boundary_normal.to_xyz())
        for sg in (1.0, -1.0):
            q = p + sg * np.radians(SEED_OFFSET_DEG) * n
            out.append(q / np.linalg.norm(q))
    return np.array(out).reshape(-1, 3)


def diverging_points(snap):
    """Points on a snapshot's diverging boundaries (where seeds() puts new seafloor either side), and the full opening
    rate there (km/Myr): every SEED_DEG along every boundary opening faster than DIVERGE_KM_PER_MYR, not subduction
    zones."""
    stats = snap.calculate_plate_boundary_statistics(np.radians(SEED_DEG))
    pts, rate = [], []
    for s in stats:
        v = s.convergence_velocity_orthogonal
        if not np.isfinite(v) or v > -DIVERGE_KM_PER_MYR:
            continue
        if s.boundary_feature.get_feature_type() == pygplates.FeatureType.gpml_subduction_zone:
            continue
        pts.append(s.boundary_point.to_xyz())
        rate.append(-v)
    return np.array(pts, float).reshape(-1, 3), np.array(rate, float)


def rift_points(cfg, path, sample_deg=1.0, start=None, end=0):
    """Every Myr from start to end, the diverging-boundary points (diverging_points) to an npz: time (Ma), lon, lat,
    opening_km_per_myr.  No tracking: where and when the reconstruction opens, for models that start something at
    rifting (IPV's keel drip fronts)."""
    trenches, _ = folder_for(cfg, sample_deg)
    start = int(cfg['start_ma'] if start is None else start)
    if trenches._model is None:
        from slablet.gplates.trench_samples import load_model
        trenches._model = load_model(*trenches.files)
    rm, feats = trenches._model
    tm = pygplates.TopologicalModel(feats, rm, anchor_plate_id=int(cfg.get('anchor_plate_id', 0)))
    out = {'time': [], 'lon': [], 'lat': [], 'opening_km_per_myr': []}
    for t in range(start, end - 1, -1):
        xyz, rate = diverging_points(tm.topological_snapshot(float(t)))
        out['time'].append(np.full(len(rate), float(t)))
        out['lon'].append(np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])))
        out['lat'].append(np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0))))
        out['opening_km_per_myr'].append(rate)
    out = {k: np.concatenate(v) for k, v in out.items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def velocity_grid(snap):
    lats = np.arange(-90 + VEL_DEG / 2, 90, VEL_DEG)
    lons = np.arange(-180 + VEL_DEG / 2, 180, VEL_DEG)
    lo, la = np.meshgrid(lons, lats)
    pts = [pygplates.PointOnSphere(float(a), float(b)) for a, b in zip(la.ravel(), lo.ravel())]
    vel = snap.get_point_velocities(pts, velocity_units=pygplates.VelocityUnits.kms_per_my)
    ve = np.full(len(pts), np.nan)
    vn = np.full(len(pts), np.nan)
    ok = [i for i, v in enumerate(vel) if v is not None]
    if ok:
        ned = pygplates.LocalCartesian.convert_from_geocentric_to_north_east_down([pts[i] for i in ok],
                                                                                   [vel[i] for i in ok])
        ve[ok] = [v.get_y() for v in ned]
        vn[ok] = [v.get_x() for v in ned]
    return ve.reshape(la.shape), vn.reshape(la.shape)


def trench_ages(samples, tree, ages):
    n = len(samples['lon'])
    out = np.full(n, np.nan)
    if n == 0 or tree is None:
        return out
    xyz = unit(samples['lon'], samples['lat'])
    az = np.asarray(samples.get('downdip_azimuth_deg', np.full(n, np.nan)), float)
    probe = xyz.copy()
    known = np.isfinite(az)
    if known.any():
        probe[known] = offset_points(xyz[known], (az[known] + 180.0) % 360.0, PROBE_KM)
    d, i = tree.query(probe, distance_upper_bound=chord(PROBE_RADIUS_KM))
    hit = np.isfinite(d)
    out[hit] = ages[i[hit]]
    return out


def age_grid(tree, ages):
    lats = np.arange(-89.5, 90, 1.0)
    lons = np.arange(-179.5, 180, 1.0)
    lo, la = np.meshgrid(lons, lats)
    out = np.full(lo.size, np.nan)
    if tree is not None:
        d, i = tree.query(unit(lo.ravel(), la.ravel()), distance_upper_bound=chord(GRID_RADIUS_KM))
        hit = np.isfinite(d)
        out[hit] = ages[i[hit]]
    return out.reshape(lo.shape).astype(np.float32)


def folder_for(cfg, sample_deg=1.0):
    trenches = CachedTrenchSamples.for_config(cfg, sample_deg)
    return trenches, os.path.join(trenches.folder, VERSION)


def track(cfg, sample_deg=1.0, start=None, end=0, grid_every=GRID_EVERY_MYR, grids_only=None):
    """Track the seafloor from start to end and write the cache.  grids_only: a folder to write only the age grid,
    every grid_every Myr, as <folder>/t<t>.npz (age_grid), leaving the cache untouched."""
    trenches, out = folder_for(cfg, sample_deg)
    if grids_only is not None:
        out = grids_only
    os.makedirs(out, exist_ok=True)
    start = int(cfg['start_ma'] if start is None else start)
    if trenches._model is None:
        from slablet.gplates.trench_samples import load_model
        trenches._model = load_model(*trenches.files)
    rm, feats = trenches._model
    anchor = int(cfg.get('anchor_plate_id', 0))
    tm = pygplates.TopologicalModel(feats, rm, anchor_plate_id=anchor)
    pts = np.zeros((0, 3))
    ages = np.zeros(0)
    t0 = time.perf_counter()
    for t in range(start, end - 1, -1):
        snap = tm.topological_snapshot(float(t))
        new = seeds(snap)
        pts = np.concatenate([pts, new])
        ages = np.concatenate([ages, np.zeros(len(new))])
        tree = cKDTree(pts) if len(pts) else None
        if grids_only is not None:
            rec = {'trench_age': np.zeros(0)}
            if t % grid_every == 0:
                np.savez_compressed(os.path.join(out, 't{:04d}.npz'.format(t)), age_grid=age_grid(tree, ages))
        else:
            rec = {'trench_age': trench_ages(trenches.samples(float(t)), tree, ages), 'n_points': len(pts)}
            rec['vel_e'], rec['vel_n'] = velocity_grid(snap)
            if t % grid_every == 0:
                rec['age_grid'] = age_grid(tree, ages)
            np.savez_compressed(os.path.join(out, 't{:04d}.npz'.format(t)), **rec)
        if t > end and len(pts):
            span = tm.reconstruct_geometry(pygplates.MultiPointOnSphere([tuple(p) for p in pts]), initial_time=float(t),
                                           oldest_time=float(t), youngest_time=float(t - 1), time_increment=1.0,
                                           deactivate_points=pygplates.ReconstructedGeometryTimeSpan.DefaultDeactivatePoints())
            moved = span.get_geometry_points(float(t - 1), return_inactive_points=True)
            keep = np.array([p is not None for p in moved])
            pts = np.array([p.to_xyz() for p in moved if p is not None]).reshape(-1, 3)
            ages = ages[keep] + 1.0
        if t % 50 == 0:
            a = rec['trench_age']
            print('{:4d} Ma: {:6d} points, trench ages known {:3.0f} %, median {:5.1f} Myr  ({:.0f} s)'.format(
                t, len(pts), 100 * np.isfinite(a).mean() if len(a) else 0, np.nanmedian(a) if np.isfinite(a).any() else np.nan,
                time.perf_counter() - t0), flush=True)
    return out


class SeafloorCache:
    """Read the tracker's per-Myr output (nearest whole Myr)."""

    def __init__(self, cfg, sample_deg=1.0):
        self.folder = folder_for(cfg, sample_deg)[1]
        if not os.path.isdir(self.folder):
            raise SystemExit('no seafloor cache at {}: run slablet/gplates/seafloor_tracker.py first'.format(self.folder))

    def at(self, t):
        t = int(round(t))
        if getattr(self, '_last', (None,))[0] != t:
            with np.load(os.path.join(self.folder, 't{:04d}.npz'.format(t))) as f:
                self._last = (t, {k: f[k] for k in f.files})
        return self._last[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    ap.add_argument('--start', type=float)
    ap.add_argument('--end', type=float, default=0)
    ap.add_argument('--grid-every', type=int, default=GRID_EVERY_MYR, help='Myr between age grids')
    ap.add_argument('--grids-only', help='write only the age grids, to this folder (the cache is left alone)')
    ap.add_argument('--rift-points', help='write only the diverging-boundary points every Myr, to this npz (no tracking)')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.rift_points:
        print(rift_points(cfg, args.rift_points, start=args.start, end=int(args.end)))
        return
    print(track(cfg, start=args.start, end=int(args.end), grid_every=args.grid_every, grids_only=args.grids_only))


if __name__ == '__main__':
    main()
