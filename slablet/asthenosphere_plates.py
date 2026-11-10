"""Drive the asthenosphere layer (slablet/asthenosphere.py) with any plate history, through the plate-history contract.

    forcing, info = forcing_from_step(mesh, step, params)        step: a PlateStep (slablet/plate_history.py)
    for t, h, forcing, info, log in run_layer(step_at, mesh, params, start_ma=80):   step_at(t) -> PlateStep
        ...
    fields = layer_fields(mesh, h, forcing, params)

The same physics for a real reconstruction (slablet.gplates.plate_history.RealPlateHistory(cfg).step) and a
synthetic world (PlateDriver's plate_step).  Over each step [t + dt, t], from the step:
  plates     plate identity of each cell centre and that plate's rotation vector, so the drag across each edge is
             exact for rigid plates (sphere_mesh.rigid_edge_flux);
  walls      mesh edges whose centre-to-centre arc crosses a trench line (always: the slab is a curtain);
  trench sink (trench_mode 'sink' only) the layer that the subducting plate's drag brings to a trench leaves with
             the slab: each trench sample takes h x (1/2) x (subducting plate velocity towards the trench) x its
             length from the cell 60 km on the subducting side; none where that plate moves away.  The 1/2 is the
             depth-mean of the drag (Couette) profile;
  keels      cells whose centre is craton (keel_mode 'craton') or continent (keel_mode 'continent');
  supply     uniform over open cells, or (supply_where 'ocean') only where the cell is not continent, as Buck;
  ridges     sink = normal divergence x lithosphere_km x length, into the cell holding each ridge sample; none
             where that cell is continent (rifts) or keel.
run_layer moves the keels between steps (uncovered cells start empty, covered layer is bulldozed or deleted, the
params' covered_mode), can share the supply out unevenly (share_at, slablet/asthenosphere_supply.py) and can cap
the thickness (cap_km: any excess is spread over the supply cells, keeping the volume).  All geometry is on the
sphere.  No pygplates: this is shared physics.
"""
import numpy as np

from slablet import asthenosphere as A
from slablet import sphere_mesh as SM
from slablet.sphere import lonlat_to_xyz, move_along_azimuth

TRENCH_SIDE_KM = 60.0


def _segments(lines):
    """Trench lines (lists of (n, 2) lon/lat arrays) as short great-circle arcs (a, b), unit vectors."""
    a, b = [], []
    for line in lines or []:
        xyz = lonlat_to_xyz(line[:, 0], line[:, 1])
        if len(xyz) > 1:
            a.append(xyz[:-1])
            b.append(xyz[1:])
    if not a:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(a), np.concatenate(b)


def _trench_spacing_km(step, radius_km):
    """Length of trench each sample stands for: the median spacing of neighbouring samples."""
    tr = step.trench
    if len(tr['lon']) < 2:
        return 0.0
    xyz = lonlat_to_xyz(tr['lon'], tr['lat'])
    gaps = np.arccos(np.clip(np.einsum('ij,ij->i', xyz[:-1], xyz[1:]), -1, 1)) * radius_km
    return float(np.median(gaps[gaps > 0]))


def forcing_from_step(mesh, step, p):
    """Forcing over one plate step, and what went into it (for maps and checks)."""
    if step.trench_lines is None:
        raise ValueError('the asthenosphere layer needs trench_lines in the plate step (its trench walls)')
    ids = np.asarray(step.plate_id_at(mesh.lon, mesh.lat), int)
    om = np.array([step.omega.get(int(i), np.zeros(3)) for i in ids], float)          # rad/Myr
    qi, qj = SM.rigid_edge_flux(mesh, om[mesh.edge_i], om[mesh.edge_j])
    continent = np.asarray(step.continent_at(mesh.lon, mesh.lat), bool)
    if p.keel_mode == 'continent':
        keel = continent.copy()
    elif p.keel_mode == 'craton':
        keel = np.asarray(step.craton_at(mesh.lon, mesh.lat), bool)
    else:
        raise ValueError('keel_mode {!r} is not supported on a plate history'.format(p.keel_mode))
    seg_a, seg_b = _segments(step.trench_lines)
    blocked = SM.edges_crossing(mesh, seg_a, seg_b)
    outflow, n_trench = None, 0
    if p.trench_mode == 'sink':
        ts = step.trench
        ok = np.isfinite(ts['downdip_azimuth_deg'])
        az = ts['downdip_azimuth_deg'][ok]
        lo, la = move_along_azimuth(ts['lon'][ok], ts['lat'][ok], az + 180.0, TRENCH_SIDE_KM, mesh.radius_km)
        c = SM.nearest_cell(mesh, lo, la)
        v = np.cross(om[c], lonlat_to_xyz(lo, la) * mesh.radius_km)            # subducting plate velocity, km/Myr
        east = np.column_stack([-np.sin(np.radians(lo)), np.cos(np.radians(lo)), np.zeros(len(lo))])
        north = np.column_stack([-np.sin(np.radians(la)) * np.cos(np.radians(lo)),
                                 -np.sin(np.radians(la)) * np.sin(np.radians(lo)), np.cos(np.radians(la))])
        u_in = (np.einsum('ij,ij->i', v, east) * np.sin(np.radians(az))
                + np.einsum('ij,ij->i', v, north) * np.cos(np.radians(az)))         # towards the trench
        length = _trench_spacing_km(step, mesh.radius_km)
        outflow = np.bincount(c, 0.5 * np.maximum(u_in, 0.0) * length, minlength=mesh.n)
        n_trench = int(ok.sum())
    s = step.ridge
    cell = SM.nearest_cell(mesh, s['lon'], s['lat'])
    on_land = continent[cell] | keel[cell]
    sink = np.bincount(cell[~on_land], (s['divergence_km_per_myr'] * p.lithosphere_km * s['length_km'])[~on_land],
                       minlength=mesh.n)
    f = A.Forcing(qi, qj, blocked=blocked, closed=keel, sink=sink, outflow=outflow, supply=p.supply_mode,
                  supply_mask=~continent if p.supply_where == 'ocean' else None, balance_outflow=True)
    info = dict(plate_id=ids, omega=om, keel=keel, continent=continent, trench_a=seg_a, trench_b=seg_b,
                ridge=s, ridge_on_land=on_land, n_trench_sink_samples=n_trench)
    return f, info


def _apply_cap(mesh, h, T, cap_km, closed, f):
    """Cut h to cap_km and spread the excess over the fed cells (open, and receiving supply) in proportion to area,
    keeping the volume; the excess keeps its tracer mixture.  Returns (h, T)."""
    if cap_km is None or not (h > cap_km).any():
        return h, T
    cut = np.maximum(h - cap_km, 0.0) * mesh.area
    excess = cut.sum()
    h = np.minimum(h, cap_km)
    fed = ~closed & (f.supply_mask if f.supply_mask is not None else True)
    if T is not None:
        amount = T * h * mesh.area + ((T * cut).sum(axis=1) / excess)[:, None] * np.where(
            fed, excess * mesh.area / mesh.area[fed].sum(), 0.0)
    h[fed] += excess / mesh.area[fed].sum()
    if T is not None:
        T = np.where(h > 1e-12, amount / np.where(h > 1e-12, h * mesh.area, 1.0), 0.0)
    return h, T


def run_layer(step_at, mesh, p, start_ma, end_ma=0, delta_time=1.0, cap_km=None, h0=None, share_at=None,
              tracers0=None, supply_tracers_at=None):
    """Run the layer from start_ma to end_ma through a plate history; yields (t, h, forcing, info, log) after each
    step over [t + delta_time, t].  step_at(t) returns the PlateStep for that interval.  The layer starts uniform at
    h_a outside keels (or h0).  Between steps the keels move (asthenosphere.move_keels, p.covered_mode).
    share_at(t, forcing) -> per-cell supply share (slablet/asthenosphere_supply.py); None: by area.
    tracers0 (k, n): passive tracer concentrations at the start, carried with the layer (log.tracers each step);
    keels bulldoze them with the layer, and a cap's excess keeps its mixture.  supply_tracers_at(t, forcing) -> (k,)
    or (k, n): what the supply carries (default 0).
    cap_km holds after the keels move as well as after each step, so a pile bulldozed ahead of a keel never enters
    a step above it (it would force many short sub-steps there: mobility grows as h^3)."""
    h, closed_old = None, None
    T = None if tracers0 is None else np.atleast_2d(np.asarray(tracers0, float))
    n = int(round((start_ma - end_ma) / delta_time))
    for k in range(1, n + 1):
        t = start_ma - k * delta_time
        f, info = forcing_from_step(mesh, step_at(t), p)
        if share_at is not None:
            f.supply_share = share_at(t, f)
        if T is not None and supply_tracers_at is not None:
            f.supply_tracers = supply_tracers_at(t, f)
        closed = np.asarray(f.closed, bool)
        if h is None:
            h = np.where(closed, 0.0, p.h_a_km if h0 is None else h0)
        else:
            if T is not None:           # move_keels is linear in what it moves, so tracer volume moves alike
                amount = np.array([A.move_keels(mesh, T[q] * h, closed_old, closed, mode=p.covered_mode)[0]
                                   for q in range(len(T))])
            h, _ = A.move_keels(mesh, h, closed_old, closed, mode=p.covered_mode)
            if T is not None:
                T = np.where(h > 1e-12, amount / np.where(h > 1e-12, h, 1.0), 0.0)
            h, T = _apply_cap(mesh, h, T, cap_km, closed, f)
        closed_old = closed
        h, log = A.advance(mesh, h, f, p, delta_time, tracers=T)
        h, T = _apply_cap(mesh, h, log.tracers, cap_km, closed, f)
        if T is not None:
            log.tracers = T
        yield t, h, f, info, log


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
