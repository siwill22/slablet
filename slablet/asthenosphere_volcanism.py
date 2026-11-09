"""Volcanism driven by the asthenosphere layer's flow (decision 18 of docs/plans/2026-11-08-cross-repo-reconciliation.md).

A melt source in the layer moves with the layer while the plate moves over it.  Where the layer, flowing relative to
the plate, moves towards thinner lithosphere, it rises along the base of the plate and can melt as it decompresses:

    u_rel = u_layer - u_plate          depth-mean layer velocity minus the plate's (km/Myr, per cell)
    w     = -u_rel . grad L            how fast the layer rises along the plate's base (km/Myr; + = up)

with L the lithosphere thickness: from the cooling law and the seafloor age under oceans (slablet.cooling, GDH1 by
default), a fixed thickness under continents, smoothed over a few cells (about 150 km on the 40,000-cell mesh) because
the base of the plate is smooth on scales below its own thickness while cell-to-cell noise in seafloor age is not.  The base of the plate moves with the plate, so it is the flow
relative to the plate that matters (the plan wrote u_layer . grad L; the two differ by the plate's own motion).

A chain grows along u_rel: its young end lies in the direction of u_rel, its spacing set by |u_rel|.  A source fixed
in the mantle (u_layer = 0) gives -u_plate, the familiar plate-motion direction; plate drag alone (u_layer = u_plate
/ 2, Couette flow) gives the same direction at half the speed.  So the angle between u_rel and -u_plate is the
layer's pressure-driven flow at work, the balance of case 1.

Not built yet: shear-driven upwelling where the sheared layer crosses thickness changes (Conrad et al. 2010);
fertility (melt is taken in proportion to w); a solidus depth limit; plumes.  No pygplates: shared physics on the
plate-history contract (slablet/plate_history.py).
"""
import numpy as np

from slablet import cooling
from slablet import sphere_mesh as SM

CONTINENT_KM = 150.0                            # lithosphere under continents outside the keels (an assumption)
SMOOTH_PASSES = 4                               # neighbour-averaging passes applied to L before its gradient


def plate_velocity(mesh, step):
    """(n, 3) plate velocity at each cell centre, km/Myr: omega x r of the plate there; zero off plates."""
    pid = np.asarray(step.plate_id_at(mesh.lon, mesh.lat))
    om = np.zeros((mesh.n, 3))
    for p in np.unique(pid):
        if int(p) in step.omega:
            om[pid == p] = step.omega[int(p)]
    return np.cross(om, mesh.xyz) * mesh.radius_km


def layer_velocity(mesh, h, forcing, p, h_min=1e-3):
    """(n, 3) depth-mean layer velocity, km/Myr: the layer's drag and pressure-driven flux over h; zero where empty."""
    from slablet.asthenosphere import edge_flux_parts
    drag, press = edge_flux_parts(mesh, h, forcing, p)
    v = SM.cell_vectors(mesh, drag + press) / np.maximum(h, h_min)[:, None]
    return np.where((h > h_min)[:, None], v, 0.0)


def lithosphere_km(mesh, step, law=None, continent_km=CONTINENT_KM):
    """Lithosphere thickness per cell (km): the cooling law's at the step's seafloor age under oceans, continent_km
    under continents (keels included; the layer is absent there anyway)."""
    age = step.seafloor_age
    lat, lon = age['lat'].values, age['lon'].values
    i = np.clip(np.round((mesh.lat - lat[0]) / (lat[1] - lat[0])).astype(int), 0, len(lat) - 1)
    j = np.round(((mesh.lon - lon[0]) % 360.0) / (lon[1] - lon[0])).astype(int) % len(lon)
    a = age.values[i, j]
    continent = np.asarray(step.continent_at(mesh.lon, mesh.lat), bool)
    ocean_l = cooling.thickness_km(cooling.get(law), np.where(np.isfinite(a), np.maximum(a, 0.0), 0.0))
    return np.where(continent | ~np.isfinite(a), continent_km, ocean_l)


def open_edges(mesh, forcing):
    """Edges the layer can cross: not a trench wall, and not touching a keel.  None means every edge."""
    if forcing is None:
        return None
    closed = np.zeros(mesh.n, bool) if forcing.closed is None else np.asarray(forcing.closed, bool)
    ok = ~(closed[mesh.edge_i] | closed[mesh.edge_j])
    if forcing.blocked is not None:
        ok &= ~np.asarray(forcing.blocked, bool)
    return ok


def gradient(mesh, values, open_edge=None):
    """(n, 3) gradient of a cell field (per km), tangent to the sphere: the directional differences across each edge,
    reconstructed per cell as a flux would be (exact for a uniform gradient on a Voronoi mesh).  Edges not in
    open_edge count as flat, so there is no slope across a wall or into a keel."""
    d = (values[mesh.edge_j] - values[mesh.edge_i]) / mesh.edge_dist * mesh.edge_len
    if open_edge is not None:
        d = d * open_edge
    return SM.cell_vectors(mesh, d)


def smooth(mesh, values, passes=SMOOTH_PASSES, open_edge=None):
    """Neighbour averaging: each pass gives a cell half its own value and half its neighbours' (weighted by edge
    length, over open_edge only, so nothing is averaged across a wall).  Keeps a constant field constant."""
    v = np.asarray(values, float).copy()
    i, j = mesh.edge_i, mesh.edge_j
    w = mesh.edge_len if open_edge is None else mesh.edge_len * open_edge
    total = np.bincount(i, w, minlength=mesh.n) + np.bincount(j, w, minlength=mesh.n)
    for _ in range(passes):
        nb = np.bincount(i, w * v[j], minlength=mesh.n) + np.bincount(j, w * v[i], minlength=mesh.n)
        v = np.where(total > 0, 0.5 * v + 0.5 * nb / np.maximum(total, 1e-300), v)
    return v


def azimuth_deg(mesh, vec):
    """Azimuth of tangent vectors, degrees clockwise from north."""
    e, n = SM.east_north(mesh, vec)
    return np.degrees(np.arctan2(e, n)) % 360.0


def flow_volcanism(mesh, step, u_layer, h, law=None, continent_km=CONTINENT_KM, min_h_km=10.0,
                   smooth_passes=SMOOTH_PASSES, forcing=None):
    """The flow-driven volcanism fields for one step, per cell:
      u_rel        (n, 3) layer velocity relative to the plate (km/Myr): a chain grows this way
      w            how fast the layer rises along the plate's base (km/Myr), 0 where the layer is thinner than min_h
      chain_az     azimuth of u_rel (degrees from north), towards the young end of a chain
      fixed_az     azimuth of -u_plate: the chain a source fixed in the mantle would make
      turn_deg     angle between the two (0-180)
      lithosphere  L (km), smoothed
    With the layer's forcing for the step (asthenosphere_plates.forcing_from_step), the smoothing and the slope of L
    stop at trench walls and keels, which the layer cannot cross."""
    u_plate = plate_velocity(mesh, step)
    u_rel = u_layer - u_plate
    ok = open_edges(mesh, forcing)
    L = smooth(mesh, lithosphere_km(mesh, step, law, continent_km), smooth_passes, ok)
    w = -np.einsum('ij,ij->i', u_rel, gradient(mesh, L, ok))
    w = np.where(h >= min_h_km, w, 0.0)
    chain = azimuth_deg(mesh, u_rel)
    fixed = azimuth_deg(mesh, -u_plate)
    turn = np.abs((chain - fixed + 180.0) % 360.0 - 180.0)
    return dict(u_rel=u_rel, w=w, chain_az=chain, fixed_az=fixed, turn_deg=turn, lithosphere=L,
                speed=np.linalg.norm(u_rel, axis=1))
