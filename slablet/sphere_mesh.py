"""Spherical Voronoi mesh of Fibonacci points, for finite volumes on the sphere (slablet/asthenosphere.py).

    mesh = voronoi_mesh(n_cells)

Cells are the Voronoi regions of n near-uniform Fibonacci points (about 110 km across for 40,000 cells); each cell
edge is a great-circle arc perpendicular to the line joining the two cell centres, so the two-point flux
(difference / centre distance) is consistent.  Areas, edge lengths and centre distances are true spherical values
(scipy.spatial.SphericalVoronoi).  No pole problem, which matters because Antarctica is in the region of interest.

Edges are oriented so that the flux of a rigid rotation omega (rad/Myr) from cell i to cell j across edge (a, b)
is exactly R^2 omega . (b - a) (the stream-function form): rigid_edge_flux().
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import SphericalVoronoi

from slablet.asthenosphere import Mesh
from slablet.sphere import xyz_to_lonlat

EARTH_RADIUS_KM = 6371.0


@dataclass
class SphereMesh(Mesh):
    xyz: np.ndarray = None                      # (n, 3) unit cell centres
    lon: np.ndarray = None
    lat: np.ndarray = None
    edge_a: np.ndarray = None                   # (m, 3) unit vectors, edge end points (Voronoi vertices)
    edge_b: np.ndarray = None
    radius_km: float = EARTH_RADIUS_KM


def fibonacci_points(n):
    """n near-uniform unit vectors on the sphere (golden-angle spiral)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * k
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _arc(u, v):
    """Great-circle angle between unit vectors (rows), stable for small angles."""
    return np.arctan2(np.linalg.norm(np.cross(u, v), axis=-1), np.einsum('ij,ij->i', u, v))


def voronoi_mesh(n_cells, radius_km=EARTH_RADIUS_KM):
    pts = fibonacci_points(n_cells)
    sv = SphericalVoronoi(pts, radius=1.0)
    sv.sort_vertices_of_regions()
    area = sv.calculate_areas() * radius_km ** 2
    verts = sv.vertices / np.linalg.norm(sv.vertices, axis=1, keepdims=True)

    # every (cell, vertex k, vertex k+1) side; each edge appears once from each of its two cells
    cell, va, vb = [], [], []
    for c, reg in enumerate(sv.regions):
        r = np.asarray(reg)
        cell.append(np.full(len(r), c))
        va.append(r)
        vb.append(np.roll(r, -1))
    cell, va, vb = np.concatenate(cell), np.concatenate(va), np.concatenate(vb)
    nv = len(verts)
    key = np.minimum(va, vb).astype(np.int64) * nv + np.maximum(va, vb)
    order = np.argsort(key, kind='stable')
    key, cell, va, vb = key[order], cell[order], va[order], vb[order]
    if not (np.all(key[0::2] == key[1::2]) and len(key) % 2 == 0):
        raise RuntimeError('Voronoi edges do not pair up')
    ei, ej = cell[0::2], cell[1::2]
    a, b = verts[va[0::2]], verts[vb[0::2]]
    keep = _arc(a, b) > 0                       # drop degenerate (zero-length) edges from near-duplicate vertices
    ei, ej, a, b = ei[keep], ej[keep], a[keep], b[keep]

    # orient (a, b) so that the normal -(a x b) points from cell i to cell j
    flip = np.einsum('ij,ij->i', np.cross(a, b), pts[ej] - pts[ei]) > 0
    a[flip], b[flip] = b[flip].copy(), a[flip].copy()

    lon, lat = xyz_to_lonlat(pts)
    return SphereMesh(area=area, edge_i=ei, edge_j=ej,
                      edge_len=_arc(a, b) * radius_km, edge_dist=_arc(pts[ei], pts[ej]) * radius_km,
                      xyz=pts, lon=lon, lat=lat, edge_a=a, edge_b=b, radius_km=radius_km)


def rigid_edge_flux(mesh, omega_i, omega_j=None):
    """Normal plate velocity integrated over each edge (km^2/Myr, + from cell i to cell j) for rigid rotations.

    omega_i, omega_j: (m, 3) or (3,) rotation vectors (rad/Myr) of the plate on each side of each edge (omega_j
    defaults to omega_i).  Exact: R^2 omega . (b - a), so a single rigid rotation has zero net flux out of every cell.
    """
    d = (mesh.edge_b - mesh.edge_a) * mesh.radius_km ** 2
    qi = np.einsum('ij,ij->i', np.broadcast_to(omega_i, d.shape), d)
    qj = qi if omega_j is None else np.einsum('ij,ij->i', np.broadcast_to(omega_j, d.shape), d)
    return qi, qj


def nearest_cell(mesh, lon, lat):
    """Index of the cell holding each point (a Voronoi cell = the points nearest its centre, so exact)."""
    from scipy.spatial import cKDTree
    from slablet.sphere import lonlat_to_xyz
    if getattr(mesh, '_tree', None) is None:
        mesh._tree = cKDTree(mesh.xyz)
    return mesh._tree.query(lonlat_to_xyz(np.asarray(lon, float), np.asarray(lat, float)))[1]


def cell_vectors(mesh, edge_flux):
    """Per-cell flux vector (n, 3), km^2/Myr, tangent to the sphere, from the volume flux across each edge
    (km^3/Myr, + from cell i to cell j): q_i = (1/A_i) sum_e F_e,out (x_e - x_i), Perot's reconstruction, exact for
    a uniform field on a Voronoi mesh."""
    mid = mesh.edge_a + mesh.edge_b
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    R = mesh.radius_km
    v = np.zeros((mesh.n, 3))
    for k in range(3):
        v[:, k] = (np.bincount(mesh.edge_i, edge_flux * R * (mid[:, k] - mesh.xyz[mesh.edge_i, k]), minlength=mesh.n)
                   - np.bincount(mesh.edge_j, edge_flux * R * (mid[:, k] - mesh.xyz[mesh.edge_j, k]),
                                 minlength=mesh.n))
    v /= mesh.area[:, None]
    return v - np.einsum('ij,ij->i', v, mesh.xyz)[:, None] * mesh.xyz


def east_north(mesh, vec):
    """East and north components of tangent vectors (n, 3) at the cell centres."""
    lon, lat = np.radians(mesh.lon), np.radians(mesh.lat)
    east = np.column_stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)])
    north = np.column_stack([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    return np.einsum('ij,ij->i', vec, east), np.einsum('ij,ij->i', vec, north)


def regrid(mesh, values, spacing_deg=1.0):
    """Cell values on a regular lon/lat grid of cell centres (nearest cell, i.e. the cell each grid point is in).
    Returns lons, lats (ascending) and an array (nlat, nlon) or (..., nlat, nlon)."""
    lons = np.arange(-180 + spacing_deg / 2, 180, spacing_deg)
    lats = np.arange(-90 + spacing_deg / 2, 90, spacing_deg)
    lo, la = np.meshgrid(lons, lats)
    idx = nearest_cell(mesh, lo.ravel(), la.ravel())
    v = np.asarray(values)
    return lons, lats, v[..., idx].reshape(v.shape[:-1] + lo.shape)


def edges_crossing(mesh, seg_a, seg_b):
    """Mesh edges whose centre-to-centre great-circle arc crosses any of the arcs seg_a[k] -> seg_b[k] (unit
    vectors (k, 3)).  Used to put walls along trench lines: flow between the two cells would cross the line."""
    from scipy.spatial import cKDTree
    if len(seg_a) == 0:
        return np.zeros(len(mesh.edge_i), bool)
    ci, cj = mesh.xyz[mesh.edge_i], mesh.xyz[mesh.edge_j]
    emid = ci + cj
    emid /= np.linalg.norm(emid, axis=1, keepdims=True)
    smid = seg_a + seg_b
    smid /= np.linalg.norm(smid, axis=1, keepdims=True)
    reach = (np.max(mesh.edge_dist) / mesh.radius_km + np.max(_arc(seg_a, seg_b))) / 2 + 1e-6   # chord ~ angle
    pairs = cKDTree(emid).query_ball_tree(cKDTree(smid), reach)
    e = np.repeat(np.arange(len(pairs)), [len(p) for p in pairs])
    s = np.concatenate([np.asarray(p, int) for p in pairs]) if len(e) else np.zeros(0, int)
    if len(e) == 0:
        return np.zeros(len(mesh.edge_i), bool)
    a1, b1, a2, b2 = ci[e], cj[e], seg_a[s], seg_b[s]
    n1, n2 = np.cross(a1, b1), np.cross(a2, b2)
    p = np.cross(n1, n2)
    norm = np.linalg.norm(p, axis=1)
    ok = norm > 1e-15
    p[ok] /= norm[ok, None]
    hit = np.zeros(len(e), bool)
    for sg in (1.0, -1.0):
        q = sg * p

        def on(a, b, n):
            return (np.einsum('ij,ij->i', np.cross(a, q), n) >= 0) & (np.einsum('ij,ij->i', np.cross(q, b), n) >= 0)
        hit |= ok & on(a1, b1, n1) & on(a2, b2, n2)
    out = np.zeros(len(mesh.edge_i), bool)
    out[e[hit]] = True
    return out
