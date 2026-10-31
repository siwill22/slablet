"""A conserved pile on the core-mantle boundary, pushed by slabs (llsvp: pile).

The state is the pile thickness h (km, nlat × nlon on the mantle grid).  It starts as a
uniform layer p.pile_mean_thickness_km thick, and its volume never changes.  The pile
flows down the gradient of a pressure head

    P = h + p.pile_slab_head_km × L

where L is the slab load on the CMB: the slablets in the bottom p.pile_slab_layer_km,
each counted with weight × exp(-age / slab_thermal_tau_myr), per square degree of
equatorial area, smoothed with slab_sigma_deg.  So slabs lying on the CMB push the pile
away, and the pile spreads under its own weight (the h term).  The volume flux across
a cell face is

    -p.pile_mobility_km_per_myr × h_face × (P_neighbour - P) × G

with G the face length over the centre-to-centre distance on the sphere (1 / cos(lat)
for east-west faces, cos(face lat) for north-south faces; no flux across the poles).
That is a finite-volume form of ∂h/∂t = ∇·(κ h ∇P): every face flux leaves one cell and
enters its neighbour, so the volume is conserved to rounding.  The spreading part is
implicit (backward Euler, h_face the mean of the two cells from the previous step), so
the small polar cells set no time-step limit; the slab push is explicit and upwind (the
donor cell's h) with as many sub-steps as keep any cell from losing more than half its
pile in one.  Any rounding below zero is clipped and the volume rescaled.
"""
import numpy as np
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import spsolve

from .sphere import spherical_gaussian_smooth

CMB_RADIUS_KM = 3480.0


def _grid(p):
    s = p.mantle_grid_spacing_deg
    lons = np.arange(-180 + s / 2, 180, s)
    lats = np.arange(-90 + s / 2, 90, s)
    d = np.radians(s)
    lat_r = np.radians(lats)
    area = CMB_RADIUS_KM ** 2 * d * (np.sin(lat_r + d / 2) - np.sin(lat_r - d / 2))   # km², per row
    return lons, lats, area


def new_pile(p):
    lons, lats, _ = _grid(p)
    return np.full((len(lats), len(lons)), float(p.pile_mean_thickness_km))


def slab_load(slab_tracers, p):
    """L: thermally weighted slab load in the bottom p.pile_slab_layer_km, per square
    degree of equatorial area, smoothed with p.slab_sigma_deg."""
    lons, lats, _ = _grid(p)
    load = np.zeros((len(lats), len(lons)))
    if len(slab_tracers):
        s = slab_tracers[slab_tracers['active'].astype(bool)
                         & (slab_tracers['depth_km'].astype(float) >= p.cmb_depth_km - p.pile_slab_layer_km)]
        if len(s):
            w = s['weight'].values.astype(float) * np.exp(-s['age'].values.astype(float) / p.slab_thermal_tau_myr)
            spacing = p.mantle_grid_spacing_deg
            i = np.clip(np.floor((s['lat'].values.astype(float) + 90) / spacing).astype(int), 0, len(lats) - 1)
            j = np.clip(np.floor((s['lon'].values.astype(float) + 180) / spacing).astype(int), 0, len(lons) - 1)
            np.add.at(load, (i, j), w)
            load /= spacing ** 2 * np.cos(np.radians(lats))[:, None]
    return spherical_gaussian_smooth(load, lats, lons, p.slab_sigma_deg)


def _faces(nlat, nlon):
    """Index pairs (a, b) of cells sharing a face: east-west faces (one per cell, wrapping
    in longitude), then north-south faces between neighbouring rows."""
    idx = np.arange(nlat * nlon).reshape(nlat, nlon)
    return (idx.ravel(), np.roll(idx, -1, axis=1).ravel()), (idx[:-1].ravel(), idx[1:].ravel())


def update_pile(h, slab_tracers, delta_time, p):
    lons, lats, area_row = _grid(p)
    nlat, nlon = len(lats), len(lons)
    area = np.repeat(area_row, nlon)                       # km² per cell, flattened row-major
    (ea, eb), (na, nb) = _faces(nlat, nlon)
    lat_r = np.radians(lats)
    g_ew = np.repeat(1.0 / np.cos(lat_r), nlon)            # east-west faces, one per cell
    lat_face = np.radians(lats[:-1] + p.mantle_grid_spacing_deg / 2)
    g_ns = np.repeat(np.cos(lat_face), nlon)               # north-south faces between rows
    a = np.concatenate([ea, na])
    b = np.concatenate([eb, nb])
    g = np.concatenate([g_ew, g_ns])
    kappa = p.pile_mobility_km_per_myr
    volume = (h.ravel() * area).sum()

    # 1. slab push, explicit and upwind: flux a->b = kappa * h_donor * (P_a - P_b) * g
    head = p.pile_slab_head_km * slab_load(slab_tracers, p).ravel()
    hf = h.ravel().copy()
    dhead = head[a] - head[b]
    rate = kappa * g * np.abs(dhead)                       # km² per Myr per km of donor pile
    donor = np.where(dhead > 0, a, b)
    out = np.zeros(len(hf))
    np.add.at(out, donor, rate)
    max_frac = (out / area).max() * delta_time if len(out) else 0.0
    nsub = max(1, int(np.ceil(max_frac / 0.5)))
    dt = delta_time / nsub
    for _ in range(nsub):
        flow = kappa * g * dhead * hf[donor] * dt          # km³ from a to b
        np.add.at(hf, a, -flow / area[a])
        np.add.at(hf, b, flow / area[b])

    # 2. spreading under the pile's own weight, implicit: (I + dt A) h_new = h
    cond = kappa * g * 0.5 * (hf[a] + hf[b])               # km²/Myr per km of head difference
    n = len(hf)
    rows = np.concatenate([a, b, a, b])
    cols = np.concatenate([a, b, b, a])
    vals = np.concatenate([cond, cond, -cond, -cond])
    A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A = A.multiply(1.0 / area[:, None]).tocsr()
    hf = spsolve((identity(n, format='csr') + delta_time * A).tocsc(), hf)

    hf = np.maximum(hf, 0.0)
    hf *= volume / (hf * area).sum()
    return hf.reshape(nlat, nlon)


def pile_layer_fractions(h, depths_g, depth_interval_km, cmb_depth_km):
    """Fraction (ndepth, nlat, nlon) of each depth layer's mantle part (the layer above the
    CMB) that lies inside the pile, which fills the column from cmb_depth_km - h to the CMB."""
    top = cmb_depth_km - h[None, :, :]
    lo = depths_g[:, None, None] - depth_interval_km / 2
    hi = np.minimum(depths_g[:, None, None] + depth_interval_km / 2, cmb_depth_km)
    overlap = np.clip(hi - np.maximum(lo, top), 0.0, None)
    return overlap / np.maximum(hi - lo, 1e-9)
