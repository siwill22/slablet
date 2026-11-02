"""A conserved pile on the core-mantle boundary, pushed by slabs (llsvp: pile).

The state is the pile thickness h (km, nlat × nlon on the mantle grid).  It starts as a
uniform layer p.pile_mean_thickness_km thick, and its volume never changes.  The pile
flows down the gradient of a pressure head

    P = h + p.pile_slab_head_km × L

where L is the slab load on the CMB: the slablets in the bottom p.pile_slab_layer_km,
each counted with weight × exp(-age / slab_thermal_tau_myr), per square degree of
equatorial area, smoothed with slab_sigma_deg.  So slabs lying on the CMB push the pile
away, and the pile spreads under its own weight (the h term).  That is pile_drive: local.
At rest h = constant - pile_slab_head_km × L wherever h > 0: a shallow layer over every
slab-free area.

pile_drive: flow replaces L by ψ, the solution of ∇²ψ = -(L - mean L) on the unit sphere
(flow_potential): slabs arriving at the base drive a flow away from them, and its potential
depends on all slabs, not only those overhead.  ψ is weighted towards long wavelengths
(degree l scaled by 1 / (l (l + 1))), so at rest the pile collects in a few compact piles
where that flow converges, far from all slabs.

With p.pile_flow_screen_deg = λ > 0 the equation is screened, ∇²ψ - ψ / λ² = -(L - mean L)
(degree l scaled by 1 / (l (l + 1) + 1 / λ²), λ in radians): a slab's push reaches about λ, so
the pile also collects in the gaps between neighbouring slabs, as hot ridges do in convection
models; small λ tends to pile_drive: local, large λ to plain flow.

The volume flux across a cell face is

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
from scipy.sparse import coo_matrix, diags, identity
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
    degree of equatorial area, smoothed with p.slab_sigma_deg (p.slab_sigma_cmb_deg when the
    slab width profile is on)."""
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
    # with the slab width profile on, the load is as wide as slabs are at the CMB
    sigma = p.slab_sigma_cmb_deg if p.slab_sigma_cmb_deg > 0 else p.slab_sigma_deg
    return spherical_gaussian_smooth(load, lats, lons, sigma)


PILE_DRIVES = ('local', 'flow')
_POISSON = {}


def flow_potential(load, p):
    """ψ with ∇²ψ = -(load - its area-weighted mean) on the unit sphere (pile_drive: flow).

    The same finite-volume faces as update_pile: M ψ = area × (load - mean), with M the
    face-conductance (g) Laplacian and areas in steradians.  M is singular (any constant
    can be added), so a tiny multiple of the areas is added to its diagonal; the right-hand
    side has zero sum, so this only fixes ψ's mean at about zero.  ψ is largest under slabs
    and lowest far from all of them, weighted towards long wavelengths: a degree-l pattern
    of the load becomes load_l / (l (l + 1)) in ψ."""
    lons, lats, area_row = _grid(p)
    nlat, nlon = len(lats), len(lons)
    screen = float(p.pile_flow_screen_deg)
    key = (nlat, nlon, screen)
    if key not in _POISSON:
        from scipy.sparse.linalg import factorized
        area = np.repeat(area_row, nlon) / CMB_RADIUS_KM ** 2
        a, b, g = _face_arrays(lats, nlat, nlon, p)
        M = coo_matrix((np.concatenate([g, g, -g, -g]),
                        (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))), shape=(len(area), len(area)))
        k2 = 1.0 / np.radians(screen) ** 2 if screen > 0 else 1e-9
        _POISSON[key] = (factorized((M + diags(k2 * area)).tocsc()), area)
    solve, area = _POISSON[key]
    f = load.ravel()
    rhs = area * (f - (f * area).sum() / area.sum())
    return solve(rhs).reshape(nlat, nlon)


def _face_arrays(lats, nlat, nlon, p):
    """Cell index pairs (a, b) of every face and the face's geometric factor g."""
    (ea, eb), (na, nb) = _faces(nlat, nlon)
    lat_r = np.radians(lats)
    g_ew = np.repeat(1.0 / np.cos(lat_r), nlon)
    lat_face = np.radians(lats[:-1] + p.mantle_grid_spacing_deg / 2)
    g_ns = np.repeat(np.cos(lat_face), nlon)
    return np.concatenate([ea, na]), np.concatenate([eb, nb]), np.concatenate([g_ew, g_ns])


def _faces(nlat, nlon):
    """Index pairs (a, b) of cells sharing a face: east-west faces (one per cell, wrapping
    in longitude), then north-south faces between neighbouring rows."""
    idx = np.arange(nlat * nlon).reshape(nlat, nlon)
    return (idx.ravel(), np.roll(idx, -1, axis=1).ravel()), (idx[:-1].ravel(), idx[1:].ravel())


def update_pile(h, slab_tracers, delta_time, p):
    lons, lats, area_row = _grid(p)
    nlat, nlon = len(lats), len(lons)
    area = np.repeat(area_row, nlon)                       # km² per cell, flattened row-major
    a, b, g = _face_arrays(lats, nlat, nlon, p)
    kappa = p.pile_mobility_km_per_myr
    volume = (h.ravel() * area).sum()

    # 1. slab push, explicit and upwind: flux a->b = kappa * h_donor * (P_a - P_b) * g
    if p.pile_drive not in PILE_DRIVES:
        raise ValueError('pile_drive must be one of {}, not {!r}'.format(PILE_DRIVES, p.pile_drive))
    load = slab_load(slab_tracers, p)
    if p.pile_drive == 'flow':
        load = flow_potential(load, p)
    head = p.pile_slab_head_km * load.ravel()
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
