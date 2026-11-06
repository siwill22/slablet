"""A thin, buoyant asthenosphere layer under the plates, after Buck, Small & Ryan (2009, G-cubed 10, Q09007).

The layer (thickness h, viscosity mu, lighter than the mesosphere below by delta_rho) is dragged by the plates,
consumed at ridges, blocked by trenches and keels, and refilled by pressure-driven flow and a supply from below
(Buck eq. 7):

    q     = h (u_top + u_base) / 2  -  K(h) grad h,     K(h) = drho* g h^3 / (12 mu)
    dh/dt = -div q + supply - sinks
    d     = d_max (1 - h / h_a)                         seafloor depth anomaly, + = deeper (eq. 8)

The mesosphere is fixed (u_base = 0) except in Buck's 1-D migrating-ridge case, which is solved in the ridge frame.

Finite volumes on any mesh given as cells (areas) and edges (cell pairs, edge length, centre distance), so the same
core runs on Buck's flat 1-D strip and box and on the spherical Voronoi mesh (slablet/sphere_mesh.py).  Plate drag
enters as the normal plate velocity integrated over each edge, taken from each side's own plate (q_top_i, q_top_j,
km^2/Myr, positive from cell i to cell j): exact for rigid plates on the sphere, so a uniform layer under one rigid
rotation stays uniform to round-off.  The drag flux is upwinded from each side (no flux across a divergent edge);
the time step is explicit, sized each sub-step so that no cell can go negative.

Units inside: km, Myr, km^2/Myr, km^3/Myr.  Driven only by the plate history; separate from slablet's mantle model.
"""
from dataclasses import dataclass

import numpy as np

G = 9.81                                        # m/s^2
SEC_PER_MYR = 1e6 * 365.25 * 86400.0


@dataclass
class AsthenosphereParams:
    """Layer properties and options.  The first value of each option is v1 (plan of 2026-11-05)."""
    delta_rho: float = 10.0                     # rho_M - rho_A, kg/m^3 (Buck: 10, about 100 C hotter)
    rho_a: float = 3300.0                       # asthenosphere density, kg/m^3
    rho_w: float = 1030.0                       # sea water, kg/m^3
    h_a_km: float = 300.0                       # mean layer thickness
    viscosity_pa_s: float = 1.3e19
    lithosphere_km: float = 100.0               # layer turned into lithosphere at a ridge
    trench_mode: str = 'wall'                   # 'wall' | 'sink' (drag-delivered layer leaves with the slab)
    keel_mode: str = 'craton'                   # 'craton' | 'continent' | 'thinned'
    covered_mode: str = 'bulldoze'              # 'bulldoze' | 'delete' (cells an advancing keel covers)
    supply_mode: str = 'balance'                # 'balance' (uniform, = sinks each step) | 'constant' | 'none'
    supply_where: str = 'open'                  # 'open' (all open cells) | 'ocean' (not under continent, as Buck)
    ridge_sink_mode: str = 'axis'               # 'axis' | 'age'
    cfl: float = 0.9                            # fraction of the positivity limit used per sub-step
    plume_delta_rho: object = None              # rho_M - rho of plume material, kg/m^3 (None: one density, v1);
                                                # when set, advance() can carry the plume fraction c of the layer

    @property
    def rho_m(self):
        return self.rho_a + self.delta_rho

    @property
    def delta_rho_star(self):
        """(rho_A - rho_W)(rho_M - rho_A)/(rho_M - rho_W), kg/m^3 (Buck eq. 6)."""
        return (self.rho_a - self.rho_w) * self.delta_rho / (self.rho_m - self.rho_w)

    @property
    def d_max_km(self):
        """Depth anomaly where the layer is used up (Buck eq. 8)."""
        return self.delta_rho / (self.rho_m - self.rho_w) * self.h_a_km

    def mobility(self, h_km):
        """K(h) = drho* g h^3 / (12 mu), in km^2/Myr."""
        h_m = np.asarray(h_km, float) * 1e3
        return self.delta_rho_star * G * h_m ** 3 / (12.0 * self.viscosity_pa_s) * SEC_PER_MYR / 1e6

    def ramberg(self, u0_km_per_myr, h_km=None):
        """Rm = drho* g h_a^2 / (mu u0) (Buck eq. 10)."""
        h_m = (self.h_a_km if h_km is None else h_km) * 1e3
        u0 = u0_km_per_myr * 1e3 / SEC_PER_MYR
        return self.delta_rho_star * G * h_m ** 2 / (self.viscosity_pa_s * u0)

    def delta_rho_of(self, c):
        """rho_M - rho of a layer holding plume fraction c (kg/m^3)."""
        if c is None or self.plume_delta_rho is None:
            return self.delta_rho
        return self.delta_rho + np.asarray(c, float) * (self.plume_delta_rho - self.delta_rho)

    def depth_anomaly_km(self, h_km, c=None):
        """Seafloor depth anomaly (km, + = deeper) from isostasy of the layer column, whose buoyancy is
        (rho_M - rho(c)) h.  Without c: Buck's d_max (1 - h / h_a)."""
        h = np.asarray(h_km, float)
        if c is None or self.plume_delta_rho is None:
            return self.d_max_km * (1.0 - h / self.h_a_km)
        return (self.delta_rho * self.h_a_km - self.delta_rho_of(c) * h) / (self.rho_m - self.rho_w)


@dataclass
class Mesh:
    """Finite-volume mesh: cell areas and the edges between neighbouring cells."""
    area: np.ndarray                            # (n,) km^2
    edge_i: np.ndarray                          # (m,) int
    edge_j: np.ndarray                          # (m,) int
    edge_len: np.ndarray                        # (m,) km
    edge_dist: np.ndarray                       # (m,) km, centre to centre

    @property
    def n(self):
        return len(self.area)


def strip_mesh(n, dx_km):
    """Buck's 1-D case: n cells of width dx along x, 1 km wide across; the two ends are walls."""
    i = np.arange(n - 1)
    return Mesh(area=np.full(n, float(dx_km)), edge_i=i, edge_j=i + 1,
                edge_len=np.ones(n - 1), edge_dist=np.full(n - 1, float(dx_km)))


def box_mesh(nx, ny, dx_km):
    """Buck's flat 2-D box: nx x ny square cells (row-major, x fastest); the sides are walls.
    Returns the mesh and each edge's unit normal (x, y components) from cell i to cell j."""
    idx = np.arange(nx * ny).reshape(ny, nx)
    ei = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    ej = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    nxe = (ny * (nx - 1))
    normal = np.zeros((len(ei), 2))
    normal[:nxe, 0] = 1.0
    normal[nxe:, 1] = 1.0
    m = len(ei)
    return Mesh(area=np.full(nx * ny, float(dx_km) ** 2), edge_i=ei, edge_j=ej,
                edge_len=np.full(m, float(dx_km)), edge_dist=np.full(m, float(dx_km))), normal


@dataclass
class Forcing:
    """What drives the layer over one interval, all held fixed during it."""
    q_top_i: np.ndarray                         # (m,) plate normal velocity x edge length, side i's plate, km^2/Myr
    q_top_j: np.ndarray                         # (m,) the same from side j's plate
    q_base: object = 0.0                        # (m,) or scalar: the same for the layer base (0: mesosphere fixed)
    blocked: object = None                      # (m,) bool: walls (trenches), no flux
    closed: object = None                       # (n,) bool: keel cells, outside the layer
    sink: object = None                         # (n,) km^3/Myr requested (ridge sinks)
    source: object = None                       # (n,) km^3/Myr fixed sources (benchmarks)
    outflow: object = None                      # (n,) km^2/Myr: cell loses h x outflow (open edge, trench sink)
    supply: str = 'balance'                     # 'balance' | 'none' | 'constant'
    supply_mask: object = None                  # (n,) bool: cells that receive the supply (default: all open cells)
    balance_outflow: bool = True                # 'balance' also replaces what the outflow takes (Buck AAD1: no)
    supply_rate: float = 0.0                    # km/Myr over open cells, for 'constant'
    source_plume: bool = False                  # sources deliver plume material (c = 1), when c is carried


@dataclass
class StepLog:
    """Totals over one call of advance()."""
    substeps: int = 0
    taken: object = None                        # (n,) km^3 removed by sinks
    deficit: object = None                      # (n,) km^3 asked for but not there (starved: mesosphere upwelling)
    outflow: float = 0.0                        # km^3 lost through outflow
    supplied: float = 0.0                       # km^3
    c: object = None                            # (n,) plume fraction at the end, when carried


def _pressure_drive(h, c, p, i, j):
    """What drives pressure flow across each edge, as an equivalent thickness difference (km): the flux i -> j is
    -K(h_mean) x this / distance.  Without c it is h_j - h_i.  With plume fraction c the layer density rho(c)
    varies, and the mid-layer pressure gradient is (rho - rho_W) g grad e + g (h / 2) grad rho, with
    e = (rho_M - rho) h / (rho_M - rho_W) the seafloor uplift; this is that gradient over drho* g.
    Also returns each cell's coefficient a on its own h (1 where c = 0), for the time-step limit."""
    if c is None or p.plume_delta_rho is None:
        return h[j] - h[i], np.ones_like(h)
    dr = p.delta_rho_of(c)
    rho = p.rho_m - dr
    a = (rho - p.rho_w) * dr / (p.rho_m - p.rho_w) / p.delta_rho_star
    hm = 0.5 * (h[i] + h[j])
    return a[j] * h[j] - a[i] * h[i] + 0.5 * hm * (rho[j] - rho[i]) / p.delta_rho_star, a


def _edge_fluxes(mesh, h, f, p, open_edge, c=None):
    """Volume flux across each edge, km^3/Myr, positive i -> j, and each cell's positivity rate limit."""
    i, j = mesh.edge_i, mesh.edge_j
    vi = 0.5 * (f.q_top_i + f.q_base)           # depth-mean drag velocity x edge length, side i
    vj = 0.5 * (f.q_top_j + f.q_base)
    drag = h[i] * np.maximum(vi, 0.0) + h[j] * np.minimum(vj, 0.0)
    cond = p.mobility(0.5 * (h[i] + h[j])) * mesh.edge_len / mesh.edge_dist
    dh, a = _pressure_drive(h, c, p, i, j)
    flux = (drag - cond * dh) * open_edge
    # rate at which each cell could lose its own content: outgoing drag + conductance
    out = np.bincount(i, (np.maximum(vi, 0.0) + cond * a[i]) * open_edge, minlength=mesh.n)
    out += np.bincount(j, (np.maximum(-vj, 0.0) + cond * a[j]) * open_edge, minlength=mesh.n)
    return flux, out


def advance(mesh, h, forcing, p, duration_myr, max_dt_myr=None, c=None):
    """Advance h (km, per cell) by duration_myr under fixed forcing.  Returns (h, StepLog).
    With p.plume_delta_rho set and c given (plume fraction per cell), c is carried with the layer (upwind),
    sinks take the local mixture, the uniform supply carries none, and sources carry c = 1 when
    forcing.source_plume; the end state is log.c."""
    f = forcing
    n = mesh.n
    h = np.array(h, float)
    closed = np.zeros(n, bool) if f.closed is None else np.asarray(f.closed, bool)
    open_edge = ~(closed[mesh.edge_i] | closed[mesh.edge_j])
    if f.blocked is not None:
        open_edge &= ~np.asarray(f.blocked, bool)
    open_edge = open_edge.astype(float)
    open_cell = ~closed
    fed = open_cell if f.supply_mask is None else open_cell & np.asarray(f.supply_mask, bool)
    fed_area = mesh.area[fed].sum()
    sink = np.zeros(n) if f.sink is None else np.where(open_cell, f.sink, 0.0)
    source = np.zeros(n) if f.source is None else np.where(open_cell, f.source, 0.0)
    outflow = np.zeros(n) if f.outflow is None else np.where(open_cell, f.outflow, 0.0)
    h[closed] = 0.0
    track = c is not None and p.plume_delta_rho is not None
    if track:
        c = np.where(open_cell, np.clip(np.asarray(c, float), 0.0, 1.0), 0.0)
    log = StepLog(taken=np.zeros(n), deficit=np.zeros(n))
    t = 0.0
    while t < duration_myr - 1e-12:
        flux, out = _edge_fluxes(mesh, h, f, p, open_edge, c if track else None)
        rate = np.max((out + outflow) / mesh.area)
        if track:                       # the composition term is not proportional to h: also bound what leaves
            leave = (np.bincount(mesh.edge_i, np.maximum(flux, 0.0), minlength=n)
                     + np.bincount(mesh.edge_j, np.maximum(-flux, 0.0), minlength=n))
            full = h > 1e-6
            if full.any():
                rate = max(rate, np.max(leave[full] / (h[full] * mesh.area[full])))
        dt = duration_myr - t
        if rate > 0:
            dt = min(dt, p.cfl / rate)
        if max_dt_myr is not None:
            dt = min(dt, max_dt_myr)
        dvol = (np.bincount(mesh.edge_j, flux, minlength=n) - np.bincount(mesh.edge_i, flux, minlength=n)) * dt
        lost = h * outflow * dt
        vol = h * mesh.area + dvol + source * dt - lost
        if track:                       # plume volume, carried upwind with the flux
            cf = flux * np.where(flux > 0, c[mesh.edge_i], c[mesh.edge_j])
            cvol = (c * h * mesh.area + (np.bincount(mesh.edge_j, cf, minlength=n)
                                          - np.bincount(mesh.edge_i, cf, minlength=n)) * dt
                    + (source * dt if f.source_plume else 0.0) - c * lost)
            cvol = np.clip(cvol, 0.0, np.maximum(vol, 0.0))
        log.outflow += lost.sum()
        want = sink * dt
        take = np.minimum(want, np.maximum(vol, 0.0))
        if track:                       # sinks take the local mixture
            cvol -= cvol / np.where(vol > 0, vol, np.inf) * take
        vol -= take
        log.taken += take
        log.deficit += want - take
        if f.supply == 'balance':
            add = take.sum() - source.sum() * dt + (lost.sum() if f.balance_outflow else 0.0)
        elif f.supply == 'constant':
            add = f.supply_rate * fed_area * dt
        else:
            add = 0.0
        if add > 0.0:
            vol[fed] += add * mesh.area[fed] / fed_area
            log.supplied += add
        elif add < 0.0:                 # sources exceed sinks: take the excess back in proportion to what each fed
            have = np.maximum(vol[fed], 0.0)    # cell holds, so no cell goes negative
            add = max(add, -have.sum())
            frac = -add / max(have.sum(), 1e-300)
            vol[fed] -= have * frac
            if track:
                cvol[fed] *= 1.0 - frac
            log.supplied += add
        h = np.where(open_cell, vol / mesh.area, 0.0)
        if track:
            ok = open_cell & (vol > 1e-12)
            c = np.where(ok, np.clip(cvol / np.where(ok, vol, 1.0), 0.0, 1.0), 0.0)
        t += dt
        log.substeps += 1
    log.c = c if track else None
    return h, log


def edge_flux_parts(mesh, h, forcing, p, c=None):
    """The two parts of the volume flux across each edge (km^3/Myr, + from cell i to cell j) for the current h:
    plate drag and pressure-driven (from thick towards thin layer).  Walls and keels carry none."""
    f = forcing
    closed = np.zeros(mesh.n, bool) if f.closed is None else np.asarray(f.closed, bool)
    open_edge = ~(closed[mesh.edge_i] | closed[mesh.edge_j])
    if f.blocked is not None:
        open_edge &= ~np.asarray(f.blocked, bool)
    i, j = mesh.edge_i, mesh.edge_j
    vi, vj = 0.5 * (f.q_top_i + f.q_base), 0.5 * (f.q_top_j + f.q_base)
    drag = (h[i] * np.maximum(vi, 0.0) + h[j] * np.minimum(vj, 0.0)) * open_edge
    press = (-p.mobility(0.5 * (h[i] + h[j])) * mesh.edge_len / mesh.edge_dist
             * _pressure_drive(h, c, p, i, j)[0] * open_edge)
    return drag, press


def move_keels(mesh, h, closed_old, closed_new, mode='bulldoze'):
    """Update the layer where keels have moved on the fixed mesh (plan, section 4).

    Uncovered cells (keel left) start empty, h = 0: the mesosphere fills the gap and the flow refills it.
    Covered cells (keel arrived): 'bulldoze' pushes their layer to the nearest open cells (ahead of the keel),
    passed cell by cell down the hop distance to open cells and shared by edge length, so no layer is lost;
    'delete' removes it (a sink).  Returns (h, deleted volume in km^3)."""
    closed_old, closed_new = np.asarray(closed_old, bool), np.asarray(closed_new, bool)
    h = np.array(h, float)
    vol = np.where(closed_new & ~closed_old, h * mesh.area, 0.0)
    h[closed_new | closed_old] = 0.0
    if mode == 'delete' or vol.sum() == 0.0:
        return h, (vol.sum() if mode == 'delete' else 0.0)
    i, j, w = mesh.edge_i, mesh.edge_j, mesh.edge_len
    hops = np.where(closed_new, np.inf, 0.0)
    while True:                                 # hop distance from every keel cell to the nearest open cell
        new = hops.copy()
        np.minimum.at(new, i, hops[j] + 1)
        np.minimum.at(new, j, hops[i] + 1)
        if np.array_equal(new, hops):
            break
        hops = new
    if not np.isfinite(hops[vol > 0]).all():
        raise ValueError('covered layer has no open cell to go to')
    for d in range(int(hops[vol > 0].max()), 0, -1):
        down_ij = (hops[i] == d) & (hops[j] == d - 1)   # edges leading one hop nearer to open cells
        down_ji = (hops[j] == d) & (hops[i] == d - 1)
        wsum = np.bincount(i, w * down_ij, minlength=mesh.n) + np.bincount(j, w * down_ji, minlength=mesh.n)
        share = np.where((hops == d) & (wsum > 0), vol / np.where(wsum > 0, wsum, 1.0), 0.0)
        recv = np.bincount(j, share[i] * w * down_ij, minlength=mesh.n) + \
            np.bincount(i, share[j] * w * down_ji, minlength=mesh.n)
        vol = np.where(hops == d, 0.0, vol) + recv
    h += vol / mesh.area
    return h, 0.0


def plate_velocity_1d(u_cells):
    """For a strip: per-edge normal plate velocities from each side's cell (u in km/Myr, +x)."""
    u = np.asarray(u_cells, float)
    return u[:-1].copy(), u[1:].copy()


def edge_velocity_2d(mesh, normal, u_cells):
    """For the flat box: per-edge normal plate velocity x edge length from each side's cell (u: (n, 2) km/Myr)."""
    u = np.asarray(u_cells, float)
    qi = np.einsum('ek,ek->e', u[mesh.edge_i], normal) * mesh.edge_len
    qj = np.einsum('ek,ek->e', u[mesh.edge_j], normal) * mesh.edge_len
    return qi, qj


# --- Buck's analytic steady states (Appendix A), for the benchmarks --------------------------------------------

def a1_profile(p, W_km, u0_km_per_myr, x_km):
    """Plate-driven flow, no sources or sinks, trench (wall) at x = 0, ridge at x = W, plate moving towards the
    trench at u0: h = h0 (1 - C0 x)^(1/3), with h0 fixed by mean h = h_a (A2).  Returns (h, h0)."""
    from scipy.optimize import brentq
    k = 18.0 * u0_km_per_myr / (12.0 * p.mobility(1.0))        # 18 mu u0 / (drho* g), km^2

    def mean_minus_ha(h0):
        c = k / h0 ** 3 * W_km
        if c >= 1.0:
            return -np.inf
        return h0 * (1.0 - (1.0 - c) ** (4.0 / 3.0)) / (4.0 / 3.0 * c) - p.h_a_km

    lo = (k * W_km) ** (1.0 / 3.0) * (1 + 1e-12)           # h(W) = 0 at this h0
    h0 = lo if mean_minus_ha(lo) >= -1e-9 * p.h_a_km else brentq(mean_minus_ha, lo, 10 * p.h_a_km)
    c0 = k / h0 ** 3
    return h0 * np.clip(1.0 - c0 * np.asarray(x_km, float), 0.0, None) ** (1.0 / 3.0), h0


def a1_critical_viscosity(p, W_km, u0_km_per_myr):
    """Viscosity at which h(W) = 0 (Rm = (3/2)^5 W / h_a, h0 = 4/3 h_a)."""
    rm_cr = 1.5 ** 5 * W_km / p.h_a_km
    u0 = u0_km_per_myr * 1e3 / SEC_PER_MYR
    return p.delta_rho_star * G * (p.h_a_km * 1e3) ** 2 / (rm_cr * u0)


def a2_profile(p, W_km, q_s_km2_per_myr, x_km):
    """Source-sink flow, no plates: line source q_s at x = 0, line sink q_s at x = W:
    h = h0 (1 - C1 x)^(1/4), h0 fixed by mean h = h_a (A7).  Returns (h, h0)."""
    from scipy.optimize import brentq
    k = 4.0 * q_s_km2_per_myr / p.mobility(1.0)                # 48 mu q_s / (drho* g), km^3

    def mean_minus_ha(h0):
        c = k / h0 ** 4 * W_km
        if c >= 1.0:
            return -np.inf
        return h0 * (1.0 - (1.0 - c) ** 1.25) / (1.25 * c) - p.h_a_km

    lo = (k * W_km) ** 0.25 * (1 + 1e-12)                  # h(W) = 0 at this h0
    h0 = lo if mean_minus_ha(lo) >= -1e-9 * p.h_a_km else brentq(mean_minus_ha, lo, 10 * p.h_a_km)
    c1 = k / h0 ** 4
    return h0 * np.clip(1.0 - c1 * np.asarray(x_km, float), 0.0, None) ** 0.25, h0


def a2_critical_flux(p, W_km):
    """Line flux q_s at which h(W) = 0 (h0 = 5/4 h_a)."""
    return p.mobility(1.25 * p.h_a_km) / 4.0 / W_km * (1.25 * p.h_a_km)
