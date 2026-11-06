"""Tests for slablet.asthenosphere and slablet.sphere_mesh: Buck et al. (2009) Appendix A steady states, mass
conservation, positivity, a rigid rotation on the sphere, and the mesh geometry."""
import numpy as np
import pytest

from slablet import asthenosphere as a
from slablet import sphere_mesh as sm


@pytest.fixture(scope='module')
def sphere():
    return sm.voronoi_mesh(5000)


def _a1_run(factor, dx=50.0, W=5000.0, u0=50.0, myr=200.0):
    p = a.AsthenosphereParams(h_a_km=225.0)
    p.viscosity_pa_s = a.a1_critical_viscosity(p, W, u0) / factor          # Rm = factor x Rm_cr
    n = int(2 * W / dx)
    x = (np.arange(n) + 0.5) * dx
    qi, qj = a.plate_velocity_1d(np.where(x < W, -u0, u0))               # trenches (walls) at both ends
    h, _ = a.advance(a.strip_mesh(n, dx), np.full(n, p.h_a_km), a.Forcing(qi, qj, supply='none'), p, myr)
    h_an, h0 = a.a1_profile(p, W, u0, x[x < W])
    return h[x < W], h_an, h0, p


def test_a1_plate_driven_matches_analytic():
    h, h_an, _, _ = _a1_run(2.0)
    assert np.abs(h - h_an).max() < 1.0                                    # km


def test_a1_critical_state():
    h, _, h0, p = _a1_run(1.0)
    assert h0 == pytest.approx(4.0 / 3.0 * p.h_a_km, rel=1e-6)
    assert h[0] == pytest.approx(h0, rel=0.01)                             # at the trench
    assert h[-1] < 0.3 * p.h_a_km                                          # near zero at the axis (singular there)


def test_a2_source_sink_matches_analytic():
    W, dx = 5000.0, 50.0
    p = a.AsthenosphereParams(h_a_km=225.0, viscosity_pa_s=1e19)
    n = int(W / dx)
    x = (np.arange(n) + 0.5) * dx
    qs = 0.5 * a.a2_critical_flux(p, W)
    src, snk = np.zeros(n), np.zeros(n)
    src[0], snk[-1] = qs, qs
    f = a.Forcing(np.zeros(n - 1), np.zeros(n - 1), source=src, sink=snk, supply='none')
    h, log = a.advance(a.strip_mesh(n, dx), np.full(n, p.h_a_km), f, p, 300.0)
    h_an, _ = a.a2_profile(p, W, qs, x)
    assert np.abs(h - h_an)[:-3].max() < 2.0
    assert log.deficit.sum() == pytest.approx(0.0, abs=1e-6)


def test_mesh_area_and_edges(sphere):
    assert sphere.area.sum() == pytest.approx(4 * np.pi * sphere.radius_km ** 2, rel=1e-12)
    assert np.all(sphere.edge_len > 0) and np.all(sphere.edge_dist > 0)
    # the edge normal -(a x b) points from cell i to cell j
    assert np.all(np.einsum('ij,ij->i', np.cross(sphere.edge_a, sphere.edge_b),
                            sphere.xyz[sphere.edge_j] - sphere.xyz[sphere.edge_i]) < 0)


def test_rigid_edge_flux_matches_midpoint_velocity(sphere):
    w = np.radians([0.3, -0.4, 0.9])
    q, _ = sm.rigid_edge_flux(sphere, w)
    mid = sphere.edge_a + sphere.edge_b
    mid /= np.linalg.norm(mid, axis=1, keepdims=True)
    d = sphere.xyz[sphere.edge_j] - sphere.xyz[sphere.edge_i]
    nhat = d - np.einsum('ij,ij->i', d, mid)[:, None] * mid
    nhat /= np.linalg.norm(nhat, axis=1, keepdims=True)
    approx = np.einsum('ij,ij->i', np.cross(w, mid * sphere.radius_km), nhat) * sphere.edge_len
    assert np.abs(q - approx).max() < 1e-3 * np.abs(approx).max()


def test_uniform_layer_under_one_rigid_rotation_stays_uniform(sphere):
    p = a.AsthenosphereParams()
    qi, qj = sm.rigid_edge_flux(sphere, np.radians([0.0, 0.5, 0.8]))
    h, _ = a.advance(sphere, np.full(sphere.n, p.h_a_km), a.Forcing(qi, qj, supply='none'), p, 2.0)
    assert np.abs(h - p.h_a_km).max() < 1e-8


def _two_plates(sphere):
    w = np.where(sphere.xyz[:, 2:3] > 0, np.radians([0.0, 0.0, 1.0]), np.radians([0.6, 0.0, 0.0]))
    return sm.rigid_edge_flux(sphere, w[sphere.edge_i], w[sphere.edge_j])


def test_mass_conserved_with_walls_and_keels(sphere):
    p = a.AsthenosphereParams()
    rng = np.random.default_rng(1)
    qi, qj = _two_plates(sphere)
    closed = sphere.lat > 70.0
    blocked = rng.random(len(sphere.edge_i)) < 0.05
    h0 = np.where(closed, 0.0, rng.uniform(50.0, 400.0, sphere.n))
    h, _ = a.advance(sphere, h0, a.Forcing(qi, qj, blocked=blocked, closed=closed, supply='none'), p, 2.0)
    assert (h * sphere.area).sum() == pytest.approx((h0 * sphere.area).sum(), rel=1e-12)
    assert h.min() >= 0.0 and np.all(h[closed] == 0.0)


def test_starved_sink_stays_positive_and_balanced_supply_keeps_volume(sphere):
    p = a.AsthenosphereParams()
    qi, qj = _two_plates(sphere)
    sink = np.where(np.abs(sphere.lat) < 2.0, 1e9, 0.0)                    # far more than the cells hold
    h0 = np.full(sphere.n, p.h_a_km)
    h, log = a.advance(sphere, h0, a.Forcing(qi, qj, sink=sink, supply='balance'), p, 2.0)
    assert h.min() >= 0.0
    assert log.deficit.sum() > 0.0                                          # the sink went unmet: starved
    assert (h * sphere.area).sum() == pytest.approx((h0 * sphere.area).sum(), rel=1e-10)


def _box_keel(nx, ny, x0, x1, y0, y1):
    idx = np.arange(nx * ny)
    ix, iy = idx % nx, idx // nx
    return (ix >= x0) & (ix < x1) & (iy >= y0) & (iy < y1)


def test_bulldozed_layer_goes_ahead_and_uncovered_cells_start_empty():
    mesh, _ = a.box_mesh(20, 20, 100.0)
    h0 = np.full(mesh.n, 300.0)
    old = _box_keel(20, 20, 5, 15, 5, 10)
    new = _box_keel(20, 20, 5, 15, 7, 12)               # keel moved two rows north
    h0[old] = 0.0
    h, deleted = a.move_keels(mesh, h0, old, new)
    assert deleted == 0.0
    assert (h * mesh.area).sum() == pytest.approx((h0 * mesh.area).sum(), rel=1e-12)
    assert np.all(h[new] == 0.0)
    assert np.all(h[old & ~new] == 0.0)                 # uncovered behind the keel: empty
    ahead = _box_keel(20, 20, 5, 15, 12, 13)
    assert np.all(h[ahead] > 300.0)                     # the covered layer piled up in front
    behind = _box_keel(20, 20, 5, 15, 4, 5)
    assert np.allclose(h[behind], 300.0)                # nothing pushed backwards
    h_del, deleted = a.move_keels(mesh, h0, old, new, mode='delete')
    assert deleted == pytest.approx(2 * 10 * 300.0 * 100.0 ** 2)


def test_outflow_is_logged_and_optionally_replaced():
    mesh, normal = a.box_mesh(10, 10, 100.0)
    p = a.AsthenosphereParams()
    qi, qj = a.edge_velocity_2d(mesh, normal, np.tile([0.0, 60.0], (mesh.n, 1)))
    north = np.arange(mesh.n) // 10 == 9
    out = np.where(north, 60.0 * 100.0, 0.0)            # h x plate speed x edge length
    h0 = np.full(mesh.n, 300.0)
    v0 = (h0 * mesh.area).sum()
    for replace in (False, True):
        f = a.Forcing(qi, qj, outflow=out, balance_outflow=replace)
        h, log = a.advance(mesh, h0, f, p, 2.0)
        v = (h * mesh.area).sum()
        assert log.outflow > 0 and h.min() >= 0.0
        assert v == pytest.approx(v0 if replace else v0 - log.outflow, rel=1e-10)


def test_supply_mask_feeds_only_its_cells_and_keeps_volume():
    mesh, normal = a.box_mesh(10, 10, 100.0)
    p = a.AsthenosphereParams(viscosity_pa_s=1e30)                   # no flow: only sink and supply act
    qi, qj = a.edge_velocity_2d(mesh, normal, np.zeros((mesh.n, 2)))
    sink = np.zeros(mesh.n)
    sink[55] = 1e5
    fed = np.arange(mesh.n) < 50
    h0 = np.full(mesh.n, 300.0)
    h, log = a.advance(mesh, h0, a.Forcing(qi, qj, sink=sink, supply_mask=fed), p, 0.01)
    assert (h * mesh.area).sum() == pytest.approx((h0 * mesh.area).sum(), rel=1e-12)
    assert np.all(h[fed] > 300.0)
    unfed = ~fed
    unfed[55] = False
    assert np.allclose(h[unfed], 300.0)


def test_source_beyond_sinks_is_taken_back_without_negative_cells():
    mesh, normal = a.box_mesh(10, 10, 100.0)
    p = a.AsthenosphereParams(viscosity_pa_s=1e30)
    qi, qj = a.edge_velocity_2d(mesh, normal, np.zeros((mesh.n, 2)))
    h0 = np.full(mesh.n, 300.0)
    h0[:10] = 0.0                                                    # empty cells must stay at zero, not go negative
    src = np.zeros(mesh.n)
    src[55] = 1e7
    h, log = a.advance(mesh, h0, a.Forcing(qi, qj, source=src), p, 0.1)
    assert (h * mesh.area).sum() == pytest.approx((h0 * mesh.area).sum(), rel=1e-12)
    assert h.min() >= 0.0 and np.all(h[:10] < 1e-6) and log.supplied < 0


def _plume_box():
    mesh, normal = a.box_mesh(21, 21, 100.0)
    qi, qj = a.edge_velocity_2d(mesh, normal, np.zeros((mesh.n, 2)))
    return mesh, qi, qj


def test_plume_tracking_with_no_plume_matches_one_density():
    mesh, qi, qj = _plume_box()
    h0 = np.where(np.arange(mesh.n) % 7 == 0, 400.0, 250.0)
    sink = np.zeros(mesh.n)
    sink[100] = 2e5
    f = a.Forcing(qi, qj, sink=sink)
    h1, _ = a.advance(mesh, h0, f, a.AsthenosphereParams(), 5.0)
    h2, log = a.advance(mesh, h0, f, a.AsthenosphereParams(plume_delta_rho=30.0), 5.0, c=np.zeros(mesh.n))
    assert np.allclose(h1, h2, rtol=1e-12, atol=1e-9) and np.all(log.c == 0.0)


def test_uniform_plume_layer_stays_put_and_blob_spreads_conserving_plume_volume():
    mesh, qi, qj = _plume_box()
    p = a.AsthenosphereParams(plume_delta_rho=30.0)
    h0 = np.full(mesh.n, 300.0)
    h, log = a.advance(mesh, h0, a.Forcing(qi, qj, supply='none'), p, 5.0, c=np.ones(mesh.n))
    assert np.allclose(h, 300.0) and np.allclose(log.c, 1.0)
    c0 = np.zeros(mesh.n)
    c0[220] = 1.0                                                    # centre cell of 21 x 21
    h, log = a.advance(mesh, h0, a.Forcing(qi, qj, supply='none'), p, 2.0, c=c0)
    assert (h * mesh.area).sum() == pytest.approx((h0 * mesh.area).sum(), rel=1e-12)
    assert (log.c * h * mesh.area).sum() == pytest.approx(300.0 * mesh.area[220], rel=1e-9)
    assert h[220] < 300.0 and log.c[221] > 0.0 and log.c.max() <= 1.0
    assert p.depth_anomaly_km(h0, c0)[220] < 0.0 < p.depth_anomaly_km(h0, c0)[0] + 1e-12   # shallower over it


def test_plume_source_carries_plume_material():
    mesh, qi, qj = _plume_box()
    p = a.AsthenosphereParams(plume_delta_rho=30.0)
    src = np.zeros(mesh.n)
    src[220] = 1e5
    h, log = a.advance(mesh, np.full(mesh.n, 300.0), a.Forcing(qi, qj, source=src, source_plume=True,
                                                               supply='none'), p, 1.0, c=np.zeros(mesh.n))
    assert (log.c * h * mesh.area).sum() == pytest.approx(1e5, rel=1e-9)
