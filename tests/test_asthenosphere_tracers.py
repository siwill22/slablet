"""Passive tracers carried by the asthenosphere layer (asthenosphere.advance(tracers=...), run_layer(tracers0=...)):
they leave h untouched, a well-mixed layer stays well mixed, tracer volume is conserved by the flow and by keel moves,
and concentrations stay within the range they started in and the supply brings, also beside a starved ridge."""
import numpy as np
import pytest
import xarray as xr

from slablet import asthenosphere as A
from slablet import plate_history as PH
from slablet import sphere_mesh as SM
from slablet.asthenosphere_plates import forcing_from_step, run_layer


@pytest.fixture(scope='module')
def mesh():
    return SM.voronoi_mesh(3000)


def _step(omega_deg=1.0, ridge=None, craton_at=None, two_plates=False):
    """One plate turning about the spin axis, or two (west half 1, east half 2) spreading apart at a ridge on 0E."""
    lons, lats = PH.cell_centred_grid(2.0)
    w = np.radians(omega_deg)
    pid = (lambda lon, lat: np.where(np.asarray(lon) < 0, 1, 2)) if two_plates else \
        (lambda lon, lat: np.ones(np.shape(lon), int))
    return PH.check(PH.PlateStep(
        time=0.0, delta_time=1.0, plate_id_at=pid,
        omega={1: np.array([0.0, 0.0, -w if two_plates else w]), 2: np.array([0.0, 0.0, w])},
        trench=PH.empty_trench(), ridge=PH.empty_ridge() if ridge is None else ridge,
        seafloor_age=xr.DataArray(np.full((len(lats), len(lons)), 50.0), dims=('lat', 'lon'),
                                  coords={'lat': lats, 'lon': lons}),
        continent_at=lambda lon, lat: np.zeros(np.shape(lon), bool),
        craton_at=craton_at or (lambda lon, lat: np.zeros(np.shape(lon), bool)), trench_lines=[]))


def _ridge_on_0E(rate_km_per_myr=50.0):
    lat = np.arange(-79.5, 80, 1.0)
    return dict(lon=np.zeros_like(lat), lat=lat, divergence_km_per_myr=rate_km_per_myr * np.cos(np.radians(lat)),
                length_km=np.full(lat.size, 111.2))


P = A.AsthenosphereParams(keel_mode='craton', supply_where='open')


def test_tracers_leave_h_untouched(mesh):
    f, _ = forcing_from_step(mesh, _step(ridge=_ridge_on_0E(), two_plates=True), P)
    h0 = np.full(mesh.n, 300.0)
    plain, log0 = A.advance(mesh, h0, f, P, 3.0)
    carried, log1 = A.advance(mesh, h0, f, P, 3.0, tracers=np.vstack([mesh.lon < -30, mesh.lat > 0]).astype(float))
    assert np.array_equal(plain, carried) and log0.substeps == log1.substeps
    assert log0.tracers is None and log1.tracers.shape == (2, mesh.n)


def test_a_well_mixed_layer_stays_well_mixed(mesh):
    f, _ = forcing_from_step(mesh, _step(ridge=_ridge_on_0E(), two_plates=True), P)
    f.supply_tracers = [0.3]
    h, log = A.advance(mesh, np.full(mesh.n, 300.0), f, P, 5.0, tracers=np.full((1, mesh.n), 0.3))
    full = h > 1e-6
    assert np.allclose(log.tracers[0][full], 0.3, atol=1e-12)


def test_rotation_conserves_tracer_volume_and_moves_the_front_east(mesh):
    f, _ = forcing_from_step(mesh, _step(omega_deg=1.0), P)
    c0 = ((mesh.lon > -20) & (mesh.lon < 20)).astype(float)
    h, log = A.advance(mesh, np.full(mesh.n, 300.0), f, P, 20.0, tracers=c0[None])
    c = log.tracers[0]
    assert (c * h * mesh.area).sum() == pytest.approx((c0 * 300.0 * mesh.area).sum(), rel=1e-12)
    eq = np.abs(mesh.lat) < 30
    centre = np.degrees(np.angle(np.sum((c * mesh.area)[eq] * np.exp(1j * np.radians(mesh.lon[eq])))))
    assert centre == pytest.approx(10.0, abs=1.5)          # the layer moves at half the plate's 1 degree per Myr
    assert c.min() >= 0.0 and c.max() <= 1.0


def test_starved_ridge_keeps_concentrations_in_range(mesh):
    f, _ = forcing_from_step(mesh, _step(omega_deg=0.5, ridge=_ridge_on_0E(80.0), two_plates=True), P)
    f.supply_tracers = [0.2]
    c0 = np.where(mesh.lon < 0, 1.0, 0.5)
    h, log = A.advance(mesh, np.full(mesh.n, 40.0), f, P, 30.0, tracers=c0[None])
    assert log.deficit.sum() > 0 and (h < 1e-3).any()       # the ridge is starved somewhere
    assert log.tracers.min() >= 0.0 and log.tracers[0][h > 1e-6].min() >= 0.2 - 1e-12
    assert log.tracers.max() <= 1.0


def test_run_layer_bulldozes_tracers_with_the_layer(mesh):
    def step_at(t):                                          # a keel slides east 2 degrees per Myr, no sinks
        x = 2.0 * (10 - t)
        return _step(craton_at=lambda lon, lat: (np.abs(np.asarray(lon) - x) < 10) & (np.abs(lat) < 20))
    p = A.AsthenosphereParams(keel_mode='craton', supply_where='open', supply_mode='none')
    c0 = (mesh.lat > 0).astype(float)
    total = None
    for t, h, f, info, log in run_layer(step_at, mesh, p, start_ma=10, end_ma=0, tracers0=c0[None]):
        amount = (log.tracers[0] * h * mesh.area).sum()
        total = amount if total is None else total
        assert amount == pytest.approx(total, rel=1e-9)
    assert 0.0 <= log.tracers.min() and log.tracers.max() <= 1.0
