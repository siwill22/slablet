"""Flow-driven volcanism (slablet/asthenosphere_volcanism.py): the mesh gradient is right, plate velocity is omega x r,
plate drag alone gives the plate-motion chain direction, pressure-driven flow turns it, and the layer rises where it
flows towards thinner lithosphere."""
import numpy as np
import pytest
import xarray as xr

from slablet import asthenosphere as A
from slablet import asthenosphere_volcanism as V
from slablet import plate_history as PH
from slablet import sphere_mesh as SM
from slablet.asthenosphere_plates import forcing_from_step


@pytest.fixture(scope='module')
def mesh():
    return SM.voronoi_mesh(5000)


def _step(age=50.0, omega_deg=1.0):
    lons, lats = PH.cell_centred_grid(2.0)
    a = age(lons[None, :], lats[:, None]) if callable(age) else np.full((len(lats), len(lons)), age)
    return PH.check(PH.PlateStep(
        time=0.0, delta_time=1.0, plate_id_at=lambda lon, lat: np.ones(np.shape(lon), int),
        omega={1: np.array([0.0, 0.0, np.radians(omega_deg)])}, trench=PH.empty_trench(), ridge=PH.empty_ridge(),
        seafloor_age=xr.DataArray(np.broadcast_to(a, (len(lats), len(lons))).astype(float), dims=('lat', 'lon'),
                                  coords={'lat': lats, 'lon': lons}),
        continent_at=lambda lon, lat: np.zeros(np.shape(lon), bool),
        craton_at=lambda lon, lat: np.zeros(np.shape(lon), bool), trench_lines=[]))


def test_gradient_of_a_linear_field(mesh):
    g = V.gradient(mesh, 1000.0 * mesh.xyz[:, 2])                # 1000 z: gradient (z_hat - z r) 1000 / R
    exact = (np.array([0.0, 0.0, 1.0]) - mesh.xyz[:, 2:3] * mesh.xyz) * 1000.0 / mesh.radius_km
    mid = np.abs(mesh.lat) < 60
    err = np.linalg.norm(g - exact, axis=1)[mid] / np.linalg.norm(exact, axis=1)[mid]
    assert np.median(err) < 0.02


def test_plate_velocity_is_omega_cross_r(mesh):
    v = V.plate_velocity(mesh, _step(omega_deg=1.0))
    e, n = SM.east_north(mesh, v)
    eq = np.abs(mesh.lat) < 2
    assert np.allclose(e[eq], np.radians(1.0) * mesh.radius_km * np.cos(np.radians(mesh.lat[eq])), rtol=1e-6)
    assert np.allclose(n, 0.0, atol=1e-9)


def test_plate_drag_alone_gives_the_plate_motion_direction(mesh):
    p = A.AsthenosphereParams(keel_mode='craton', supply_where='open')
    step = _step()
    f, _ = forcing_from_step(mesh, step, p)
    h = np.full(mesh.n, 300.0)
    out = V.flow_volcanism(mesh, step, V.layer_velocity(mesh, h, f, p), h)
    mid = np.abs(mesh.lat) < 60
    assert np.median(out['turn_deg'][mid]) < 1.0                 # u_rel = -u_plate / 2: same direction
    assert np.median(out['speed'][mid] / np.linalg.norm(V.plate_velocity(mesh, step), axis=1)[mid]) == \
        pytest.approx(0.5, abs=0.01)


def test_pressure_flow_turns_the_chain(mesh):
    step = _step(omega_deg=0.0)                                  # a plate at rest over a layer flowing north
    u_layer = np.cross(np.array([np.radians(0.5), 0.0, 0.0]) * mesh.radius_km, mesh.xyz)
    h = np.full(mesh.n, 300.0)
    out = V.flow_volcanism(mesh, step, u_layer, h)
    assert np.allclose(out['speed'], np.linalg.norm(u_layer, axis=1))
    lon0 = (np.abs(mesh.lat) < 10) & (np.abs(mesh.lon - 90.0) < 10)
    assert np.median(np.abs((out['chain_az'][lon0] + 180.0) % 360.0 - 180.0)) < 2.0   # north, at 90E


def test_the_layer_rises_flowing_towards_thinner_lithosphere(mesh):
    young_east = _step(age=lambda lon, lat: np.clip(80.0 - (lon + 180.0) / 4.0, 1.0, 80.0) + 0 * lat, omega_deg=0.0)
    east = np.cross(np.array([0.0, 0.0, np.radians(0.5)]) * mesh.radius_km, mesh.xyz)
    out = V.flow_volcanism(mesh, young_east, east, np.full(mesh.n, 300.0))
    band = (np.abs(mesh.lat) < 30) & (mesh.lon > -150) & (mesh.lon < 100)
    assert np.median(out['w'][band]) > 0
    west = V.flow_volcanism(mesh, young_east, -east, np.full(mesh.n, 300.0))
    assert np.median(west['w'][band]) < 0


def test_smoothing_keeps_a_constant_and_damps_noise(mesh):
    assert np.allclose(V.smooth(mesh, np.full(mesh.n, 7.0)), 7.0)
    noise = np.random.default_rng(0).normal(size=mesh.n)
    assert V.smooth(mesh, noise).std() < 0.5 * noise.std()


def test_no_slope_across_a_wall(mesh):
    p = A.AsthenosphereParams(keel_mode='craton', supply_where='open')
    line = np.column_stack([np.full(61, 0.0), np.linspace(-30, 30, 61)])
    step = _step(age=lambda lon, lat: np.where(lon < 0, 100.0, 1.0) + 0 * lat, omega_deg=0.0)
    step.trench_lines = [line]
    f, _ = forcing_from_step(mesh, step, p)
    ok = V.open_edges(mesh, f)
    L = V.lithosphere_km(mesh, step)
    g_wall = np.linalg.norm(V.gradient(mesh, V.smooth(mesh, L, 4, ok), ok), axis=1)
    g_free = np.linalg.norm(V.gradient(mesh, V.smooth(mesh, L, 4)), axis=1)
    near = (np.abs(mesh.lon) < 4) & (np.abs(mesh.lat) < 25)
    assert g_wall[near].max() < 0.2 * g_free[near].max()
