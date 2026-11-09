"""The asthenosphere layer driven through the plate-history contract (slablet/asthenosphere_plates.py), on toy plate
steps: the drag is exact for a rigid plate, trench lines become walls, ridge samples become sinks off the keels, and
run_layer moves keels and keeps the layer's volume."""
import numpy as np
import pytest
import xarray as xr

from slablet import asthenosphere as A
from slablet import plate_history as PH
from slablet import sphere_mesh as SM
from slablet.asthenosphere_plates import forcing_from_step, run_layer


@pytest.fixture(scope='module')
def mesh():
    return SM.voronoi_mesh(5000)


def _step(**changes):
    """One plate everywhere (id 1) turning about the spin axis at 1 degree per Myr, no trenches or ridges."""
    lons, lats = PH.cell_centred_grid(2.0)
    kw = dict(time=0.0, delta_time=1.0,
              plate_id_at=lambda lon, lat: np.ones(np.shape(lon), int),
              omega={1: np.array([0.0, 0.0, np.radians(1.0)])},
              trench=PH.empty_trench(), ridge=PH.empty_ridge(),
              seafloor_age=xr.DataArray(np.full((len(lats), len(lons)), 50.0), dims=('lat', 'lon'),
                                        coords={'lat': lats, 'lon': lons}),
              continent_at=lambda lon, lat: np.zeros(np.shape(lon), bool),
              craton_at=lambda lon, lat: np.zeros(np.shape(lon), bool),
              trench_lines=[])
    kw.update(changes)
    return PH.check(PH.PlateStep(**kw))


P = A.AsthenosphereParams(keel_mode='craton', supply_where='open')


def test_trench_lines_are_required(mesh):
    with pytest.raises(ValueError):
        forcing_from_step(mesh, _step(trench_lines=None), P)


def test_one_rigid_plate_leaves_a_uniform_layer_uniform(mesh):
    f, _ = forcing_from_step(mesh, _step(), P)
    h, _ = A.advance(mesh, np.full(mesh.n, 300.0), f, P, 5.0)
    assert np.allclose(h, 300.0, rtol=1e-9)


def test_a_trench_line_is_a_wall(mesh):
    line = np.column_stack([np.full(61, 30.0), np.linspace(-30, 30, 61)])     # along 30E, 30S-30N
    f, info = forcing_from_step(mesh, _step(trench_lines=[line]), P)
    assert f.blocked.sum() > 0
    # every blocked edge joins a cell west of 30E to one east of it
    lo_i, lo_j = mesh.lon[mesh.edge_i[f.blocked]], mesh.lon[mesh.edge_j[f.blocked]]
    assert np.all((lo_i - 30.0) * (lo_j - 30.0) < 0)
    assert len(info['trench_a']) == 60


def test_ridge_samples_are_sinks_except_on_keels(mesh):
    ridge = dict(lon=np.array([0.0, 100.0]), lat=np.array([0.0, 0.0]), divergence_km_per_myr=np.array([50.0, 50.0]),
                 length_km=np.array([55.0, 55.0]))
    keel_at = lambda lon, lat: (np.asarray(lon) > 90) & (np.asarray(lon) < 110) & (np.abs(lat) < 10)  # noqa: E731
    f, info = forcing_from_step(mesh, _step(ridge=ridge, craton_at=keel_at), P)
    assert f.sink.sum() == pytest.approx(50.0 * P.lithosphere_km * 55.0)
    assert f.sink[SM.nearest_cell(mesh, [0.0], [0.0])[0]] > 0
    assert list(info['ridge_on_land']) == [False, True]
    assert f.closed.sum() > 0


def test_run_layer_moves_keels_and_keeps_the_volume(mesh):
    def keel_at_time(t):
        lo0 = 20.0 + 2.0 * (10.0 - t)            # a keel moving east 2 degrees per Myr
        return lambda lon, lat: (np.asarray(lon) > lo0) & (np.asarray(lon) < lo0 + 20) & (np.abs(lat) < 20)

    def step_at(t):
        return _step(time=t, craton_at=keel_at_time(t), omega={1: np.zeros(3)})

    vols, uncovered_empty = [], []
    prev = None
    for t, h, f, info, log in run_layer(step_at, mesh, P, start_ma=10, end_ma=0):
        vols.append((h * mesh.area).sum())
        if prev is not None:
            unc = prev & ~f.closed
            uncovered_empty.append(np.all(h[unc] < 300.0))
        prev = np.asarray(f.closed, bool)
    assert t == 0 and len(vols) == 10
    assert np.allclose(vols, vols[0], rtol=1e-9)          # bulldozing and refilling keep the volume
    assert all(uncovered_empty)                            # uncovered cells start empty and only partly refill
