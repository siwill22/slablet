"""The plate-history contract (slablet/plate_history.py) and the real-reconstruction driver.

The driver tests use Müller et al. (2022) from gprm's download cache (configs/muller2022_refined.yaml)
and are skipped when it is not there.
"""
import os

import numpy as np
import pytest
import xarray as xr

from slablet import plate_history as PH

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fibonacci(n):
    i = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1 - 2 * i / n))
    lon = (np.degrees(np.pi * (1 + 5 ** 0.5) * i) + 180) % 360 - 180
    return lon, lat


def _toy_step(**changes):
    """One plate everywhere (id 1) turning about the spin axis at 1 degree per Myr."""
    lons, lats = PH.cell_centred_grid(2.0)
    kw = dict(time=0.0, delta_time=1.0,
              plate_id_at=lambda lon, lat: np.ones(np.shape(lon), int),
              omega={1: np.array([0.0, 0.0, np.radians(1.0)])},
              trench=PH.empty_trench(), ridge=PH.empty_ridge(),
              seafloor_age=xr.DataArray(np.full((len(lats), len(lons)), 50.0), dims=('lat', 'lon'),
                                        coords={'lat': lats, 'lon': lons}),
              continent_at=lambda lon, lat: np.zeros(np.shape(lon), bool),
              craton_at=lambda lon, lat: np.zeros(np.shape(lon), bool))
    kw.update(changes)
    return PH.PlateStep(**kw)


def test_cell_centred_grid_starts_half_a_cell_in():
    lons, lats = PH.cell_centred_grid(1.0)
    assert (lons[0], lons[-1], len(lons)) == (-179.5, 179.5, 360)
    assert (lats[0], lats[-1], len(lats)) == (-89.5, 89.5, 180)


def test_velocity_is_omega_cross_r():
    lat = np.array([0.0, 30.0, 60.0, -45.0])
    east, north = PH.velocity_at(_toy_step(), np.zeros(4), lat)
    np.testing.assert_allclose(east, np.radians(1.0) * PH.RADIUS_KM * np.cos(np.radians(lat)), rtol=1e-12)
    np.testing.assert_allclose(north, 0.0, atol=1e-9)


def test_velocity_is_nan_off_plates():
    step = _toy_step(plate_id_at=lambda lon, lat: np.zeros(np.shape(lon), int))
    east, _ = PH.velocity_at(step, [0.0], [0.0])
    assert np.isnan(east).all()


def test_check_accepts_a_good_step_and_names_what_is_wrong():
    PH.check(_toy_step())
    bad_trench = dict(PH.empty_trench(), lon=np.array([1.0]))          # arrays of different length
    node_grid = xr.DataArray(np.zeros((181, 361)), dims=('lat', 'lon'),
                             coords={'lat': np.arange(-90, 91.0), 'lon': np.arange(-180, 181.0)})
    with pytest.raises(ValueError, match='trench sample arrays differ'):
        PH.check(_toy_step(trench=bad_trench))
    with pytest.raises(ValueError, match='cell-centred'):
        PH.check(_toy_step(seafloor_age=node_grid))
    with pytest.raises(ValueError, match='omega'):
        PH.check(_toy_step(omega={1: np.array([0.0, np.nan, 0.0])}))


def test_on_grid_evaluates_a_query_at_cell_centres():
    da = PH.on_grid(lambda lon, lat: lat > 0, 10.0)
    assert PH.is_cell_centred(da) and da.sel(lat=85.0).all() and not da.sel(lat=-85.0).any()


# ---- the real-reconstruction driver ----------------------------------------------------------

@pytest.fixture(scope='module')
def muller2022():
    pytest.importorskip('pygplates')
    pytest.importorskip('ptt')
    gprm = pytest.importorskip('gprm')                                   # noqa: F841
    from pooch import os_cache
    if not os.path.isdir(os.path.join(str(os_cache('gprm')), 'Muller2022')):
        pytest.skip('Muller2022 not in gprm\'s cache')
    import yaml
    from slablet.gplates.plate_history import RealPlateHistory
    with open(os.path.join(REPO, 'configs', 'muller2022_refined.yaml')) as fh:
        cfg = yaml.safe_load(fh)
    cwd = os.getcwd()
    os.chdir(REPO)                      # the config's cache_dir is relative to the repository
    try:
        history = RealPlateHistory(cfg)
        yield history, history.step(50.0)
    finally:
        os.chdir(cwd)


def test_real_step_meets_the_contract(muller2022):
    _, step = muller2022
    PH.check(step)
    assert len(step.trench['lon']) > 100 and len(step.ridge['lon']) > 100
    assert np.isfinite(step.trench['downdip_azimuth_deg']).mean() > 0.9      # polarity known almost everywhere
    assert step.trench_lines
    if 'no seafloor tracker cache' in step.seafloor_age.attrs.get('source', ''):
        assert np.isnan(step.seafloor_age.values).all()      # a fresh clone: ages need the tracker run first
    else:
        assert np.isfinite(step.seafloor_age.values).any()


def test_real_plate_ids_match_the_plate_partitioner(muller2022):
    import pygplates
    history, step = muller2022
    lon, lat = _fibonacci(4000)
    resolved = []
    pygplates.resolve_topologies(history.topo, history.rm, resolved, 50.0,
                                 resolve_topology_types=pygplates.ResolveTopologyType.boundary)
    part = pygplates.PlatePartitioner(resolved, history.rm)
    want = []
    for lo, la in zip(lon, lat):
        r = part.partition_point(pygplates.PointOnSphere(la, lo))
        want.append(r.get_feature().get_reconstruction_plate_id() if r else 0)
    np.testing.assert_array_equal(step.plate_id_at(lon, lat), want)


def test_real_velocities_match_pygplates(muller2022):
    import pygplates
    history, step = muller2022
    lon, lat = _fibonacci(500)
    pid = step.plate_id_at(lon, lat)
    east, north = PH.velocity_at(step, lon, lat)
    for k in range(0, 500, 25):
        rot = history.rm.get_rotation(50.0, int(pid[k]), 51.0)
        p = pygplates.PointOnSphere(lat[k], lon[k])
        v = pygplates.calculate_velocities([p], rot, 1.0, pygplates.VelocityUnits.kms_per_my)
        ned = pygplates.LocalCartesian.convert_from_geocentric_to_north_east_down([p], v)[0]
        # pygplates' Earth radius is 6371.009 km, slablet's 6371 km: agreement to about 1.4e-6
        np.testing.assert_allclose([east[k], north[k]], [ned.get_y(), ned.get_x()], rtol=1e-5, atol=1e-6)


def test_real_masks_match_the_reconstructed_polygons(muller2022):
    import pygplates
    history, step = muller2022
    lon, lat = _fibonacci(2000)
    for kind, query in (('continent', step.continent_at), ('craton', step.craton_at)):
        rfgs = []
        for fc in history.polygons[kind]:
            pygplates.reconstruct(fc, history.rm, rfgs, 50.0)
        polys = [r.get_reconstructed_geometry() for r in rfgs]
        want = [any(poly.is_point_in_polygon(pygplates.PointOnSphere(la, lo)) for poly in polys)
                for lo, la in zip(lon, lat)]
        np.testing.assert_array_equal(query(lon, lat), want, err_msg=kind)
