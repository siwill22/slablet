"""Tests for the shared mantle package.

    conda run -n pygmt17 python -m pytest tests

tests/data/mantle_golden.npz was made with the code as it was before it moved into
mantle/ (self_driving_plates.update_slab_tracers etc.), on the small problem built
in _golden_run below.  The mantle code must reproduce it exactly.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import mantle  # noqa: E402

GOLDEN = os.path.join(REPO, 'tests', 'data', 'mantle_golden.npz')
FORBIDDEN = ('self_driving_plates', 'run_SDP', 'plate_driver', 'reconstructions',
             'pygplates', 'geopandas', 'shapely')


def test_mantle_imports_nothing_from_either_stream():
    code = ('import sys, mantle; '
            'print(",".join(m for m in {!r} if m in sys.modules))'.format(FORBIDDEN))
    out = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '', 'mantle imported: ' + out.stdout.strip()


def _golden_run():
    """Initial slab columns, 10 steps of 2 Myr, then the dVs field, from PlateDriver-style
    boundaries through the sdp adapter (the same path run_SDP.py takes)."""
    gpd = pytest.importorskip('geopandas')
    from shapely.geometry import LineString
    import self_driving_plates as sdp

    def boundaries(step):
        # three trenches (one crossing the dateline, one at high latitude, one slow) and a ridge
        lines = [LineString([(170, -30), (-170, -10)]), LineString([(-60, 60), (-20, 72)]),
                 LineString([(30, -5), (50, 5)]), LineString([(0, -50), (10, -20)])]
        div = [-60.0 - step, -25.0, -4.0, 30.0]
        return gpd.GeoDataFrame({'divergence': div}, geometry=lines, crs=4326)

    p = mantle.MantleParams(mantle_grid_spacing_deg=10.0, mantle_depth_interval_km=200.0,
                            slab_thermal_tau_myr=4.0, tomo_smooth_horiz_deg=5.0,
                            tomo_smooth_depth_km=150.0, initial_slab_depth_km=700.0)
    slabs = mantle.initial_slab_columns(sdp.trench_samples(boundaries(0)), 2.0, p)
    n_init = len(slabs)
    cmb = mantle.new_cmb_heat(p)
    for step in range(1, 11):
        slabs = mantle.seed_slabs(mantle.sink_slabs(slabs, 2.0, p),
                                  sdp.trench_samples(boundaries(step)), 2.0, p)
        cmb = mantle.update_cmb_heat(cmb, slabs, 2.0, p)
    ds = mantle.dvs_field(slabs, cmb, p)
    mid_lon, mid_lat = np.array([175.0, -40.0]), np.array([-20.0, 66.0])
    side = mantle.lonlat_to_xyz(np.array([178.0, -42.0]), np.array([-19.0, 68.0])) \
        - mantle.lonlat_to_xyz(mid_lon, mid_lat)
    z = mantle.slab_depth_near(mid_lon, mid_lat, side, slabs.lon.values, slabs.lat.values, slabs.depth_km.values)
    return n_init, slabs, cmb, ds, z


def test_reproduces_pre_extraction_code_exactly():
    g = np.load(GOLDEN)
    n_init, slabs, cmb, ds, z = _golden_run()
    assert n_init == int(g['n_init'])
    for col in ('lon', 'lat', 'depth_km', 'weight', 'age'):
        np.testing.assert_array_equal(slabs[col].values.astype(float), g[col], err_msg=col)
    np.testing.assert_array_equal(cmb, g['cmb_heat'])
    np.testing.assert_array_equal(ds.dVs.values, g['dVs'])
    np.testing.assert_array_equal(z, g['slab_depth'])


def test_dvs_has_zero_area_weighted_mean_at_each_depth():
    _, slabs, cmb, ds, _ = _golden_run()
    w = np.cos(np.radians(ds.lat.values))[:, None] * np.ones(len(ds.lon))
    means = (ds.dVs.values * w).sum(axis=(1, 2)) / w.sum()
    np.testing.assert_allclose(means, 0.0, atol=1e-12)


def test_slab_weight_per_myr_does_not_depend_on_time_step():
    p = mantle.MantleParams()
    samples = {'lon': np.array([10.0, 20.0]), 'lat': np.array([0.0, 5.0]),
               'convergence_km_per_myr': np.array([50.0, 4.0])}
    empty = mantle.core.pd.DataFrame(columns=mantle.SLAB_COLUMNS)
    one = mantle.seed_slabs(empty, samples, 2.0, p)
    two = mantle.seed_slabs(mantle.seed_slabs(empty, samples, 1.0, p), samples, 1.0, p)
    assert one['weight'].sum() == pytest.approx(two['weight'].sum())


def test_params_read_config_keys_and_ignore_others():
    p = mantle.MantleParams.from_config({'slab_thermal_tau_myr': 4, 'slab_pull': 6, 'cmb_sigma_deg': 7.0})
    assert p.slab_thermal_tau_myr == 4 and p.cmb_sigma_deg == 7.0
    assert p.slab_max_sink_rate == mantle.MantleParams().slab_max_sink_rate


def _ns_slab(lat0, depth_km=1050.0, age=0.0):
    """A north-south slab 20° long, one slablet per great-circle degree, centred on lat0."""
    lat = np.arange(lat0 - 10, lat0 + 10, 1.0) + 0.5
    n = len(lat)
    return mantle.core.pd.DataFrame({'lon': np.full(n, 32.5), 'lat': lat, 'depth_km': depth_km, 'age': age,
                                     'active': True, 'sink_rate_km_per_myr': 50.0, 'weight': 1.0,
                                     'attached': True})


def _integrated_anomaly(p, lat0):
    ds = mantle.dvs_field(_ns_slab(lat0), None, p)
    f = ds.dVs.sel(depth=1050).values
    f = f - f.min()                                   # undo the zero-mean shift
    return (f * np.cos(np.radians(ds.lat.values))[:, None]).sum()


def test_area_normalised_binning_counts_a_slab_the_same_at_every_latitude():
    p_off = mantle.MantleParams(slab_thermal_tau_myr=1e9)
    p_on = mantle.MantleParams(slab_thermal_tau_myr=1e9, mantle_bin_area_normalise=True)
    ratio_off = _integrated_anomaly(p_off, 60.0) / _integrated_anomaly(p_off, 0.0)
    ratio_on = _integrated_anomaly(p_on, 60.0) / _integrated_anomaly(p_on, 0.0)
    assert ratio_off < 0.6                            # the old binning halves a slab at 60°
    assert ratio_on == pytest.approx(1.0, abs=0.05)


def test_slabs_rest_at_the_cmb_until_their_anomaly_fades():
    p = mantle.MantleParams(slab_at_cmb='rest', slab_thermal_tau_myr=10.0)
    slabs = _ns_slab(0.0, depth_km=2880.0)
    out = mantle.sink_slabs(slabs, 1.0, p)
    assert len(out) == len(slabs) and (out['depth_km'] == p.cmb_depth_km).all()
    old = mantle.sink_slabs(_ns_slab(0.0, depth_km=2880.0, age=100.0), 1.0, p)   # exp(-101/10) < 1e-3
    assert len(old) == 0
    removed = mantle.sink_slabs(slabs, 1.0, mantle.MantleParams())
    assert len(removed) == 0


def test_cmb_suppression_can_follow_the_thermal_decay():
    young, aged = _ns_slab(0.0, depth_km=2850.0), _ns_slab(0.0, depth_km=2850.0, age=400.0)
    heat = np.full((36, 72), 1.0)
    for flag, expect_equal in ((False, True), (True, False)):
        p = mantle.MantleParams(cmb_suppression_thermal_decay=flag)
        a = mantle.update_cmb_heat(heat, young, 1.0, p).min()
        b = mantle.update_cmb_heat(heat, aged, 1.0, p).min()
        assert (a == b) == expect_equal
        if flag:
            assert b > a                              # the aged slab suppresses less


def _cell_area(p):
    return mantle.pile._grid(p)[2][:, None]


def test_pile_conserves_volume_and_moves_away_from_slabs():
    p = mantle.MantleParams(llsvp='pile', slab_at_cmb='rest')
    h0 = mantle.new_llsvp_state(p)
    slabs = _ns_slab(0.0, depth_km=p.cmb_depth_km)                 # a slab lying on the CMB at 32.5°E
    slabs['weight'] = 100.0                                         # about 100 Myr of subduction
    h = h0
    for _ in range(50):
        h = mantle.update_llsvp(h, slabs, 1.0, p)
    area = _cell_area(p)
    assert (h * area).sum() == pytest.approx((h0 * area).sum(), rel=1e-9)
    assert (h >= 0).all() and np.isfinite(h).all()
    lons, lats = mantle.mantle_grid(p.mantle_grid_spacing_deg)
    under = h[np.abs(lats) < 5][:, np.abs(lons - 32.5) < 3].mean()
    far = h[np.abs(lats) < 5][:, np.abs(((lons - 32.5 + 180) % 360) - 180) > 90].mean()
    assert under < 0.5 * p.pile_mean_thickness_km < far


def test_pile_spreads_back_to_a_uniform_layer_without_slabs():
    p = mantle.MantleParams(llsvp='pile')
    h = mantle.new_llsvp_state(p)
    h[10:20, 10:20] *= 4.0
    empty = mantle.core.pd.DataFrame(columns=mantle.SLAB_COLUMNS)
    spread0 = h.std()
    for _ in range(200):
        h = mantle.update_llsvp(h, empty, 1.0, p)
    assert h.std() < 0.3 * spread0


def test_pile_layer_fractions():
    depths = np.array([2650.0, 2750.0, 2850.0])
    f = mantle.pile.pile_layer_fractions(np.array([[150.0]]), depths, 100.0, 2890.0)[:, 0, 0]
    np.testing.assert_allclose(f, [0.0, 0.6, 1.0])     # pile from 2740 km to the CMB at 2890 km


@pytest.mark.parametrize('l, pattern', [(2, lambda lon, lat: 1.5 * np.sin(lat) ** 2 - 0.5),
                                        (3, lambda lon, lat: np.cos(lat) ** 3 * np.cos(3 * lon))])
def test_flow_potential_solves_poisson_on_the_sphere(l, pattern):
    p = mantle.MantleParams(pile_drive='flow')
    lons, lats = mantle.mantle_grid(p.mantle_grid_spacing_deg)
    lon, lat = np.meshgrid(np.radians(lons), np.radians(lats))
    load = pattern(lon, lat)
    psi = mantle.pile.flow_potential(load, p)
    np.testing.assert_allclose(psi, load / (l * (l + 1)), atol=0.01 * np.abs(load).max() / (l * (l + 1)))


def test_flow_pile_collects_on_the_far_side_from_a_slab():
    """A slab on the CMB at 32.5°E: the local pile only thins under it; the flow pile also
    piles up on the opposite side (147.5°W), more than a quarter of the way round (122.5°E)."""
    far = {}
    for drive in ('local', 'flow'):
        p = mantle.MantleParams(llsvp='pile', slab_at_cmb='rest', pile_drive=drive, pile_slab_head_km=3000.0)
        h0 = mantle.new_llsvp_state(p)
        slabs = _ns_slab(0.0, depth_km=p.cmb_depth_km)
        slabs['weight'] = 1000.0                                    # a long-lived slab graveyard
        h = h0
        for _ in range(100):
            h = mantle.update_llsvp(h, slabs, 1.0, p)
        area = _cell_area(p)
        assert (h * area).sum() == pytest.approx((h0 * area).sum(), rel=1e-9)
        lons, lats = mantle.mantle_grid(p.mantle_grid_spacing_deg)
        eq = h[np.abs(lats) < 5].mean(axis=0)
        far[drive] = eq[np.argmin(np.abs(lons + 147.5))] / eq[np.argmin(np.abs(lons - 122.5))]
    assert far['local'] == pytest.approx(1.0, abs=0.05)
    assert far['flow'] > 1.5


def _dip_samples(azimuth=90.0):
    return {'lon': np.array([0.0]), 'lat': np.array([0.0]), 'convergence_km_per_myr': np.array([50.0]),
            'downdip_azimuth_deg': np.array([azimuth]), 'trench_east_km_per_myr': np.array([0.0]),
            'trench_north_km_per_myr': np.array([0.0])}


def test_dipping_slab_moves_towards_the_overriding_plate_in_the_upper_mantle_only():
    p = mantle.MantleParams(slab_dip_deg=45.0)                    # sinks 50 km/Myr in the upper mantle
    slabs = mantle.seed_slabs(mantle.core.pd.DataFrame(columns=mantle.SLAB_COLUMNS), _dip_samples(), 1.0, p)
    for _ in range(6):                                            # to 300 km: 300 km east (45 deg)
        slabs = mantle.sink_slabs(slabs, 1.0, p)
    first = slabs.iloc[0]
    assert first.depth_km == pytest.approx(300.0)
    assert abs(first.lat) < 1e-9
    km = np.radians(first.lon) * 6371.0
    assert 300.0 < km < 300.0 * 6371.0 / (6371.0 - 300.0)        # moved at depth: a bit more than 300 km of surface
    for _ in range(40):
        slabs = mantle.sink_slabs(slabs, 1.0, p)
    deep = slabs.iloc[0]
    assert deep.depth_km > 1000.0
    lon_at_670 = np.degrees(sum(50.0 / (6371.0 - z) for z in np.arange(0.0, 700.0, 50.0)))
    assert deep.lon == pytest.approx(lon_at_670, rel=1e-6)       # no horizontal motion below 670 km


def test_vertical_slabs_ignore_the_dip_keys():
    p = mantle.MantleParams()
    empty = mantle.core.pd.DataFrame(columns=mantle.SLAB_COLUMNS)
    plain = {k: v for k, v in _dip_samples().items() if k in ('lon', 'lat', 'convergence_km_per_myr')}
    a = mantle.sink_slabs(mantle.seed_slabs(empty, _dip_samples(), 1.0, p), 1.0, p)
    b = mantle.sink_slabs(mantle.seed_slabs(empty, plain, 1.0, p), 1.0, p)
    assert list(a.columns) == list(b.columns) == mantle.SLAB_COLUMNS
    mantle.core.pd.testing.assert_frame_equal(a, b)


def test_resting_slab_warms_at_the_cmb():
    p = mantle.MantleParams(slab_at_cmb='rest', slab_cmb_tau_myr=20.0, slab_thermal_tau_myr=1e9)
    slabs = _ns_slab(0.0, depth_km=p.cmb_depth_km)
    for _ in range(20):
        slabs = mantle.sink_slabs(slabs, 1.0, p)
    np.testing.assert_allclose(slabs['weight'].astype(float), np.exp(-1.0))


def test_warm_layer_above_the_pile_decays_upward():
    lons, lats = mantle.mantle_grid(5.0)
    h = np.zeros((len(lats), len(lons)))
    h[18, 36] = 400.0                                             # one pile column, top at 2490 km
    on = mantle.MantleParams(llsvp='pile', pile_halo_km=300.0, pile_halo_dVs=-1.0, pile_halo_h_km=300.0)
    off = mantle.MantleParams(llsvp='pile')
    empty = mantle.core.pd.DataFrame(columns=mantle.SLAB_COLUMNS)
    d = (mantle.dvs_field(empty, h, on).dVs - mantle.dvs_field(empty, h, off).dVs)
    col = (d.isel(lat=18, lon=36) - d.isel(lat=0, lon=0)).sel(depth=[1950.0, 2250.0, 2450.0]).values
    np.testing.assert_allclose(col, -np.exp(-np.array([540.0, 240.0, 40.0]) / 300.0), rtol=1e-6)


def test_rotate_samples_keeps_directions_and_speeds():
    from reconstructions.run_mantle import rotate_samples, rotation_matrix
    s = {**_dip_samples(45.0), 'trench_east_km_per_myr': np.array([10.0]), 'trench_north_km_per_myr': np.array([5.0])}
    spin = rotate_samples(s, rotation_matrix(90.0, 0.0, 10.0))  # about the spin axis: only longitude changes
    assert spin['lon'][0] == pytest.approx(10.0) and spin['lat'][0] == pytest.approx(0.0, abs=1e-9)
    assert spin['downdip_azimuth_deg'][0] == pytest.approx(45.0)
    tilt = rotate_samples(s, rotation_matrix(0.0, 0.0, 30.0))   # about (0 N, 0 E): the point is on the axis
    assert tilt['downdip_azimuth_deg'][0] == pytest.approx(15.0)  # directions turn 30 deg anticlockwise
    assert np.hypot(tilt['trench_east_km_per_myr'], tilt['trench_north_km_per_myr'])[0] == pytest.approx(np.hypot(10, 5))


def test_shallow_dip_near_the_trench_then_the_main_dip():
    p = mantle.MantleParams(slab_dip_deg=45.0, slab_dip_shallow_deg=25.0, slab_dip_shallow_km=100.0)
    slabs = mantle.seed_slabs(mantle.core.pd.DataFrame(columns=mantle.SLAB_COLUMNS), _dip_samples(), 1.0, p)
    for _ in range(6):                                            # 0-100 km at 25 deg, 100-300 km at 45 deg
        slabs = mantle.sink_slabs(slabs, 1.0, p)
    expected = 100.0 / np.tan(np.radians(25.0)) + 200.0
    km = np.radians(slabs.iloc[0].lon) * 6371.0
    assert expected < km < expected * 6371.0 / (6371.0 - 300.0)
    cols = mantle.initial_slab_columns(_dip_samples(), 1.0, p, depth_km=300.0)
    deepest = cols.iloc[cols.depth_km.astype(float).argmax()]
    assert np.radians(deepest.lon) * 6371.0 == pytest.approx(expected, rel=0.06)   # same shape at the start
