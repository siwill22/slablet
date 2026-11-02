"""Tests for slablet's mantle model (the core: no pygplates).

    conda run -n pygmt17 python -m pytest tests

tests/data/mantle_golden.npz was made with the code as it was before it moved out of
PlateDriver's self_driving_plates.py (update_slab_tracers etc.), on the small problem built
in _golden_run below; the code must reproduce it exactly.  Its trench samples, from
PlateDriver's adapter (self_driving_plates.trench_samples on three trenches and a ridge),
are saved in tests/data/golden_samples.npz, so this test needs no PlateDriver.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

import slablet

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
GOLDEN = os.path.join(DATA, 'mantle_golden.npz')
FORBIDDEN = ('self_driving_plates', 'run_SDP', 'plate_driver', 'reconstructions',
             'pygplates', 'geopandas', 'shapely', 'gprm', 'plate_model_manager')


def test_core_imports_no_plate_model_code():
    code = ('import sys, slablet; '
            'print(",".join(m for m in {!r} if m in sys.modules))'.format(FORBIDDEN))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '', 'slablet imported: ' + out.stdout.strip()


def _golden_samples(step):
    with np.load(os.path.join(DATA, 'golden_samples.npz')) as f:
        return {k: f['s{}_{}'.format(step, k)] for k in ('lon', 'lat', 'convergence_km_per_myr')}


def _golden_run():
    """Initial slab columns, 10 steps of 2 Myr, then the dVs field, from the saved samples."""
    p = slablet.MantleParams(mantle_grid_spacing_deg=10.0, mantle_depth_interval_km=200.0,
                             slab_thermal_tau_myr=4.0, tomo_smooth_horiz_deg=5.0,
                             tomo_smooth_depth_km=150.0, initial_slab_depth_km=700.0)
    slabs = slablet.initial_slab_columns(_golden_samples(0), 2.0, p)
    n_init = len(slabs)
    cmb = slablet.new_cmb_heat(p)
    for step in range(1, 11):
        slabs = slablet.seed_slabs(slablet.sink_slabs(slabs, 2.0, p), _golden_samples(step), 2.0, p)
        cmb = slablet.update_cmb_heat(cmb, slabs, 2.0, p)
    ds = slablet.dvs_field(slabs, cmb, p)
    mid_lon, mid_lat = np.array([175.0, -40.0]), np.array([-20.0, 66.0])
    side = slablet.lonlat_to_xyz(np.array([178.0, -42.0]), np.array([-19.0, 68.0])) \
        - slablet.lonlat_to_xyz(mid_lon, mid_lat)
    z = slablet.slab_depth_near(mid_lon, mid_lat, side, slabs.lon.values, slabs.lat.values, slabs.depth_km.values)
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
    p = slablet.MantleParams()
    samples = {'lon': np.array([10.0, 20.0]), 'lat': np.array([0.0, 5.0]),
               'convergence_km_per_myr': np.array([50.0, 4.0])}
    empty = slablet.core.pd.DataFrame(columns=slablet.SLAB_COLUMNS)
    one = slablet.seed_slabs(empty, samples, 2.0, p)
    two = slablet.seed_slabs(slablet.seed_slabs(empty, samples, 1.0, p), samples, 1.0, p)
    assert one['weight'].sum() == pytest.approx(two['weight'].sum())


def test_params_read_config_keys_and_ignore_others():
    p = slablet.MantleParams.from_config({'slab_thermal_tau_myr': 4, 'slab_pull': 6, 'cmb_sigma_deg': 7.0})
    assert p.slab_thermal_tau_myr == 4 and p.cmb_sigma_deg == 7.0
    assert p.slab_max_sink_rate == slablet.MantleParams().slab_max_sink_rate


def _ns_slab(lat0, depth_km=1050.0, age=0.0):
    """A north-south slab 20° long, one slablet per great-circle degree, centred on lat0."""
    lat = np.arange(lat0 - 10, lat0 + 10, 1.0) + 0.5
    n = len(lat)
    return slablet.core.pd.DataFrame({'lon': np.full(n, 32.5), 'lat': lat, 'depth_km': depth_km, 'age': age,
                                     'active': True, 'sink_rate_km_per_myr': 50.0, 'weight': 1.0,
                                     'attached': True})


def _integrated_anomaly(p, lat0):
    ds = slablet.dvs_field(_ns_slab(lat0), None, p)
    f = ds.dVs.sel(depth=1050).values
    f = f - f.min()                                   # undo the zero-mean shift
    return (f * np.cos(np.radians(ds.lat.values))[:, None]).sum()


def test_area_normalised_binning_counts_a_slab_the_same_at_every_latitude():
    p_off = slablet.MantleParams(slab_thermal_tau_myr=1e9)
    p_on = slablet.MantleParams(slab_thermal_tau_myr=1e9, mantle_bin_area_normalise=True)
    ratio_off = _integrated_anomaly(p_off, 60.0) / _integrated_anomaly(p_off, 0.0)
    ratio_on = _integrated_anomaly(p_on, 60.0) / _integrated_anomaly(p_on, 0.0)
    assert ratio_off < 0.6                            # the old binning halves a slab at 60°
    assert ratio_on == pytest.approx(1.0, abs=0.05)


def test_slabs_rest_at_the_cmb_until_their_anomaly_fades():
    p = slablet.MantleParams(slab_at_cmb='rest', slab_thermal_tau_myr=10.0)
    slabs = _ns_slab(0.0, depth_km=2880.0)
    out = slablet.sink_slabs(slabs, 1.0, p)
    assert len(out) == len(slabs) and (out['depth_km'] == p.cmb_depth_km).all()
    old = slablet.sink_slabs(_ns_slab(0.0, depth_km=2880.0, age=100.0), 1.0, p)   # exp(-101/10) < 1e-3
    assert len(old) == 0
    removed = slablet.sink_slabs(slabs, 1.0, slablet.MantleParams())
    assert len(removed) == 0


def test_cmb_suppression_can_follow_the_thermal_decay():
    young, aged = _ns_slab(0.0, depth_km=2850.0), _ns_slab(0.0, depth_km=2850.0, age=400.0)
    heat = np.full((36, 72), 1.0)
    for flag, expect_equal in ((False, True), (True, False)):
        p = slablet.MantleParams(cmb_suppression_thermal_decay=flag)
        a = slablet.update_cmb_heat(heat, young, 1.0, p).min()
        b = slablet.update_cmb_heat(heat, aged, 1.0, p).min()
        assert (a == b) == expect_equal
        if flag:
            assert b > a                              # the aged slab suppresses less


def _cell_area(p):
    return slablet.pile._grid(p)[2][:, None]


def test_pile_conserves_volume_and_moves_away_from_slabs():
    p = slablet.MantleParams(llsvp='pile', slab_at_cmb='rest')
    h0 = slablet.new_llsvp_state(p)
    slabs = _ns_slab(0.0, depth_km=p.cmb_depth_km)                 # a slab lying on the CMB at 32.5°E
    slabs['weight'] = 100.0                                         # about 100 Myr of subduction
    h = h0
    for _ in range(50):
        h = slablet.update_llsvp(h, slabs, 1.0, p)
    area = _cell_area(p)
    assert (h * area).sum() == pytest.approx((h0 * area).sum(), rel=1e-9)
    assert (h >= 0).all() and np.isfinite(h).all()
    lons, lats = slablet.mantle_grid(p.mantle_grid_spacing_deg)
    under = h[np.abs(lats) < 5][:, np.abs(lons - 32.5) < 3].mean()
    far = h[np.abs(lats) < 5][:, np.abs(((lons - 32.5 + 180) % 360) - 180) > 90].mean()
    assert under < 0.5 * p.pile_mean_thickness_km < far


def test_pile_spreads_back_to_a_uniform_layer_without_slabs():
    p = slablet.MantleParams(llsvp='pile')
    h = slablet.new_llsvp_state(p)
    h[10:20, 10:20] *= 4.0
    empty = slablet.core.pd.DataFrame(columns=slablet.SLAB_COLUMNS)
    spread0 = h.std()
    for _ in range(200):
        h = slablet.update_llsvp(h, empty, 1.0, p)
    assert h.std() < 0.3 * spread0


def test_pile_layer_fractions():
    depths = np.array([2650.0, 2750.0, 2850.0])
    f = slablet.pile.pile_layer_fractions(np.array([[150.0]]), depths, 100.0, 2890.0)[:, 0, 0]
    np.testing.assert_allclose(f, [0.0, 0.6, 1.0])     # pile from 2740 km to the CMB at 2890 km


@pytest.mark.parametrize('l, pattern', [(2, lambda lon, lat: 1.5 * np.sin(lat) ** 2 - 0.5),
                                        (3, lambda lon, lat: np.cos(lat) ** 3 * np.cos(3 * lon))])
def test_flow_potential_solves_poisson_on_the_sphere(l, pattern):
    p = slablet.MantleParams(pile_drive='flow')
    lons, lats = slablet.mantle_grid(p.mantle_grid_spacing_deg)
    lon, lat = np.meshgrid(np.radians(lons), np.radians(lats))
    load = pattern(lon, lat)
    psi = slablet.pile.flow_potential(load, p)
    np.testing.assert_allclose(psi, load / (l * (l + 1)), atol=0.01 * np.abs(load).max() / (l * (l + 1)))


def test_flow_pile_collects_on_the_far_side_from_a_slab():
    """A slab on the CMB at 32.5°E: the local pile only thins under it; the flow pile also
    piles up on the opposite side (147.5°W), more than a quarter of the way round (122.5°E)."""
    far = {}
    for drive in ('local', 'flow'):
        p = slablet.MantleParams(llsvp='pile', slab_at_cmb='rest', pile_drive=drive, pile_slab_head_km=3000.0)
        h0 = slablet.new_llsvp_state(p)
        slabs = _ns_slab(0.0, depth_km=p.cmb_depth_km)
        slabs['weight'] = 1000.0                                    # a long-lived slab graveyard
        h = h0
        for _ in range(100):
            h = slablet.update_llsvp(h, slabs, 1.0, p)
        area = _cell_area(p)
        assert (h * area).sum() == pytest.approx((h0 * area).sum(), rel=1e-9)
        lons, lats = slablet.mantle_grid(p.mantle_grid_spacing_deg)
        eq = h[np.abs(lats) < 5].mean(axis=0)
        far[drive] = eq[np.argmin(np.abs(lons + 147.5))] / eq[np.argmin(np.abs(lons - 122.5))]
    assert far['local'] == pytest.approx(1.0, abs=0.05)
    assert far['flow'] > 1.5


def _dip_samples(azimuth=90.0):
    return {'lon': np.array([0.0]), 'lat': np.array([0.0]), 'convergence_km_per_myr': np.array([50.0]),
            'downdip_azimuth_deg': np.array([azimuth]), 'trench_east_km_per_myr': np.array([0.0]),
            'trench_north_km_per_myr': np.array([0.0])}


def test_dipping_slab_moves_towards_the_overriding_plate_in_the_upper_mantle_only():
    p = slablet.MantleParams(slab_dip_deg=45.0)                    # sinks 50 km/Myr in the upper mantle
    slabs = slablet.seed_slabs(slablet.core.pd.DataFrame(columns=slablet.SLAB_COLUMNS), _dip_samples(), 1.0, p)
    for _ in range(6):                                            # to 300 km: 300 km east (45 deg)
        slabs = slablet.sink_slabs(slabs, 1.0, p)
    first = slabs.iloc[0]
    assert first.depth_km == pytest.approx(300.0)
    assert abs(first.lat) < 1e-9
    km = np.radians(first.lon) * 6371.0
    assert 300.0 < km < 300.0 * 6371.0 / (6371.0 - 300.0)        # moved at depth: a bit more than 300 km of surface
    for _ in range(40):
        slabs = slablet.sink_slabs(slabs, 1.0, p)
    deep = slabs.iloc[0]
    assert deep.depth_km > 1000.0
    lon_at_670 = np.degrees(sum(50.0 / (6371.0 - z) for z in np.arange(0.0, 700.0, 50.0)))
    assert deep.lon == pytest.approx(lon_at_670, rel=1e-6)       # no horizontal motion below 670 km


def test_vertical_slabs_ignore_the_dip_keys():
    p = slablet.MantleParams()
    empty = slablet.core.pd.DataFrame(columns=slablet.SLAB_COLUMNS)
    plain = {k: v for k, v in _dip_samples().items() if k in ('lon', 'lat', 'convergence_km_per_myr')}
    a = slablet.sink_slabs(slablet.seed_slabs(empty, _dip_samples(), 1.0, p), 1.0, p)
    b = slablet.sink_slabs(slablet.seed_slabs(empty, plain, 1.0, p), 1.0, p)
    assert list(a.columns) == list(b.columns) == slablet.SLAB_COLUMNS
    slablet.core.pd.testing.assert_frame_equal(a, b)


def test_resting_slab_warms_at_the_cmb():
    p = slablet.MantleParams(slab_at_cmb='rest', slab_cmb_tau_myr=20.0, slab_thermal_tau_myr=1e9)
    slabs = _ns_slab(0.0, depth_km=p.cmb_depth_km)
    for _ in range(20):
        slabs = slablet.sink_slabs(slabs, 1.0, p)
    np.testing.assert_allclose(slabs['weight'].astype(float), np.exp(-1.0))


def test_warm_layer_above_the_pile_decays_upward():
    lons, lats = slablet.mantle_grid(5.0)
    h = np.zeros((len(lats), len(lons)))
    h[18, 36] = 400.0                                             # one pile column, top at 2490 km
    on = slablet.MantleParams(llsvp='pile', pile_halo_km=300.0, pile_halo_dVs=-1.0, pile_halo_h_km=300.0)
    off = slablet.MantleParams(llsvp='pile')
    empty = slablet.core.pd.DataFrame(columns=slablet.SLAB_COLUMNS)
    d = (slablet.dvs_field(empty, h, on).dVs - slablet.dvs_field(empty, h, off).dVs)
    col = (d.isel(lat=18, lon=36) - d.isel(lat=0, lon=0)).sel(depth=[1950.0, 2250.0, 2450.0]).values
    np.testing.assert_allclose(col, -np.exp(-np.array([540.0, 240.0, 40.0]) / 300.0), rtol=1e-6)


def test_rotate_samples_keeps_directions_and_speeds():
    from slablet.gplates.run_mantle import rotate_samples, rotation_matrix
    s = {**_dip_samples(45.0), 'trench_east_km_per_myr': np.array([10.0]), 'trench_north_km_per_myr': np.array([5.0])}
    spin = rotate_samples(s, rotation_matrix(90.0, 0.0, 10.0))  # about the spin axis: only longitude changes
    assert spin['lon'][0] == pytest.approx(10.0) and spin['lat'][0] == pytest.approx(0.0, abs=1e-9)
    assert spin['downdip_azimuth_deg'][0] == pytest.approx(45.0)
    tilt = rotate_samples(s, rotation_matrix(0.0, 0.0, 30.0))   # about (0 N, 0 E): the point is on the axis
    assert tilt['downdip_azimuth_deg'][0] == pytest.approx(15.0)  # directions turn 30 deg anticlockwise
    assert np.hypot(tilt['trench_east_km_per_myr'], tilt['trench_north_km_per_myr'])[0] == pytest.approx(np.hypot(10, 5))


def test_shallow_dip_near_the_trench_then_the_main_dip():
    p = slablet.MantleParams(slab_dip_deg=45.0, slab_dip_shallow_deg=25.0, slab_dip_shallow_km=100.0)
    slabs = slablet.seed_slabs(slablet.core.pd.DataFrame(columns=slablet.SLAB_COLUMNS), _dip_samples(), 1.0, p)
    for _ in range(6):                                            # 0-100 km at 25 deg, 100-300 km at 45 deg
        slabs = slablet.sink_slabs(slabs, 1.0, p)
    expected = 100.0 / np.tan(np.radians(25.0)) + 200.0
    km = np.radians(slabs.iloc[0].lon) * 6371.0
    assert expected < km < expected * 6371.0 / (6371.0 - 300.0)
    cols = slablet.initial_slab_columns(_dip_samples(), 1.0, p, depth_km=300.0)
    deepest = cols.iloc[cols.depth_km.astype(float).argmax()]
    assert np.radians(deepest.lon) * 6371.0 == pytest.approx(expected, rel=0.06)   # same shape at the start


@pytest.mark.parametrize('l, pattern', [(2, lambda lon, lat: 1.5 * np.sin(lat) ** 2 - 0.5),
                                        (3, lambda lon, lat: np.cos(lat) ** 3 * np.cos(3 * lon))])
def test_screened_flow_potential_damps_each_degree_by_l_l1_plus_screen(l, pattern):
    p = slablet.MantleParams(pile_drive='flow', pile_flow_screen_deg=20.0)
    lons, lats = slablet.mantle_grid(p.mantle_grid_spacing_deg)
    lon, lat = np.meshgrid(np.radians(lons), np.radians(lats))
    load = pattern(lon, lat)
    k = l * (l + 1) + 1.0 / np.radians(20.0) ** 2
    np.testing.assert_allclose(slablet.pile.flow_potential(load, p), load / k, atol=0.01 * np.abs(load).max() / k)


def test_slabs_on_the_cmb_spread_away_from_the_slab_pile_and_shallower_ones_do_not():
    p = slablet.MantleParams(slab_at_cmb='rest', slab_thermal_tau_myr=1e9, slab_cmb_spread_km_per_myr=20.0)
    deep = _ns_slab(0.0, depth_km=p.cmb_depth_km)
    deep['lon'] = 32.5 + np.tile([-1.5, 1.5], len(deep) // 2)       # two lines 3° apart, either side of 32.5°E
    mid = _ns_slab(0.0, depth_km=1500.0)
    out = slablet.sink_slabs(slablet.core.pd.concat([deep, mid], ignore_index=True), 5.0, p)
    d, m = out[out.depth_km == p.cmb_depth_km], out[out.depth_km < 2000]
    assert (np.abs(d.lon.values - 32.5) > 1.5 + 0.2).mean() > 0.8   # moved apart, away from the slab
    assert (m.lon.values == 32.5).all()                              # the mid-mantle slab is untouched
    still = slablet.sink_slabs(deep, 5.0, slablet.MantleParams(slab_at_cmb='rest', slab_thermal_tau_myr=1e9))
    np.testing.assert_array_equal(still.lon.values, deep.lon.values)  # off by default


def test_slab_width_profile_is_narrow_above_660_and_widens_to_the_cmb():
    p = slablet.MantleParams(slab_sigma_deg=0.75, slab_sigma_lower_deg=2.0, slab_sigma_cmb_deg=3.5)
    sig = slablet.core.slab_sigma_profile(np.array([150.0, 650.0, 700.0, 1780.0, 2890.0]), p)
    np.testing.assert_allclose(sig, [0.75, 0.75, 2.0, 2.75, 3.5], atol=0.03)


def test_block_mean_and_upsample_round_trip_a_smooth_field():
    lons_f, lats_f = slablet.mantle_grid(1.0)
    lons_c, lats_c = slablet.mantle_grid(5.0)
    lon, lat = np.meshgrid(np.radians(lons_f), np.radians(lats_f))
    f = np.cos(lat) ** 2 * np.cos(2 * lon)                          # degree 2
    c = slablet.core._block_mean(f, lats_f, 5)
    assert c.shape == (len(lats_c), len(lons_c))
    back = slablet.core._upsample(c, lats_c, lons_c, lats_f, lons_f)
    assert np.abs(back - f)[np.abs(lats_f) < 80].max() < 0.02


def test_fine_field_keeps_the_slab_and_narrows_it_in_the_upper_mantle():
    coarse = slablet.MantleParams(slab_thermal_tau_myr=1e9, mantle_bin_area_normalise=True)
    fine = slablet.MantleParams(slab_thermal_tau_myr=1e9, mantle_bin_area_normalise=True, mantle_field_spacing_deg=1.0,
                               slab_sigma_deg=0.75, slab_sigma_lower_deg=2.0, slab_sigma_cmb_deg=3.5,
                               tomo_smooth_horiz_deg=5.0, dvs_save_native=True)
    slab = _ns_slab(0.0, depth_km=350.0)
    widths = {}
    for name, p in (('coarse', coarse), ('fine', fine)):
        ds = slablet.dvs_field(slab, None, p)
        f = ds['dVs_native' if name == 'fine' else 'dVs'].sel(depth=350).values
        f = f - np.median(f)                                         # undo the zero-mean shift
        eq = f[np.abs(ds.lat.values) < 5].mean(axis=0)
        widths[name] = (eq > 0.5 * eq.max()).sum() * (ds.lon.values[1] - ds.lon.values[0])   # degrees above half max
        if name == 'fine':
            assert 'dVs' in ds and ds.dVs.shape == ds.dVs_native.shape == (29, 180, 360)
    assert widths['fine'] < 0.5 * widths['coarse']


def test_fine_field_has_the_same_integrated_slab_anomaly_as_the_model_grid():
    base = dict(slab_thermal_tau_myr=1e9, mantle_bin_area_normalise=True, slab_sigma_deg=3.0)
    out = {}
    for name, extra in (('coarse', {}), ('fine', {'mantle_field_spacing_deg': 1.0})):
        ds = slablet.dvs_field(_ns_slab(0.0, depth_km=1050.0), None, slablet.MantleParams(**base, **extra))
        f = ds.dVs.sel(depth=1050).values.astype(float)
        f = f - np.median(f)
        cell = np.radians(ds.lon.values[1] - ds.lon.values[0]) ** 2 * np.cos(np.radians(ds.lat.values))[:, None]
        out[name] = (f * cell).sum()
    assert out['fine'] == pytest.approx(out['coarse'], rel=0.05)


def _slab_line(step_deg, depth_km=1050.0):
    """A north-south slab 20° long at 32.5° E, one slablet every step_deg."""
    lat = np.arange(-10, 10, step_deg) + step_deg / 2
    n = len(lat)
    return slablet.core.pd.DataFrame({'lon': np.full(n, 32.5), 'lat': lat, 'depth_km': depth_km, 'age': 0.0,
                                     'active': True, 'sink_rate_km_per_myr': 50.0, 'weight': 1.0,
                                     'attached': True})


def _volume_anomaly(ds, var='dVs'):
    f = ds[var].values.astype(float)
    f = f - np.median(f, axis=(1, 2), keepdims=True)                 # undo the zero-mean shift per layer
    cell = np.radians(ds.lon.values[1] - ds.lon.values[0]) ** 2 * np.cos(np.radians(ds.lat.values))[:, None]
    return (f * cell).sum() * (ds.depth.values[1] - ds.depth.values[0])


def test_resolution_independent_slab_strength_does_not_depend_on_the_grids():
    base = dict(slab_thermal_tau_myr=1e9, mantle_bin_area_normalise=True, slab_sigma_deg=3.0,
                resolution_independent=True)
    ref = _volume_anomaly(slablet.dvs_field(_ns_slab(0.0), None, slablet.MantleParams(**base)))
    for extra in ({'mantle_grid_spacing_deg': 2.5}, {'mantle_depth_interval_km': 50.0},
                  {'mantle_field_spacing_deg': 0.5}, {'mantle_grid_spacing_deg': 2.5, 'mantle_field_spacing_deg': 0.5}):
        got = _volume_anomaly(slablet.dvs_field(_ns_slab(0.0), None, slablet.MantleParams(**base, **extra)))
        assert got == pytest.approx(ref, rel=0.05), extra
    # without the option, 50 km layers halve the slab
    old = slablet.MantleParams(**{**base, 'resolution_independent': False, 'mantle_depth_interval_km': 50.0})
    assert _volume_anomaly(slablet.dvs_field(_ns_slab(0.0), None, old)) == pytest.approx(0.5 * ref, rel=0.05)


def test_resolution_independent_seeding_weights_per_length_of_trench():
    def total(step, flag):
        lat = np.arange(-10, 10, step) + step / 2
        s = {'lon': np.full(len(lat), 30.0), 'lat': lat, 'convergence_km_per_myr': np.full(len(lat), 50.0)}
        p = slablet.MantleParams(slab_sample_deg=step, resolution_independent=flag)
        empty = slablet.core.pd.DataFrame(columns=slablet.SLAB_COLUMNS)
        return slablet.seed_slabs(empty, s, 1.0, p)['weight'].sum(), \
            slablet.initial_slab_columns(s, 1.0, p, depth_km=700.0)['weight'].sum()
    assert total(0.5, True) == pytest.approx(total(1.0, True))
    assert total(0.5, False)[0] == pytest.approx(2 * total(1.0, False)[0])


def test_resolution_independent_cmb_suppression_does_not_depend_on_the_grid():
    removed = {}
    for spacing in (5.0, 2.5):
        p = slablet.MantleParams(mantle_grid_spacing_deg=spacing, resolution_independent=True, cmb_heating_rate=0.0,
                                cmb_sigma_deg=0.0, cmb_slab_suppression=0.01, mantle_bin_area_normalise=True)
        lons, lats = slablet.mantle_grid(spacing)
        heat = np.ones((len(lats), len(lons)))
        after = slablet.update_cmb_heat(heat, _ns_slab(0.0, depth_km=2850.0), 1.0, p)
        removed[spacing] = ((heat - after) * np.radians(spacing) ** 2 * np.cos(np.radians(lats))[:, None]).sum()
    assert removed[2.5] == pytest.approx(removed[5.0], rel=1e-6)


def test_fixed_lower_mantle_rate_does_not_depend_on_convergence():
    p = slablet.MantleParams(slab_lower_rate_km_per_myr=12.0)
    s = _ns_slab(0.0, depth_km=1000.0).iloc[:2].copy()
    s['sink_rate_km_per_myr'] = [10.0, 100.0]
    out = slablet.sink_slabs(s, 1.0, p)
    np.testing.assert_allclose(out['depth_km'].values, [1012.0, 1012.0])
    old = slablet.sink_slabs(s, 1.0, slablet.MantleParams(slab_slowing_factor=4.0))
    np.testing.assert_allclose(old['depth_km'].values, [1002.5, 1025.0])


def test_seafloor_age_sets_slab_weight_and_thermal_time():
    p = slablet.MantleParams(slab_age_ref_myr=50.0)
    samples = {'lon': np.array([10.0, 20.0, 30.0]), 'lat': np.zeros(3), 'convergence_km_per_myr': np.full(3, 50.0),
               'seafloor_age_myr': np.array([12.5, 200.0, np.nan])}
    new = slablet.seed_slabs(slablet.core.pd.DataFrame(columns=slablet.SLAB_COLUMNS), samples, 1.0, p)
    np.testing.assert_allclose(new['weight'].values, [0.5, np.sqrt(2.0), 1.0])     # 12.5/50, 100/50 (capped), default
    new['age'] = 100.0
    np.testing.assert_allclose(slablet.core.thermal_tau(new, p), [50.0, 400.0, 200.0])
    off = slablet.seed_slabs(slablet.core.pd.DataFrame(columns=slablet.SLAB_COLUMNS), samples, 1.0, slablet.MantleParams())
    assert 'plate_age' not in off and (off['weight'] == 1.0).all()


def test_lower_mantle_slabs_move_with_a_fraction_of_the_plate_velocity():
    lats, lons = np.arange(-89.0, 90, 2.0), np.arange(-179.0, 180, 2.0)
    flow = {'lats': lats, 'lons': lons, 'vel_e': np.full((len(lats), len(lons)), 100.0),
            'vel_n': np.zeros((len(lats), len(lons)))}
    s = _ns_slab(0.0, depth_km=1000.0).iloc[[10]].copy()
    s['lat'] = 0.0
    p = slablet.MantleParams(slab_flow_top=0.5, slab_flow_bottom=0.5, slab_slowing_factor=1e9)
    out = slablet.sink_slabs(s, 1.0, p, flow)
    assert out['lon'].iloc[0] - 32.5 == pytest.approx(np.degrees(50.0 / 6371.0), rel=1e-6)
    upper = slablet.sink_slabs(s.assign(depth_km=300.0), 1.0, p, flow)         # upper-mantle slabs do not drift
    assert upper['lon'].iloc[0] == 32.5


def test_fixed_lower_mantle_rate_can_scale_with_the_slab_feed():
    p = slablet.MantleParams(slab_lower_rate_km_per_myr=12.0, slab_lower_rate_exponent=0.5)
    s = _ns_slab(0.0, depth_km=1000.0).iloc[:2].copy()
    s['sink_rate_km_per_myr'] = [25.0, 100.0]
    out = slablet.sink_slabs(s, 1.0, p)
    np.testing.assert_allclose(out['depth_km'].values, [1012.0, 1024.0])


def test_plume_sites_are_active_where_the_hot_layer_is_thick():
    p = slablet.MantleParams(llsvp='pile', plume_dVs=-1.5, plume_source='sites', plume_spacing_deg=10.0,
                            plume_min_fraction=0.2)
    lons, lats = slablet.mantle_grid(p.mantle_grid_spacing_deg)
    h = np.where(np.abs(lats)[:, None] < 30, 300.0, 0.0) * np.ones(len(lons))   # a band of pile
    lon, lat, strength = slablet.core.plume_sources(h, p)
    assert len(slablet.core.plume_sites(10.0)[0]) == 413                 # 41253 square degrees / 100
    assert len(lon) > 20 and np.all(np.abs(lat) < 33) and np.all(strength > 0.2)


def test_slab_field_can_saturate():
    base = dict(slab_thermal_tau_myr=1e9, mantle_bin_area_normalise=True, slab_sigma_deg=0.0, slab_sigma_depth_km=0.0)
    slab = slablet.core.pd.concat([_ns_slab(0.0)] * 20, ignore_index=True)        # 20 slablets per cell
    raw = slablet.dvs_field(slab, None, slablet.MantleParams(**base)).dVs.sel(depth=1050).values
    cap = slablet.dvs_field(slab, None, slablet.MantleParams(**base, slab_dVs_max=4.0)).dVs.sel(depth=1050).values
    assert raw.max() - np.median(raw) > 8.0
    assert cap.max() - np.median(cap) < 4.0 + 1e-6
    upper_only = slablet.dvs_field(slab, None, slablet.MantleParams(**base, slab_dVs_max=4.0, slab_dVs_max_depth_km=670.0))
    np.testing.assert_allclose(upper_only.dVs.sel(depth=1050).values, raw)          # 1050 km is below the cap


def test_mantle_wedge_is_slow_on_the_overriding_side():
    p = slablet.MantleParams(slab_thermal_tau_myr=1e9, mantle_bin_area_normalise=True, slab_sigma_deg=1.0,
                            slab_sigma_depth_km=0.0, wedge_dVs=-2.0, wedge_offset_km=500.0, wedge_sigma_deg=1.0)
    slab = _ns_slab(0.0, depth_km=250.0)
    slab['dip_azimuth_deg'] = 90.0                               # overriding plate to the east
    f = slablet.dvs_field(slab, None, p).dVs.sel(depth=250).values
    eq = f[np.abs(slablet.mantle_grid(5.0)[1]) < 5].mean(axis=0)
    lons = slablet.mantle_grid(5.0)[0]
    east = lons[np.argmin(eq)]
    assert eq.min() < eq.max() and 32.5 < east < 42.5                # slowest about 4.5 deg east of the slab
    none = slablet.dvs_field(slab.drop(columns='dip_azimuth_deg'), None, p).dVs.sel(depth=250).values
    assert none.min() >= np.median(none) - 1e-9                       # no azimuth: no wedge
