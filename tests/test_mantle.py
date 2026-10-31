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
