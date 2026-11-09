"""Tests for slablet.dyntopo and slablet.sphharm (dynamic topography from the mantle field).

The full solution is compared with HC by scripts/check_against_hc.py (it needs an HC build); these tests check
the pieces and the physical limits without HC.
"""
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

pytest.importorskip('pyshtools')

import slablet                                                    # noqa: E402
from slablet import dyntopo                                       # noqa: E402
from slablet.sphharm import Grid, degree_correlation, band_correlation  # noqa: E402

LATS = np.arange(-89.5, 90, 1.0)
LONS = np.arange(-179.5, 180, 1.0)


def _random_coeffs(lmax, seed=0):
    c = np.tril(np.random.default_rng(seed).normal(size=(2, lmax + 1, lmax + 1)))
    c[1, :, 0] = 0.0
    return c


def test_dyntopo_imports_no_plate_model_code():
    code = ('import sys, slablet.dyntopo; print(",".join(m for m in ("pygplates", "gprm", "shapely", '
            '"geopandas", "plate_model_manager") if m in sys.modules))')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ''


def test_analysis_inverts_synthesis_exactly_on_the_cell_centred_grid():
    g = Grid(LATS, LONS, 40)
    c = _random_coeffs(40)
    assert np.abs(g.analyse(g.synth(c)) - c).max() < 1e-12


def test_hessian_trace_is_the_surface_laplacian():
    g = Grid(LATS, LONS, 20)
    c = _random_coeffs(20, 1)
    htt, _, hpp = g.hessian(c)
    L = np.arange(21) * (np.arange(21) + 1.0)
    lap = g.synth(-c * L[None, :, None])
    assert np.abs(htt + hpp - lap).max() < 1e-9 * np.abs(lap).max()


def test_degree_correlation_of_a_field_with_itself_is_one():
    c = _random_coeffs(10, 2)
    assert np.allclose(degree_correlation(c, c)[1:], 1.0)
    assert band_correlation(c, -c, 1, 3) == pytest.approx(-1.0)


@pytest.mark.parametrize('l', [1, 2, 5, 12])
def test_propagator_matrix_has_the_stokes_and_laplace_exponents(l):
    ev = np.sort(np.linalg.eigvals(dyntopo._matrix(l, 7.0)).real)
    assert np.allclose(ev, np.sort([l + 1, l - 1, -l, -l - 2, l, -l - 1]), atol=1e-9)


def test_prem_density():
    rho = dyntopo.prem_density([100.0, 1000.0, 2891.0])
    assert rho[0] == pytest.approx(3372.5, abs=0.5)
    assert rho[1] == pytest.approx(4580.1, abs=0.5)
    assert rho[2] == pytest.approx(5566.0, abs=1.0)


@pytest.mark.parametrize('viscosity', ['uniform', 'two_layer', 'four_layer'])
def test_kernels_compensate_loads_at_the_surface_and_at_the_cmb(viscosity):
    """A load just below the surface is balanced by surface topography (kernel -1) and makes no geoid; a load just
    above the CMB is balanced by CMB topography and lifts nothing at the surface."""
    depth = np.concatenate([[2.0], np.arange(50, 2850, 100.0), [2889.0]])
    dp = dyntopo.DynTopoParams(viscosity=viscosity, lmax=8)
    K = dyntopo.FlowSolver(1 - depth[::-1] / dp.radius_km, dp).kernels()
    for l in range(1, 9):
        assert K['topography'][l, -1] == pytest.approx(-1.0, abs=0.01)
        assert abs(K['geoid'][l, -1]) < 0.01
        assert K['cmb'][l, 0] == pytest.approx(-1.0, abs=0.01)
        assert abs(K['topography'][l, 0]) < 0.01


def test_kernels_do_not_depend_on_the_viscosity_scale():
    depth = np.arange(50, 2890, 100.0)
    r = 1 - depth[::-1] / 6371.0
    k1 = dyntopo.FlowSolver(r, dyntopo.DynTopoParams(viscosity=[[0, 1e21], [660, 3e22]], lmax=6)).kernels()
    k2 = dyntopo.FlowSolver(r, dyntopo.DynTopoParams(viscosity=[[0, 1e23], [660, 3e24]], lmax=6)).kernels()
    for name in k1:
        assert np.allclose(k1[name][1:], k2[name][1:], rtol=1e-8, atol=1e-12)


def test_splitting_layers_for_growth_does_not_change_the_solution():
    depth = np.arange(50, 2890, 100.0)
    r = 1 - depth[::-1] / 6371.0
    a = dyntopo.FlowSolver(r, dyntopo.DynTopoParams(lmax=12, max_growth=1e4)).kernels()
    b = dyntopo.FlowSolver(r, dyntopo.DynTopoParams(lmax=12, max_growth=1e2)).kernels()
    for name in a:
        assert np.allclose(a[name][1:], b[name][1:], rtol=1e-8, atol=1e-10)


def _dataset(seed=3):
    """A mantle field: a dense blob in the mid mantle and a slow one above the CMB, plus a shallow one."""
    depth = np.arange(50, 2890, 100.0)
    lon2, lat2 = np.meshgrid(LONS, LATS)

    def blob(lon0, lat0, sigma):
        d = np.degrees(np.arccos(np.clip(np.sin(np.radians(lat2)) * np.sin(np.radians(lat0)) + np.cos(np.radians(lat2))
                                         * np.cos(np.radians(lat0)) * np.cos(np.radians(lon2 - lon0)), -1, 1)))
        return np.exp(-0.5 * (d / sigma) ** 2)

    f = np.zeros((len(depth), len(LATS), len(LONS)))
    f[12] += 1.0 * blob(140.0, 10.0, 15.0)
    f[-2] += -3.0 * blob(0.0, -10.0, 25.0)
    f[1] += 2.0 * blob(-60.0, 40.0, 10.0)                         # above the cutoff: ignored
    w = np.cos(np.radians(LATS))[:, None]
    f -= ((f * w).sum((1, 2)) / (w.sum() * len(LONS)))[:, None, None]
    return xr.Dataset({'dVs_native': (('depth', 'lat', 'lon'), f.astype(np.float32))},
                      coords={'depth': depth, 'lat': LATS, 'lon': LONS})


def test_compute_gives_a_low_over_a_dense_slab_and_ignores_shallow_anomalies():
    ds = _dataset()
    dp = dyntopo.DynTopoParams(lmax=16, volume=False)
    res = dyntopo.compute(ds, dp)
    topo = res['dyn_topo']
    assert float(topo.sel(lat=10.5, lon=140.5)) < 0                 # over the dense (fast) blob
    assert float(topo.sel(lat=-10.5, lon=0.5)) > 0                  # over the light (slow) blob near the CMB
    ds2 = ds.copy(deep=True)
    ds2['dVs_native'][1] = 0.0                                     # remove the shallow blob
    res2 = dyntopo.compute(ds2, dp)
    assert np.allclose(res['dyn_topo_coeffs'], res2['dyn_topo_coeffs'])


def test_water_loading_scales_topography_by_the_density_contrasts():
    ds = _dataset()
    w = dyntopo.compute(ds, dyntopo.DynTopoParams(lmax=8, volume=False, loading='water'))
    a = dyntopo.compute(ds, dyntopo.DynTopoParams(lmax=8, volume=False, loading='air'))
    assert np.allclose(w['dyn_topo_coeffs'], a['dyn_topo_coeffs'] * 3380.0 / (3380.0 - 1030.0))
    assert np.allclose(w['geoid_coeffs'], a['geoid_coeffs'])


def test_volume_velocities_are_consistent_with_incompressibility():
    """The radial velocity's horizontal pattern is zero mean on every layer, and the surface has no radial flow."""
    res = dyntopo.compute(_dataset(), dyntopo.DynTopoParams(lmax=12))
    w = np.cos(np.radians(LATS))[:, None]
    ur = res['u_r'].values.astype(float)
    assert np.abs((ur * w).sum((1, 2))).max() < 1e-6 * np.abs(ur).max() * w.sum() * len(LONS)
    assert np.abs(res['u_r'].isel(depth=0)).max() < np.abs(res['u_r']).max()


def test_source_components_sum_to_the_field_and_change_nothing():
    """dvs_save_components writes each source; they add up to dVs_native, and the field is unchanged."""
    p = dict(mantle_grid_spacing_deg=10.0, mantle_depth_interval_km=200.0, slab_thermal_tau_myr=4.0,
             tomo_smooth_horiz_deg=5.0, tomo_smooth_depth_km=150.0, initial_slab_depth_km=700.0,
             dvs_save_native=True, llsvp='pile', pile_halo_km=300.0, pile_halo_dVs=-1.0, pile_dVs=-3.0)
    samples = {'lon': np.array([140.0, 141.0, -70.0]), 'lat': np.array([30.0, 31.0, -20.0]),
               'convergence_km_per_myr': np.array([50.0, 50.0, 40.0])}
    out = []
    for comps in (False, True):
        mp = slablet.MantleParams(**p, dvs_save_components=comps)
        slabs = slablet.initial_slab_columns(samples, 2.0, mp)
        state = slablet.new_llsvp_state(mp)
        for _ in range(5):
            slabs = slablet.seed_slabs(slablet.sink_slabs(slabs, 2.0, mp), samples, 2.0, mp)
            state = slablet.update_llsvp(state, slabs, 2.0, mp)
        out.append(slablet.dvs_field(slabs, state, mp))
    plain, split = out
    assert np.array_equal(plain['dVs'].values, split['dVs'].values)
    assert np.array_equal(plain['dVs_native'].values, split['dVs_native'].values)
    names = [v for v in split.data_vars if v.startswith('dVs_native__')]
    assert {'dVs_native__slabs', 'dVs_native__pile', 'dVs_native__pile_halo'} <= set(names)
    total = sum(split[v].values.astype(float) for v in names)
    assert np.abs(total - split['dVs_native'].values).max() < 1e-5
    assert not any(v.startswith('dVs_native__') for v in plain.data_vars)
