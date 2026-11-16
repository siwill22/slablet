"""Tests for slablet.cooling: each law against its published form, and depth, heat flow and thickness against the
one temperature field they must all come from (decision 11 of docs/plans/2026-11-08-cross-repo-reconciliation.md)."""
import numpy as np
import pytest

from slablet import cooling

trapezoid = getattr(np, 'trapezoid', None) or np.trapz     # np.trapz is gone from numpy 2.4 on

T = np.array([0.5, 1, 5, 10, 19.9, 20, 30, 55, 56, 80, 100, 150, 200, 300.0])
ANALYTIC = ('half_space', 'gdh1', 'ps77', 'crosby2007')


def test_default_is_gdh1_and_unknown_names_are_refused():
    assert cooling.get().name == 'gdh1'
    with pytest.raises(ValueError):
        cooling.get('gdh2')


def test_gdh1_matches_stein_and_stein_closed_forms():
    # Stein & Stein (1992) give closed-form fits to their plate model; the series is within a few metres of them
    g = cooling.get('gdh1')
    ss = np.where(T < 20, 2600 + 365 * np.sqrt(T), 5651 - 2473 * np.exp(-0.0278 * T))
    assert np.max(np.abs(cooling.depth_m(g, T) - ss)) < 3.0
    q = np.where(T <= 55, 510 / np.sqrt(T), 48 + 96 * np.exp(-0.0278 * T))
    assert np.allclose(cooling.heat_flow_mw_m2(g, T), q, rtol=3e-3)


def test_half_space_limits():
    # GDH1's properties without a base give 2600 + 365 sqrt(t); PS77's young seafloor gives 2500 + 350 sqrt(t)
    assert np.allclose(cooling.depth_m(cooling.get('half_space'), T), 2600 + 365 * np.sqrt(T), atol=0.5 + 0.3 * np.sqrt(T))
    young = T < 25
    assert np.allclose(cooling.depth_m(cooling.get('ps77'), T[young]), 2500 + 350 * np.sqrt(T[young]), atol=0.5)


def test_crosby2007_matches_pybacktrack():
    # pyBacktrack's _age_to_depth_CROSBY_2007 (a port of Crosby's age2depth.c), at T, computed 2026-11-09
    ref = [2845.479, 2947.157, 3376.255, 3697.743, 4146.874, 4150.709, 4486.629, 5070.539, 5092.262, 5586.921,
           5461.811, 5688.719, 5738.534, 5760.232]
    assert np.allclose(cooling.depth_m(cooling.get('crosby2007'), T), ref, atol=0.01)


@pytest.mark.parametrize('name', ANALYTIC)
def test_depth_and_heat_flow_come_from_the_temperature_field(name):
    law = cooling.get(name)
    z = np.linspace(0.0, 600.0, 60001)
    for t in (0.3, 3.0, 30.0, 120.0):
        temp = cooling.temperature_c(law, t, z)
        tm = law.mantle_temperature_c
        d = law.ridge_depth_m + cooling._amplitude_m(law) / tm * trapezoid(tm - temp, z * 1e3)
        if law.crosby_bump:
            d = d - cooling._crosby_bump_m(t)
        assert abs(d - cooling.depth_m(law, t)) < 0.05
        q = 1e3 * law.conductivity_w_m_k * (temp[1] - temp[0]) / ((z[1] - z[0]) * 1e3)
        assert q == pytest.approx(float(cooling.heat_flow_mw_m2(law, t)), rel=1e-3)


@pytest.mark.parametrize('name', ANALYTIC)
def test_thickness_is_the_isotherm(name):
    law = cooling.get(name)
    for t in (2.0, 20.0, 90.0, 250.0):
        th = float(cooling.thickness_km(law, t))
        assert float(cooling.temperature_c(law, t, th)) == pytest.approx(0.9 * law.mantle_temperature_c, rel=2e-3)


def test_plate_series_meets_half_space_at_the_switch():
    # no step where the plate series takes over from the half-space form
    g = cooling.get('gdh1')
    a = g.plate_thickness_km * 1e3
    t0 = cooling._SWITCH * a ** 2 / (np.pi ** 2 * g.diffusivity_m2_s * cooling.SECONDS_PER_MYR)
    for f in (cooling.depth_m, cooling.heat_flow_mw_m2):
        lo, hi = f(g, t0 * (1 - 1e-9)), f(g, t0 * (1 + 1e-9))
        assert abs(hi - lo) < 1e-6 * abs(lo)


rhcw18 = pytest.mark.skipif(not __import__('os').path.isdir(cooling.data_dir()),
                            reason='RHCW18 output not in $SLABLET_DATA/CoolingModels')


@rhcw18
def test_rhcw18_reads_the_authors_output():
    law = cooling.get('rhcw18')
    dep = np.loadtxt(cooling.data_dir() + '/depth-1333-130-2500.dat')
    assert np.allclose(cooling.depth_m(law, dep[::97, 0]), dep[::97, 1])
    assert cooling.depth_m(law, 0.0) == pytest.approx(2500.0, abs=0.01)
    assert cooling.depth_m(law, 400.0) == cooling.depth_m(law, 500.0)        # held at its last age (300 Myr)
    th = cooling.thickness_km(law, np.array([5.0, 30.0, 80.0, 150.0]))
    assert np.all(np.diff(th) > 0) and 100 < th[-1] < 130
    assert float(cooling.temperature_c(law, 80.0, th[2])) == pytest.approx(0.9 * 1333, rel=2e-3)
