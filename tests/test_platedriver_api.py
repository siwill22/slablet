"""The functions PlateDriver calls, called the way PlateDriver calls them.

PlateDriver (~/GIT/PlateDriver) uses an editable install of slablet, so a change of name,
signature or keyword here reaches it at its next run. These tests make such a change fail in
slablet's own suite instead (decision 7 of docs/plans/2026-11-08-cross-repo-reconciliation.md).
The calls copy run_SDP.py and self_driving_plates.py; when PlateDriver's use of slablet changes,
change this file in the same session.
"""
import numpy as np
import pandas as pd
import xarray as xr

import slablet


def _platedriver_params():
    """A PlateDriver params dict: mantle keys mixed with the simulator's own, which from_config ignores."""
    return {'slab_sink_rate_factor': 1.0, 'slab_thermal_tau_myr': 4.0, 'slab_tracers_enabled': True,
            'div_threshold_subduction': -1e-4, 'raster_spacing': 0.25, 'random_seed': 42}


def _samples():
    return {'lon': np.array([140.0, 141.0, 142.0]), 'lat': np.array([-5.0, -5.5, -6.0]),
            'convergence_km_per_myr': np.array([60.0, 55.0, 50.0])}


def test_version_info_gives_the_git_state():
    info = slablet.version_info()
    assert 'version' in info


def test_params_from_platedriver_config():
    p = slablet.MantleParams.from_config(_platedriver_params())    # run_SDP.py
    assert p.slab_sink_rate_factor == 1.0
    assert p.slab_sample_deg > 0                                    # run_SDP._trench_samples


def test_slab_columns_hold_what_platedriver_reads_and_writes():
    for col in ('lon', 'lat', 'depth_km', 'attached'):              # detach_colliding_slabs writes 'attached'
        assert col in slablet.SLAB_COLUMNS
    assert len(pd.DataFrame(columns=slablet.SLAB_COLUMNS)) == 0     # run_SDP.py


def test_empty_samples_is_what_trench_samples_returns_for_no_trench():
    s = slablet.empty_samples()                                     # self_driving_plates.trench_samples
    assert set(s) >= {'lon', 'lat', 'convergence_km_per_myr'} and len(s['lon']) == 0


def test_one_platedriver_step():
    p = slablet.MantleParams.from_config(_platedriver_params())
    dt = 1.0
    slabs = slablet.initial_slab_columns(_samples(), dt, p)                          # run_SDP.py
    state = slablet.new_llsvp_state(p)
    for t in (1.0, 2.0):                                                             # forward model time
        slabs = slablet.seed_slabs(slablet.sink_slabs(slabs, dt, p), _samples(), dt, p, time_myr=t)
        state = slablet.update_llsvp(state, slabs, dt, p)
    assert isinstance(slablet.dvs_field(slabs, state, p), xr.Dataset)               # run_SDP output
    assert len(slabs) > 0


def test_slab_depth_near_with_platedriver_keywords():
    p = slablet.MantleParams.from_config(_platedriver_params())
    slabs = slablet.initial_slab_columns(_samples(), 1.0, p)
    side = np.tile([0.0, 0.0, -1.0], (1, 1))                                         # one boundary midpoint
    depth = slablet.slab_depth_near(np.array([141.0]), np.array([-5.5]), side,      # self_driving_plates.py
                                    slabs['lon'].values, slabs['lat'].values, slabs['depth_km'].values,
                                    radius_deg=5.0, line_tol_deg=1.0, depth_cap_km=1320.0)
    assert depth.shape == (1,)
