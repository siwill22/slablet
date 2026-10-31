"""Simplified mantle model shared by PlateDriver and the real-reconstruction stream.

    from mantle import MantleParams, sink_slabs, seed_slabs, update_cmb_heat, dvs_field

Never imports PlateDriver (self_driving_plates, run_SDP), reconstructions/, pygplates,
geopandas or shapely; tests/test_mantle.py checks this.  See mantle/core.py for the
trench-sample and state contract.
"""
from .params import MantleParams
from .sphere import lonlat_to_xyz, xyz_to_lonlat, spherical_gaussian_smooth
from .core import (SLAB_COLUMNS, SLAB_FULL_WEIGHT_CONVERGENCE_KM_PER_MYR, empty_samples,
                   sink_slabs, seed_slabs, initial_slab_columns, slab_depth_near,
                   mantle_grid, new_cmb_heat, update_cmb_heat, dvs_field)
