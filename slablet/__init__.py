"""slablet: a simplified, kinematic mantle model driven by subduction history.

    from slablet import MantleParams, sink_slabs, seed_slabs, update_llsvp, dvs_field

Used by PlateDriver (synthetic plates) and by slablet.gplates (real GPlates reconstructions).
The core (this package's top level) never imports pygplates, gprm, geopandas, shapely or
PlateDriver; tests/test_core.py checks this.  See slablet/core.py for the trench-sample and
state contract.
"""
from .params import MantleParams
from .sphere import lonlat_to_xyz, xyz_to_lonlat, spherical_gaussian_smooth
from . import pile
from .core import (SLAB_COLUMNS, SLAB_FULL_WEIGHT_CONVERGENCE_KM_PER_MYR, empty_samples,
                   sink_slabs, seed_slabs, initial_slab_columns, slab_depth_near,
                   mantle_grid, new_cmb_heat, update_cmb_heat, dvs_field,
                   LLSVP_MODES, new_llsvp_state, update_llsvp)
