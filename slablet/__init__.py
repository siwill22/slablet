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
                   mantle_grid, new_cmb_heat, update_cmb_heat, dvs_field, current_dvs,
                   LLSVP_MODES, new_llsvp_state, update_llsvp, STALL_COLUMNS, stall_sites, stall_times, trench_retreat)

__version__ = '0.1.0'


def version_info():
    """slablet's version and, for a git checkout, its commit (with '-dirty' if it has
    uncommitted changes): recorded with every run so results can be traced to the code."""
    import os
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    info = {'version': __version__}
    try:
        git = ['git', '-C', here]
        commit = subprocess.run(git + ['rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(git + ['status', '--porcelain', '--untracked-files=no'], capture_output=True,
                               text=True, check=True).stdout.strip()
        info['git'] = commit + ('-dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        pass
    return info
