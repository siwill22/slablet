"""Where paths point.

Relative paths in run configs (output_dir, cache_dir) and the research scripts' fixed folders
(runs_real/...) resolve against WORK: $SLABLET_WORK if set, else the current directory.

Local data that cannot be downloaded (tomography models, OPT1, Slab1.0, residual-depth grids,
cooling models) live under DATA: $SLABLET_DATA if set, else ~/Data. A path in a config may start
with $SLABLET_DATA, or with $GPRM_CACHE (gprm's download folder, which differs between operating
systems); data_path expands both, any other environment variable and ~.
"""
import os

WORK = os.path.abspath(os.environ.get('SLABLET_WORK', os.getcwd()))
DATA = os.path.expanduser(os.environ.get('SLABLET_DATA', '~/Data'))


def gprm_cache():
    """gprm's download folder (pooch's cache for 'gprm')."""
    from pooch import os_cache
    return str(os_cache('gprm'))


def data_path(path):
    """A path from a config, with $SLABLET_DATA, $GPRM_CACHE, other environment variables and ~ expanded."""
    path = str(path)
    for name, value in (('SLABLET_DATA', lambda: DATA), ('GPRM_CACHE', gprm_cache)):
        if '$' + name in path or '${' + name + '}' in path:
            v = value()
            path = path.replace('${' + name + '}', v).replace('$' + name, v)
    return os.path.expanduser(os.path.expandvars(path))
