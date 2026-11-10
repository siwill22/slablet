"""The reference mantle run (configs/muller2022_refined.yaml) again, saving every 5 Myr, then dynamic topography and the
instantaneous deep flow at every saved age back to 300 Ma, for the Earth viewer.

    conda run -n pygmt17 python viewer/prep/mantle_5myr.py          # from ~/GIT/slablet

Writes runs_real/muller2022_refined_5myr/ (mantle_temp_<age>Ma.nc every 5 Myr, dyntopo_<age>Ma.nc for 0-300 Ma).
Same model as the reference run; only the output interval differs (checked at 0 Ma against the reference).
"""
import os
import time

import numpy as np
import xarray as xr
import yaml

SLABLET = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('SLABLET_WORK', SLABLET)

from slablet import dyntopo                              # noqa: E402
from slablet.gplates import run_mantle                   # noqa: E402

OLDEST_MA = 300

if __name__ == '__main__':
    cfg = yaml.safe_load(open(os.path.join(SLABLET, 'configs', 'muller2022_refined.yaml')))
    ref_dir = os.path.join(SLABLET, cfg['output_dir'])
    cfg['output_every_myr'] = 5
    cfg['output_dir'] = cfg['output_dir'] + '_5myr'
    cfg.setdefault('dynamic_topography', {})['after_run'] = False
    t0 = time.time()
    run_mantle.run(cfg)
    print('mantle run: {:.0f} s'.format(time.time() - t0), flush=True)
    out = os.path.join(SLABLET, cfg['output_dir'])
    a = xr.open_dataset(os.path.join(out, 'mantle_temp_0000Ma.nc'))['dVs_native'].values
    b = xr.open_dataset(os.path.join(ref_dir, 'mantle_temp_0000Ma.nc'))['dVs_native'].values
    print('0 Ma dVs against the reference run: largest difference {:.2e}'.format(np.nanmax(np.abs(a - b))), flush=True)
    t0 = time.time()
    dyntopo.run(cfg, ages=[float(t) for t in range(0, OLDEST_MA + 1, 5)])
    print('dynamic topography: {:.0f} s'.format(time.time() - t0), flush=True)
