"""The reference mantle run (configs/muller2022_refined.yaml) again, saving every 5 Myr, then dynamic topography and the
instantaneous deep flow at every saved age back to 300 Ma, for the Earth viewer.

    conda run -n pygmt17 python viewer/prep/mantle_5myr.py [--variant NAME]                  # from ~/GIT/slablet
    conda run -n pygmt17 python viewer/prep/mantle_5myr.py [--variant NAME] --viscosity NAME  # flow only

Writes runs_real/muller2022_refined_5myr/ (mantle_temp_<age>Ma.nc every 5 Myr, dyntopo_<age>Ma.nc for 0-300 Ma).
Same model as the reference run; only the output interval differs (checked at 0 Ma against the reference).

A variant (VARIANTS) changes one mantle choice from the reference and writes runs_real/muller2022_refined_5myr_<name>/:
  lm10, lm20   slabs sink through the lower mantle at 10 or 20 km/Myr (x the slab feed^0.5) instead of 15, bracketing
               the 1-2 cm/yr of slab-remnant estimates (van der Meer et al. 2010; Butterworth et al. 2014)
With --viscosity, only the flow is recomputed for a run's saved fields with another radial viscosity profile
(slablet.dyntopo VISCOSITY_PRESETS), into <run folder>_<viscosity>/, whose mantle_temp files link to the run's:
  sc2006       Steinberger & Calderwood (2006)
  four_layer   a stiff 100 km lid (1e22) over 1e21 to 660 km and 3e22 below
The reference flow is the two-layer profile (1e21 above 660 km, 3e22 below).
"""
import argparse
import glob
import os
import re
import time

import numpy as np
import xarray as xr
import yaml

SLABLET = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('SLABLET_WORK', SLABLET)

from slablet import dyntopo                              # noqa: E402
from slablet.gplates import run_mantle                   # noqa: E402

OLDEST_MA = 300
VARIANTS = {'ref': {}, 'lm10': {'slab_lower_rate_km_per_myr': 10}, 'lm20': {'slab_lower_rate_km_per_myr': 20}}
VISCOSITY = {'two_layer': 'two_layer', 'sc2006': 'steinberger_calderwood2006', 'four_layer': 'four_layer'}


def config(variant):
    """The run's config: the reference with the variant's mantle changes, saving every 5 Myr into its own folder."""
    cfg = yaml.safe_load(open(os.path.join(SLABLET, 'configs', 'muller2022_refined.yaml')))
    cfg['mantle'].update(VARIANTS[variant])
    cfg['output_every_myr'] = 5
    cfg['output_dir'] = cfg['output_dir'] + '_5myr' + ('' if variant == 'ref' else '_' + variant)
    cfg.setdefault('dynamic_topography', {})['after_run'] = False
    return cfg


def flow_dir(variant, viscosity):
    """Folder (relative to slablet) holding dyntopo_<age>Ma.nc for this mantle run and viscosity profile."""
    d = config(variant)['output_dir']
    return d if viscosity == 'two_layer' else d + '_' + viscosity


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='ref', choices=sorted(VARIANTS))
    ap.add_argument('--viscosity', choices=sorted(VISCOSITY))
    args = ap.parse_args()
    cfg = config(args.variant)
    ages = [float(t) for t in range(0, OLDEST_MA + 1, 5)]
    if args.viscosity:
        run_dir = os.path.join(SLABLET, cfg['output_dir'])
        cfg['output_dir'] = flow_dir(args.variant, args.viscosity)
        cfg['dynamic_topography']['viscosity'] = VISCOSITY[args.viscosity]
        out = os.path.join(SLABLET, cfg['output_dir'])
        os.makedirs(out, exist_ok=True)
        for f in glob.glob(os.path.join(run_dir, 'mantle_temp_*Ma.nc')):
            name = os.path.basename(f)
            if int(re.search(r'_(\d+)Ma\.nc$', name).group(1)) <= OLDEST_MA and not os.path.exists(
                    os.path.join(out, name)):
                os.symlink(f, os.path.join(out, name))
    else:
        ref_dir = os.path.join(SLABLET, config('ref')['output_dir'].replace('_5myr', ''))
        t0 = time.time()
        run_mantle.run(cfg)
        print('mantle run: {:.0f} s'.format(time.time() - t0), flush=True)
        if args.variant == 'ref':
            out = os.path.join(SLABLET, cfg['output_dir'])
            a = xr.open_dataset(os.path.join(out, 'mantle_temp_0000Ma.nc'))['dVs_native'].values
            b = xr.open_dataset(os.path.join(ref_dir, 'mantle_temp_0000Ma.nc'))['dVs_native'].values
            print('0 Ma dVs against the reference run: largest difference {:.2e}'.format(np.nanmax(np.abs(a - b))),
                  flush=True)
    t0 = time.time()
    dyntopo.run(cfg, ages=ages)
    print('dynamic topography: {:.0f} s'.format(time.time() - t0), flush=True)
