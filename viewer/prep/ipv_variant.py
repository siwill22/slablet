"""IPV (mantlememory) on Müller 2022 from 300 Ma with one choice changed from its reference config
(~/GIT/IPV/mantlememory/examples/config_muller2022_300Ma.yaml), for the Earth viewer.

    cd ~/GIT/IPV && conda run -n pygmt17 python ~/GIT/slablet/viewer/prep/ipv_variant.py NAME

Variants:
  front14, front33   keel drip fronts migrate at 14 or 33 km/Myr instead of 20: the range Gernon et al. (2025) give
  age                fronts start from the seafloor age beside each passive margin (IPV's original rule), on slablet's
                     tracker ages, instead of from the reconstruction's rifting
Writes runs_real/ipv_muller2022_300Ma_<NAME>/ (its config as config.yaml, and viewer_data/ every 5 Myr).
"""
import os
import sys

import yaml

sys.path.append(os.path.expanduser('~/GIT/IPV'))
from mantlememory import MantleSimulation, SimulationConfig    # noqa: E402

BASE = os.path.expanduser('~/GIT/IPV/mantlememory/examples/config_muller2022_300Ma.yaml')
RUNS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'runs_real'))
VARIANTS = {'front14': {'keel_front_speed': 14.0},
            'front33': {'keel_front_speed': 33.0},
            'age': {'keel_front_trigger': 'seafloor_age'}}


def output_dir(name):
    return os.path.realpath(os.path.join(RUNS, 'ipv_muller2022_300Ma_' + name))


if __name__ == '__main__':
    name = sys.argv[1]
    cfg = yaml.safe_load(open(BASE))
    unknown = set(VARIANTS[name]) - set(cfg)
    assert not unknown, 'not in the reference config: {}'.format(unknown)
    cfg.update(VARIANTS[name])
    cfg['output_dir'] = output_dir(name)
    os.makedirs(cfg['output_dir'], exist_ok=True)
    path = os.path.join(cfg['output_dir'], 'config.yaml')
    with open(path, 'w') as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    MantleSimulation(SimulationConfig.from_yaml(path)).run()
