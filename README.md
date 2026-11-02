# slablet

A simplified, kinematic mantle model driven by subduction history. Slab pieces ("slablets")
are seeded at trenches, sink through the upper and lower mantle, push a basal pile on the
core-mantle boundary, and are turned into a synthetic shear-velocity anomaly field (dVs) that
can be compared with seismic tomography.

It is driven by trench samples (positions in a mantle frame and convergence rates) from any
source: synthetic plate models (PlateDriver) or real GPlates reconstructions (`slablet.gplates`).

## Layout

| | |
|---|---|
| `slablet/` (core) | the model: `MantleParams`, `seed_slabs`, `sink_slabs`, `update_llsvp`, `dvs_field`. Needs numpy, pandas, xarray, scipy, scikit-learn only |
| `slablet/gplates/` | real reconstructions (pygplates): trench samples, seafloor tracking, lithosphere, the run driver |
| `slablet/analysis/` | scoring against tomography and OPT1, parameter sweeps |
| `slablet/frame/` | fitting and scoring absolute reference frames |
| `configs/` | run configs and sweeps |
| `tests/` | `pytest tests` |

## Install

    conda run -n pygmt17 pip install -e .

## Run a real reconstruction

    conda run -n pygmt17 python -m slablet.gplates.run_mantle configs/muller2022_refined.yaml

Relative paths in configs resolve against the current folder (or `$SLABLET_WORK`).
