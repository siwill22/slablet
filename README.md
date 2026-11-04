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

    conda run -n pygmt17 python -m slablet.gplates.seafloor_tracker configs/muller2022_refined.yaml   # once
    conda run -n pygmt17 python -m slablet.gplates.run_mantle configs/muller2022_refined.yaml

Relative paths in configs (`output_dir`, `cache_dir`) resolve against the current folder, or
`$SLABLET_WORK` if set. Each run writes `run_info.yaml` with the full parameters, the files the
reconstruction resolved to, and slablet's version and commit.

### Choosing the reconstruction

A config names its reconstruction in a `reconstruction:` block (details in
`slablet/gplates/models.py`):

```yaml
reconstruction:                    # gprm (GPlatesReconstructionModel)
  source: gprm
  model: Muller2022                # gprm's fetch_Muller2022
  options: {NNR: false}            # keyword arguments of the fetch function
  craton_files: [Cratons/*.gpml]   # gprm has no cratons; needed only for the lithosphere lid
  coastline_files: [Coastlines/shapes_coastlines_Merdith_etal.gpmlz]   # for maps and the Geode viewer

reconstruction:                    # GPlately's PlateModelManager
  source: plate_model_manager
  model: muller2022
  data_dir: ~/.cache/slablet/plate_model_manager

reconstruction:                    # files on disk
  source: files
  model_dir: ~/Data/MyModel
  rotation_files: [rotations/*.rot]
  topology_files: [topologies/*.gpml]
  continent_files: [continents.gpml]   # optional, for the lithosphere lid
  craton_files: [cratons.gpml]
```

Any source also takes `rotation_files` to swap in another frame. Slablets sink vertically in the
anchor plate's frame, so it must be fixed to the mantle. gprm records each model's frames, and a
frame fixed to the spin axis is refused unless the config sets `allow_spin_axis_frame: true`. For
other sources, or after a `rotation_files` swap, the run prints a note that the frame is unknown.
Configs with `model_dir`, `rotation_files` and `topology_files` at the top level (the original
form) still work. The same files reach the same trench-sample cache whichever source named them.

## Notebook: run a model and view it in Geode

`notebooks/run_and_view.ipynb` (kernel: pygmt17). Set the inputs in its first code cell (reconstruction, ages,
time step, mantle overrides, what the viewer opens on), then run all cells. It tracks seafloor ages if needed,
runs the model, draws the present-day field, writes a Geode Archive of the run (`slablet.geode.export_run`) and
shows the Geode globe viewer in the last cell, with a link to open it in its own tab (`slablet.geode.show`).
The viewer is Geode's tomography viewer, built once into `~/.cache/slablet/geode-viewer` from a Geode checkout
(`~/GIT/Geode` or `$GEODE`, with `npm install` done in `viewer/`) and served from a local web server; nothing is
written into the Geode checkout. Clennett 2020 from 250 Ma takes about 3.5 min the first time (seafloor
tracking) and about 1.5 min after.
