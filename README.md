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
| `slablet/dyntopo.py`, `slablet/sphharm.py` (core) | dynamic topography, geoid and instantaneous flow from the field (needs pyshtools: `slablet[dyntopo]`) |
| `slablet/plate_history.py` (core) | the plate-history contract: what a driver (a PlateDriver world or a real reconstruction) hands the response models each step (`PlateStep`, `check`, `velocity_at`) |
| `slablet/asthenosphere.py`, `slablet/asthenosphere_plates.py`, `slablet/asthenosphere_supply.py` (core) | the Buck-type asthenosphere layer (finite volumes on any mesh), its forcing from any plate step (`forcing_from_step`, `run_layer`), for real reconstructions and PlateDriver worlds alike, its supply as a mixture of uniform, plume-fed and deep-upwelling parts from a mantle run, and passive tracers (fertility, mantle domains) carried upwind with its flux |
| `slablet/cooling.py` (core) | cooling laws (half-space, GDH1 default, PS77, Crosby 2007, RHCW18): depth, heat flow and lithosphere thickness from one temperature field per law; RHCW18 reads the authors' output from `$SLABLET_DATA/CoolingModels/RHCW18_Plate_Model` |
| `slablet/gplates/` | real reconstructions (pygplates): the plate-history driver (`plate_history.RealPlateHistory`), trench samples, seafloor tracking, lithosphere, the run driver |
| `slablet/analysis/` | scoring against tomography and OPT1, parameter sweeps, the stall sweep; residual depth under any cooling law from Seton (2020) ages, SRTM15+ and GlobSed (`residual_depth.py`, needs pygmt) |
| `slablet/frame/` | fitting and scoring absolute reference frames |
| `configs/` | run configs and sweeps |
| `scripts/check_against_hc.py` | compares `slablet.dyntopo` with HC |
| `CONTEXT.md` | the project's terms |
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

## Dynamic topography

    conda run -n pygmt17 python -m slablet.dyntopo configs/muller2022_refined.yaml          # 0 Ma; --ages, --all-ages

writes `dyntopo_<age>Ma.nc` beside each `mantle_temp_<age>Ma.nc`, with:
- surface dynamic topography measured from the geoid (water-loaded by default);
- the geoid;
- CMB topography;
- topography, CMB and geoid kernels;
- velocities and radial and shear stresses at every layer depth;
- spherical-harmonic coefficients of the surface fields;
- the same per source of the field (slabs, pile, ...), when the run saved them (`mantle: dvs_save_components: true`).

The flow is Hager & O'Connell's (1981): incompressible, self-gravitating as in HC, with viscosity varying only with
depth (presets `two_layer` (default), `uniform`, `four_layer`, `steinberger_calderwood2006`, or a table). Density
comes from `dVs_native` below 300 km, with d ln rho / d ln Vs = 0.25 times PREM density. Settings go in a
`dynamic_topography:` block of the run config (`slablet.dyntopo.DynTopoParams`). `after_run: true` computes it at
the end of `run_mantle`; it is off by default.

`scripts/check_against_hc.py` runs HC (github.com/geodynamics/hc, built with `make all_no_gmt`) on the same field
and compares degree by degree. Surface and CMB stress, geoid, velocities and stresses agree to about 1e-5 for all
four viscosity presets.

`slablet/analysis/dyntopo_refs.py` loads present-day references (Steinberger's flow models, Holdt et al. 2022
residual topography, Valentine & Davies 2020, the observed non-hydrostatic geoid) and the measures used to compare
with them.

## Stalls in the transition zone

Off by default (`mantle: slab_stall_fraction: 0`). With it on, slab from one stall patch (the nearest of a fixed
lattice of sites, `slab_stall_site_deg` apart, times a window of seeding time, `slab_stall_window_myr`) shares one
stall time: with probability `slab_stall_fraction` the patch stalls, for `slab_stall_time_myr` (fixed) or a time
drawn about that mean (exponential); otherwise it goes straight through. Each slablet of a stalled patch is held
1 km above `upper_mantle_km`, fixed in the mantle frame, for that time from its arrival, then sinks at its
lower-mantle rate. Its thermal decay runs on while it is held. `slab_stall_seed` picks the draws, so an ensemble
is the same config with different seeds; the same seed gives nested draws across settings (patches that stall at
one fraction also stall at a larger one). Initial slab columns never stall. The caller passes the time to
`seed_slabs(..., time_myr=)`.

    conda run -n pygmt17 python -m slablet.analysis.stall_sweep configs/muller2022_refined.yaml   # about 200 runs
    conda run -n pygmt17 python -m slablet.analysis.stall_summary

The sweep runs each setting with 10 seeds and scores fast regions at 600-900 km and the usual scores at
1000-2800 km against each tomography model, plus present-day dynamic topography. The summary calls a setting
helpful in a depth range when its members' mean beats stalls off by more than twice its standard error for every
tomography model.

## Slab carried with its trench

Off by default (`mantle: slab_trench_follow: false`). With it on, a slablet above `upper_mantle_km` also moves
with the velocity its trench had when it was seeded, times `slab_trench_follow_fraction` (1: attached; 0-1:
partly anchored in the mantle). With `slab_trench_follow_retreating_only: true`, only slablets whose trench was
retreating (moving away from the overriding plate, in the mantle frame) faster than
`slab_trench_follow_retreat_km_per_myr` at seeding follow it; the rest stay fixed in the mantle. Needs the trench
samples' down-dip azimuth and trench velocity (real reconstructions; PlateDriver has none).

Upper-mantle slablets can also move with a fraction of the plate velocity above them, `slab_flow_upper` (< 0:
return flow; off by default), as `slab_flow_top` / `slab_flow_bottom` do in the lower mantle; both need the
seafloor tracker's velocity grid (`seafloor: true`).

Frame tests (run config, not `mantle:`): `frame_lon_drift_deg_per_myr` rotates every trench sample about the spin
axis by drift x age; `frame_lon_drift_until_ma` stops it growing at that age, and `frame_lon_drift_zero_by_ma`
brings it back to zero by that age. Diagnostics of longitude errors, not corrections.
