# Status: where the work stands

Written 12 November 2026, at a pause of a few days; updated 16 November, when the public release of slablet and
PlateDriver began (PlateDriver `docs/public_release_plan.md`). The work spans four repositories (PlateDriver,
slablet, IPV and defamation), and its plans, results and open problems are spread over several documents. This page
is the entry point for picking it up again. It says what exists, what is uncommitted, what is planned next and what is known to be
wrong, and points to where each is written up in full. Update it when any of these changes.

## Where the details live

| Document | What it holds |
| --- | --- |
| [`plans/2026-11-08-cross-repo-reconciliation.md`](plans/2026-11-08-cross-repo-reconciliation.md) | The register: ten conceptual inconsistencies between the repositories, the 20 decisions that settle them, v1 (the AAD region on Müller 2022) in gated stages, and each stage's result (section 8) |
| [`plans/2026-11-05-buck-asthenosphere-test.md`](plans/2026-11-05-buck-asthenosphere-test.md) | The Buck-type asthenosphere layer: its decisions, the AAD result and stress test, and the list of issues to deal with (section 9, items 0–14, planning only) |
| [`plans/2026-11-11-s40rts-filter.md`](plans/2026-11-11-s40rts-filter.md) | Replacing the generic tomography blur with S40RTS's resolution operator (stage 0, the blur's removal, done) |
| [`adr/0001-one-package-for-shared-physics-and-real-world-tools.md`](adr/0001-one-package-for-shared-physics-and-real-world-tools.md) | Why the shared physics and the real-world tools are one package |
| [`../CONTEXT.md`](../CONTEXT.md) | The project's terms |
| [`../README.md`](../README.md), [`../viewer/README.md`](../viewer/README.md) | What slablet and its Earth viewer do, and how to run them |
| PlateDriver `README.md`, `docs/force_model_v1_plan.md` to `v3`, `docs/viewer_plan.md` | The synthetic plate model, its force-model phases and its viewer |
| PlateDriver `docs/public_release_plan.md` | The public release of both repositories: its stages, the decisions taken and a progress section |
| PlateDriver `show-me/<date>-<topic>/` | Every gate's figures, `make_figures.py` and README (gitignored, so local only); the register names the folder for each result |
| IPV `mantlememory/README.md`, `mantlememory/docs/OPEN_ISSUES.md` | IPV's model, its viewer, and its open issues |
| `~/GIT/aspect/gplates-kunanyi/README.md` | GPlates-driven ASPECT runs on kunanyi (separate from this work) |

## What exists

**slablet** (`~/GIT/slablet`, branch `asthenosphere-layer`, with `main` at the same commit; private until the
release, MIT licence) holds the shared physics and the real-world tools (README, layout table):
- the kinematic mantle model (slablets, the basal pile, plume sites; reference config `configs/muller2022_refined.yaml`);
- dynamic topography, geoid and instantaneous flow (Hager & O'Connell, checked against HC to about 1e-5);
- the plate-history contract and its real-reconstruction driver;
- five cooling laws, and residual depth under any of them;
- the asthenosphere layer with its supply mixture, passive tracers and flow-driven volcanism;
- the seafloor tracker, which can also write yearly age grids and list where the reconstruction rifts;
- tomography and OPT1 scoring, sweeps, and frame fitting.

Its Earth viewer (`viewer/`, http://localhost:5180 with `npm run dev`) shows all the layers together on Müller 2022,
300 Ma to now every 5 Myr, with alternative runs of each model (below).

**PlateDriver** (`~/GIT/PlateDriver`, branch `mantle-extract`, 78 commits ahead of `master`; public on GitHub as
OldDriver, to be renamed) makes synthetic plate histories. Force model v1 (the physical default) and v2 (rifting,
collisions, merging) are done. v3 (subduction initiation, then back-arc rifting, plume weakening and teaching
outputs) has been paused since 5 October 2026, after stage S1's first step (passive margins logged as candidates).
Since then the work has been on the mantle and the asthenosphere. PlateDriver now uses slablet only optionally and
hands it its state through the contract (`self_driving_plates.plate_step`); the release plan makes it a requirement
once slablet is public. Its viewer shows PlateDriver runs at http://localhost:5173. All run outputs, for both
repositories, are in `PlateDriver/runs_real/` (95 GB, gitignored; slablet's `runs_real` links there).

**IPV** (`~/GIT/IPV`, branch `main`) is MantleMemory: mantle fertility on particles and lithospheric vulnerability,
which give intraplate volcanism. It now runs on Müller 2022 (uncommitted, below), and its drip fronts can start where
the reconstruction rifts. None of its physics has moved into slablet yet (decision 6).

**defamation** has not changed since the review of 8 November (cases 9 and 10 of the register).

### The model runs in the Earth viewer

Each alternative changes one choice from the reference; the viewer picks one run per model. Scripts are in
`viewer/prep/`, and the run table is in `viewer/README.md`.

| Model | Reference | Alternatives | What changes (10 November) |
| --- | --- | --- | --- |
| Mantle structure | lower-mantle slab sinking 15 km/Myr | 10, 20 km/Myr | at 10 km/Myr the field correlates 0.78–0.93 with the reference at 1050–2650 km; at 20 km/Myr 0.88–0.97 |
| Deep flow viscosity | two-layer (1e21, 3e22 below 660 km) | Steinberger & Calderwood (2006); stiff 100 km lid | dynamic topography hardly changes (r ≥ 0.99, rms within 10%) |
| Asthenosphere layer | Buck values, craton keels, uniform supply | mobility ×0.5, ×2; capped at 600 km; continents as keels (capped); deep-upwelling supply | continents as keels leave 59% of the surface open (80% with cratons), and double the layer's seafloor-depth effect (rms 616 against 319 m today) |
| IPV | drip fronts 20 km/Myr, started by rifting | 14, 33 km/Myr; IPV's seafloor-age trigger | at 100 Ma fronts are active over 4.8%, 3.1% and 1.9% of the area at 14, 20 and 33 km/Myr |

## Uncommitted work

Check this first: some of it is someone else's work in progress.

| Repository | What | State |
| --- | --- | --- |
| PlateDriver | The asthenosphere viewer of 7 November (`viewer/asthenosphere.html`, `viewer/src/asthenosphere/`, `viewer/prep/export_asthenosphere.py`, `viewer/scripts/check_asthenosphere.mjs`, and its section of `viewer/README.md`) | Stashed in PlateDriver on 17 November (`git stash list`: "Asthenosphere viewer of 7-10 Nov"); slablet's Earth viewer shows the same fields and more |
| IPV | Müller 2022 support (`reconstruction`), fixed normalisation (`vulnerability_normalisation: fixed`), drip fronts from rift points (`keel_front_trigger: rift_points`), `examples/config_muller2022_300Ma.yaml`, and the matching notes in `docs/OPEN_ISSUES.md` and `README.md` | Uncommitted, on `main` |
| IPV | Simon's own notebook edits (`transforms.ipynb`, `isopolatred.ipynb`, `test_data/test_isochrons.gpml`) | Uncommitted; not to be committed with the above |

## Planned next

1. **The public release** (PlateDriver `docs/public_release_plan.md`): slablet first, then PlateDriver; then the
   synthetic-Earth stages D–F of that plan.
2. **The S40RTS filter**, stage 1 onwards (`plans/2026-11-11-s40rts-filter.md`): pass the model through S40RTS's
   resolution operator and compare like with like.
3. **v1, the AAD region** (register, decision 20):
   - Stage 5: a forward run from the best-fitting 80 Ma domain map, and fertility's sources.
   - Stage 6: the observations Simon deferred on 10 November. These are seamount chains from SWOT (no grids on disk
     yet), the orientation test against layer flow, fabric and plate motion, and chains as tracks.
   - Alongside: anisotropy, by asking De Laat et al. for their model, with digitising their Figure 7 as a fallback.
     Not started.
4. **The asthenosphere issues list** (Buck plan, section 9, items 0–14): planning only, none started.
5. **Deferred from v1** (register, end of decision 20): continental deformation from defamation; defamation's trench
   side and IPV's continent definition; PlateDriver's `contains_xy` fix; a spherical rasterise function in gprm; the
   global seamount catalogues; moving the rest of IPV into slablet; the volcanism mixture and the layer in synthetic
   worlds; cratons distinct from continents in synthetic worlds.
6. **PlateDriver force model v3**, from stage S1, then the teaching outputs (`docs/force_model_v3_plan.md`).
7. **IPV calibration**: no agreed targets yet (`OPEN_ISSUES.md`, section 1).

## Known issues

### Mantle model and deep flow
- **Dynamic topography is large.** Its rms is about 2 km in the reference run and its largest values about 7 km,
  10 km with the slowest lower-mantle sinking and the stiff lid. Earlier work claims nothing beyond degrees 1–3, and
  the cause of the size has not been looked into.
- **The deep flow cannot drive the plates.** It is driven by buoyancy below 300 km with a free-slip surface, so it
  does not move with the plates, while the layer assumes a still mantle beneath it. The viewer shows both as they
  are.
- **The frame conclusion depends on the mantle model.** Müller 2022's optimised frame was rejected at 100–200 Ma with
  the old mantle model, but not with the refined one (PlateDriver `show-me/2026-11-02-refinements/`).
- **Stalls do not help globally.** Freely chosen transition-zone stalls fit no better than chance. The one regional
  gain, a delay of NW Pacific slab, is at 1200–1500 km. Young slab fits tomography about 4.5° further east, which
  points at the frame (PlateDriver `show-me/2026-11-06-overnight-summary/`, `2026-11-06-frame-families/`).
- **The field is compared with tomography unfiltered.** The old blur removed far more fine structure than tomography
  loses and hardly changed the scores, so it was removed (S40RTS plan, section 1). Until the S40RTS resolution
  operator is built (its stage 1), scores compare the unfiltered field with tomography.

### Asthenosphere layer
- **Piles under advancing keels.** Bulldozing piles the layer into enclosed pockets without limit. The reference
  reaches 4750 km in 0.011% of cells, and with every continent a keel the run all but stalls. A cap (`run_layer
  cap_km`, held after keels move since slablet `a0f30d6`) fixes it and barely changes the reference (r 0.96–0.999),
  but the reference still runs without one.
- **The layer fails on the sphere outside the AAD** (register, stage 4 and the layer sweep). The EPR is used up
  beneath 21% of its samples, the PAR deepens towards the EPR, and no mobility fits both the AAD and the PAR.
- **Grid-cell tracers smear fronts** to 400–800 km in 10–40 Myr, so the domain test uses particles.
- **The westward migration of the isotopic boundary is too slow** at about 10 mm/yr, against 25–40 observed (stage 5).
- **Flow-driven volcanism is incomplete.** It has no solidus limit and no fertility sources, and the shear-forced
  upwelling exists only in the viewer's exporter, not in `asthenosphere_volcanism.py`.

### Seafloor ages and reconstructions
- **Müller 2022 keeps old seafloor that never subducts.** 2–3% of the ocean is older than 300 Myr (up to 850 Myr
  today; Arctic, Tethys, off Antarctica), in the published grids and the tracker alike. Any rule that reads the age
  beside a margin as time since break-up is misled by it.
- **The published Müller 2022 212 Ma age grid is all zeros**; the tracker's grids replace the published ones.
- **The tracker's ages inside continents** reach 200 Myr and more; consumers mask them with the continent query.
- **The seafloor and lithosphere caches did not record where tracking started** (fixed 17 November 2026). Runs
  tracked from 250 Ma and from 1000 Ma shared one cache, so a run's 50-km lid depended on which config built it
  first (up to 3% dVs). Each start now has its own folders (`seafloor_v1_from<start>`, `lithosphere_v1_from<start>`);
  the existing caches were renamed to their starts, so earlier runs read the same data.
- **Not built:** the Crosby thermal-boundary-layer cooling law (its definition is unknown) and Grose & Afonso (2013)
  as a law of its own.

### IPV
- Its open issues are in `mantlememory/docs/OPEN_ISSUES.md`. Three of them matter most here:
  - Strain rate marks plate boundaries only.
  - The rift term never fires.
  - Edge-driven convection still finds passive margins from seafloor age, so it meets the old-seafloor problem
    above.

### PlateDriver
- A synthetic world has no cratons distinct from its continents, so keels cannot differ from continents there.
- Where moved plates overlapped, the lower plate ID kept the cells, whatever the polarity. The current model now gives
  them to the overriding plate (`overlap: polarity`). The `simple` preset keeps the old rule, because with the new one
  its undamped speeds ran away further; `simple` suits runs of up to about 200 Myr, and in 3 of 6 runs of 500 Myr the
  world ended as one plate (PlateDriver `show-me/2026-11-16-overlap_polarity/`, `2026-11-16-simple_overlap/`).
- The shapely `contains_xy` test for initial seafloor ages remains. Douglas–Peucker simplification in degrees shapes
  the trench samples.
- v2's known issues: hidden continent lost at a rift, calibration of the stiff contact, and per-segment trench
  polarity (`docs/force_model_v3_plan.md`, "Open").

### Viewers and tooling
- **No point layer in the Earth viewer.** IPV's particles and seamounts and the mantle run's plume sites, which IPV's
  own viewer draws, are missing.
- **The flat-map rule is not everywhere yet.** In slablet's Earth viewer, a sideways drag or scroll on a flat map
  moves the central meridian. PlateDriver's viewers and Geode's core still slide the map instead.
- **Disk space.**
  - The Earth viewer's data is 12 GB (local, `viewer/public/data/earth/`).
  - Its runs take about 35 GB of `runs_real`, mostly flow files: 3 GB per viscosity variant. Each mantle variant
    also keeps 5 GB of saves, including those older than 300 Ma.
  - These can be deleted once exported, and rebuilt with the commands in `viewer/README.md`.
- **IPV's viewer depends on unreleased petrify work** (`OPEN_ISSUES.md`, section 6).
