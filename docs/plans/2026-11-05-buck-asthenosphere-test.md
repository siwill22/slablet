# Plan: a Buck et al. (2009) asthenosphere layer driven by the plate history

Handoff plan. Stage 0 (the open choices) was agreed with Simon on 2026-11-05; the decisions are in section 4.
Nothing has been built yet. Start at Stage 1. Each stage ends at a gate: figures, a short README, then wait for Simon.

## 1. Aim

Exploratory, not a hypothesis test:
- With plausible parameters, does a Buck-type asthenosphere layer driven by the real plate history give a starved
  ridge between Australia and Antarctica, something like the Australian–Antarctic Discordance (AAD)?
- The same run covers the rest of the world for free: what does it do at other ridges?
- What pattern of upper-mantle flow does this simple picture imply (flow drawn towards the "holes" that spreading
  makes, against plate drag that carries the layer away)?

Not in scope: a contest with the Gurnis et al. (1998) deep-slab explanation (slablet's deep dynamic topography has
already been tried for the AAD and struggles), fitting on the EPR–PAR step, pass/fail scores, or adding the result
to deep dynamic topography. These may come later as separate work.

Paper: `~/Library/CloudStorage/OneDrive-UniversityofTasmania/pdflib/Buck_etal_2009_AAD.pdf`
(Buck, Small & Ryan 2009, G-cubed 10, Q09007, doi:10.1029/2009GC002373). Sections 3–4 and Appendix A.

## 2. The model in the paper

Thin-layer flow of an asthenosphere of thickness h, viscosity μ, density lower than the mesosphere below by
ρM − ρA, under rigid plates moving at u_p, with zero velocity at its base (eq. 2–3):

    q = u_p h / 2  −  (Δρ* g h³ / 12μ) ∇h                       (flux, m²/s: drag part, pressure-driven part)
    ∂h/∂t = −∇·q + S − sinks                                       (eq. 7)
    Δρ* = (ρA − ρW)(ρM − ρA) / (ρM − ρW)        (≈ 10 kg/m³ for ρM − ρA = 10)
    d = d_max (1 − h / h_a),  d_max = (ρM − ρA)/(ρM − ρW) · h_a    (eq. 8; seafloor depth anomaly, + = deeper)
    Rm = Δρ* g h_a² / (μ u0)                                       (Ramberg number, eq. 10)

Spreading thins the layer two ways: it is turned into lithosphere at the axis, and the plates drag it away towards
the trenches. Pressure-driven flow and the supply refill it. At the AAD the keels of Australia and East Antarctica
block the refill from the sides.

What the paper states and what it leaves open (checked 2026-11-05):
- Explicit finite differences on a fixed uniform grid (100 km), Courant time step. Mesosphere compensates locally
  and instantly (fine for mesosphere viscosity ≤ 10²¹ Pa s).
- Sink: all conversion to 100 km lithosphere at the ridge axis; gradual conversion "did not qualitatively affect
  the results".
- Supply: uniform, balancing the sinks; **not delivered under the continental blocks**.
- AAD1: layer 300 km "everywhere except under the model continental blocks" (so h = 0 under them); AUS moves north
  6 cm/yr over the fixed grid. How cells uncovered or covered by a moving block are treated is **not stated**.
- Trenches: reflecting walls in EPR-PAR1; advective outflow (h × plate speed) on AAD1's north edge, which lowers the
  mean h by about 20 % in 40 Myr.
- Preferred values: Δρ = 10 kg/m³, h_a = 250–300 km, μ ≈ 10¹⁹ (1.3 × 10¹⁹ in the 2-D cases).
- Near-critical behaviour: small parameter changes give large changes.

Benchmarks:
- **A1** (1-D plate-driven, S = 0, symmetric basin of half-width W): h(x) = h(0)(1 − C0 x)^{1/3}; axial thickness
  reaches zero at Rm_cr = (3/2)^5 · W/h_a ≈ 7.6 W/h_a, with h(0) = (4/3) h_a.
- **A2** (1-D source–sink): h(x) = h(0)(1 − C1 x)^{1/4}.
- **Fig. 3** (1-D numerical): 10,000 km basin, μ = 10¹⁹, h_a = 225 km, 100 km grid; 5 → 10 cm/yr full rate thins
  the sub-axial layer by about 150 km; ridge migration about 75 km.
- **EPR-PAR1** (Fig. 4–5): 10,000 × 10,000 km box, reflecting sides; PAC 9 cm/yr west, ANT fixed, NAZ 6 cm/yr east;
  ridge at x = 7,500 km; EPR 15 cm/yr migrating 1.5 cm/yr west, PAR 9 cm/yr migrating 4.5 cm/yr west; Δρ = 10,
  h_a = 300 km, μ = 1.3 × 10¹⁹; 50 Myr. Model EPR about 400 m deeper than PAR.
- **AAD1** (Fig. 8–9): two impermeable blocks 3,000 km E–W × 2,000 km N–S, 800 km apart at the start, ridge
  midway; AUS 6 cm/yr north, ANT fixed; S, E, W reflecting, N advective outflow; same Δρ, h_a, μ; 40 Myr. Axis
  between the blocks exhausted after a few hundred km of opening; smooth axial depth anomaly over several thousand
  km. At full resolution Fig. 9a shows a deep pocket (about -800 m) just behind AUS's trailing edge, as expected if
  cells the block uncovers start empty (our v1 rule); no bulge ahead of AUS (the north edge is an outflow).

## 3. What slablet already has

Read `README.md` and `CONTEXT.md` (glossary; the asthenosphere terms were added at Stage 0) first.
- `slablet/gplates/models.py`: reconstruction from a config's `reconstruction:` block. Reference config:
  `configs/muller2022_refined.yaml` (Müller 2022, optimised mantle frame, anchor 0, from 1000 Ma).
- `slablet/gplates/lithosphere.py`: craton and continent polygons (Müller 2022: `Cratons/shapes_cratons_Merdith_etal.gpml`,
  `Continents/shapes_continents.gpml`) reconstructed with pygplates `PlatePartitioner`. Copy the masking pattern; the
  1° grid itself is not used (keels are tested at mesh cell centres).
- `slablet/gplates/trench_samples.py`: trench samples per Myr (convergence, down-dip azimuth, trench velocity),
  cached. Pattern for `ridge_samples.py`.
- `slablet/sphere.py`: lon/lat ↔ xyz, `move_along_azimuth`, spherical smoothing.
- `slablet/analysis/dyntopo_refs.py`: `dunn_points()` (Dunn et al. 2026, Southern Ocean residual depths),
  `holdt_points()`. Import only.
- Data: `~/Data/ResidualBasementDepth/res_basement_*.nc`; bathymetry from PyGMT's `@earth_relief`.

Present-day Merdith cratons cover western and central Australia and almost all of East Antarctica, close to Buck's
3,000 km blocks. A small "craton" at about 144–150°E, 42–48°S (Tasmania / South Tasman Rise) partly blocks the
eastern gateway towards the AAD; check it before the first sphere run and decide with Simon whether to keep it.

**slablet has uncommitted work from another chat** (modified `core.py`, `params.py`, `run_mantle.py`, `README.md`,
`pyproject.toml`, `__init__.py`, `test_core.py`; untracked `dyntopo.py`, `sphharm.py`, `dyntopo_refs.py`,
`stall_*.py`, `dip_fit.py` and others). Do not edit, revert, reformat or commit any of it. Never `pkill` broad patterns.

## 4. Decisions (Stage 0, agreed 2026-11-05)

v1 is the first entry in each row; the others are options, off by default, each tested later as its own stage.

| Choice | v1 | Options for later |
|---|---|---|
| Keels | Merdith craton polygons fully impermeable; other continent is open layer | all continents impermeable; channel thinned under thick lithosphere (h ≤ h_max − lithosphere) |
| Moving keels (fixed mesh) | uncovered cell starts at h = 0 (mesosphere fills, the flow refills); covered cell's layer bulldozed into open neighbours ahead; no supply under keels | covered layer deleted (counted as a sink) |
| Trenches | wall: no flux across mesh edges crossed by a trench line; walls move with trenches each Myr | sink: outflow h × convergence. **Both will be run and reported.** |
| Supply | uniform S over open cells, reset each step to balance that step's sinks (mean h stays h_a) | S constant in time; plume sites |
| Ridge sink | at axis: normal divergence × 100 km × sample arc length, into the cell holding the sample | gradual with age (5/√age km/Myr) |
| Mesh | Fibonacci spherical Voronoi, ~40,000 cells (~110 km); plate velocity from plate ID + rotation at cell centres each Myr | ~160,000 cells (resolution check); icosahedral mesh if grid noise shows; implicit time step if slow |
| Frame | `muller2022_refined` (Müller 2022 optimised mantle frame, anchor 0) | other mantle-fixed frames (a hotspot frame; NNR as a reference). Never a palaeomagnetic frame |
| Start | 250 Ma, h = h_a in open cells, h = 0 under keels | 150 Ma and 400 Ma |
| Parameters | Buck's values: Δρ = 10 kg/m³, h_a = 300 km, μ = 1.3 × 10¹⁹ Pa s | sweep decided after review; candidate box Δρ {5, 10, 20} × h_a {150, 225, 300} × μ {10¹⁸, 3×10¹⁸, 10¹⁹, 3×10¹⁹, 10²⁰} |

Edge rules:
- **Positivity:** a cell can't give more than it holds. The unmet ridge sink is recorded as the starved deficit
  (mesosphere upwelling), the starved-ridge diagnostic.
- **Divergent boundaries in continents** (rifts in the topologies): no ridge sink unless the sample lies on a
  non-continent cell. Count how many such samples there are.
- **Ridge-less corners:** with walls and uniform supply, poorly connected regions without ridges can thicken
  (≈ 0.5 km/Myr of supply). Every run reports the ten fastest-thickening places.
- **Start-time test:** the 0 Ma SE Indian Ridge axial profile and global depth map should differ by less than about
  10 % of the AAD-region low between starts. If not, report long memory as a finding.

Spherical geometry: true Voronoi cell areas and edge lengths; centre-to-centre great-circle distances for the
gradient; trench-edge crossings by pygplates distance = 0 between the edge arc and the trench polyline; keels and
plate IDs by pygplates polygon tests at cell centres; ridge sample arc lengths on the sphere. The flat metric is used
only in Stage 2, because the paper's cases are flat.

## 5. Outputs

For each output time (every 5 Myr):
- maps of h and d; total flux as arrows over h; the pressure-driven part alone (the "flow towards the holes");
  the drag part alone; flux divergence with supply and sinks; starved deficit;
- NetCDF of all fields remapped to a regular 1° grid (flux as east/north components), for PyGMT.

Projections: Mollweide global; Lambert azimuthal equal-area centred near the AAD (~125°E, 50°S).
Comparisons (description, not scores): observed SE Indian Ridge axial depth across the AAD (median `@earth_relief`
within 50 km of the reconstruction's 0 Ma ridge, every ~50 km along strike) against model d; other ridges likewise,
with known hotspot segments labelled; Dunn's points; `res_basement_*.nc` for the off-axis trace.

## 6. Stages (each a gate: figures in `~/GIT/PlateDriver/show-me/<date>-<topic>/` with `make_figures.py` and README)

1. **Solver and 1-D benchmarks.** `slablet/asthenosphere.py` (`AsthenosphereParams` dataclass with all options, off
   by default, and the solver: upwinded drag flux u_p·n h/2 per edge, minus (Δρ* g h̄³/12μ)(Δh / centre distance);
   explicit, adaptive stability limit). `tests/test_asthenosphere.py`: A1 critical solution and A2 profile on a 1-D
   strip, mass conservation to round-off with walls, uniform layer under a rigid rotation stays uniform, positivity,
   mesh area sums to 4πR². Gate figures: numerical vs analytic profiles; Fig. 3 reproduced.
2. **Buck's flat-box AAD1 and EPR-PAR1**, same solver with a flat metric. Reproduce Fig. 9 and Fig. 5, judged on
   the features that are not near-critical (walls, shape and width of the low, where the layer is used up). Stage 1
   found Buck's Fig. 3 fast case on a cliff (axis 80 km at mu = 0.95e19, 0 at 1.0e19), so axial values near zero
   are reported as sensitivity, not failure. A mismatch in the robust features means a solver or reading error:
   stop and report.
3. **Sphere, present-day plates held fixed until steady.** An engineering check of mesh, walls, ridge sinks and
   keels on the sphere without moving geometry, plus a first look at the AAD region. Not a result: fixed present
   plates over-starve the AAD compared with the real history. Includes the Tasmania/STR craton check and
   `slablet/gplates/ridge_samples.py` (cached like trench samples).
4. **Through time, 250 → 0 Ma, v1 settings.** Driver `slablet/gplates/run_asthenosphere.py`; plate velocities,
   ridges, trenches and keels per Myr; NetCDF beside the run output. Check when the AAD-region low appears
   (fast Australia–Antarctica spreading from ~45 Ma). This is the main result.
5. **Options, one at a time, each reviewed:** trench sink; parameter sweep; start-time checks; keel alternatives;
   frame comparison; gradual sink; constant supply; finer mesh.

Rough cost (explicit, worst case h = 400 km): μ = 10¹⁹ about 550 sub-steps/Myr, about 2 min per 200 Myr at ~40k
cells, about 30 min at ~160k; μ = 10¹⁸ is 10× slower. About 4 workers in total for any sweep.

New files only: `slablet/asthenosphere.py`, `slablet/sphere_mesh.py`, `slablet/gplates/ridge_samples.py`,
`slablet/gplates/run_asthenosphere.py`, `slablet/analysis/asthenosphere_check.py`, `tests/test_asthenosphere.py`.
The layer is driven only by the plate history; no coupling to slablet's slabs or plume sites. Commit only these
files, and only when Simon asks.

## 7. Working rules

- Python: `conda run -n pygmt17 python ...`. Maps with PyGMT; matplotlib only for non-map plots.
- No shapely/fiona/geopandas; pygplates or spherical array maths.
- Earth values are a reference, not a target: don't tune until a map "looks right".
- The asthenosphere signal is short-scale; judge it along ridge axes and at points, never in the degree 1–3 headline band.
- Lead each gate with maps, and with anything that contradicts expectations. Explain plainly, then wait for Simon.
- Never list Simon as an author of generated reports. Use they/them for anyone else if a pronoun is needed.

## 8. Known weaknesses to state in every write-up

- No crustal-thickness feedback: the AAD amplitude is expected short by about 400 m.
- Near-critical behaviour: only statements that hold across parameters count.
- Boundary choices (trenches, keels, moving keels, supply) are assumptions with real leverage; report each one's effect.
- Constant μ in the layer; not one self-consistent rheology with the deep flow solver.
- Residual-depth data near Antarctica carry large glacial-isostatic and crust/lithosphere correction errors.
- Fixed-mesh treatment of moving keels is our choice; the paper doesn't state its own (Fig. 9a's pocket behind AUS
  suggests uncovered cells start empty there too).

## 9. Results so far

### Significant result (2026-11-06): with plate history, a Buck-type layer puts the SEIR low at the AAD

**Finding.** A Buck-like layer run through time puts the deepest point of the Southeast Indian Ridge at 122°E, inside
the observed AAD (120–128°E). Values: Δρ 10 kg/m³, h_a 300 km, μ 1.3 × 10¹⁹ Pa s; trench walls; Merdith craton keels;
supply beneath ocean.

| Set-up (Buck-like layer) | Deepest on the SEIR at 0 Ma | AAD contrast* |
|---|---|---|
| Antarctica fixed (anchor 802), 80 → 0 Ma | −510 m at 122°E | −126 m |
| Müller 2022 mantle frame (anchor 0), 80 → 0 Ma | −388 m at 122°E | −89 m |
| Present-day plates held fixed 80 Myr, Antarctica fixed | −167 m at 98°E | +1 m |
| Present-day plates held fixed, mantle frame, μ × 1–16 (fit) | no local low at any μ | — |

\*Mean over 120–128°E minus the mean of 95–110°E and 135–150°E. Observed: −875 m raw, about −475 m without the crustal
term. Amplitude is not the test (it scales with Δρ h_a, and other processes such as subduction may add a component);
location is.

- **The history makes it; the frame only sizes it.** At 30–20 Ma the layer converges on the cells that Australia's
  craton keels uncover at their trailing southern edge near 125–130°E. These start empty (mesosphere fills the gap), the
  flow refills them, and the spot becomes the deepest part of the ridge, staying there as the basin widens.
- **Iaffaldano-like layer** (h_a 110 km, μ 6 × 10¹⁹ Pa s): too thin and stiff to feed any ridge. 80–88 % of the global
  ridge sink is unmet, and every ridge sits on a used-up strip, so nothing is localised.
- **Buoyant plumes** (55 hotspots, plume material 30 kg/m³ lighter, total = Sleep's flux at 30 kg/m³) add swells of a
  few hundred metres. They raise the St Paul/Amsterdam end of the SEIR. They don't move the low.
- **Steady-state fits on present-day plates failed:** viscosity alone, plumes as dense as the layer, and buoyant plumes
  in steady state. The ridge crosses the starving threshold as a whole or from one end.
- **What the result rests on** (untested):
  - uncovered keel cells start empty (Buck's rule);
  - the Merdith craton outlines as full-depth keels;
  - trench walls;
  - Buck's near-critical values;
  - one start (80 Ma, uniform layer).

Figures, code and READMEs: `~/GIT/PlateDriver/show-me/2026-11-06-asthenosphere-antarctica-frame/` (main result),
`2026-11-06-asthenosphere-hotspots/`, `2026-11-06-asthenosphere-aad-fit/`, `2026-11-05-asthenosphere-stage1..3/`.
Solver options added for it: `plume_delta_rho` (plume fraction carried with the layer), balance supply that takes back
an excess of sources. 15 tests in `tests/test_asthenosphere.py`.

### Stress test (2026-11-06): one arbitrary choice at a time

Twenty variants of the mantle-frame result; see `~/GIT/PlateDriver/show-me/2026-11-06-asthenosphere-aad-stress/`.
- **Location holds (deepest at 117–123°E)** under changes to:
  - frame;
  - trench rule (walls or sinks);
  - supply placement;
  - start time (60, 80 or 120 Ma);
  - mesh (20,000 to 80,000 cells);
  - ridge-sink thickness.
- **All continents as keels makes the low much stronger:** −549 m, a Buck AAD1-like basin closed in by the margins and
  the South Tasman Rise. Müller 2019 with continents as keels: deepest at 130°E, −380 m.
- **No low** with almost no keels, or when uncovered cells are refilled to h_a.
- **It needs layer mobility within about a factor of 2 of Buck's.**
  - More mobile: the pocket forms at 20 Ma but is refilled by 0 Ma.
  - Stiffer or thinner: the whole corridor starves.
- **Found on the way:** bulldozing piles the layer without limit into enclosed pockets (Karakoram cell 4,564 km at
  0 Ma in the baseline; far worse with continents as keels). The continent-keel runs cap h at 560 km. Any global use
  needs a physical rule for this.
