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

### Issues to deal with (list started 2026-11-08)

Planning only: none of these is started. Each would be its own gated stage.

0. **Literature review: how original is this?** A quick search (2026-11-08) shows most pieces exist: the AAD
   mechanism (Buck et al. 2009), pressure-driven flow on plate reconstructions (Stotz et al. 2021; Wang et al. 2023;
   Proc. R. Soc. A 2025), a global plume-fed lubrication layer (Yamamoto et al. 2007), shear-driven volcanism
   (Conrad et al. 2011), depth-rotating anisotropy (Natarov & Conrad 2012; EPSL 2025). Read them and pin down what
   each did. Key question: does any earlier model give the layer its own memory (thickness conserved and carried
   forward, opened seafloor starting empty, keels displacing it)? Ours does: present-day plates held fixed give no
   AAD low, while starts at 60, 80 or 120 Ma all do. In the Stotz/Bunge/Wang models the flow seems to be
   recomputed at each time from an imposed pressure field (mantle circulation model, or sources and sinks).
1. **Volcanism likelihood, and the trend of seamount trails made by migrating asthenosphere.**
   - The present proxy is the shear-forced upwelling w at 50 km into the layer. It ignores two things volcanism needs.
     First, enough total rise (melt depends on how far a parcel climbs, roughly 1 % per km above the solidus, not on
     the rate). Second, rise at shallow enough depth (dry ambient mantle starts melting at about 60–80 km).
   - Add the upwelling forced by flow climbing the sloping base of the plate, w = u · ∇(plate thickness), with
     thickness from the age grids and a cooling model. Ridge-ward refill (e.g. south of Australia) climbs that slope
     and may outweigh the shear term.
   - Proxy in rising order of effort: (a) total w = shear + slope terms; (b) only where the plate is thinner than a
     tunable cutoff; (c) rise accumulated along layer flow paths, volcanism where it passes a threshold.
   - Trend: the axis of the top-of-layer shear s = u_p − 6 v_P, younging towards −s (as in the shear-lineation
     figures). Test against non-hotspot seamount chains and volcanic ridges in the SE Indian Ocean.
   - Eastern Australia: Aus22 places all Cenozoic intraplate volcanism (dated, last eruptions 0–40 Ma) on thin
     Phanerozoic lithosphere east of the cratonic edge. A time-dependent test of keel-edge upwelling and of the
     thin-plate condition, as well as the hotspot tracks (East Australia, Tasmantid, Lord Howe).
2. **Likelihood of complex spreading fabric (microplates, propagators, rotated or "swirly" fabric).** A regional
   tendency in space and time, not individual microplates.
   - Model: for every ridge point in every 5 Myr snapshot, sample both flanks 150–300 km off-axis (avoiding the
     ridge sink) for pressure-flow strength, cross-ridge contrast in h and in pressure flow, along-axis layer flow,
     along-axis gradient and variability of h, and layer-minus-plate velocity with its curl. Show as ridge-coloured
     map sequences.
   - Turn time into space: take each present-day ocean cell back to where it formed (age grid + reconstruction,
     great-circle nearest ridge point) and read the tendency there and then, giving a present-day map of tendency
     at the time of formation.
   - Observations: pseudofaults, extinct ridges, discordant zones and fracture-zone spacing (e.g. Matthews et al.
     2011); known microplates; angle between observed fabric and the reconstruction's spreading direction; small
     plates and ridge jumps in Müller 2022 (not independent: the layer runs on that plate model).
   - Compare within spreading-rate bins (rate is the main control on ridge style), with along-ridge and time
     shuffles as nulls. Read relative contrasts, given the East Pacific Rise sign problem. Mostly Pacific crust
     before about 80 Ma.
3. **Match to seismic azimuthal anisotropy.**
   - Prediction: olivine fast axes align with the direction of shear (simple shear at large strain). In the layer
     u(ζ) = u_p(1 − ζ) + 6 v_P ζ(1 − ζ), ζ = z/h, so the shear du/dz ∝ −u_p + 6 v_P(1 − 2ζ) turns with depth: along
     u_p − 6 v_P at the top, along plate motion at mid-layer, along u_p + 6 v_P at the base. Where pressure flow
     is weak all depths follow plate motion; where it is strong the fast axis rotates with depth. That depth
     rotation is the model's distinctive signature.
   - Data: depth-dependent surface-wave azimuthal anisotropy models (2ψ fast azimuth and amplitude at about
     100–300 km), smoothed comparably (resolution about 500–1000 km). SKS splitting where available (ocean
     islands, OBS), predicted by integrating over depth. The lithosphere above holds frozen-in fabric from
     spreading, so compare only below the plate.
   - Nulls: absolute plate motion direction (the standard explanation) and fossil spreading direction. The model
     only adds something where its shear differs from plate motion, so score it on that subset, with the axial
     (180°-periodic) angle misfit weighted by observed amplitude.
   - Refinements: finite strain accumulated along flow paths (fabric lags where flow changes fast, e.g. near
     ridges); D-Rex style texture later. Amplitude is a weaker test than direction.
   - Frame: shear relative to a resting mesosphere needs the mantle frame; the Antarctica-fixed run is not a
     physical prediction here. Anisotropy could therefore also test frames.
   - Regional model: Aus22 (de Laat et al. 2023, GJI 234: 153–189), waveform tomography with azimuthal anisotropy
     to 660 km, about 150 km node spacing, slices 56–485 km, robust directions flagged by a 90°-rotation test. Its
     reading of the AAD region (their Fig. 7, section 5.3): a strong N–S, spreading-parallel fast direction in the
     asthenosphere beneath the AAD, continuing towards southern Australia and northern Antarctica; east of the AAD
     the fast direction turns ridge-parallel towards the Balleny hotspot, which they read as hot plume material
     flowing along the ridge. Our layer predicts along-axis flow converging on the AAD from both ends (westward east
     of it), so ridge-parallel shear east of the AAD from refill alone, and plate-parallel (N–S) shear at the
     meeting point. To check against the run before claiming: direction and depth (top and base of the layer
     versus mid-layer), and whether a westward-flow signal shows west of the AAD.
   - What anisotropy cannot test (2026-11-10). In Stotz (2025, Fig. 13) the direction misfit stays at 25–33° from
     10¹⁸ to 10²¹ Pa s along the post-glacial rebound trade-off, because the direction pattern hardly changes with
     viscosity. In his model anisotropy therefore cannot constrain the layer, and it says nothing about pressure
     amplitude, which is what dynamic topography needs (issue 12). In ours the pattern does change with mobility
     (the AAD low needs ×0.5–1 of Buck's), so this test may separate values that his cannot.
4. **Geographic mesosphere** (2026-11-06). Let the supply depend on the deep mantle beneath each cell, using slablet's
   refined mantle model (dVs every 10 Myr): less refill above slabs (e.g. under the AAD at 30 Ma, +0.6 % at
   1200–2000 km), more above hot upwellings (EPR, about −1 % at 700–2000 km). Keep the global total balancing the
   sinks; one strength parameter and a depth window. It would address the global misfit that the starved-ridge logic
   puts the deepest hole on the East Pacific Rise.
5. **Ridge-segment predictions** (2026-11-07). In the baseline the AAD axis is thinned (about 210 km against 250 km)
   but never used up, so it predicts a depth anomaly without a cold or thin-crust segment. The stiffer and
   continent-keel runs use the axis up, and so predict cold mantle there, as observed. Along-axis layer flow converges
   on the AAD from both ends (1–4 cm/yr; meeting at about 118°E at 20 Ma and 122°E today), near the Indian–Pacific
   isotopic boundary (~126°E). To do:
   - a passive-tracer run (plume_delta_rho = delta_rho) labelling layer east and west of the AAD;
   - per-segment diagnostics through time (axis h, share fed by mesosphere, along-axis flow, flank upwelling).
   Overlaps with issue 2.
6. **Present-day depth and residual depth.**
   - Compare the model depth anomaly with observed residual depth (bathymetry corrected for plate cooling, sediment
     load and, where known, crustal thickness), at long wavelengths only. Amplitude is not a target; score pattern
     and sign (AAD low, East Pacific Rise, western Pacific) and along-ridge profiles.
   - Through time: the model gives each place a depth-anomaly history, which could be checked against palaeodepth
     or subsidence anomalies at drill sites.
7. **Geochemistry.**
   - Isotopic provenance: a passive tracer labelling layer by where it came from (issue 5) predicts where the
     Indian–Pacific isotopic boundary sits along the SEIR and how it has migrated. Test with samples along flowlines
     on older crust, as well as on axis.
   - Melting conditions: Na8 and Fe8 (degree and pressure of melting) record mantle temperature beneath the ridge.
     Predict them from an axis temperature proxy (below), using segment averages from a global MORB compilation.
   - Plume contribution: in the plume run the plume fraction c at the ridge predicts enriched signatures where
     ridges approach hotspots.
   - Off-axis volcanism (issue 1): provenance of the layer feeding it.
8. **Crustal thickness.**
   - Crust records melt supply when it formed, so it depends on mantle temperature beneath the axis and on spreading
     rate (thin crust at slow rates). Predict it from the axis temperature proxy; control for spreading rate.
   - Time into space as in issue 2: predicted thickness at the time of formation, painted on present-day seafloor.
     Compare with seismic refraction compilations and gravity-derived crustal thickness. Links to issue 6: thin crust
     also makes seafloor deeper.

Shared missing ingredient for issues 7 and 8 (and 1): the model has no temperature. Simplest proxy: the layer is hot
(Δρ = 10 kg/m³ is about 100 K at α = 3 × 10⁻⁵ /K), the mesosphere is not, so axis temperature follows the share of
axis supply drawn from the layer against from the mesosphere (e.g. AAD axis used up in the stiffer and continent-keel
runs → cold: thin crust, high Na8 and low Fe8, as reported for the AAD). Shared tool with issue 2: sampling the ridge each snapshot and
mapping it onto present-day crust by age.

9. **Stotz recomputed with continents.** Stotz (2025, Proc. R. Soc. A 481: 20250085; read 2026-11-08) builds
   asthenosphere flow as Couette (v_plate/2) plus Poiseuille from prescribed sources (25 plumes, radial outflow) and
   sinks (slabs), in a uniform 110 km, 5 × 10¹⁹ Pa s channel with "no distinction between continents and oceans"
   (listed as a limitation). The flow is recomputed at each time with no continuity or thickness memory; ridges are
   passive; anisotropy is taken as the flow direction, strength V/L. Against SL2016svA/svAr at 200 km (median
   axial misfit): Couette alone 18° (smooth) / 26° (rough), Poiseuille 29° / 32°, all combined about 25–33°, random
   44–45°. Fits are good under the Australian plate (Couette and Poiseuille alike), poor under Antarctica. Near the
   EPR the anisotropy departs from Couette for about 500 km (read as ~10 Myr fabric memory); at the MAR it doesn't
   (read as active lateral flow across the ridge).
   - Recompute with continents, using his metric and data so numbers compare directly: our shear axis at the
     matching depth, per plate and globally. The bar to beat is plate motion alone (18° against the smooth model).
   - Our differences to test, one at a time: keels (and all continents) as barriers; ridges as sinks (ridge-ward
     flow, against his cross-ridge flow at the MAR); mass conservation with thickness memory; shear direction at
     depth rather than mean flow direction.
   - Also Stotz et al. (2021): with our layer, compute the drive on the Australian plate through time (basal drag
     plus layer pressure on keel edges) and compare with their no-continent case for the Eocene separation.
   - His random null (44–45°) is what any random axial angle gives, so it carries no information. Plate motion alone
     (18°) is the null that matters, and his full model (25° with equal plumes, smooth model) does not beat it.
     (2026-11-10)
10. **Keels from Aus22.** de Laat et al. (2023, Fig. 15) map cratonic (dVs > 5 % at 110–150 km), intermediate and
    Phanerozoic lithosphere beneath Australia. The southern margins of the Yilgarn and Gawler cratons, the keel
    edges facing the AAD, lack a cratonic root, possibly eroded during Australia–Antarctica rifting. Rerun with this
    outline as keels (rotated with the Australian plate), and with intermediate lithosphere as partial keels, to
    test how much the AAD low rests on the Merdith keel polygons. Caveat: a present-day shape applied back in time.
    Aus22 also images fast material beneath the AAD (the Gurnis et al. slab), a possible second component.
11. **Arctic pile and North Atlantic flow; plume weighting** (2026-11-08). In the 150 Ma viewer runs (baseline and
    plumes alike) a strong pressure-driven flow runs from the Arctic through the Fram Strait gap (about 4 cm/yr),
    south-west past Iceland, and weakly into the Labrador Sea.
    - Cause: the Arctic layer is a long-standing pile, mean h north of 75°N 314 km at 145 Ma, about 400–430 km from
      120 to 60 Ma, 369 km today (h_a 300, global ocean mean 288). From 120 to 60 Ma the Arctic was enclosed
      (continents, keels, trench walls) with almost no ridge sink, while supply balancing the global sinks is spread
      evenly beneath all ocean, so it inflated. Keels pushing layer into the closed basin may add to it (not yet
      separated). The opening of the Norwegian–Greenland Sea and Eurasia Basin (from about 55 Ma) gave it an outlet;
      thickness falls Arctic 360 → Greenland Sea 335 → off Iceland 320 → North Atlantic 305 → Labrador Sea 200–260 km.
    - Labrador Sea: opened 62–33 Ma with an empty layer, boxed in by the Canadian Shield and Greenland keels, refilled
      only through its mouth; North America moving NW in the mantle frame keeps thinning it along the Labrador margin
      (minimum about 200 km). Inflow is weak (0.2–1.4 cm/yr) because the layer there is thin.
    - Plume run: Iceland barely registers (plume fraction 0.11–0.14, h 306–309 km, below the Arctic's 340–370 km).
      Equal shares of Sleep's total give Iceland about 1 Mg/s, about 1 × 10⁶ km³/Myr of layer, against an Arctic
      outflow through the Fram gap of about 7–8 × 10⁶ km³/Myr; the ridge near Iceland consumes about 2 × 10⁶ km³/Myr
      on the spot. Reversing the flow would need Iceland at perhaps 5–10 × its present supply.
    - Read as an artefact of the even-supply rule (one of the forced choices that is not justifiable globally), and
      at odds with the usual picture of North Atlantic layer spreading out from Iceland, including northward.
    - To do, in order, about 15 min each at 4 workers:
      1. a supply rule that does not inflate enclosed basins without ridges (geographic mesosphere, issue 4, or supply
         tied to nearby sinks), then see whether the Arctic-to-Atlantic flow survives;
      2. separate supply from keel push in the Arctic;
      3. weight plumes by a published catalogue (Sleep 1990 per-hotspot values, or Hoggard et al., where Iceland is
         the strongest) instead of equal shares, and see whether Iceland then sets the North Atlantic pattern;
      4. note that hotspots are fixed at present positions in the mantle frame for 150 Myr (matters for the Greenland
         and Labrador opening history, not for present-day flow).
    - Catalogue notes (from Stotz 2025, 2026-11-10). King & Adam (2014) derive plume flux partly from plate speed, so
      plumes under fast plates come out strongest (Hawaii first); for a layer driven by plate motions that is
      circular. Hoggard et al. (2020) use swell volume, independent of plate speed (Iceland first, Hawaii
      comparable), so they are the better choice for step 3. Stotz's supplement merges both into 25 hotspots with
      scale factors (tables S1–S2). His table S3 and fig. S17 give a plume on/off history for 0–120 Ma from large
      igneous province ages, an alternative to the fixed hotspots of step 4.
12. **Dynamic topography from upper-mantle flow** (2026-11-10). Vilacís et al. (2022, Proc. R. Soc. A 478:
    20210764) and Stotz (2025) rest on a sound idea. In a thin channel beneath a rigid plate, the channel pressure p
    sets the topography the plate holds up, the pressure-driven flux q = −(D³/12μ)∇p, and the traction on the plate
    base τ = (D/2)|∇p|. The traction does not depend on viscosity, and integrated along a flow line it gives (D/2) ×
    the pressure drop, so the force on a plate depends only on the dynamic topography amplitude and the channel
    thickness D. Viscosity sets how fast the layer moves, not how hard it pushes. Their execution does not compute
    topography from flow:
    - Topography is an input. Every plume gets 1.4 km (Δp = ρgh with ρ = 3300 kg/m³, about 45 MPa), and velocity
      follows from it. The only topography–flow comparison is the area where pressure flow beats plate drag by
      0.5 cm/yr, which grows under slow plates by construction.
    - Δp/Δx with Δx the distance from the plume is not the gradient of any pressure field. The resulting v ∝ 1/r has
      no balancing sink; on a closed sphere sources and sinks must balance, and 25 plumes plus slab points do not.
    - Mean speed × 2πrD implies about 20 km³/yr (2 × 10⁷ km³/Myr) from each plume: about twice Hawaii's (Sleep, at
      30 kg/m³) and 20 × the Iceland supply in our plume run (issue 11). Iceland, Afar and Tristan get the same.
    - Their plate force, 2.25 × 10¹² N/m, is close to (D/2)Δp = 55 km × 45 MPa ≈ 2.5 × 10¹² N/m. The viscosity and
      the 5 cm/yr cancel, so it restates the assumed 1.4 km.
    - Flow is recomputed from each snapshot (no memory), the channel is uniform (no keels), and the plate-drag part
      depends on the absolute frame (their 133 Ma Tristan area changes about threefold between frames, Fig. 11).

    Ways to compute it:
    - A, the evolving layer (what we have: Buck eq. 8 from h, with plume fraction). Plume topography follows from a
      chosen flux, the right direction of cause, and history is kept. Before a global map:
      - pile-ups: cap_km now exists but is a numerical limit, so dynamic topography near piles reads the cap value;
      - trench walls pile the layer against trenches, possibly the wrong sign there (compare trench_mode 'sink');
      - continents: an empty layer under a keel reads as a fixed low of 1.3 km (0.9 km air-loaded), an artefact.
        Present continental elevation already includes whatever layer lies beneath, so on continents only change
        through time and keel-edge effects carry meaning (this suits issue 14). Continents are air-loaded, not
        water-loaded;
      - lithosphere made from layer subsides, which the cooling law (`slablet/cooling.py`) may already count; avoid
        counting it twice;
      - the mobility conflict with post-glacial rebound (issue 13).
    - B, a no-memory pressure solve (their idea done properly). At each time solve the channel's mass balance on the
      sphere, ∇·[(D³/12μ)∇p] = ∇·(D ū_drag) + sinks − sources, with D varying (keels D → 0) and mean p = 0;
      dynamic topography = p/(Δρ g). Plate drag makes pressure at ridges, trenches and moving keel edges (rigid
      rotation is divergence-free on the sphere, so inside a plate only u_p·∇D acts). Sources must balance sinks.
      One sparse solve on the existing mesh. At 0 Ma it is the failed steady-state fit; run at every step it is the
      no-memory control for A, and where A and B differ is where history matters.
    - C, full Stokes with radial viscosity (extend `slablet/dyntopo.py`): a low-viscosity channel in the viscosity
      profile (the presets have none; two_layer is 10²¹ / 3 × 10²² Pa s), the layer's density as sheets at
      100–400 km, and imposed plate velocities at the surface instead of free slip (Hager & O'Connell 1981). It has
      no keels, but it checks A's local isostasy (for a shallow load at long wavelength the kernel should be near 1)
      and fixes the overlap: deep dynamic topography is cut off at 300 km while the layer spans 100–400 km, so
      300–400 km is in both and the layer's own density is in neither solve.
    - D, coupling to the deep flow. Use the deep flow's horizontal velocity at about 400 km as the layer's base
      velocity (fixed at zero now; stage 4 already uses u_r there for supply). In the thin-channel limit, surface
      dynamic topography ≈ deep normal stress passed through the channel + the channel's own pressure, so the two
      add.
    - E, what needs a numerical solver: keels with lateral viscosity changes, and non-Newtonian plug flow (Semple &
      Lenardic 2018), in which dynamic topography stops growing at high flux. ASPECT on kunanyi could test A–C as
      approximations, not test the Earth.
    - Outputs, whatever the method: dynamic topography and its rate of change, split by cause (layer thickness,
      plume fraction, plate-driven pressure, deep load), and the traction (D/2)∇p on each plate, so a plate-motion
      check comes for free (issue 9, Stotz et al. 2021).
    - Order: B and C's isostasy check first. Both are cheap and show which of A's choices move the result.
      Validation is a separate step; Simon has ideas to explore later.
13. **Post-glacial rebound and layer mobility** (2026-11-10). Post-glacial rebound constrains roughly h³/μ of a
    channel (Paulson & Richards 2009, as plotted in Stotz 2025, Fig. 13A). Scaling Stotz's point (110 km,
    5 × 10¹⁹ Pa s) to 300 km gives about 10²¹ Pa s, a channel roughly 30–80 × less mobile than Buck's (300 km,
    1.3 × 10¹⁹ Pa s). The AAD low needs ×0.5–1 of Buck's, and the Iaffaldano-like layer (about ×0.01) starved every
    ridge. The rebound data sample the mantle beneath former ice sheets (cratons), at a different stress and
    timescale, so the conflict is not decisive. Two readings, to weigh before relying on the AAD result or on issue
    12: (1) the oceanic channel is much weaker than the mantle under cratons (hotter, wetter, thinner plate,
    non-Newtonian), or (2) ridges are fed mostly from below, and lateral layer flow, with the dynamic topography it
    carries, is regional rather than plate-scale. Read the band from Paulson & Richards (2009) directly before
    quoting the factor; the 30–80 × here comes from reading Stotz's figure.
14. **Continental vertical motion from hiatus maps** (2026-11-10; Vilacís et al. 2022, read 2026-11-10). Hiatus maps
    (Hayek et al. 2020, Proc. R. Soc. A 476: 20200390; correction 2021) record, for each geological series since the
    Late Jurassic, where the Americas, Europe, Africa and Australia stood high (missing record) or low. They are
    smoothed to spherical-harmonic degree 15 and say nothing about amplitude. Our list tests vertical motion only on
    the seafloor (issue 6), so these are the continental counterpart.
    - AAD region (Vilacís Fig. 7, proxy elevation curves): eastern Australia high in the Late Cretaceous, during Lord
      Howe Rise (Tasman Sea) spreading; in the Paleocene western Australia (west of 129°E) rises and eastern
      Australia falls, before fast Australia–Antarctica spreading in the Eocene; near continent-wide hiatus at the
      base of the Miocene; mostly none at the base of the Pliocene.
    - Our prediction: the change in depth anomaly through time under non-keel Australia, and keel-edge effects
      (issue 12, A). Sign, not amplitude. Only the baseline (cratons as keels) has layer under continent; the
      all-continents-as-keels runs have none.
    - Their horizontal claim (uplift precedes a spreading-rate change by one series) is not directly testable, since
      plate motions are our input; it becomes a check on the Australian-plate drive in issue 9.
    - Weaknesses: correlations judged by eye on a handful of events, with no null; misses absorbed by "except where
      plate-boundary forces matter"; every region peaks in the Oligocene, which they partly put down to glacial
      sea-level fall; Cretaceous series are about 40 Myr long, so a lag of "one series" means 10–40 Myr; a hiatus
      also records erosion, sea level and tectonics, not only dynamic topography. A qualitative pattern to check
      against, not something to fit.
