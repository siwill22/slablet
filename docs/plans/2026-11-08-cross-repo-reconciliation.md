# Plan: reconciling the physical assumptions of PlateDriver, slablet, IPV and defamation

Recorded on 2026-11-08 from a read-only review of the four repositories, then settled in a grilling session on
2026-11-08 and 2026-11-09 whose decisions are in section 8. The architecture, the scoring rule and v1 (the AAD
region on Müller 2022, built in the stages of decision 20) are decided. Stages 0 and 1 were built on 2026-11-09
(slablet `2500381`, PlateDriver `50743e0`), and stages 2–4 the same day; stage 5 is next. The cases deferred from v1 are listed at the end of
decision 20.

## 1. Aim

Four repositories model parts of the same Earth system. PlateDriver makes synthetic plate histories. slablet, IPV and
defamation each take a plate history and work out how the Earth responds: slablet below the plates (slabs, the basal
pile, the asthenosphere layer, dynamic topography), IPV in the asthenosphere and lithosphere (intraplate volcanism),
and defamation in the continental crust (stretching, tectonothermal age, heat flux). They were written separately,
and where two of them touch the same quantity they often make different physical assumptions about it.

Here we record those conceptual inconsistencies, the constraints that any reconciliation has to respect, and the
observations each repository is compared against. Duplicated code is listed briefly in section 6 because it bears on
where shared code would live, but the subject of this plan is the physics.

## 2. Constraints

Agreed with Simon on 2026-11-08:

1. A model that runs only on synthetic PlateDriver worlds stays available. It uses no observations, and Earth values
   are a reference for it, not a target.
2. A distinct set of tools is aimed at the real world: real reconstructions and fits to observations.
3. slablet and IPV may stay private. Merging repositories is open for discussion if it turns out to be the logical
   step, but it is not a goal in itself.
4. Where several mechanisms could produce the same observable, the aim is a framework that weights the mechanisms and
   compares the mixtures against observations, not a contest with one winner. Intraplate volcanism (case 5) is the
   example that prompted this, and the same framing applies to the other cases of type W below.
5. The methods for comparing models with observations are part of the reconciliation, not something added afterwards
   (section 5).

Standing rules that also apply: new physics leaves the simple formulation selectable and no slower; a failed test
rules out a set-up, not the hypothesis; plumes are never seeded at LLSVP edges; present-day Earth values are a
reference, not a target.

## 3. The four repositories

| | PlateDriver | slablet | IPV (`mantlememory`) | defamation |
|---|---|---|---|---|
| Role | Source of synthetic plate histories | Response below the plates | Response in asthenosphere and lithosphere | Response in continental crust |
| Plate input | Own forward model, plates rebuilt from a raster each step | Real reconstructions (gprm, PMM, files); spin-axis frames refused | Merdith 2021 v1.1, hard-wired | Rotations and continent polygons only, no topologies (ADR-0001) |
| Main state | Plate-ID raster, polygons, slab tracers | Slablets, 5° or 1° mantle grid; asthenosphere on a ~110 km Voronoi mesh | 50,000 two-dimensional particles; 1° grids | HEALPix tracking points (rigidly rotated); deformation grids |
| Time | Elapsed model time, no mapping to Ma | Ma, old to young, 1 Myr | Ma, old to young, 1 Myr | Ma, old to young, 1 Myr (2 Myr Basin & Range) |
| Tests | 2 | 93 (core only) | 0 | 326 |
| GitHub | public (OldDriver) | private | private | private (deformer) |

## 4. Conceptual inconsistencies

Each case is given a type:

- D, decision: a convention where one answer is needed and the choice is ours.
- L, link: each repository has one side of the physics, and connecting their outputs resolves it.
- W, weight: several mechanisms or a free parameter, to be weighted or fitted against observations.
- X, data limit: observations cannot constrain it everywhere, so it stays a sensitivity axis.

### Case 1. The asthenosphere layer (L, with one W)

slablet's Buck et al. (2009) layer and IPV's mantle reservoir model the same layer with different state and opposite
assumptions at ridges. The Buck layer carries thickness h and its flow (`slablet/asthenosphere.py`); IPV carries
composition, its "fertility", on particles and never uses thickness (`mantlememory/core/mantle_reservoir.py`).

The disagreement separates into four parts:

| Part | slablet (Buck layer) | IPV (reservoir) | Type |
|---|---|---|---|
| State | Thickness h, conserved | Fertility on particles; particle count drifts and is remeshed onto a fixed Fibonacci set | L |
| Ridges | Sink: layer turned into 100 km of lithosphere (`lithosphere_km`) | Source: particles spawned with fertility 0.2 | L |
| Trenches, by default | Wall (`trench_mode='wall'`), nothing leaves | Sink of the full strip v⊥·dt (`subduction_entrainment_fraction=1.0`) | D |
| What moves the layer | Plate drag plus pressure-driven flow, K(h) ∝ h³ | Drag only, 0.4 × plate velocity (`mantle_velocity_factor`) | W |

IPV's spawning at ridges fills gaps that divergence opens between particles; it is a numerical device, not a claim
that ridges supply asthenosphere. If fertility were carried as a passive tracer on the Buck layer's flow, the ridge
contradiction and the mass-balance problem would both go away, and IPV's 0.4 × plate velocity would remain as the
simple option. Drag alone with a fixed base gives a depth-averaged 0.5 × plate velocity, so IPV is close to Buck's
drag-only limit. The open physical question is the balance between drag and pressure-driven flow (Buck's Rm), which
is a parameter to fit, for example against residual depth, rather than a choice between codes. IPV's own open issues
note that fertility under continents rises without limit, which a conserved layer would also address.

### Case 2. Lithospheric cooling: one process, five formulas (D)

Every repository models a thermal boundary layer that cools after a reset, with different laws:

- slablet `gplates/lithosphere.py`: 10·√age km, capped at 100 km (reached at 100 Myr); unknown age 60 Myr, while the
  slab code uses 50 Myr (`slab_age_default_myr`).
- IPV `utils/plate_interface.py:oceanic_lithosphere_thickness`: 2.32·√(κt) ≈ 13.0·√age km, capped at 100 km
  (reached at 59 Myr). IPV also puts the peak of small-scale convection at 80 Myr, with none below 50 Myr
  (`ssc_peak_age`, `ssc_window_min`). Small-scale convection is usually what stops thickening, so the cap and the
  convection window disagree within IPV.
- slablet Buck layer: 100 km of lithosphere forms at the ridge axis in the default `ridge_sink_mode='axis'`; an
  `'age'` mode exists.
- PlateDriver: GDH1 age–depth (Stein & Stein 1992), a plate model that flattens; slab pull grows as √(min(age, 80)/80).
- defamation `pipeline/compute_heat_flux.py`: q = q_bg + (q_active − q_bg)·exp(−age/τ) with τ = 200 Myr. Conductive
  cooling of ~125 km lithosphere gives a time constant of ~50 Myr (McKenzie 1978).

One plate-cooling model, with half-space cooling as a flag, could supply thickness, depth and basal heat flux from a
thermal age. One limit applies: defamation finds that tectonothermal age explains little continental heat flux
(R² ≈ 0.07 at fitted parameters, −0.50 at the shipped ones; docs/13), which points to crustal heat production. The
shared law should therefore supply the mantle contribution only, and defamation should keep the crust.

### Case 3. Time since the last thermal reset (L)

Three quantities measure the same thing in different places. In the oceans it is seafloor age, from slablet's
tracker (`gplates/seafloor_tracker.py`), IPV's published Merdith age grids or PlateDriver's tracers. On continents it
is defamation's tectonothermal age, the last step with significant change in log β. At passive margins it is IPV's
time since break-up, read from the age grid with a 15 Myr rift lead (`keel_rift_lead`). No repository combines the
ocean and continent parts into one thermal-age field.

Break-up time is defined twice: by IPV from seafloor age at the margin, and by defamation from the end of divergence
between rigid polygon groups. At a conjugate margin the two must agree, because the end of rifting is the start of
spreading. Comparing them on the same reconstruction is both a link and a test of each.

### Case 4. Continental roots: obstacle, erodible material or stretched lithosphere (L, with one W)

slablet treats keels as fixed polygons that block and bulldoze the asthenosphere layer (150 km continent, 250 km
craton; `keel_mode`, `covered_mode`). IPV treats them as material (120 km continent, 200 km craton) that drip fronts
strip after break-up and deliver under young ocean (`core/continental_keel.py`). defamation thins the crust by β but
tracks no lithosphere. IPV's rift vulnerability term never fires because continental thickness is fixed at 120 km,
which is the quantity defamation could supply.

Lithosphere thickness of L₀/β (uniform stretching), using defamation's β, could feed both IPV's rift term and slablet's
keel barriers. Whether roots are removed at all after break-up, as in Gernon et al. (2023), or stay in place, is a
hypothesis; the stripped fraction is a parameter to fit (W).

### Case 5. Sources of intraplate volcanism (W)

The repositories contain several mechanisms that could produce intraplate volcanism, each in a different place:

- Plumes: PlateDriver seeds seamounts at fixed random hotspots (`self_driving_plates.generate_hotspot_locations`);
  slablet draws plume conduits at random sites or at the pile's highs (`core.plume_sites`, `plume_sources`) and can
  carry plume material in the Buck layer (`plume_delta_rho`); defamation's plan (docs/06) adds heat pulses along
  observed hotspot trails.
- Fertile asthenosphere under vulnerable lithosphere: IPV's P = fertility × vulnerability, deliberately without plumes,
  where vulnerability combines small-scale convection, edge-driven convection, plate speed, strain rate, rifting and
  keel-front instabilities.

These are not exclusive; real volcanism is probably a mixture. The aim is a framework in which each mechanism
produces a rate field (eruptions or volume per unit area per Myr), the mechanisms are combined with weights, and the
weighted mixtures are scored against observations: seamount and volcanic catalogues with ages, the presence or absence
of age progressions along chains, volumes, and where possible geochemistry. Two things are needed first. Each
mechanism has to give a rate in common units; IPV currently normalises its probability map to sum to one, so its
absolute rate is set elsewhere. And the scores have to be comparable across mechanisms and data sets (section 5).

### Case 6. LLSVPs: a mobile pile or fixed provinces (D)

slablet models the LLSVPs as a conserved pile that slabs push around (`slablet/pile.py`). defamation's plan for mantle
contributions (docs/06, not built) raises the mantle heat flux near LLSVP polygons (Torsvik et al. 2014) "roughly
fixed in the mantle frame", and calls a no-net-rotation or palaeomagnetic frame a "mantle reference frame", although
a palaeomagnetic frame is fixed to the spin axis. The standing rules (no plume generation zones at LLSVP edges;
classify a frame by what it is fixed to) side with slablet. If defamation builds this step, it should take the mantle
field from slablet. The same plan proposes edge-driven convection at craton margins, back-arc wedge heating and slab
windows, which would be a third implementation of physics that IPV and slablet already contain.

### Case 7. What is stationary (X)

The repositories assume different things are at rest. PlateDriver's plates feel basal drag against a static mantle
fixed to its grid (`plate_area_tensors`). The Buck layer has a fixed mesosphere at its base (`q_base = 0`). slablet's
lower-mantle slabs drift with the plate-velocity flow (`flow_lower_slabs`), so its mantle is not static. IPV drags its
layer at 0.4 × plate velocity in Merdith 2021's anchor-0 frame, which gprm records as a palaeomagnetic, spin-axis
frame (`gprm/datasets/_frames.py`), so IPV assumes a mantle at rest in a frame that is not tied to the mantle. slablet
would refuse that frame. IPV's plate-speed vulnerability and edge-driven convection damping also depend on the frame.

IPV should run in a mantle frame. However, mantle frames are weakly constrained before ~150–200 Ma, and slablet's own
result is that the verdict on Müller et al. (2022) at 100–200 Ma depends on the mantle model. IPV's 250 Myr span
crosses into that range, so the frame has to stay a sensitivity axis.

### Case 8. The same slab with two geometries and two ages in PlateDriver (D)

Inside PlateDriver, synthetic earthquakes assume a planar dipping slab (depth = distance × tan(dip),
`generate_synthetic_earthquakes`), while slablet's slabs sink vertically, because PlateDriver passes no down-dip
azimuth and slablet's dipping slabs are silently off. Slab pull uses the true age of the subducting seafloor, while
slablet receives no age and would use its 50 Myr default if its age options were on. PlateDriver knows both polarity
and age, so passing them through the trench-sample contract resolves this.

### Case 9. Intraplate deformation from rigid plates (L for continents, gap for oceans)

IPV's "stress" vulnerability is the strain-rate invariant of rigid-plate velocities (`utils/velocity.py`), which is
zero inside plates by construction and so marks plate boundaries only; IPV's open issues say the same. defamation
models the diffuse continental deformation that IPV is trying to represent. Its strain-rate grids could replace the
rigid-plate term on continents. None of the four repositories models intraplate deformation of oceanic lithosphere,
for example in the Indian Ocean.

### Case 10. Two smaller decisions (D)

What counts as a continent: IPV masks continents where the age grid has no value but places keels from polygons;
slablet and defamation use polygons (defamation ADR-0010); PlateDriver uses a craton raster. Polygons are the clearer
definition, since a gap in an age grid is not a continent (Greater India is the example in IPV's open issues).

Which side of a trench is heated: defamation's subduction influence (`pipeline/compute_subduction_influence.py`)
spreads to both sides with no polarity test, while slablet and PlateDriver know polarity. Arc heating belongs on the
overriding plate.

### Case 11. What fills the space a continental root leaves (W)

Found on 2026-11-09 while settling v1. slablet's Buck layer gets its AAD low from the cells that Australia's keels
uncover at 30–20 Ma, which start empty by Buck's rule and fill only through flow and supply
(`docs/plans/2026-11-05-buck-asthenosphere-test.md` §9). IPV assumes the opposite for the same kind of event: keel
material stripped after break-up is delivered into the asthenosphere about 200 km oceanward of the margin
(`ContinentalKeel._deliver`, `MantleReservoir.add_keel_material`). The two models disagree about what occupies the
space a retreating or stripped continental root vacates, and the AAD is where that space opened. Whether the
vacated space starts empty, takes stripped keel material (with its own density and fertility), or takes ambient
asthenosphere is a choice to test against residual depth; it links cases 1, 4 and 5.

In a fully spherical convection model (CitcomS, ASPECT, TERRA) nothing is ever empty: the vacated volume fills at once
with whichever neighbouring material flows in. Buck's empty cell is the thin-layer version of filling from below by the
denser mantle beneath the layer (slablet's mesosphere), with the buoyant layer arriving later by lateral flow. It
yields a deficit only if the asthenosphere is a distinct, more buoyant material (hotter, as Buck's Δρ = 10 kg/m³
implies, or compositionally lighter). If the asthenosphere is merely the low-viscosity part of an isochemical mantle
at the same potential temperature, the mantle that fills the gap is asthenosphere, which is option (c), and the
mechanism gives no AAD low. In such a model the three options are end-members set by the density and viscosity
contrasts of keel, asthenosphere and deeper mantle, by the layer's thickness relative to keel depth, and by the rate
of keel retreat against the rate of lateral inflow. A GPlates-driven ASPECT run on Müller 2022
(`~/GIT/aspect/gplates-kunanyi`) with a keel compositional field and a hotter low-viscosity layer could test which
end-member applies beneath the AAD, given enough resolution there.

## 5. Observations and comparison methods

The repositories compare against different observations with different scoring methods. Constraint 2 places these
tools in the real-world set, and the weighting in case 5 requires their scores to be comparable.

| Repo | Observations | Scores, nulls and weighting |
|---|---|---|
| slablet | Tomography (REVEAL, SEMUCB-WM1, S40RTS, SMEAN, GypsumS, S362ANI, GAP-P4, UU-P07) and vote maps; Müller et al. (2022) OPT1; Slab1.0 dips; residual depth (Holdt et al. 2022, Dunn et al. 2026), Steinberger and V&D 2020 dynamic topography, EGM2008 geoid; hotspot fits for frames | Slow/fast IoU at quantile thresholds, r_2_3, r_1_L, longitude-rotation null, skill against a ceiling (`analysis/sweep.py`); degree correlations; cos(lat) block averages; dynamic topography reported at degrees 1–3 only; parameter search by process-pool sweeps and an evolution strategy (`frame/fit_frame.py`) |
| defamation | IHFC 2024 heat flow; LITHO1.0 Moho and GSRM v2.1 strain rates (benchmark only); Artemieva (2006) TC1 tectonothermal age; Basin & Range reconstructions (McQuarrie & Wernicke 2005, Bahadori et al. 2018); thermochronology loaders (AusGeochem, USGS), comparison not built | R² and weighted statistics; RMS in km against a null model; HEALPix-declustered χ² (`assimilation/misfit.py`); Gaussian-process emulator and Bayesian optimisation over expensive parameters |
| IPV | Gernon et al. (2023, 2025) targets listed in `docs/OPEN_ISSUES.md` §1 (Kaapvaal ~114 Ma, Western Australian keel at 126–130 Ma, Walvis timing, Christmas Island peak 40–60 Myr after break-up) | None in code; no comparison with seamount catalogues |
| PlateDriver | Earth as a reference only: Forsyth–Uyeda plots, ocean-age distributions | Map sequences and gate tables; no fitting, by design |

The methods disagree in four ways. Area weighting uses cos(lat) blocks in slablet and HEALPix declustering in
defamation. Scores are overlaps and correlations in slablet but χ², R² and RMS in defamation, so they cannot be added.
Nulls differ: a longitude rotation in slablet, a no-deformation model in defamation. Searches use sweeps and an
evolution strategy in slablet but an emulator with Bayesian optimisation in defamation.

Several cases call for observations that no repository scores yet: seamount and volcanic catalogues with ages and
age progressions (case 5); oceanic heat flow and lithosphere thickness from seismology (case 2); break-up ages at
conjugate margins (case 3); residual depth for the asthenosphere (case 1; slablet has the references); seismic
anisotropy and geochemistry (listed in slablet's asthenosphere issues).

## 6. Code-level findings that bear on reconciliation

Duplication. There are four reconstruction loaders (slablet `gplates/models.py` is the most capable), at least ten
implementations of velocity from rotations across the repositories, four sets of spherical helpers, two lithosphere
functions (case 2), four trench-sample routes (slablet, PlateDriver's adapter, IPV through gprm, defamation's
tessellated trenches) and four routes to a Geode or petrify globe.

Conventions. Grid registration is cell-centred in slablet, node-registered with 360 and 361 columns joined implicitly
in IPV, gridline with both ±180° columns in defamation, and 0.25° with a duplicated seam in PlateDriver. The Earth
radius is 6371.0 km in three repositories and 6371.009 km (pygplates) in PlateDriver. Velocities are in km/Myr,
mm/yr or cm/yr depending on the repository. Config handling ranges from defamation's strict typed schema to
PlateDriver's flat dictionary; slablet's `MantleParams.from_config` silently ignores unknown keys.

Problems that need attention regardless of any merge, as found on 2026-11-08:

1. PlateDriver's uncommitted `run_SDP.py` calls `seed_slabs(..., time_myr=t)`, which only slablet's uncommitted
   working tree accepts. Committing either side alone breaks PlateDriver's physical runs.
2. slablet holds a large amount of uncommitted work (dyntopo, sphharm, CONTEXT.md, the stall code in `core.py`).
3. Public PlateDriver imports private slablet, so nobody else can run PlateDriver.
4. slablet's run outputs live in `PlateDriver/runs_real`, and the asthenosphere driver scripts sit in PlateDriver's
   git-ignored `show-me/`.
5. Departures from standing preferences: shapely in IPV and PlateDriver; `fit_frame` uses Doubrovine et al. (2012)
   rather than Torsvik et al. (2008); several tools default to 8–10 workers.

### Plate-ID lookups, planar geometry and shapely (checked 2026-11-08)

We timed the plate-ID lookups found across the repositories on 40,000 Fibonacci points against Müller et al. (2022)
topologies at 100 Ma, single process, taking pygplates' PlatePartitioner as the correct answer:

| Method | Used in | Time | Agreement | Geometry |
|---|---|---|---|---|
| `pygplates.PlatePartitioner`, point by point | slablet asthenosphere and lithosphere, gprm | 0.59 s | reference | spherical |
| `ptt.utils.points_in_polygons.find_polygons` | gprm internals, ptt | 0.13 s | 100% | spherical |
| gprm `rasterize_polygons` (rasterio) then nearest cell | PlateDriver, defamation, IPV (geocube) | 0.08 s | 98.9% at 1°, 99.7% at 0.25°, 99.8% at 0.1° | planar in lon/lat, shapely |
| `pygplates.partition_into_plates` | defamation seeding and assimilation, IPV keels | 11.8 s | not checked (output order differs) | spherical |

Building a whole plate-ID grid with `find_polygons` at cell centres took 0.15 s at 1°, 0.37 s at 0.5° and 1.2 s at
0.25° (plus 1.3 s to make the points at 0.25°, which can be done once). PlateDriver's median step is 2.6 s
(`runs/new_mantle4`), so a spherical 0.25° plate-ID grid would add about 45% to a step, and a 0.5° grid about 15%.
The rasterio mismatches lie along plate boundaries at all latitudes, and persist at 0.1°, so they come from
straight lon/lat edges as well as cell size.

Shapely and planar geometry, by repository:

- slablet: none; a test enforces it for the core.
- PlateDriver: the plate rebuild cycle is raster → polygons (xrspatial polygonize, topojson Douglas–Peucker in degrees,
  shapely `make_valid` and `clip`) → rigid rotation (pygplates, spherical) → raster (gprm rasterio). Also shapely
  `contains_xy` for initial seafloor ages, lon/lat cells for tracer thinning, pygmt blockmedian for the age grid, and
  planar slab geometry for earthquakes. Its boundaries, and so its trench samples, inherit the planar simplification.
- IPV: shapely Points to write seamount catalogues (output only); geocube (rasterio, shapely) for craton masks;
  continent mask from age-grid gaps; margins from 8-neighbour grid adjacency.
- defamation: production uses gprm `reconstruct_and_rasterize_polygons` (rasterio, shapely through geopandas) for the
  rigid-group footprints and exclusion polygons; `partition_into_plates` for seeding; geopandas in experiments only.
- gprm: `rasterize_polygons` and `reconstruct_and_rasterize_polygons` are planar (rasterio over geopandas).

## 7. Reasons for caution about merging

The discretisations follow from the science (slablets, a Voronoi layer, two-dimensional particles, HEALPix crustal
points), so a shared mesh would be a mistake. The plate inputs differ by design: defamation uses no topologies, and
PlateDriver's plates have no persistent topology, so a shared layer has to be a contract on samples and grids rather
than a pygplates reconstruction. Bit-identity tests in PlateDriver (`regression_check.py --tol 0`), slablet (golden
test) and defamation (sphere refactor) make refactors of shared helpers expensive; changing PlateDriver's radius alone
would fail its check. Editable installs already caused the break in section 6, item 1. slablet changes daily, so
making IPV and defamation depend on it would import that churn unless shared pieces are released at tagged versions.
slablet's core has a test that it never imports plate-model code, and merging IPV would bring geopandas, shapely and
geocube with it. IPV, with no tests, would need work before it could sit beside defamation's standards.

## 8. Decisions

Settled during the grilling, in order:

1. One set of physics with two drivers (2026-11-08). The response models are the same code whether a synthetic world
   or a real reconstruction drives them; only real reconstructions are compared with observations. The one
   asymmetry is slab feedback: in a synthetic world slablets may drive the plates through slab pull, while in a real
   reconstruction plate motion is prescribed. defamation is real-only, because it finds deformation from rigid
   polygon groups in a rotation file.
2. Intraplate volcanism is shared physics and runs on synthetic worlds too; scoring against catalogues is
   real-only. PlateDriver's hotspot seamounts become the plume mechanism within it (2026-11-08).
3. One trench flux feeds slablets, slab pull and wedge enrichment of the asthenosphere, as a trial (2026-11-08).
4. The plate-history contract (2026-11-08). Each step a driver supplies: plate identity at any point; one rotation
   vector per plate over the step; trench samples with convergence, down-dip azimuth, trench velocity and subducting
   seafloor age; ridge samples with spreading rate; a seafloor-age grid; continent and craton masks. Plates are rigid,
   so deforming networks count as rigid plates in the shared physics and continental deformation comes only from
   defamation. Shared grids are cell-centred. For real reconstructions, plate identity and masks come from exact
   spherical point-in-polygon tests (ptt `find_polygons` is the fastest exact method measured), with no rasterio or
   shapely. A cached lookup grid is not part of the contract unless a benchmark shows it gives identical results on
   the queries actually made and is much faster; a plain nearest-cell grid fails the first test (section 6). In a
   synthetic world the plate-ID raster is PlateDriver's state itself, not a cache.

   Built in stage 1 (2026-11-09): `slablet/plate_history.py` (the contract: `PlateStep`, `check`, `velocity_at`,
   `on_grid`, `cell_centred_grid`), `slablet/gplates/plate_history.py` (`RealPlateHistory`) and PlateDriver's
   `self_driving_plates.plate_step`. Four details settled while building it, to confirm: the continent and craton
   masks are point queries like plate identity (exact for real reconstructions; `on_grid` gives a grid when one is
   wanted), since a mask grid would fail the same identical-results test as a plate-ID grid; an optional
   `trench_lines` field carries the subduction zones as lines, because the asthenosphere layer's walls need lines,
   not samples; the real driver supports only 1 Myr steps, the interval of its cached samples; and a PlateDriver
   world has no craton polygons, so its craton query is its continent query (confirmed 2026-11-09 as a known
   limitation; separating cratons from continents in synthetic worlds is planned, see the end of decision 20). The
   seafloor tracker's cache for
   Müller et al. (2022) holds ages of 200 Myr and more inside and around continents (tracked points carried onto
   continents keep ageing), so consumers mask ages with the continent query.
5. Small numerical inconsistencies are acceptable in synthetic worlds, because there is no truth to be wrong about;
   they are not acceptable in the real-world tools (2026-11-08). On that basis PlateDriver's plate rebuild cycle
   (polygonize, topojson, shapely `make_valid` and `clip`, rasterio) is an accepted exception to the no-shapely
   rule: on two saved steps of `runs/new_mantle4` it differs from an exact spherical rasterisation in 17–36 of 1.04
   million cells (0.002–0.004% of area) and is about 150 times faster (0.01 s against 1.5 s). Still to fix in
   PlateDriver: the shapely `contains_xy` test for initial seafloor ages (replace with an exact spherical test), and,
   at lower priority, Douglas–Peucker simplification in degrees, which shapes the trench samples. The grids
   PlateDriver hands over through the contract must be cell-centred, although its own raster is not.
6. Packaging (2026-11-08; ADR-0001). One private package, slablet, holds four layers: the plate-history contract,
   the response models that run on both drivers (mantle, asthenosphere, and volcanism moved in from IPV), the
   real-reconstruction driver, and the observation tools. Import tests enforce the layers: the physics never imports
   pygplates, the driver or the observation tools. PlateDriver depends only on the contract and the physics.
   defamation stays its own repository. IPV's physics moves in on the contract, starting from the simplest piece,
   and only once its shapely and geocube use is gone; the IPV repository keeps its notebooks and old runs. A
   spherical rasterise function (using ptt) goes into gprm, because PlateDriver and defamation both call gprm's
   planar one.
7. PlateDriver keeps an editable install of slablet, with guard rails (2026-11-08): `gate.py` and
   `regression_check.py` stop when slablet has uncommitted changes, and ordinary runs write a warning into
   `run_info.yaml`; slablet gets a test that calls every function PlateDriver uses, the way PlateDriver calls it, so a
   signature change fails in slablet's own tests (`tests/test_platedriver_api.py`); and a change to that set lands
   in the same session as the matching PlateDriver commit. Built in stage 0: rather than make the private
   `_spherical_kernel` public, PlateDriver keeps its own copies of the three generic sphere helpers it used from
   slablet (`lonlat_to_xyz`, `xyz_to_lonlat`, the spherical Gaussian kernel), which decision 8 needs anyway; a
   PlateDriver test checks them against slablet's whenever slablet is installed.
8. slablet stays private and becomes optional in public PlateDriver (2026-11-08): it is imported only when a mantle
   or slab option needs it, with an error naming the option otherwise, so PlateDriver stays runnable for teaching
   with its simpler slab-pull modes.
9. Scores are reported as skill between a null and a ceiling, for every observation set (v1, 2026-11-09). Skill is
   computed from the native score that suits each set: a likelihood where an error model exists (seamount
   catalogues, heat flow), an overlap or correlation where it does not (tomography). Skills of different sets are
   shown side by side and only summed with explicitly chosen weights. Each set needs a defensible null and ceiling,
   chosen as a scientific decision.
10. v1 starts in the region of the Australian–Antarctic Discordance (2026-11-09), scoring the asthenosphere layer
    (case 1) and the volcanism mixture (case 5) together. Its observations: residual depth made with the shared
    cooling law (case 2), or under several laws side by side; seamount chains mapped from SWOT as connected lines,
    because the catalogues are old, miss many chains that SWOT shows near the AAD, and reduce chains to unconnected
    points; upper-mantle Vs from Aus22 (De Laat et al. 2023); and azimuthal anisotropy, which would compare the
    predicted asthenosphere flow direction (plate drag against pressure-driven flow towards the starved ridge) with
    fast directions. The Aus22 supplement file (`Aus22_DeLaat_etal_2023.nc`) holds isotropic Vs and Vp only, at 0.5°
    and 56–661 km, although the paper and EarthScope's EMC page describe an azimuthally anisotropic model; the EMC
    file could not be checked (HTTP 403). Local alternatives: Debayle & Ricard (2012) with azimuthal anisotropy on a
    2° grid (`~/Data/SeismicTomography/DR2012.model`), and an SKS splitting database (`.../Splitting`). v1 task:
    request the anisotropic model from the authors, and in parallel digitise Figure 7 of De Laat et al. (2023) as a
    labelled fallback. The figure is a raster (1998 × 2165 px JPEG, page 9) with eight depth slices (56–485 km) of
    sticks (orientation = fast direction, length ∝ amplitude, white = robust, grey = unreliable) over an amplitude
    colour field scaled to each slice's printed maximum. Steps: georeference each panel by fitting the oblique
    projection to coastlines and ticks, checked by rendering the file's Vs in the fitted projection against the
    paper's Vs figures; extract sticks by their outlines and principal axes, converting image angle to azimuth with
    the local north direction; take amplitude from both stick length and the inverted colour scale; replot the
    extracted sticks over the figure as the check. Overlapping long sticks south of Australia at 110–260 km are the
    hardest part. By eye, fast directions there run about N–S to NNE, parallel to both present plate motion and the
    old spreading direction, so the discriminating signal is likely confined to deviations near the AAD. Global
    catalogues (Kim & Wessel census, Chase & Wessel 2021 dated Pacific seamounts) follow as the second observation
    set. Local holdings for residual depth: Holdt et al. (2022), Dunn (2026), Stephenson's global residual
    topography, and residual basement depth grids at 1.5° against GDH1, two plate models and a thermal boundary layer
    (`~/Data/ResidualBasementDepth`).
11. Cooling laws (2026-11-09). All are selectable: half-space cooling, GDH1 (Stein & Stein 1992), Parsons & Sclater
    (1977), the Crosby plate and thermal-boundary-layer models of the existing residual basement grids, and models
    with temperature-dependent properties (Richards et al. 2018, Grose & Afonso 2013). GDH1 is the default, and the
    fits are explored across all of them. Within each law, lithosphere thickness, seafloor depth and heat flow come
    from one temperature field, with the lithosphere base defined as one isotherm, and residual depth is made with
    the same law as the model it is compared with.

    Built in stage 2 (2026-11-09): `slablet/cooling.py` with half-space (GDH1's properties), GDH1, PS77, Crosby
    (2007, with its empirical 94 Myr term) and RHCW18 (read from the authors' output, Tp 1333 °C, 130 km, 2500 m);
    the isotherm is 0.9 of the law's mantle temperature, as in IPV. Not built: the Crosby thermal-boundary-layer law
    (its definition is unknown; the old residual grid matches none of the analytic forms) and Grose & Afonso (2013)
    as a law of its own (no output to check a port against; RHCW18 uses their property laws). slablet's lithosphere
    lid and the Buck layer's ridge sink do not use the cooling module yet; that change belongs with stage 3.
    `slablet/analysis/residual_depth.py` makes residual depth under any law from Seton (2020) ages, SRTM15+ v2.7 and
    GlobSed, with a compaction-aware sediment correction and no crustal correction. In the AAD region the pattern is
    the same under all five laws, and the AAD box sits 533–613 m below the regional median; the laws differ mainly in
    overall level (median −106 m under GDH1, −399 m under RHCW18). Figures and numbers:
    PlateDriver `show-me/2026-11-09-cooling-laws-residual-depth/`.
12. v1 runs on Müller et al. (2022) in its optimised mantle frame, slablet's reference reconstruction, and only on it;
    no second reconstruction is assigned (2026-11-09). IPV's physics moves off Merdith 2021 for v1, which settles
    case 7 for the AAD, whose history lies well within the span where mantle frames are constrained.
13. Seafloor ages in v1 (2026-11-09): residual depth uses the observed present-day ages of Seton et al. (2020)
    (`~/Data/AgeGrids/2020`), so the observation stays independent of the model; the model runs on the tracker's
    ages through time. At 0 Ma the two are compared and their differences mapped. The Seton (2020) full spreading-rate
    grid checks the ridge samples.

    Checked in stage 2 (2026-11-09): the tracker's 0 Ma ages on Müller 2022 are within 5 Myr of Seton's over 83% of
    the ocean (89% in the AAD region). 60% of ridge length is within 5 mm/yr of Seton's rate. A further 29% has a
    normal rate under half of Seton's, mostly transform and oblique segments, where the samples carry the normal rate
    by design.
14. In v1, space vacated by a continental root starts empty and fills by flow and supply (Buck's rule, as in the
    existing AAD result) (2026-11-09). Filling it with stripped keel material or ambient asthenosphere (case 11) is
    recorded but not scheduled.
15. The asthenosphere in v1 (2026-11-09) is a distinct, hotter, buoyant layer (Buck's Δρ against the mantle beneath)
    supplied from below; a plume-fed layer is an option; an asthenosphere that is merely the low-viscosity part of an
    isochemical mantle is recorded as the alternative that would remove the AAD mechanism. Test: if the AAD gap was
    filled from below by cooler, denser mantle, Aus22 Vs should be faster beneath the AAD at about 80–200 km than
    beneath the ridge segments either side, where the model's layer is thin. Under a distinct layer, IPV's
    replenishment becomes the composition of the layer's supply.

    Plumes are not the only supply. The supply may vary with the state of the deep mantle beneath: more above broad
    deep upwellings (the model's pile, not fixed LLSVP outlines), less or none above cold downwellings. The hypothesis
    to test is that the AAD lies over a lower-mantle slab that starves the layer, while the East Pacific Rise lies
    over an active deep upwelling that is more than a plume. slablet's flow solver already gives radial velocity u_r
    (cm/yr) at every depth for any saved age (`slablet.dyntopo`), driven by the field below 300 km, from which the
    supply beneath the layer can be weighted. A check of `runs_real/muller2022_refined` at 0 Ma: beneath the East
    Pacific Rise (100–115°W, 20°S–10°N) dVs is negative at all depths, reaching −4.2% at 2750 km, as the hypothesis
    needs; beneath the AAD (120–128°E, 45–55°S) there is no distinct slab at 450–1050 km (−0.2 to −0.3%, near the
    layer mean), and the fast material lies below 1950 km (+4.5% at 2750 km) and is as strong beneath 140–150°E east of
    the AAD (+5.3%). Deep-flow-weighted supply from this field would therefore starve the AAD and the ridge east of it
    alike. Gurnis et al. (1998) placed their slab near the transition zone; whether tomography shows one there (Aus22
    reaches 661 km) and whether slablet can produce one are prerequisites for the AAD half of the test. Supply that
    varies at the AAD's scale (degree ~20) is beyond the degrees where dynamic-topography claims are made; the
    AAD–EPR contrast is long-wavelength and safe.
16. The layer's supply is a weighted mixture of three components (2026-11-09): uniform (the current baseline);
    plume-fed at plume sites, the same plumes as the volcanism mixture's plume mechanism; and deep upwelling,
    proportional to max(u_r, 0) at the base of the layer (about 400 km) from slablet's flow solver at each saved age.
    The total still balances the sinks each step; only its distribution changes, and the weights are fitted against
    residual depth as skills. A downwelling withholds supply by default; draining the layer where u_r < 0 is an
    option. The same weighting framework serves the volcanism mixture (case 5).
17. IPV's fertility rides on the Buck layer (2026-11-09). It moves with the layer's flow (0.4 × plate velocity stays
    as the simple mode); IPV's ridge spawning goes, because Buck ridges are sinks; trench enrichment comes from the
    shared trench flux; trenches stay walls in v1. Numerically, fertility is a concentration per cell carried upwind
    with the layer flux, like the plume fraction, so it vanishes where the layer does. Before it is trusted it is
    benchmarked against IPV-style particles moved by the layer's cell velocities (a fertility front carried round a
    rigid rotation and past a starved ridge), to measure the smearing against the scales the observations resolve.
18. Volcanism in v1 at the AAD (2026-11-09). The priority is volcanism tied to the general direction of
    asthenospheric flow over a wide area, not plumes. Main mechanism: flow-driven melting in the Buck layer, where the
    layer rises as it flows, up the slope of the lithosphere base (w = −u_layer · ∇L, with L from the cooling law) and
    where shear crosses thickness changes (Conrad-type shear-driven upwelling), with melt production modulated by the
    fertility carried in the layer. Such a source moves with the layer while the plate moves over it, so its chain
    runs along u_layer − u_plate; with drag alone that points against plate motion, and pressure-driven flow towards a
    starved ridge rotates it, so chain azimuths measure the balance of case 1. Plumes remain a component with low
    priority, their weight free to reach zero. Chain azimuths are scored against three predicted directions: the
    layer–plate relative flow, spreading fabric and fracture zones (gprm `SeafloorFabric`, reconstructed with the
    plates), and absolute plate motion, so that fabric-parallel chains are not taken as flow evidence. The
    Indian–Pacific mantle isotopic boundary at the AAD and its westward migration (Pyle et al. 1992; Christie et al.
    2005) is added as an observation, with two compositional domains carried in the layer like fertility.
19. Seamount chains from SWOT (2026-11-09; may be deferred beyond v1). Extraction is hybrid: automatic lineament
    candidates on a SWOT gravity or vertical-gravity-gradient grid of the AAD region, then manual curation with every
    decision recorded. Orientation is scored with axial statistics (doubled angles) against the three predicted
    directions, with uniform random orientations as the null and automatic-against-curated agreement as the
    ceiling; chain density is scored against the model's rate field, per chain. SWOT gives no ages, so v1 tests
    geometry, not age progression. No SWOT grids are on disk yet. If the chains are deferred, v1 still tests the flow
    through residual depth, Aus22 Vs, anisotropy and the isotopic boundary, while flow-driven volcanism is computed but
    unscored; the existing catalogues (Kim & Wessel, SIO) could score chain density in the region as a stopgap,
    though not orientation.
20. v1 is built in gated stages, each ending with the map sequence and figures in `show-me/` before Simon decides
    (2026-11-09):
    - Stage 0, housekeeping: fix the `seed_slabs` seam and commit slablet's pending work; add the guard rails;
      make slablet optional in PlateDriver (decisions 7, 8).
    - Stage 1, the contract in slablet: exact plate IDs (ptt), masks, trench samples with polarity, cell-centred
      grids; PlateDriver's adapter emits it; a test of every function PlateDriver calls (decisions 4, 7).
    - Stage 2, cooling laws (GDH1 default, all selectable) and residual depth for the AAD from Seton (2020) ages,
      with the map of tracker against Seton ages (decisions 11, 13).
    - Stage 3, reproduce the Buck AAD baseline on the contract, and the Aus22 Vs test beneath the AAD
      (decisions 14, 15). The existing result must survive the move before any new physics is added.
    - Stage 4, the supply mixture: uniform, plume, deep upwelling from u_r through time (decision 16).
    - Stage 5, fertility and the Indian–Pacific mantle domains on the layer: the upwind-against-particles benchmark,
      then the isotopic-boundary observation (decisions 17, 18).
    - Stage 6, flow-driven volcanism, and the orientation test once SWOT chains exist (decisions 18, 19).
    - Alongside: anisotropy, by request to the authors and by digitising Figure 7 as a fallback (decision 10).

    Stage 3 built (2026-11-09): `slablet/asthenosphere_plates.py` drives the layer from any plate step
    (`forcing_from_step`, `run_layer`); `gplates/run_asthenosphere.PlateSetting` is a thin wrapper over it. Plate
    identity where polygons overlap now takes the highest plate ID, as pygplates' PlatePartitioner does; cells in
    gaps between plates have no plate motion. The mantle-frame AAD baseline is bit-identical to the 2026-11-06 run
    (deepest SEIR point −387 m at 120.2°E); the Antarctica-fixed run differs by at most 0.13 km of layer thickness.
    Vs test (decision 15): at 80–200 km the AAD is faster than the ridge east of it in Aus22 and SL2013sv (4.3%,
    5.2%) and faster than the ridge west of it in SL2013sv only (2.0%; Aus22 0.1%); the fast anomaly is gone by
    300 km. Along 88–158°E, Vs correlates with the layer's thinness at r = 0.74 and 0.77 (ceiling 0.87; skill 0.88
    and 0.92 against a longitude-shift null), but the profile is one broad east–west swing, so the evidence rests on
    few independent samples and does not separate the layer from other causes of a warm ridge east of the AAD.
    Figures: PlateDriver `show-me/2026-11-09-asthenosphere-on-contract/`.

    Stage 4 built (2026-11-09): `slablet/asthenosphere_supply.py` (uniform, plume and deep parts, mixed by weight;
    `Forcing.supply_share`; `run_layer(share_at=...)`), with u_r at 400 km and plume sites from the mantle run
    `runs_real/muller2022_refined` (flow solved every 10 Myr, 0–80 Ma). Six mixtures beside the uniform baseline:
    the SEIR low stays at 120–122°E in all, and the AAD contrast grows from −89 to −121 m with deep supply; the SEIR
    profile skill is 0.98–0.99 for all, so it cannot rank them. The EPR is tested as Buck et al. (2009) tested it,
    by its step below the slower Pacific-Antarctic Ridge (observed residual depth −354 m; Small & Danyushevsky 2003
    about −400 m): uniform supply gives −393 m, and deep or plume supply weakens it to −125 to −260 m by feeding the
    EPR. Ridge depth therefore gives no support for extra supply beneath the EPR in this mantle run, and the deep
    part of decision 16 worsens the fit. (An EPR – AAD contrast scored first mixes two anomalies and the AAD's
    crustal part, and is not a test.) What fails is the layer on the sphere, not the supply: the model EPR is used
    up beneath 21% of its ridge samples, the model PAR deepens steadily towards the EPR where the observed one is
    flat, the AAD is not deep relative to the PAR (+92 m model, −899 m observed), and the all-ridge skill is
    0.06–0.07. Buck's properties were set in closed 10,000 km boxes; the wider real Pacific and the western trench
    walls need a more mobile layer (critical Rm h_a / W ≈ 7). Not built: draining above downwellings, buoyant plume
    material in the mixture, catalogue plumes, a finer weight fit. Figures: PlateDriver
    `show-me/2026-11-09-asthenosphere-supply-mixture/`.

    Not in v1, each for its own session: continental deformation from defamation (case 9); defamation's trench side and
    IPV's continent definition (case 10); PlateDriver's `contains_xy` fix; the spherical rasterise function in gprm;
    the global catalogues; moving the rest of IPV beyond fertility into slablet; the volcanism mixture and the
    asthenosphere layer in synthetic worlds; cratons distinct from continents in synthetic worlds (2026-11-09: a
    PlateDriver world's craton raster marks all continental lithosphere, so keels cannot yet differ from continents
    there).

## 9. Proposed order for the grilling

1. Architecture: what belongs to the synthetic model, what to the real-world tools, what physics both share, and where
   shared code lives (gprm, a new package, slablet, or nowhere).
2. The observation and scoring framework (section 5), because the weighting in cases 1, 4 and 5 depends on it.
3. The thermal state: cases 2, 3 and 4, aiming at one thermal-age and lithosphere state.
4. The asthenosphere and volcanism: cases 1 and 5.
5. The deep mantle and frames: cases 6 and 7.
6. The small decisions: cases 8, 9 and 10.
