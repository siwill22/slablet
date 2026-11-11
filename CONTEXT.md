# slablet

A kinematic mantle model: slabs from a plate reconstruction sink through the mantle and build piles on the
core–mantle boundary, giving a seismic-velocity anomaly field to compare with tomography and with physical
(dynamic) mantle models.

## Language

### Plate histories

**Plate history**:
The sequence of plate states through time (boundaries, velocities, seafloor ages, continents) that drives the
mantle model and the other response models; supplied either by a synthetic world or by a real reconstruction.
_Avoid_: plate model

**Synthetic world**:
A plate history made by PlateDriver's forward model, for which Earth values are a reference and never a target.

**Real reconstruction**:
A published plate reconstruction taken as the record of Earth's plate history, and the only kind of plate history
that is compared with observations.
_Avoid_: plate model, GPlates model

**Plate step**:
One interval of a plate history as a driver hands it over: plate identity at any point, a rotation vector per
plate, trench and ridge samples, the seafloor age, and the continent and craton at any point.
_Avoid_: snapshot (pygplates' term for one resolved time, without rates)

**Response model**:
A model of how the Earth responds to a plate history: the mantle model, the asthenosphere layer, intraplate
volcanism, crustal deformation. All but crustal deformation run on synthetic worlds as well as real reconstructions.

**Trench flux**:
The slab entering the mantle along a stretch of trench per unit time, set by trench-normal convergence; the one
quantity that seeds slablets, sets slab pull in a synthetic world and enriches the asthenosphere above the slab.

**Slab feedback**:
Slab pull on a synthetic world's plates computed from its slablets, so the mantle model helps drive the plates;
absent in a real reconstruction, where plate motion is prescribed.

**Cooling law**:
The relation between the time since lithosphere last formed or was reset and its temperature (a plate or a
half-space model), from which lithosphere thickness, seafloor depth and heat flow all follow.
_Avoid_: age–depth curve (that is one consequence of a cooling law)

### Comparing with observations

**Null**:
The score a model gets when the property under test is removed (the field rotated in longitude, a uniform rate),
i.e. what chance or geography alone achieves.
_Avoid_: baseline (in prose about scores)

**Ceiling**:
The best score an observation set can support, from scoring it against an independent version of itself (one
tomography model against another, a catalogue against a cross-validated smoothing of itself).

**Skill**:
A score rescaled so that the null reads 0 and the ceiling 1; skills of different observation sets are read side by
side and are only combined with weights chosen explicitly.

### Mantle model

**Slablet**:
A parcel of subducted slab, seeded at a trench and followed as it sinks.
_Avoid_: tracer, particle

**Pile**:
The layer of material on the core–mantle boundary built from slab push and spreading; slablet's stand-in for
the LLSVPs.
_Avoid_: LLSVP (keep that for the seismic observation)

**Stall**:
A hold on slab at the base of the upper mantle before it enters the lower mantle.
_Avoid_: delay, trapping

**Stall patch**:
The slab from one stretch of trench during one window of time, which shares a single stall time, so different
slabs can stall for different times while neighbouring slablets behave as one slab.
_Avoid_: stalling slablet (single slablets never stall on their own)

**Stall fraction**:
The share of stall patches that stall at all; the rest go straight into the lower mantle, as some slabs in
tomography do.

**Stall time**:
How long a stalled patch is held, drawn per patch, either fixed or exponential about a mean (a constant chance of
breaking through per Myr).

**Avalanche**:
Release of all of a stall patch's held slab at one moment, instead of each part after the same delay; a possible
later variant of stalling.
_Avoid_: flush, breakthrough (for the steady case)

### Dynamic topography

**Dynamic topography**:
Vertical deflection of Earth's surface predicted from mantle density anomalies through a mantle flow
calculation.
_Avoid_: mantle topography, DT (in prose)

**Deep dynamic topography**:
The dynamic topography from density anomalies below the cutoff depth (300 km by default), through
the flow calculation; what slablet's mantle field can speak to.

**Short-scale dynamic topography**:
Contributions from shallow processes above the cutoff depth (plume heads, edge-driven convection,
drips), estimated heuristically from plate motions and lithosphere; kept separate from deep dynamic
topography so nothing is counted twice.

**Headline band**:
Spherical-harmonic degrees 1–3, where results are claimed; degrees 4–8 are shown as secondary and
nothing above 8 is reported.

**Geoid scale**:
The single factor that best fits a model's geoid to the observed non-hydrostatic geoid over degrees 2–8
(degree 1 is set by the frame); reported beside a model's dynamic topography to calibrate its nominal amplitude, never applied
silently.

**Residual topography**:
Observed topography after removing the isostatic effects of crust and lithosphere; the observation that
dynamic topography is compared with (e.g. Hoggard et al.). In the oceans it is called residual depth: the depth a
cooling law predicts from the observed seafloor age minus the sediment-corrected depth, so it depends on the law.
_Avoid_: observed dynamic topography

**Reference dynamic topography**:
A present-day dynamic topography field from another model (e.g. Steinberger's flow models driven by
tomography), used as a comparison, not as truth.
_Avoid_: true dynamic topography

### Asthenosphere layer

**Asthenosphere layer**:
A thin, buoyant, low-viscosity layer below the plates (after Buck et al. 2009), dragged by the plates, consumed at
ridges and refilled by flow and a supply from below; computed from the plate history alone, separate from the
mantle model.
_Avoid_: channel (keep that for the thin-layer flow equation), LVZ

**Mesosphere**:
The denser mantle below the asthenosphere layer; fixed in the mantle frame and compensating the layer locally.

**Ridge sink**:
The asthenosphere converted to new lithosphere at a spreading ridge: normal divergence × 100 km per unit ridge length.

**Keel barrier**:
Continental lithosphere thick enough that the layer flows around it, never under it (v1: the craton polygons).
_Avoid_: root (in prose about the model)

**Bulldozing**:
Moving the layer from cells that an advancing keel covers into the open cells ahead of it, so no layer is lost.

**Layer cap**:
The thickness the layer may not exceed (600 km where used, about the base of the upper mantle below a plate), its
excess spread over the supplied cells. Without it, bulldozing piles the layer without limit in pockets that keels
close.

**Starved ridge**:
A stretch of ridge where the layer is used up (h = 0) and the ridge sink is not met; the shortfall is filled by
upwelling mesosphere and is recorded as the starved deficit.

**Fertility**:
How melt-productive the asthenosphere layer is, a concentration carried with the layer: enriched above slabs,
depleted by melting at ridges.

**Mantle domain**:
A compositional population of the asthenosphere layer carried with its flow, such as the Indian and Pacific
mid-ocean-ridge basalt sources whose boundary lies at the AAD.

**Flow-driven volcanism**:
Intraplate volcanism from melting where the asthenosphere layer rises as it flows, up the slope of the lithosphere
base or where shear crosses changes in thickness; its chains run along the layer's velocity relative to the plate.
_Avoid_: shear-driven upwelling (one form of it)

**Volcanism mixture**:
The weighted sum of the rate fields of several volcanism mechanisms (flow-driven, plume, fertility and vulnerability),
whose weights are fitted against observations.

**Supply**:
The influx of asthenosphere from below that balances the sinks (v1: uniform over open cells, reset each step).
_Avoid_: source (ambiguous with plume source)

### Continental keels in IPV

**Metasomatised keel**:
The volatile-rich base of continental lithosphere (about 30 km thick in IPV) that drip fronts strip away; full under
cratons at the start of a run, partial elsewhere.
_Avoid_: keel barrier (the layer's obstacle, a different thing)

**Drip front**:
A front of Rayleigh-Taylor drips that starts at a rifting margin and migrates towards the craton, stripping the
metasomatised keel as it passes (after Gernon et al. 2023, 2025).
_Avoid_: RT instability (the process), keel drip (in labels)

**Rift point**:
A point on a diverging plate boundary of the reconstruction at one time, with its opening rate; a keel point within
reach of one is rifting, and its drip front starts at the rift's onset, before the boundary reaches it.

**Keel material**:
Stripped keel carried in the mantle beneath the young ocean beside the margin it came from, as a tracer on IPV's
particles that raises their fertility.
