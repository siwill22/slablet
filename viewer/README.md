# Earth viewer

A local globe viewer for slablet's layers on one reconstruction, Müller et al. (2022) in its optimised mantle frame,
from 300 Ma to the present in 5 Myr steps. It answers a simple need: to look at the deep mantle, the asthenosphere
layer, the lithosphere and the surface together, at the same age and in the same frame. Here we export each model's
output onto common grids and show one field and one set of arrows at a time.

```
cd ~/GIT/slablet
conda run -n pygmt17 python viewer/prep/mantle_5myr.py        # the reference mantle run, saved every 5 Myr (~6 min),
                                                              # then dynamic topography and deep flow back to 300 Ma
conda run -n pygmt17 python viewer/prep/layer_300.py          # the asthenosphere layer from 300 Ma (~70 min)
conda run -n pygmt17 python -m slablet.gplates.seafloor_tracker configs/muller2022_refined.yaml --grid-every 1 \
    --grids-only runs_real/seafloor_ages_muller2022_1myr                  # seafloor ages every Myr (~6 min)
conda run -n pygmt17 python viewer/prep/ipv_ages.py                       # ... in IPV's layout, continents blank
conda run -n pygmt17 python -m slablet.gplates.seafloor_tracker configs/muller2022_refined.yaml --start 340 \
    --rift-points runs_real/cache/ipv/rift_points_muller2022.npz          # where and when it rifts (~1 min)
# IPV on Müller 2022 from 300 Ma (~15-40 min), in ~/GIT/IPV:
#   conda run -n pygmt17 python -c "from mantlememory import MantleSimulation, SimulationConfig; \
#     MantleSimulation(SimulationConfig.from_yaml('mantlememory/examples/config_muller2022_300Ma.yaml')).run()"
# the other model runs (below), one thread each, about 4 at a time:
conda run -n pygmt17 python viewer/prep/layer_300.py 300 --variant weak   # also strong, cap, continent, deep
                                                                          # (~30-45 min each)
conda run -n pygmt17 python viewer/prep/mantle_5myr.py --variant lm10     # also lm20 (~10 min each)
conda run -n pygmt17 python viewer/prep/mantle_5myr.py --variant ref --viscosity sc2006   # each mantle run x
                                                                          # sc2006, four_layer (~3 min each)
cd ~/GIT/IPV && conda run -n pygmt17 python ~/GIT/slablet/viewer/prep/ipv_variant.py front14   # also front33, age
conda run -n pygmt17 python viewer/prep/export.py --workers 4 # all of it -> viewer/public/data/earth/ (~5 min,
                                                              # 12 GB, local only; the runs above take ~35 GB)
cd viewer && npm install && npm run dev                       # http://localhost:5180
node scripts/check.mjs                                        # every field, arrow set, depth, age and model run;
                                                              # fails on page errors
```

## Model runs

Each model has a reference run and a few others, each changing one choice that the literature or our own tests leave
open; the panel's **Model runs** folder picks one run per model, and a field shows the run of each model it comes
from (the status box names them). All runs of a field share one colour scale, so switching runs compares like with
like.

| Model | Reference | Other runs |
| --- | --- | --- |
| Mantle structure (`mantle_5myr.py`) | slabs sink through the lower mantle at 15 km/Myr x (slab feed)^0.5 | 10 and 20 km/Myr, bracketing the 1-2 cm/yr of slab-remnant estimates (van der Meer et al. 2010; Butterworth et al. 2014) |
| Deep flow viscosity (`mantle_5myr.py --viscosity`) | two-layer, 1e21 Pa s above 660 km, 3e22 below | Steinberger & Calderwood (2006); a stiff 100 km lid (1e22) over the two-layer profile |
| Asthenosphere layer (`layer_300.py`) | Buck et al. (2009): 300 km, 1.3e19 Pa s, density contrast 10 kg/m3, craton keels, uniform supply beneath ocean | viscosity 2.6e19 and 6.5e18 Pa s (half and twice the mobility, the range over which the AAD low held); the layer capped at 600 km; every continent a keel (capped too); supply where the deep mantle rises (u_r at 400 km) |
| IPV vulnerability (`ipv_variant.py`) | drip fronts 20 km/Myr, started by the reconstruction's rifting | fronts at 14 and 33 km/Myr (Gernon et al. 2025's range); fronts started by the seafloor age beside margins (IPV's original rule) |

The deep flow, geoid, stresses and dynamic topography come from a mantle run and a viscosity profile together (every
pairing is exported); the summed seafloor elevation also takes the layer run.

Advancing keels bulldoze the layer into enclosed pockets without limit. In the reference this reaches 4750 km in a few
cells (99.9% of the layer stays below 560 km); with every continent a keel the pockets reach 2400 km by 284 Ma and the
run all but stalls, so that run is capped at 600 km (about the base of the upper mantle below a 100 km plate), and
the capped reference is there to compare it with.

## What it shows

- **Surface:** dynamic topography and the geoid from the mantle below 300 km (`slablet.dyntopo`); the asthenosphere
  layer's effect on seafloor depth; seafloor elevation from age (GDH1); and the three added together.
- **Lithosphere:** seafloor age (slablet's seafloor tracker, every Myr; the grids IPV reads); lithosphere thickness
  as the layer sees it (GDH1 under oceans, 150 km under continents, smoothed); IPV's vulnerability and each of its parts (small-scale
  convection, strain rate, plate speed, edge-driven convection, keel drip fronts (Rayleigh-Taylor, "RT"), rifting;
  exactly zero is left blank, which is most continents); IPV's keel remaining.
- **Upper mantle:** the layer's thickness, depth-mean speed and pressure-driven speed, flow divergence (+ thinning),
  shear rate at its top and the shear trend against plate motion, how fast it rises along the plate's base, and how
  far the chain direction turns from plate motion (`slablet/asthenosphere_volcanism.py`). At depths below the
  plate's base (25-400 km): the layer's speed, its shear rate, and the upwelling its shear forces (export.py's
  docstring has the profile; near the top this is PlateDriver's shear-forced upwelling, and at 50 km, away from
  keels, the two rank alike, rank correlation 0.97, with this one about 12% smaller from the profile's curvature).
- **Deep mantle,** at the chosen depth (50-2850 km): the shear-velocity anomaly as modelled and blurred as
  tomography would see it, radial flow, horizontal flow speed, shear stress on horizontal planes (log10), radial
  normal stress, and the geoid at depth; without depth, the basal pile's thickness and core-mantle boundary
  topography.
- **Arrows:** plate velocity, the layer's flow (depth mean, and at the chosen depth below the plate), its
  pressure-driven part, the shear lineation trend (fixed length, towards younging), the chain direction (layer
  minus plate, towards a chain's young end), and the deep flow at the chosen depth.
- **Depth fields** whose values change by an order of magnitude with depth carry their own encoding and colour range
  at each depth, set from four sample ages (0, 100, 200, 300 Ma) of every run; the shear-velocity anomaly keeps +-2%
  at every depth. The exporter reports any field with more than 0.5% of its values outside its encoding.
- **Overlays:** plate boundaries, coastlines, continents and keels (the layer's own masks), graticule.

Projection is the round button at the bottom right; age is the bar along the bottom. On a flat map (Robinson,
Plate Carrée) a sideways drag or sideways scroll (trackpad swipe, shift+wheel) moves the central meridian, one to one
with the pointer at the equator; vertical scroll zooms; vertical drag does nothing, as in petrify's globe host. URL
parameters: `?t=<Ma>` `?field=<id|none>` `?arrows=<id|none>` `?depth=<km>` `?below=<km below the plate>`
`?proj=robinson|plateCarree` `?lon=<central longitude>` `?view=<lon>,<lat>`, and a run per model:
`?mantle=lm10` `?viscosity=sc2006` `?layer=weak` `?ipv=front14`.

## What to keep in mind

- The deep flow is driven by slab buoyancy alone with a free-slip surface, so it does not move with the plates, and
  the layer assumes the mantle beneath it is still. Where the two levels disagree, that is the models, not the
  viewer.
- IPV's vulnerability runs on Müller 2022 here, on the seafloor tracker's ages, but is otherwise IPV's own: strain
  rate comes from rigid plates, so it marks plate boundaries only; its "shear" part is plate speed; its rifting part
  does not switch on (continents are a fixed 120 km thick); and continents are where the age grid has no value (here
  the Müller 2022 continent polygons). Strain rate, plate speed and rifting are divided by fixed references
  (`vulnerability_normalisation: fixed`), so values compare through time.
- Keel drip fronts start where the reconstruction rifts (`keel_front_trigger: rift_points`), not from the seafloor
  age beside a margin: Müller 2022 keeps unsubducted old seafloor (2-3% of the ocean older than 300 Myr, in the
  Arctic, the Tethys and off Antarctica, in its published grids and in slablet's tracker alike) beside many margins,
  which the age rule reads as ancient break-up, stripping almost every keel at once
  (PlateDriver `show-me/2026-11-10-keel-drips-comparison/`).
- The layer starts uniform at 300 Ma, so it shows nothing at 300 Ma. Its first few tens of Myr are spin-up.

## Layout

- `src/core/` — copied (not imported) from PlateDriver's viewer, itself Geode `viewer/src/core` at Geode commit
  `bbaa7ab`. `vendor/petrify/js/` — petrify commit `45768d0`, the one that Geode commit pins.
- `src/earth/` — this viewer: `main.ts` (panel, data), `fieldLayer.ts` (a byte grid through a colour map, and its
  colour bar), `arrowLayer.ts` (great-circle arrows on an equal-area lattice), `ageBar.ts`. The last three began as
  copies of PlateDriver's asthenosphere viewer.
- `prep/` — the runs and the exporter (formats in `export.py`'s docstring); `pygmt17`.
- `scripts/` — `check.mjs` (above), `shoot.mjs <out.png> [url]` (screenshot).
