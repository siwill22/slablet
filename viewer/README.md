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
conda run -n pygmt17 python viewer/prep/export.py             # all of it -> viewer/public/data/earth/ (local only)
cd viewer && npm install && npm run dev                       # http://localhost:5180
node scripts/check.mjs                                        # every field, arrow set, depth and age; fails on page errors
```

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
  at each depth, set from four sample ages (0, 100, 200, 300 Ma); the shear-velocity anomaly keeps +-2% at every depth.
- **Overlays:** plate boundaries, coastlines, continents and keels (the layer's own masks), graticule.

Projection is the round button at the bottom right; age is the bar along the bottom. On a flat map (Robinson,
Plate Carrée) a sideways drag or sideways scroll (trackpad swipe, shift+wheel) moves the central meridian, one to one
with the pointer at the equator; vertical scroll zooms; vertical drag does nothing, as in petrify's globe host. URL
parameters: `?t=<Ma>` `?field=<id|none>` `?arrows=<id|none>` `?depth=<km>` `?below=<km below the plate>`
`?proj=robinson|plateCarree` `?lon=<central longitude>` `?view=<lon>,<lat>`.

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
