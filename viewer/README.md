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
# IPV on Müller 2022 from 300 Ma (~40 min), in ~/GIT/IPV:
#   conda run -n pygmt17 python -c "from mantlememory import MantleSimulation, SimulationConfig; \
#     MantleSimulation(SimulationConfig.from_yaml('mantlememory/examples/config_muller2022_300Ma.yaml')).run()"
conda run -n pygmt17 python viewer/prep/export.py             # all of it -> viewer/public/data/earth/ (local only)
cd viewer && npm install && npm run dev                       # http://localhost:5180
node scripts/check.mjs                                        # every field, arrow set, depth and age; fails on page errors
```

## What it shows

- **Surface:** dynamic topography from the mantle below 300 km (`slablet.dyntopo`); the asthenosphere layer's
  effect on seafloor depth; seafloor elevation from age (GDH1); and the three added together.
- **Lithosphere:** seafloor age (the published Müller 2022 grids); lithosphere thickness as the layer sees it
  (GDH1 under oceans, 150 km under continents, smoothed); IPV's vulnerability and each of its parts (small-scale
  convection, strain rate, plate speed, edge-driven convection, keel drip fronts (Rayleigh-Taylor, "RT"), rifting;
  exactly zero is left blank, which is most continents); IPV's keel remaining.
- **Upper mantle:** the layer's thickness, depth-mean speed, how fast it rises along the plate's base, and how far
  the chain direction turns from plate motion (`slablet/asthenosphere_volcanism.py`).
- **Deep mantle,** at the chosen depth (50-2850 km): the shear-velocity anomaly, radial flow, and the basal pile's
  thickness.
- **Arrows:** plate velocity, the layer's flow, the chain direction (layer minus plate, towards a chain's young end),
  and the deep flow at the chosen depth.
- **Overlays:** plate boundaries, coastlines, continents and keels (the layer's own masks), graticule.

Projection is the round button at the bottom right; age is the bar along the bottom. On a flat map (Robinson,
Plate Carrée) a sideways drag or sideways scroll (trackpad swipe, shift+wheel) moves the central meridian, one to one
with the pointer at the equator; vertical scroll zooms; vertical drag does nothing, as in petrify's globe host. URL
parameters: `?t=<Ma>` `?field=<id|none>` `?arrows=<id|none>` `?depth=<km>` `?proj=robinson|plateCarree`
`?lon=<central longitude>` `?view=<lon>,<lat>`.

## What to keep in mind

- The deep flow is driven by slab buoyancy alone with a free-slip surface, so it does not move with the plates, and
  the layer assumes the mantle beneath it is still. Where the two levels disagree, that is the models, not the
  viewer.
- IPV's vulnerability runs on Müller 2022 here, but is otherwise IPV's own: strain rate comes from rigid plates, so it
  marks plate boundaries only; its "shear" part is plate speed; its rifting part does not switch on (continents are a
  fixed 120 km thick); and continents are where the age grid has no value. Strain rate, plate speed and rifting are
  divided by fixed references (`vulnerability_normalisation: fixed`), so values compare through time.
- On Müller 2022, IPV strips almost every keel in its first step: it reads the age of the seafloor beside a margin as
  the time since break-up, and Müller 2022's seafloor beside Pangea is very old (median 216 Myr at 300 Ma, against
  85-91 Myr on Merdith 2021). So the keel drip fronts have little to strip and look sparse
  (PlateDriver `show-me/2026-11-10-keel-drips-comparison/`).
- The layer starts uniform at 300 Ma, so it shows nothing at 300 Ma. Its first few tens of Myr are spin-up.
- The published 212 Ma age grid is all zeros (in the download and its zip); IPV uses the 211 Ma grid there
  (`runs_real/cache/ipv/agegrids_muller2022/README.txt`).

## Layout

- `src/core/` — copied (not imported) from PlateDriver's viewer, itself Geode `viewer/src/core` at Geode commit
  `bbaa7ab`. `vendor/petrify/js/` — petrify commit `45768d0`, the one that Geode commit pins.
- `src/earth/` — this viewer: `main.ts` (panel, data), `fieldLayer.ts` (a byte grid through a colour map, and its
  colour bar), `arrowLayer.ts` (great-circle arrows on an equal-area lattice), `ageBar.ts`. The last three began as
  copies of PlateDriver's asthenosphere viewer.
- `prep/` — the runs and the exporter (formats in `export.py`'s docstring); `pygmt17`.
- `scripts/` — `check.mjs` (above), `shoot.mjs <out.png> [url]` (screenshot).
