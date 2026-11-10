import GUI, { type Controller } from 'lil-gui';
import { Color, Scene, Vector3, WebGLRenderer, type Camera, type PerspectiveCamera } from 'three';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import '../core/lilgui-theme.css';
import { BoundaryOverlay } from '../core/boundaries';
import { lonLatToVec3 } from '../core/constants';
import { Graticule } from '../core/graticule';
import { OceanSurface } from '../core/oceanSurface';
import {
  createProjectionCamera, createProjectionControls, isFlat, lonLatToProjected, PROJECTION_ORDER,
  updateProjectionCameraAspect, type ProjectionMode,
} from '../core/projection';
import { wireProjectionToggle } from '../core/projectionToggle';
import { orientationQuaternion, toRenderFrameRotation, type Quaternion } from '../core/rotation';
import { applyChromeLightness, resolveTheme } from '../core/theme';
import { AgeBar } from './ageBar';
import { ArrowLayer, type ArrowLattice } from './arrowLayer';
import { ColorBar, FieldLayer } from './fieldLayer';

/**
 * Earth viewer: slablet's layers on Müller et al. (2022), from the surface to the core-mantle
 * boundary (prep/export.py writes them to public/data/earth/).
 *
 * Panel: one field (grouped Surface, Lithosphere, Upper mantle, Deep mantle) with its colour
 * range and opacity; one set of arrows with spacing and length; two depths, each used by the fields
 * and arrows of its set: the mantle run's depth below the surface, and the depth below the plate's
 * base within the asthenosphere layer; overlays (plate boundaries, coastlines, continents and keels,
 * graticule). Projection is the round button at the bottom right; age is the bar along the bottom.
 * A depth field may carry its encoding and colour range per depth (lists, one per depth).
 * Model runs: one variant per model (mantle structure, deep flow viscosity, asthenosphere layer,
 * IPV), each changing one choice from the reference; a field or arrow set is read from the
 * folder of the variants of the models it comes from (export.py DEPENDS).
 *
 * On a flat map (Robinson, Plate Carree), as in petrify's globe host: a sideways drag or a sideways
 * scroll (trackpad swipe, shift+wheel) moves the central meridian, one to one with the pointer at
 * the equator; vertical scroll zooms; vertical drag does nothing. The map never slides off-centre.
 *
 * URL: ?t=<age Ma> ?field=<id|none> ?arrows=<id|none> ?depth=<km> ?below=<km below the plate>
 *      ?proj=robinson|plateCarree ?<model id>=<variant id> (e.g. ?viscosity=sc2006)
 *      ?lon=<central longitude of flat maps> ?view=<lon>,<lat> (globe camera at load)
 */

const DATA = `${import.meta.env.BASE_URL}data/earth`;

interface Grid { deg: number; nlon: number; nlat: number }
type PerDepth = number | number[];
interface FieldInfo {
  id: string; group: string; label: string; units: string; path: string; grid: string; depth: string | null;
  offset: PerDepth; scale: PerDepth; nodata: number; min: PerDepth; max: PerDepth; colormap: string;
  models: string[]; ages: Record<string, number[]>;
}
interface VectorInfo {
  id: string; group: string; label: string; path: string; depth: string | null;
  models: string[]; ages: Record<string, number[]>;
}
interface DepthSet { label: string; km: number[] }
interface Model { id: string; label: string; variants: { id: string; label: string }[] }
/** Anything stored per combination of model variants. */
interface PerRun { path: string; models: string[]; ages: Record<string, number[]> }
interface RunInfo {
  label: string;
  ages: number[];
  depth_sets: Record<string, DepthSet>;
  models: Model[];
  grids: Record<string, Grid>;
  land: PerRun & { grid: string };
  fields: FieldInfo[];
  vectors: VectorInfo[];
  arrow_lattice: ArrowLattice;
}

const COAST_STYLE = { other: { stroke: 'rgba(60,60,60,0.75)', width: 0.8, label: 'Coastline' } };
const BOUNDARY_STYLE = {
  subduction: { stroke: '#111111', width: 1.8, label: 'Trench' },
  ridge: { stroke: '#2e9e3e', width: 1.6, label: 'Ridge' },
  transform: { stroke: '#444444', width: 0.9, label: 'Transform' },
  other: { stroke: '#444444', width: 0.9, label: 'Other boundary' },
};
const CACHE_ENTRIES = 48;

const params = new URLSearchParams(location.search);
let centreLon = Number(params.get('lon') ?? 0);
const theme = resolveTheme('newsprint');
applyChromeLightness(theme.lightness);

const view = {
  field: params.get('field') ?? 'dyntopo',
  range: 1.0,
  opacity: 0.9,
  arrows: params.get('arrows') ?? 'plate',
  spacing: 6,
  length: 1.0,
  boundaries: true,
  coastlines: true,
  land: true,
  graticule: true,
};
/** The chosen depth in each depth set (km): the mantle's below the surface, the layer's below the plate. */
const depthView: Record<string, number> = {
  mantle: Number(params.get('depth') ?? 650),
  layer: Number(params.get('below') ?? 50),
};
/** The chosen variant of each model (the first, the reference, unless the URL names another). */
const runView: Record<string, string> = {};

const scene = new Scene();
const renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(new Color(theme.page));
document.body.appendChild(renderer.domElement);

let mode: ProjectionMode = (PROJECTION_ORDER.find((m) => m === params.get('proj')) ?? 'globe') as ProjectionMode;
let camera: Camera = createProjectionCamera(mode, innerWidth / innerHeight);
let controls: OrbitControls = createControls();
const viewParam = params.get('view');
if (viewParam && mode === 'globe') {
  const [lon, lat] = viewParam.split(',').map(Number);
  const [x, y, z] = lonLatToVec3(lon, lat, camera.position.length());
  camera.position.set(x, y, z);
  camera.lookAt(0, 0, 0);
}

const ocean = new OceanSurface(mode);
ocean.applyTheme(theme);
scene.add(ocean.mesh);
const land = new FieldLayer(mode, 2, 1.0005);
land.setStyle({ offset: 0, scale: 1, min: 1, max: 2, nodata: 0, colormap: 'land' });
scene.add(land.mesh);
const field = new FieldLayer(mode, 3, 1.0009);
scene.add(field.mesh);
const graticule = new Graticule(30, 30);
graticule.setProjection(mode);
graticule.applyTheme(theme);
scene.add(graticule.lines);

// Canvases in paint order: coastlines, boundaries, then arrows on top.
const coastlines = new BoundaryOverlay(camera as PerspectiveCamera);
coastlines.setCamera(camera, mode);
const boundaries = new BoundaryOverlay(camera as PerspectiveCamera);
boundaries.setCamera(camera, mode);
const arrows = new ArrowLayer(camera, mode);

const status = document.createElement('div');
status.className = 'asth-box asth-status';
document.body.appendChild(status);
const colorBar = new ColorBar();

function applyProjection(next: ProjectionMode): void {
  if (next === mode) return;
  mode = next;
  controls.dispose();
  camera = createProjectionCamera(mode, innerWidth / innerHeight);
  controls = createControls();
  ocean.setProjection(mode);
  land.setProjection(mode);
  field.setProjection(mode);
  graticule.setProjection(mode);
  coastlines.setCamera(camera, mode);
  boundaries.setCamera(camera, mode);
  arrows.setCamera(camera, mode);
  applyCentreLon();
}

// --- central meridian of flat maps ------------------------------------------------------

/** Zoom only on flat maps: their drag belongs to the central meridian, never to sliding the map. */
function createControls(): OrbitControls {
  const c = createProjectionControls(mode, camera, renderer.domElement);
  if (isFlat(mode)) c.enablePan = false;
  return c;
}

/** Every layer to the current central longitude: a rotation about the pole on flat maps, none on
 *  the globe (whose overlays share the same rotation hooks). */
function applyCentreLon(): void {
  const q: Quaternion = isFlat(mode) ? orientationQuaternion(centreLon, 0, 0) : [0, 0, 0, 1];
  field.centreLon = centreLon;
  land.centreLon = centreLon;
  graticule.setOrientation(q);
  coastlines.setReferenceRotation(q);
  boundaries.setReferenceRotation(q);
  arrows.setFlatRotation(toRenderFrameRotation(q));
}

/** Screen pixels per degree of longitude along the equator, through the live camera (tracks zoom). */
function pxPerDegree(): number {
  const a = new Vector3(...lonLatToProjected(mode, 0, 0)).project(camera);
  const b = new Vector3(...lonLatToProjected(mode, 10, 0)).project(camera);
  return (Math.abs(b.x - a.x) * 0.5 * innerWidth) / 10 || 1;
}

function panLongitude(dxPixels: number): void {
  centreLon = ((centreLon + dxPixels / pxPerDegree() + 540) % 360) - 180;
  applyCentreLon();
}

let dragX: number | null = null;
renderer.domElement.addEventListener('pointerdown', (e) => {
  if (isFlat(mode) && e.button === 0) dragX = e.clientX;
});
addEventListener('pointermove', (e) => {
  if (dragX === null) return;
  panLongitude(-(e.clientX - dragX));
  dragX = e.clientX;
});
addEventListener('pointerup', () => { dragX = null; });
addEventListener('pointercancel', () => { dragX = null; });
// Capture phase on the window, so a sideways scroll never reaches OrbitControls' zoom.
addEventListener('wheel', (e) => {
  if (e.target !== renderer.domElement || !isFlat(mode)) return;
  const dx = e.deltaX !== 0 ? e.deltaX : (e.shiftKey ? e.deltaY : 0);
  if (Math.abs(dx) > Math.abs(e.shiftKey ? 0 : e.deltaY)) {
    e.preventDefault();
    e.stopImmediatePropagation();
    panLongitude(dx);
  }
}, { capture: true, passive: false });
wireProjectionToggle(document.getElementById('projection-toggle'), () => mode, applyProjection);
applyCentreLon();

// --- data -----------------------------------------------------------------------------

let run: RunInfo | null = null;
const cache = new Map<string, Promise<ArrayBuffer | null>>();
let request = 0;

/** Bytes of one exported file, kept for the last CACHE_ENTRIES files (deep fields hold every depth). */
function fetchBytes(path: string): Promise<ArrayBuffer | null> {
  let p = cache.get(path);
  if (p) {
    cache.delete(path);
  } else {
    p = fetch(`${DATA}/${path}`).then((r) => (r.ok ? r.arrayBuffer() : null)).catch(() => null);
  }
  cache.set(path, p);
  while (cache.size > CACHE_ENTRIES) cache.delete(cache.keys().next().value as string);
  return p;
}

const tag = (age: number) => String(age).padStart(4, '0');
const currentField = () => run?.fields.find((f) => f.id === view.field) ?? null;
const currentVector = () => run?.vectors.find((v) => v.id === view.arrows) ?? null;
/** Index of the chosen depth in a depth set (0 for none). */
const depthIndex = (set: string | null) => {
  const d = set ? run?.depth_sets[set]?.km ?? [] : [];
  let best = 0;
  d.forEach((z, i) => { if (Math.abs(z - depthView[set!]) < Math.abs(d[best] - depthView[set!])) best = i; });
  return best;
};
const atDepth = (x: PerDepth, k: number) => (Array.isArray(x) ? x[k] : x);
/** The chosen runs' folder for an item, its ages there, and the file at one age. */
const runFolder = (x: PerRun) => x.models.map((m) => runView[m]).join('-');
const runAges = (x: PerRun) => x.ages[runFolder(x)] ?? [];
const runFile = (x: PerRun, age: number) => {
  const c = runFolder(x);
  return `${x.path}${c ? `/${c}` : ''}/${tag(age)}.bin`;
};

function applyFieldStyle(): void {
  const f = currentField();
  if (f) {
    const k = depthIndex(f.depth);
    const min = atDepth(f.min, k), max = atDepth(f.max, k);
    // Diverging maps (centred on zero) widen about zero; the others from their lower end.
    const centred = min === -max;
    const lo = centred ? min * view.range : min;
    const hi = centred ? max * view.range : min + (max - min) * view.range;
    field.setStyle({ offset: atDepth(f.offset, k), scale: atDepth(f.scale, k), nodata: f.nodata, min: lo, max: hi,
      colormap: f.colormap });
    colorBar.set(`${f.label}${f.units ? ` (${f.units})` : ''}`, lo, hi, f.colormap);
  }
  colorBar.visible = f !== null;
  for (const [set, ctl] of depthCtls) ctl.enable(f?.depth === set || currentVector()?.depth === set);
}

async function applyAge(age: number): Promise<void> {
  if (!run) return;
  const r = run;
  const at = r.ages.reduce((a, b) => (Math.abs(b - age) < Math.abs(a - age) ? b : a));
  const mine = ++request;
  void boundaries.setAge(at);
  void coastlines.setAge(at);
  const f = currentField(), v = currentVector();
  const fOk = f !== null && runAges(f).includes(at), vOk = v !== null && runAges(v).includes(at);
  const [fb, lb, vb] = await Promise.all([
    fOk ? fetchBytes(runFile(f!, at)) : Promise.resolve(null),
    runAges(r.land).includes(at) ? fetchBytes(runFile(r.land, at)) : Promise.resolve(null),
    vOk ? fetchBytes(runFile(v!, at)) : Promise.resolve(null),
  ]);
  if (mine !== request) return;
  applyFieldStyle();
  if (f && fb) {
    const g = r.grids[f.grid];
    const n = g.nlon * g.nlat;
    field.setGrid(new Uint8Array(fb, f.depth ? depthIndex(f.depth) * n : 0, n), g.nlon, g.nlat);
  } else {
    field.setGrid(null, 1, 1);
  }
  const lg = r.grids[r.land.grid];
  land.setGrid(lb ? new Uint8Array(lb) : null, lg.nlon, lg.nlat);
  if (v && vb) {
    const n = 2 * r.arrow_lattice.row_count.reduce((a, b) => a + b, 0);
    arrows.setValues(new Float32Array(vb, v.depth ? 4 * depthIndex(v.depth) * n : 0, n));
  } else {
    arrows.setValues(null);
  }
  const where = (set: string | null) => {
    if (!set) return '';
    const km = r.depth_sets[set].km[depthIndex(set)];
    return set === 'layer' ? ` (${km} km)` : ` at ${km} km`;
  };
  const line = (what: string, info: FieldInfo | VectorInfo | null, ok: boolean) => (info === null ? ''
    : `\n${what}: ${info.label}${where(info.depth)}${ok ? '' : '  (none at this age)'}`);
  // The runs behind what is shown, one line per model used.
  const used = new Set([...(f?.models ?? []), ...(v?.models ?? [])]);
  const runs = r.models.filter((m) => used.has(m.id)).map((m) =>
    `\n${m.label}: ${m.variants.find((x) => x.id === runView[m.id])?.label ?? runView[m.id]}`).join('');
  status.textContent = `${r.label}\nAge ${at} Ma${line('Field', f, fOk)}${line('Arrows', v, vOk)}${runs}`;
}

// --- panel ----------------------------------------------------------------------------

const gui = new GUI({ title: 'Earth layers' });
gui.domElement.style.zIndex = '10';
const ageBar = new AgeBar((age) => { void applyAge(age); });
let fieldCtl: Controller = gui.add(view, 'field', {}).name('Field');
const depthCtls = new Map<string, Controller>();
const fieldFolder = gui.addFolder('Colour');
fieldFolder.add(view, 'range', 0.25, 4, 0.05).name('Range (x)').onChange(() => applyFieldStyle());
fieldFolder.add(view, 'opacity', 0, 1, 0.05).name('Opacity').onChange((v: number) => { field.opacity = v; });
const arrowFolder = gui.addFolder('Arrows');
let arrowCtl: Controller = arrowFolder.add(view, 'arrows', {}).name('Arrows');
arrowFolder.add(view, 'spacing', 2, 10, 2).name('Spacing (deg)').onChange((v: number) => { arrows.setSpacing(v); });
arrowFolder.add(view, 'length', 0.05, 2, 0.05).name('Length (deg per cm/yr)')
  .onChange((v: number) => { arrows.length = v; });
const runFolderGui = gui.addFolder('Model runs');
const overlayFolder = gui.addFolder('Overlays');
overlayFolder.add(view, 'boundaries').name('Plate boundaries').onChange((v: boolean) => { boundaries.visible = v; });
overlayFolder.add(view, 'coastlines').name('Coastlines').onChange((v: boolean) => { coastlines.visible = v; });
overlayFolder.add(view, 'land').name('Continents and keels').onChange((v: boolean) => { land.visible = v; });
overlayFolder.add(view, 'graticule').name('Graticule').onChange((v: boolean) => { graticule.lines.visible = v; });
field.opacity = view.opacity;
arrows.length = view.length;

function options(items: { id: string; group: string; label: string }[]): Record<string, string> {
  return Object.fromEntries([['None', 'none'], ...items.map((x) => [`${x.group}: ${x.label}`, x.id])]);
}

async function boot(): Promise<void> {
  run = await (await fetch(`${DATA}/run.json`)).json() as RunInfo;
  const r = run;
  if (view.field !== 'none' && !r.fields.some((f) => f.id === view.field)) view.field = r.fields[0]?.id ?? 'none';
  if (view.arrows !== 'none' && !r.vectors.some((v) => v.id === view.arrows)) view.arrows = r.vectors[0]?.id ?? 'none';
  fieldCtl.destroy();
  fieldCtl = gui.add(view, 'field', options(r.fields)).name('Field')
    .onChange(() => { applyFieldStyle(); void applyAge(ageBar.age); });
  for (const [set, { label, km: d }] of Object.entries(r.depth_sets)) {
    if (!d.length) continue;
    depthView[set] = d.reduce((a, b) => (Math.abs(b - depthView[set]) < Math.abs(a - depthView[set]) ? b : a));
    const step = d.length > 1 ? Math.min(...d.slice(1).map((z, i) => z - d[i])) : 1;
    depthCtls.set(set, gui.add(depthView, set, d[0], d[d.length - 1], step).name(label)
      .onChange(() => { void applyAge(ageBar.age); }));
  }
  for (const m of r.models) {
    const asked = params.get(m.id);
    runView[m.id] = m.variants.some((x) => x.id === asked) ? asked! : m.variants[0].id;
    runFolderGui.add(runView, m.id, Object.fromEntries(m.variants.map((x) => [x.label, x.id]))).name(m.label)
      .onChange(() => { void applyAge(ageBar.age); });
  }
  arrowCtl.destroy();
  arrowCtl = arrowFolder.add(view, 'arrows', options(r.vectors)).name('Arrows')
    .onChange(() => { applyFieldStyle(); void applyAge(ageBar.age); });
  // lil-gui appends; keep the field and depth at the top, and each folder's dropdown first.
  for (const ctl of [...depthCtls.values()].reverse()) gui.$children.prepend(ctl.domElement);
  gui.$children.prepend(fieldCtl.domElement);
  arrowFolder.$children.prepend(arrowCtl.domElement);
  arrows.setLattice(r.arrow_lattice, view.spacing);
  applyFieldStyle();
  await Promise.all([boundaries.load(`${DATA}/boundaries/manifest.json`, BOUNDARY_STYLE),
    coastlines.load(`${DATA}/coastlines/manifest.json`, COAST_STYLE)]);
  ageBar.setAges(r.ages);
  ageBar.setAge(Number(params.get('t') ?? 0));
  await applyAge(ageBar.age);
}

addEventListener('resize', () => {
  updateProjectionCameraAspect(camera, innerWidth / innerHeight);
  renderer.setSize(innerWidth, innerHeight);
  boundaries.setRect({ x: 0, y: 0, width: innerWidth, height: innerHeight });
  coastlines.setRect({ x: 0, y: 0, width: innerWidth, height: innerHeight });
  arrows.resize();
});

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
  coastlines.draw();
  boundaries.draw();
  arrows.draw();
});

void boot();
