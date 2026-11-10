import GUI, { type Controller } from 'lil-gui';
import { Color, Scene, WebGLRenderer, type Camera, type PerspectiveCamera } from 'three';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import '../core/lilgui-theme.css';
import { BoundaryOverlay } from '../core/boundaries';
import { lonLatToVec3 } from '../core/constants';
import { Graticule } from '../core/graticule';
import { OceanSurface } from '../core/oceanSurface';
import {
  createProjectionCamera, createProjectionControls, PROJECTION_ORDER, updateProjectionCameraAspect,
  type ProjectionMode,
} from '../core/projection';
import { wireProjectionToggle } from '../core/projectionToggle';
import { applyChromeLightness, resolveTheme } from '../core/theme';
import { AgeBar } from './ageBar';
import { ArrowLayer, type ArrowLattice } from './arrowLayer';
import { ColorBar, FieldLayer } from './fieldLayer';

/**
 * Earth viewer: slablet's layers on Müller et al. (2022), from the surface to the core-mantle
 * boundary (prep/export.py writes them to public/data/earth/).
 *
 * Panel: one field (grouped Surface, Lithosphere, Upper mantle, Deep mantle) with its colour
 * range and opacity; one set of arrows with spacing and length; the depth used by deep-mantle
 * fields and arrows; overlays (plate boundaries, coastlines, continents and keels, graticule). Projection
 * is the round button at the bottom right; age is the bar along the bottom.
 *
 * URL: ?t=<age Ma> ?field=<id|none> ?arrows=<id|none> ?depth=<km> ?proj=robinson|plateCarree
 *      ?view=<lon>,<lat> (globe camera at load)
 */

const DATA = `${import.meta.env.BASE_URL}data/earth`;

interface Grid { deg: number; nlon: number; nlat: number }
interface FieldInfo {
  id: string; group: string; label: string; units: string; path: string; grid: string; depth: boolean;
  offset: number; scale: number; nodata: number; min: number; max: number; colormap: string; ages: number[];
}
interface VectorInfo { id: string; group: string; label: string; path: string; depth: boolean; ages: number[] }
interface RunInfo {
  label: string;
  ages: number[];
  depths_km: number[];
  grids: Record<string, Grid>;
  land: { path: string; grid: string; ages: number[] };
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
const theme = resolveTheme('newsprint');
applyChromeLightness(theme.lightness);

const view = {
  field: params.get('field') ?? 'dyntopo',
  depth: Number(params.get('depth') ?? 650),
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

const scene = new Scene();
const renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(new Color(theme.page));
document.body.appendChild(renderer.domElement);

let mode: ProjectionMode = (PROJECTION_ORDER.find((m) => m === params.get('proj')) ?? 'globe') as ProjectionMode;
let camera: Camera = createProjectionCamera(mode, innerWidth / innerHeight);
let controls: OrbitControls = createProjectionControls(mode, camera, renderer.domElement);
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
  controls = createProjectionControls(mode, camera, renderer.domElement);
  ocean.setProjection(mode);
  land.setProjection(mode);
  field.setProjection(mode);
  graticule.setProjection(mode);
  coastlines.setCamera(camera, mode);
  boundaries.setCamera(camera, mode);
  arrows.setCamera(camera, mode);
}
wireProjectionToggle(document.getElementById('projection-toggle'), () => mode, applyProjection);

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
const depthIndex = () => {
  const d = run?.depths_km ?? [];
  let best = 0;
  d.forEach((z, i) => { if (Math.abs(z - view.depth) < Math.abs(d[best] - view.depth)) best = i; });
  return best;
};

function applyFieldStyle(): void {
  const f = currentField();
  if (f) {
    // Diverging maps (centred on zero) widen about zero; the others from their lower end.
    const centred = f.min === -f.max;
    const lo = centred ? f.min * view.range : f.min;
    const hi = centred ? f.max * view.range : f.min + (f.max - f.min) * view.range;
    field.setStyle({ ...f, min: lo, max: hi });
    colorBar.set(`${f.label}${f.units ? ` (${f.units})` : ''}`, lo, hi, f.colormap);
  }
  colorBar.visible = f !== null;
  depthCtl?.enable(Boolean(f?.depth || currentVector()?.depth));
}

async function applyAge(age: number): Promise<void> {
  if (!run) return;
  const r = run;
  const at = r.ages.reduce((a, b) => (Math.abs(b - age) < Math.abs(a - age) ? b : a));
  const mine = ++request;
  void boundaries.setAge(at);
  void coastlines.setAge(at);
  const f = currentField(), v = currentVector();
  const fOk = f !== null && f.ages.includes(at), vOk = v !== null && v.ages.includes(at);
  const [fb, lb, vb] = await Promise.all([
    fOk ? fetchBytes(`${f!.path}/${tag(at)}.bin`) : Promise.resolve(null),
    r.land.ages.includes(at) ? fetchBytes(`${r.land.path}/${tag(at)}.bin`) : Promise.resolve(null),
    vOk ? fetchBytes(`${v!.path}/${tag(at)}.bin`) : Promise.resolve(null),
  ]);
  if (mine !== request) return;
  const k = depthIndex();
  if (f && fb) {
    const g = r.grids[f.grid];
    const n = g.nlon * g.nlat;
    field.setGrid(new Uint8Array(fb, f.depth ? k * n : 0, n), g.nlon, g.nlat);
  } else {
    field.setGrid(null, 1, 1);
  }
  const lg = r.grids[r.land.grid];
  land.setGrid(lb ? new Uint8Array(lb) : null, lg.nlon, lg.nlat);
  if (v && vb) {
    const n = 2 * r.arrow_lattice.row_count.reduce((a, b) => a + b, 0);
    arrows.setValues(new Float32Array(vb, v.depth ? 4 * k * n : 0, n));
  } else {
    arrows.setValues(null);
  }
  const z = `${r.depths_km[k]} km`;
  const line = (what: string, info: FieldInfo | VectorInfo | null, ok: boolean) => (info === null ? ''
    : `\n${what}: ${info.label}${info.depth ? ` at ${z}` : ''}${ok ? '' : '  (none at this age)'}`);
  status.textContent = `${r.label}\nAge ${at} Ma${line('Field', f, fOk)}${line('Arrows', v, vOk)}`;
}

// --- panel ----------------------------------------------------------------------------

const gui = new GUI({ title: 'Earth layers' });
gui.domElement.style.zIndex = '10';
const ageBar = new AgeBar((age) => { void applyAge(age); });
let fieldCtl: Controller = gui.add(view, 'field', {}).name('Field');
let depthCtl: Controller | null = null;
const fieldFolder = gui.addFolder('Colour');
fieldFolder.add(view, 'range', 0.25, 4, 0.05).name('Range (x)').onChange(() => applyFieldStyle());
fieldFolder.add(view, 'opacity', 0, 1, 0.05).name('Opacity').onChange((v: number) => { field.opacity = v; });
const arrowFolder = gui.addFolder('Arrows');
let arrowCtl: Controller = arrowFolder.add(view, 'arrows', {}).name('Arrows');
arrowFolder.add(view, 'spacing', 2, 20, 2).name('Spacing (deg)').onChange((v: number) => { arrows.setSpacing(v); });
arrowFolder.add(view, 'length', 0.1, 5, 0.1).name('Length (deg per cm/yr)')
  .onChange((v: number) => { arrows.length = v; });
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
  if (r.depths_km.length) {
    const d = r.depths_km;
    view.depth = d.reduce((a, b) => (Math.abs(b - view.depth) < Math.abs(a - view.depth) ? b : a));
    depthCtl = gui.add(view, 'depth', d[0], d[d.length - 1], d.length > 1 ? d[1] - d[0] : 1)
      .name('Depth (km)').onChange(() => { void applyAge(ageBar.age); });
  }
  arrowCtl.destroy();
  arrowCtl = arrowFolder.add(view, 'arrows', options(r.vectors)).name('Arrows')
    .onChange(() => { applyFieldStyle(); void applyAge(ageBar.age); });
  // lil-gui appends; keep the field and depth at the top, and each folder's dropdown first.
  if (depthCtl) gui.$children.prepend(depthCtl.domElement);
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
