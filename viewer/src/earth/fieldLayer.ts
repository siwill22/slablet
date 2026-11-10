import {
  ClampToEdgeWrapping, DataTexture, LinearFilter, Mesh, NearestFilter, RedFormat, RepeatWrapping,
  RGBAFormat, ShaderMaterial, UnsignedByteType, type BufferGeometry,
} from 'three';

import { LIGHT_DIR, R_SURFACE } from '../core/constants';
import { GEOGRAPHIC_GLSL } from '../core/glsl/geographic';
import {
  createSurfaceGeometry, isFlat, PROJECTION_UNIFORM, type ProjectionMode,
} from '../core/projection';

/**
 * A byte grid draped on the surface through a colour map: one byte per node (Geode Archive
 * layout, nodes, poles included), value = offset + byte * scale, `nodata` not drawn. The
 * colour map runs from `min` to `max`; values beyond take its end colours. Each node is
 * shown as it is (nearest node), never blended with its neighbour.
 *
 * Used twice by the Earth viewer: the chosen field, and the land/keel classes underneath
 * it (two grey levels, ocean as no data). Copied from PlateDriver's asthenosphere viewer.
 */

const VERT = /* glsl */ `
varying vec3 vWorldPos;
void main() {
  vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAG = /* glsl */ `
precision highp float;
${GEOGRAPHIC_GLSL}
uniform sampler2D uValues;
uniform sampler2D uColormap;
uniform vec2 uGridSize;          // nlon, nlat
uniform float uScale;            // value per byte
uniform float uOffset;           // value of byte 0
uniform float uMin;              // value at the bottom of the colour map
uniform float uMax;              // value at the top of the colour map
uniform float uNoData;           // byte not drawn
uniform float uOpacity;
uniform float uProjectionMode;   // 0 globe, 1 Plate Carree, 2 Robinson
uniform vec3 uLightDir;
uniform float uShadeStrength;
varying vec3 vWorldPos;

void main() {
  vec2 ll;
  if (uProjectionMode < 0.5) ll = worldToGeographic(vWorldPos);
  else if (uProjectionMode < 1.5) ll = worldToGeographicFlat(vWorldPos);
  else {
    ll = worldToGeographicRobinson(vWorldPos);
    if (robinsonOffMap(ll)) discard;
  }
  float i = (ll.x + PI) / (2.0 * PI) * uGridSize.x;
  float j = (ll.y + 0.5 * PI) / PI * (uGridSize.y - 1.0);
  vec2 uv = vec2((floor(i + 0.5) + 0.5) / uGridSize.x, (floor(j + 0.5) + 0.5) / uGridSize.y);
  float b = floor(texture2D(uValues, uv).r * 255.0 + 0.5);
  if (abs(b - uNoData) < 0.5) discard;
  float x = clamp((uOffset + b * uScale - uMin) / (uMax - uMin), 0.0, 1.0);
  vec3 col = texture2D(uColormap, vec2(x, 0.5)).rgb;
  vec3 n = normalize(vWorldPos);
  float ndl = dot(n, normalize(uLightDir)) * 0.5 + 0.5;
  float shade = mix(1.0, 0.5 + 0.5 * ndl * ndl, uShadeStrength);
  gl_FragColor = vec4(col * shade, uOpacity);
}
`;

/** Colour maps as stops, interpolated to 256 entries. */
export const COLORMAPS: Record<string, string[]> = {
  // matplotlib viridis at eleven stops
  viridis: ['#440154', '#482475', '#414487', '#355f8d', '#2a788e', '#21918c',
    '#22a884', '#44bf70', '#7ad151', '#bddf26', '#fde725'],
  // GMT polar: negative blue, positive red (holes blue, swells and upwelling red)
  polar: ['#0000ff', '#ffffff', '#ff0000'],
  // matplotlib magma at nine stops
  magma: ['#000004', '#1c1044', '#4f127b', '#812581', '#b5367a', '#e55964', '#fb8761', '#fec287', '#fcfdbf'],
  // seismic: slow (negative) red, fast (positive) blue
  seis: ['#b2182b', '#ef8a62', '#fddbc7', '#ffffff', '#d1e5f0', '#67a9cf', '#2166ac'],
  // seafloor elevation: deep dark blue to shallow pale blue
  ocean: ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#eff3ff'],
  // seafloor age: young red through yellow, green and blue to old violet
  age: ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba', '#5e3c99'],
  // land underneath the field: continent light grey, keel dark grey
  land: ['#c4c4c4', '#7a7a7a'],
};

function hexRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function colormapBytes(name: string): Uint8Array {
  const rgb = (COLORMAPS[name] ?? COLORMAPS.viridis).map(hexRgb);
  const out = new Uint8Array(256 * 4);
  for (let k = 0; k < 256; k++) {
    const x = (k / 255) * (rgb.length - 1);
    const a = Math.min(Math.floor(x), rgb.length - 2);
    const f = x - a;
    for (let c = 0; c < 3; c++) out[4 * k + c] = Math.round(rgb[a][c] * (1 - f) + rgb[a + 1][c] * f);
    out[4 * k + 3] = 255;
  }
  return out;
}

export interface FieldStyle {
  offset: number;
  scale: number;
  min: number;
  max: number;
  nodata: number;
  colormap: string;
}

export class FieldLayer {
  readonly mesh: Mesh;
  private readonly material: ShaderMaterial;
  private readonly cmap: DataTexture;
  private texture: DataTexture | null = null;
  private mode: ProjectionMode;
  private hasData = false;
  private enabled = true;
  /** Radius factor of the draped surface (layers stack by it and by renderOrder). */
  private readonly lift: number;

  constructor(mode: ProjectionMode, renderOrder: number, lift: number) {
    this.mode = mode;
    this.lift = lift;
    this.cmap = new DataTexture(colormapBytes('viridis'), 256, 1, RGBAFormat, UnsignedByteType);
    this.cmap.magFilter = LinearFilter;
    this.cmap.minFilter = LinearFilter;
    this.cmap.wrapS = ClampToEdgeWrapping;
    this.cmap.needsUpdate = true;
    this.material = new ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      uniforms: {
        uValues: { value: null },
        uColormap: { value: this.cmap },
        uGridSize: { value: [720, 361] },
        uScale: { value: 1 },
        uOffset: { value: 0 },
        uMin: { value: 0 },
        uMax: { value: 1 },
        uNoData: { value: 255 },
        uOpacity: { value: 1 },
        uProjectionMode: { value: PROJECTION_UNIFORM[mode] },
        uLightDir: { value: LIGHT_DIR.clone() },
        uShadeStrength: { value: isFlat(mode) ? 0 : 1 },
      },
    });
    this.mesh = new Mesh(this.buildGeometry(mode), this.material);
    this.mesh.renderOrder = renderOrder;
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
  }

  private buildGeometry(mode: ProjectionMode): BufferGeometry {
    return createSurfaceGeometry(mode, R_SURFACE * this.lift);
  }

  setStyle(s: FieldStyle): void {
    const u = this.material.uniforms;
    u.uScale.value = s.scale;
    u.uOffset.value = s.offset;
    u.uMin.value = s.min;
    u.uMax.value = s.max > s.min ? s.max : s.min + 1;
    u.uNoData.value = s.nodata;
    (this.cmap.image.data as Uint8Array).set(colormapBytes(s.colormap));
    this.cmap.needsUpdate = true;
  }

  set opacity(v: number) { this.material.uniforms.uOpacity.value = v; }
  set visible(v: boolean) { this.enabled = v; this.mesh.visible = v && this.hasData; }

  /** One step's bytes, nlon x nlat, longitude fastest; null hides the layer. */
  setGrid(bytes: Uint8Array | null, nlon: number, nlat: number): void {
    this.hasData = bytes !== null;
    this.mesh.visible = this.enabled && this.hasData;
    if (!bytes) return;
    if (!this.texture || this.texture.image.width !== nlon || this.texture.image.height !== nlat) {
      this.texture?.dispose();
      // A copy: later steps are copied into the texture's own array, never into a caller's (cached) bytes.
      this.texture = new DataTexture(bytes.slice(), nlon, nlat, RedFormat, UnsignedByteType);
      this.texture.magFilter = NearestFilter;
      this.texture.minFilter = NearestFilter;
      this.texture.wrapS = RepeatWrapping;
      this.texture.wrapT = ClampToEdgeWrapping;
      this.texture.generateMipmaps = false;
      this.material.uniforms.uValues.value = this.texture;
      this.material.uniforms.uGridSize.value = [nlon, nlat];
    } else {
      (this.texture.image.data as Uint8Array).set(bytes);
    }
    this.texture.needsUpdate = true;
  }

  setProjection(mode: ProjectionMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    this.mesh.geometry.dispose();
    this.mesh.geometry = this.buildGeometry(mode);
    this.material.uniforms.uProjectionMode.value = PROJECTION_UNIFORM[mode];
    this.material.uniforms.uShadeStrength.value = isFlat(mode) ? 0 : 1;
  }
}

/** A horizontal colour bar with a title and end labels. */
export class ColorBar {
  readonly el: HTMLDivElement;
  private readonly title: HTMLDivElement;
  private readonly lo: HTMLSpanElement;
  private readonly hi: HTMLSpanElement;
  private readonly ctx: CanvasRenderingContext2D;

  constructor() {
    this.el = document.createElement('div');
    this.el.className = 'asth-box';
    Object.assign(this.el.style, { top: 'auto', bottom: '64px', width: '260px', display: 'none' });
    this.title = document.createElement('div');
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 1;
    Object.assign(canvas.style, { width: '100%', height: '12px', display: 'block', margin: '4px 0 2px' });
    this.ctx = canvas.getContext('2d')!;
    const ends = document.createElement('div');
    Object.assign(ends.style, { display: 'flex', justifyContent: 'space-between' });
    this.lo = document.createElement('span');
    this.hi = document.createElement('span');
    ends.append(this.lo, this.hi);
    this.el.append(this.title, canvas, ends);
    document.body.appendChild(this.el);
  }

  set(title: string, min: number, max: number, colormap: string): void {
    this.title.textContent = title;
    const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : String(Number(v.toPrecision(3))));
    this.lo.textContent = fmt(min);
    this.hi.textContent = fmt(max);
    const img = this.ctx.createImageData(256, 1);
    img.data.set(colormapBytes(colormap));
    this.ctx.putImageData(img, 0, 0);
  }

  set visible(v: boolean) { this.el.style.display = v ? 'block' : 'none'; }
}
