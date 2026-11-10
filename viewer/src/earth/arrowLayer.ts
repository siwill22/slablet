import type { Camera, PerspectiveCamera } from 'three';

import { ThreeProjector } from '../core/boundaries';
import { FlatProjector } from '../core/flatProjector';
import type { ProjectionMode } from '../core/projection';

/** The exported arrow lattice (export_asthenosphere.py arrow_lattice): rows of points. */
export interface ArrowLattice {
  spacing_deg: number;
  row_lat: number[];
  row_count: number[];
}

const DEG = Math.PI / 180;

/**
 * Vector arrows drawn on a 2D canvas over the globe or map.
 *
 * Each frame is one vector per point of the exported lattice (rows every 2 deg, points
 * about 2 deg apart along each row), east and north in km/Myr. `spacing` thins it: every
 * k-th row, and in each kept row about 360 / spacing x cos(lat) points spread evenly, so
 * the arrows stay roughly equal-area at any density. `length` is the arrow's arc in degrees
 * per cm/yr. Each tip lies on the great circle through the base in the vector's
 * direction, so arrows follow the sphere in every projection. On flat maps an arrow
 * whose tip is across the seam is skipped.
 */
export class ArrowLayer {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private projector!: ThreeProjector | FlatProjector;
  private flat: FlatProjector | null = null;
  private lattice: ArrowLattice | null = null;
  private rowStart: number[] = [];
  private picked: Int32Array = new Int32Array(0);
  private values: Float32Array | null = null;
  /** Base point unit vectors and east/north bases, per lattice point. */
  private base = new Float32Array(0);
  private east = new Float32Array(0);
  private north = new Float32Array(0);
  visible = true;
  /** Arc length in degrees per cm/yr. */
  length = 1.0;
  /** Vectors slower than this (cm/yr) are not drawn. */
  minSpeed = 0.05;
  colour = 'rgba(0,0,0,0.85)';
  lineWidth = 1.1;

  constructor(camera: Camera, mode: ProjectionMode) {
    this.canvas = document.createElement('canvas');
    Object.assign(this.canvas.style, { position: 'fixed', left: '0', top: '0', pointerEvents: 'none' });
    document.body.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d')!;
    this.setCamera(camera, mode);
    this.resize();
  }

  setCamera(camera: Camera, mode: ProjectionMode): void {
    if (mode === 'globe') {
      this.projector = new ThreeProjector(camera as PerspectiveCamera);
      this.flat = null;
    } else {
      const flat = new FlatProjector(camera);
      flat.setFlatMode(mode);
      this.projector = flat;
      this.flat = flat;
    }
  }

  resize(): void {
    const dpr = Math.min(devicePixelRatio, 2);
    Object.assign(this.canvas.style, { width: `${innerWidth}px`, height: `${innerHeight}px` });
    this.canvas.width = Math.round(innerWidth * dpr);
    this.canvas.height = Math.round(innerHeight * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  setLattice(lattice: ArrowLattice, spacingDeg: number): void {
    this.lattice = lattice;
    const n = lattice.row_count.reduce((a, b) => a + b, 0);
    this.rowStart = [];
    this.base = new Float32Array(3 * n);
    this.east = new Float32Array(3 * n);
    this.north = new Float32Array(3 * n);
    let p = 0;
    lattice.row_lat.forEach((lat, r) => {
      this.rowStart.push(p);
      const cnt = lattice.row_count[r];
      for (let k = 0; k < cnt; k++, p++) {
        const lon = (-180 + ((k + 0.5) * 360) / cnt) * DEG;
        const la = lat * DEG;
        const cl = Math.cos(la), sl = Math.sin(la), co = Math.cos(lon), so = Math.sin(lon);
        this.base.set([cl * co, cl * so, sl], 3 * p);
        this.east.set([-so, co, 0], 3 * p);
        this.north.set([-sl * co, -sl * so, cl], 3 * p);
      }
    });
    this.setSpacing(spacingDeg);
  }

  /** Thin the lattice to about `spacingDeg` between arrows. */
  setSpacing(spacingDeg: number): void {
    const L = this.lattice;
    if (!L) return;
    const step = Math.max(1, Math.round(spacingDeg / L.spacing_deg));
    const picked: number[] = [];
    for (let r = Math.floor(step / 2); r < L.row_lat.length; r += step) {
      const cnt = L.row_count[r];
      const want = Math.max(1, Math.round((360 / (step * L.spacing_deg)) * Math.cos(L.row_lat[r] * DEG)));
      const n = Math.min(want, cnt);
      for (let j = 0; j < n; j++) picked.push(this.rowStart[r] + Math.floor(((j + 0.5) * cnt) / n));
    }
    this.picked = Int32Array.from(picked);
  }

  /** One frame's east/north (km/Myr) for every lattice point, interleaved; null clears. */
  setValues(values: Float32Array | null): void { this.values = values; }

  draw(): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.restore();
    if (!this.visible || !this.values) return;
    this.projector.update(innerWidth, innerHeight);
    const v = this.values, B = this.base, E = this.east, N = this.north;
    const tip = [0, 0, 0], b = [0, 0, 0];
    ctx.save();
    ctx.strokeStyle = this.colour;
    ctx.fillStyle = this.colour;
    ctx.lineWidth = this.lineWidth;
    ctx.beginPath();
    const heads: number[] = [];
    for (const p of this.picked) {
      const e = v[2 * p], n = v[2 * p + 1];
      const speed = Math.hypot(e, n) / 10;                     // cm/yr
      if (!(speed > this.minSpeed)) continue;
      const th = speed * this.length * DEG;
      const c = Math.cos(th), s = Math.sin(th) / (speed * 10);  // direction = (e E + n N) / |v|
      for (let i = 0; i < 3; i++) {
        b[i] = B[3 * p + i];
        tip[i] = b[i] * c + (e * E[3 * p + i] + n * N[3 * p + i]) * s;
      }
      if (this.flat && this.flat.seamSplit(b, tip)) continue;
      const s0 = this.projector.project(b);
      const s1 = this.projector.project(tip);
      if (!s0 || !s1) continue;
      ctx.moveTo(s0[0], s0[1]);
      ctx.lineTo(s1[0], s1[1]);
      heads.push(s0[0], s0[1], s1[0], s1[1]);
    }
    ctx.stroke();
    ctx.beginPath();
    for (let k = 0; k < heads.length; k += 4) {
      const x0 = heads[k], y0 = heads[k + 1], x1 = heads[k + 2], y1 = heads[k + 3];
      const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy);
      if (len < 2) continue;
      const hl = Math.min(6, 0.4 * len), ux = dx / len, uy = dy / len;
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - hl * ux + 0.5 * hl * uy, y1 - hl * uy - 0.5 * hl * ux);
      ctx.lineTo(x1 - hl * ux - 0.5 * hl * uy, y1 - hl * uy + 0.5 * hl * ux);
      ctx.closePath();
    }
    ctx.fill();
    ctx.restore();
  }
}
