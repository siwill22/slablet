"""Export slablet's Earth layers on Müller et al. (2022) for the Earth viewer (viewer/index.html).

    conda run -n pygmt17 python viewer/prep/export.py [--ages 0 5 ...]          # from ~/GIT/slablet

Reads (run these first; a missing input leaves its layers out at those ages):
  runs_real/muller2022_refined_5myr/   viewer/prep/mantle_5myr.py: mantle_temp_<age>Ma.nc (shear-velocity anomaly,
                                       basal pile) and dyntopo_<age>Ma.nc (dynamic topography, deep flow)
  runs_real/layer_300Ma.npz            viewer/prep/layer_300.py: the asthenosphere layer and flow-driven volcanism
  runs_real/ipv_muller2022_300Ma_rift/viewer_data/fields_<age>Ma.nc
                                       IPV (mantlememory) on Müller 2022, keel drip fronts started by rifting:
                                       vulnerability and its parts
  runs_real/cache/ipv/agegrids_muller2022_tracker/   seafloor ages from slablet's tracker, every Myr, 1 deg nodes,
                                       continents blank (viewer/prep/ipv_ages.py); IPV reads the same grids
and writes viewer/public/data/earth/ (gitignored: local only):

  run.json                    ages, depth sets, grids, fields, vectors, arrow lattice (read by src/earth/main.ts)
  fields/<id>/<AAAA>.bin      uint8 per grid node, value = offset + byte x scale, 255 no data (IPV's vulnerability
                              fields: also where exactly 0, mostly continents); a field with depth
                              holds every depth of its set, shallowest first (depth-major).  AAAA = age in Ma.
  land/<AAAA>.bin             uint8 on the 0.5 deg grid: 0 ocean, 1 continent, 2 keel (the layer's own masks)
  vectors/<id>/<AAAA>.bin     float32, east and north (km/Myr) interleaved at every arrow-lattice point; with depth,
                              every depth, shallowest first
  boundaries/                 petrify BoundarySeries: ridges, trenches (with polarity), transforms, other
  coastlines/                 the same format: Müller 2022's coastlines (Merdith et al.) at each age, as 'other' lines

Grids use the Geode Archive layout: longitude fastest, -180 + i x 360 / nlon (no repeated seam), then latitude
ascending, -90 + j x 180 / (nlat - 1) (both poles).  Two grids: 0.5 deg (layer fields, from the 40,000-cell mesh by
the cell each node lies in) and 1 deg (seafloor ages, mantle, dynamic topography, IPV); the summed seafloor elevation
takes the ages to 0.5 deg by bilinear interpolation between the 1 deg nodes.  The mantle's 1 deg
cell-centred grids go to the nodes by bilinear interpolation, wrapping at the seam, with each pole the mean of the
row next to it.  Arrows sit on rows every 2 deg, about 2 deg apart along each row (equal area); each takes the
nearest mesh cell or mantle cell and is expressed in the arrow point's own east and north.  All in Müller 2022's
optimised mantle frame (anchor plate 0).

Two depth sets: 'mantle', the mantle run's depths (50-2850 km below the surface), and 'layer', depths below the
plate's base within the asthenosphere layer (LAYER_Z).  Within the layer the flow is plate drag over a fixed base
plus pressure-driven flow, u(z) = u_p (1 - z/h) + 6 v_P (z/h)(1 - z/h) (u_p plate velocity, v_P the depth-mean
pressure-driven velocity, recomputed from the saved thickness and the step's forcing, as the run had them).  The
upwelling that profile forces at depth z, with w = 0 at the plate's base, is w(z) = div of the flux above z:
    w(z) = u_p . grad(z - z^2 / 2h) + div(6 v_P (z^2 / 2h - z^3 / 3h^2))        (+ up; rigid plates: div u_p = 0)
which near the top is PlateDriver's shear-forced upwelling (z^2 / 2)(-div s / h + s . grad h / h^2), s = u_p - 6 v_P
(viewer/prep/export_asthenosphere.py), blank within MASK_KM of a ridge or trench and smoothed by two neighbour
averages as there.  Fields at a depth are blank where the layer is thinner than that depth.
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
import pygplates
import xarray as xr
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

SLABLET = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('SLABLET_WORK', SLABLET)

from layer_300 import PARAMS as LAYER_PARAMS                 # noqa: E402  (the layer run's own values)
from slablet import asthenosphere as A                       # noqa: E402
from slablet import asthenosphere_volcanism as V             # noqa: E402
from slablet import cooling                                  # noqa: E402
from slablet import sphere_mesh as SM                        # noqa: E402
from slablet.asthenosphere_plates import forcing_from_step   # noqa: E402
from slablet.gplates.models import load as load_reconstruction   # noqa: E402
from slablet.sphere import lonlat_to_xyz                     # noqa: E402

RUNS = os.path.join(SLABLET, 'runs_real')
MANTLE = os.path.join(RUNS, 'muller2022_refined_5myr')
LAYER = os.path.join(RUNS, 'layer_300Ma.npz')
IPV = os.path.join(RUNS, 'ipv_muller2022_300Ma_rift', 'viewer_data')
AGEGRIDS = os.path.join(RUNS, 'cache', 'ipv', 'agegrids_muller2022_tracker', 'seafloor_age_{:d}Ma.nc')
CFG = os.path.join(SLABLET, 'configs', 'muller2022_refined.yaml')
OUT = os.path.join(SLABLET, 'viewer', 'public', 'data', 'earth')
AGES = list(range(300, -1, -5))
ARROW_DEG = 2.0
MIN_LAYER_KM = 10.0            # layer fields blank where the layer is thinner than this (as flow_volcanism)
LAYER_Z = [25, 50, 75, 100, 150, 200, 250, 300, 400]     # km below the plate's base: the 'layer' depth set
MASK_KM = 250.0                # shear-forced upwelling blank this close to a ridge or trench (as PlateDriver's)
TREND_SPEED = 30.0             # km/Myr: the fixed arrow length of the shear lineation trend

# id: (group, label, units, grid, depth set (None, 'mantle', 'layer'), encode lo, encode hi, colour min, colour max,
#      colour map).  Encodings span the whole run (0.5-99.5th percentiles at 0, 100, 200 and 300 Ma), so nothing
#      clips; the colour range is where the detail is, and the panel's Range widens it.
FIELDS = {
    'dyntopo': ('Surface', 'Dynamic topography (mantle below 300 km)', 'm', 1, None, -8000, 8000, -1500, 1500,
                'polar'),
    'geoid': ('Surface', 'Geoid (mantle below 300 km)', 'm', 1, None, -1000, 1000, -500, 500, 'polar'),
    'layer_depth': ('Surface', 'Seafloor depth change from the layer (+ shallower)', 'm', 0.5, None, -1500, 1500,
                    -600, 600, 'polar'),
    'age_depth': ('Surface', 'Seafloor elevation from age (GDH1)', 'm', 1, None, -7000, -2000, -6500, -2500,
                  'ocean'),
    'total_depth': ('Surface', 'Seafloor elevation: age + layer + dynamic', 'm', 0.5, None, -9000, -1000,
                    -6500, -2500, 'ocean'),
    'age': ('Lithosphere', "Seafloor age (slablet's tracker)", 'Myr', 1, None, 0, 254, 0, 200, 'age'),
    'lithosphere': ('Lithosphere', 'Lithosphere thickness (GDH1; continents 150 km)', 'km', 0.5, None, 0, 254,
                    0, 160, 'viridis'),
    'vuln': ('Lithosphere', 'Vulnerability (IPV): total', '', 1, None, 0, 1, 0, 0.6, 'magma'),
    'vuln_ssc': ('Lithosphere', 'Vulnerability: small-scale convection', '', 1, None, 0, 1, 0, 1, 'magma'),
    'vuln_stress': ('Lithosphere', 'Vulnerability: strain rate (plate boundaries)', '', 1, None, 0, 1, 0, 1,
                    'magma'),
    'vuln_shear': ('Lithosphere', 'Vulnerability: plate speed', '', 1, None, 0, 1, 0, 1, 'magma'),
    'vuln_edc': ('Lithosphere', 'Vulnerability: edge-driven convection', '', 1, None, 0, 1, 0, 1, 'magma'),
    'vuln_rt': ('Lithosphere', 'Vulnerability: keel drip fronts (RT)', '', 1, None, 0, 1, 0, 1, 'magma'),
    'vuln_rift': ('Lithosphere', 'Vulnerability: rifting', '', 1, None, 0, 1, 0, 1, 'magma'),
    'keel_fraction': ('Lithosphere', 'Keel remaining (IPV drip fronts)', '', 1, None, 0, 1, 0, 1, 'viridis'),
    'h': ('Upper mantle', 'Layer thickness', 'km', 0.5, None, 0, 600, 0, 500, 'viridis'),
    'layer_speed': ('Upper mantle', 'Layer speed (depth mean)', 'cm/yr', 0.5, None, 0, 20, 0, 8, 'viridis'),
    'pspeed': ('Upper mantle', 'Pressure-driven flow speed (depth mean)', 'cm/yr', 0.5, None, 0, 20, 0, 8,
               'viridis'),
    'layer_div': ('Upper mantle', 'Layer flow divergence (+ thinning)', 'mm/yr', 0.5, None, -80, 80, -10, 10,
                  'polar'),
    'shear': ('Upper mantle', 'Shear rate at the top of the layer', 'per Myr', 0.5, None, 0, 5, 0, 2, 'viridis'),
    'shear_angle': ('Upper mantle', 'Shear trend vs plate motion', 'deg', 0.5, None, 0, 90, 0, 90, 'magma'),
    'rise': ('Upper mantle', 'Layer rise rate along the plate base', 'mm/yr', 0.5, None, -5, 5, -1.5, 1.5,
             'polar'),
    'turn': ('Upper mantle', 'Chain direction vs plate motion', 'deg', 0.5, None, 0, 180, 0, 90, 'magma'),
    'upwell': ('Upper mantle', 'Shear-forced upwelling, below the plate', 'mm/yr', 0.5, 'layer', -10, 10, -1.5, 1.5,
               'polar'),
    'layer_speed_z': ('Upper mantle', 'Layer speed, below the plate', 'cm/yr', 0.5, 'layer', 0, 20, 0, 8,
                      'viridis'),
    'shear_z': ('Upper mantle', 'Shear rate, below the plate', 'per Myr', 0.5, 'layer', 0, 5, 0, 2, 'viridis'),
    'dvs': ('Deep mantle', 'Shear-velocity anomaly (model)', '%', 1, 'mantle', -8, 12, -2, 2, 'seis'),
    'dvs_tomo': ('Deep mantle', 'Shear-velocity anomaly (blurred as tomography sees it)', '%', 1, 'mantle', -8, 12,
                 -2, 2, 'seis'),
    'ur': ('Deep mantle', 'Radial flow (up +)', 'cm/yr', 1, 'mantle', -40, 40, -5, 5, 'polar'),
    'uh': ('Deep mantle', 'Horizontal flow speed', 'cm/yr', 1, 'mantle', 0, 50, 0, 15, 'viridis'),
    'tau': ('Deep mantle', 'Shear stress on horizontal planes (log10)', 'log10 MPa', 1, 'mantle', -1.5, 3, -0.5, 2.5,
            'magma'),
    's_rr': ('Deep mantle', 'Radial normal stress (with potential)', 'MPa', 1, 'mantle', -1500, 1500, -150, 150,
             'polar'),
    'geoid_z': ('Deep mantle', 'Geoid at depth (-potential / g)', 'm', 1, 'mantle', -3000, 3000, -1000, 1000,
                'polar'),
    'pile': ('Deep mantle', 'Basal pile thickness', 'km', 1, None, 0, 1000, 0, 600, 'viridis'),
    'cmb_topo': ('Deep mantle', 'Core-mantle boundary topography', 'km', 1, None, -32, 32, -15, 15, 'polar'),
}
# id: (group, label, depth set)
VECTORS = {
    'plate': ('Surface', 'Plate velocity', None),
    'layer': ('Upper mantle', 'Layer flow (depth mean)', None),
    'pressure': ('Upper mantle', 'Pressure-driven flow (depth mean)', None),
    'layer_z': ('Upper mantle', 'Layer flow, below the plate', 'layer'),
    'trend': ('Upper mantle', 'Shear lineation trend (to younging; fixed length)', None),
    'chain': ('Upper mantle', 'Volcanic chain direction (layer minus plate; to young end)', None),
    'deep': ('Deep mantle', 'Deep flow', 'mantle'),
}
DEPTH_SETS = {'mantle': 'Depth (km)', 'layer': 'Below the plate (km)'}
# Depth fields whose values change by an order of magnitude with depth: each depth gets its own encoding and colour
# range, from these ages (Writer.levels); FIXED_COLOUR keep their one colour range (the seismic convention, +-2%).
# The log stress keeps one range for all depths.
PER_DEPTH = [k for k, v in FIELDS.items() if v[4] and k != 'tau']
FIXED_COLOUR = ('dvs', 'dvs_tomo')
SAMPLE_AGES = (0, 100, 200, 300)
IPV_FIELDS = {'vuln': 'vulnerability', 'vuln_ssc': 'vulnerability_ssc', 'vuln_stress': 'vulnerability_stress',
              'vuln_shear': 'vulnerability_shear', 'vuln_edc': 'vulnerability_edc', 'vuln_rt': 'vulnerability_rt',
              'vuln_rift': 'vulnerability_rift', 'keel_fraction': 'keel_fraction'}
TYPE_MAP = {pygplates.FeatureType.gpml_subduction_zone: 'subduction',
            pygplates.FeatureType.gpml_mid_ocean_ridge: 'ridge',
            pygplates.FeatureType.gpml_transform: 'transform'}


def grid_nodes(deg):
    nlon, nlat = int(round(360 / deg)), int(round(180 / deg)) + 1
    return -180 + np.arange(nlon) * 360 / nlon, -90 + np.arange(nlat) * 180 / (nlat - 1)


def arrow_lattice(spacing_deg):
    """Rows every spacing_deg from -90 + spacing_deg / 2; n = round(360 / spacing_deg x cos lat) points per row at
    -180 + (k + 0.5) x 360 / n.  Returns lon, lat, row latitudes and row counts."""
    rows = np.arange(-90 + spacing_deg / 2, 90, spacing_deg)
    counts = [max(1, int(round(360 / spacing_deg * np.cos(np.radians(la))))) for la in rows]
    lon = np.concatenate([-180 + (np.arange(n) + 0.5) * 360 / n for n in counts])
    lat = np.concatenate([np.full(n, la) for la, n in zip(rows, counts)])
    return lon, lat, rows, counts


def east_north(lon, lat):
    lam, phi = np.radians(lon), np.radians(lat)
    E = np.c_[-np.sin(lam), np.cos(lam), np.zeros_like(lam)]
    N = np.c_[-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)]
    return E, N


def encode(values, lo, hi):
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    b = np.round((np.clip(np.where(ok, v, lo), lo, hi) - lo) / (hi - lo) * 254).astype(np.uint8)
    b[~ok] = 255
    return b


def cells_to_nodes(a, deg):
    """1 deg cell-centred grid(s) (..., 180, 360) to the node grid of spacing deg: bilinear, wrapping in longitude,
    each pole the mean of the row next to it."""
    a = np.asarray(a, float)
    clon = -179.5 + np.arange(360)
    clat = -89.5 + np.arange(180)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        south = np.nanmean(a[..., :1, :], axis=-1, keepdims=True).repeat(360, axis=-1)
        north = np.nanmean(a[..., -1:, :], axis=-1, keepdims=True).repeat(360, axis=-1)
    padded = np.concatenate([south, a, north], axis=-2)
    padded = np.concatenate([padded[..., -1:], padded, padded[..., :1]], axis=-1)
    plat = np.r_[-90.0, clat, 90.0]
    plon = np.r_[clon[0] - 1, clon, clon[-1] + 1]
    glon, glat = grid_nodes(deg)
    LAT, LON = np.meshgrid(glat, glon, indexing='ij')
    pts = np.c_[LAT.ravel(), LON.ravel()]
    flat = padded.reshape(-1, padded.shape[-2], padded.shape[-1])
    out = [RegularGridInterpolator((plat, plon), f)(pts).reshape(len(glat), len(glon)) for f in flat]
    return np.array(out).reshape(a.shape[:-2] + (len(glat), len(glon)))


def coastlines_at(coast, rm, time, max_deg=1.0):
    rfgs = []
    pygplates.reconstruct(coast, rm, rfgs, float(time), anchor_plate_id=0)
    out = []
    for r in rfgs:
        g = r.get_reconstructed_geometry()
        if isinstance(g, pygplates.PolygonOnSphere):
            g = pygplates.PolylineOnSphere(list(g.get_exterior_ring_points()) + [g.get_exterior_ring_points()[0]])
        if not isinstance(g, pygplates.PolylineOnSphere):
            continue
        pts = g.to_tessellated(np.radians(max_deg)).to_lat_lon_array()
        if len(pts) > 1:
            out.append({'type': 'Feature', 'properties': {'boundary_type': 'other'},
                        'geometry': {'type': 'LineString', 'coordinates': np.round(pts[:, ::-1], 3).tolist()}})
    return {'type': 'FeatureCollection', 'features': out}


def write_frame(kind, age, gj):
    os.makedirs(os.path.join(OUT, kind), exist_ok=True)
    with open(os.path.join(OUT, kind, '{:04d}.geojson'.format(age)), 'w') as fh:
        json.dump(gj, fh, separators=(',', ':'))


def nodes1_to_05(a):
    """1 deg nodes (181, 360) to 0.5 deg nodes (361, 720): bilinear between nodes, wrapping in longitude; NaN where
    any node used is NaN."""
    right = np.roll(a, -1, axis=1)
    out = np.full((361, 720), np.nan)
    out[::2, ::2] = a
    out[::2, 1::2] = 0.5 * (a + right)
    out[1::2, ::2] = 0.5 * (a[:-1] + a[1:])
    out[1::2, 1::2] = 0.25 * (a[:-1] + a[1:] + right[:-1] + right[1:])
    return out


def smooth_finite(mesh, w, passes=2):
    """Area-weighted mean of each cell and its neighbours over finite cells; NaN stays NaN (as PlateDriver's)."""
    i, j = mesh.edge_i, mesh.edge_j
    for _ in range(passes):
        ok = np.isfinite(w)
        wa, a = np.where(ok, w * mesh.area, 0.0), np.where(ok, mesh.area, 0.0)
        num = wa + np.bincount(i, wa[j], minlength=mesh.n) + np.bincount(j, wa[i], minlength=mesh.n)
        den = a + np.bincount(i, a[j], minlength=mesh.n) + np.bincount(j, a[i], minlength=mesh.n)
        w = np.where(ok, num / np.where(den > 0, den, 1.0), np.nan)
    return w


def divergence(mesh, vec, open_edge):
    """Divergence of a tangent cell-vector field (per Myr x vec's length unit / km): the edge-mean normal component
    times edge length, out of each cell, over its area; nothing crosses an edge not in open_edge."""
    d = mesh.xyz[mesh.edge_j] - mesh.xyz[mesh.edge_i]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    flux = 0.5 * np.einsum('ij,ij->i', vec[mesh.edge_i] + vec[mesh.edge_j], d) * mesh.edge_len * open_edge
    return (np.bincount(mesh.edge_i, flux, minlength=mesh.n) - np.bincount(mesh.edge_j, flux, minlength=mesh.n)) \
        / mesh.area


def layer_profile(mesh, h, f, info, u_plate, p):
    """The layer's flow with depth below the plate (module docstring), per mesh cell.  Returns the fields without
    depth, the fields at each LAYER_Z (lists, shallowest first) and the vectors (n, 3), km/Myr."""
    drag, press = A.edge_flux_parts(mesh, h, f, p)
    gone = np.asarray(f.closed, bool) | (h < MIN_LAYER_KM)
    hh = np.maximum(h, MIN_LAYER_KM)
    v_p = np.where(gone[:, None], 0.0, SM.cell_vectors(mesh, press) / hh[:, None])
    up = np.where(gone[:, None], 0.0, u_plate)
    s = up - 6.0 * v_p                                       # top shear x h (km/Myr): plate minus 6 v_P
    total = drag + press
    div = (np.bincount(mesh.edge_i, total, minlength=mesh.n) - np.bincount(mesh.edge_j, total, minlength=mesh.n)) \
        / mesh.area
    se, sn = SM.east_north(mesh, s)
    pe, pn = SM.east_north(mesh, up)
    angle = np.degrees(np.abs(np.angle(np.exp(2j * (np.arctan2(se, sn) - np.arctan2(pe, pn)))))) / 2.0
    flat = {'pspeed': np.where(gone, np.nan, np.linalg.norm(v_p, axis=1) / 10.0),
            'layer_div': np.where(np.asarray(f.closed, bool), np.nan, div),
            'shear': np.where(gone, np.nan, np.linalg.norm(s, axis=1) / hh),
            'shear_angle': np.where(gone | (np.hypot(pe, pn) < 1.0), np.nan, angle)}
    near = np.zeros(mesh.n, bool)
    pts = [lonlat_to_xyz(info['ridge']['lon'], info['ridge']['lat'])] + \
        ([info['trench_a']] if len(info['trench_a']) else [])
    pts = np.concatenate(pts)
    if len(pts):
        near = cKDTree(pts).query(mesh.xyz)[0] * mesh.radius_km < MASK_KM
    ok_edge = V.open_edges(mesh, f).astype(float)
    at = {'upwell': [], 'layer_speed_z': [], 'shear_z': []}
    vec_z = []
    for z in LAYER_Z:
        inside = ~gone & (h >= z)
        zeta = np.minimum(z / hh, 1.0)[:, None]
        u = up * (1 - zeta) + 6.0 * v_p * zeta * (1 - zeta)
        a = np.where(h >= z, z - z * z / (2 * hh), hh / 2)          # depth integral of the drag profile / u_p
        b = np.where(h >= z, z * z / (2 * hh) - z ** 3 / (3 * hh * hh), hh / 6)    # ... of the pressure profile
        w = np.einsum('ij,ij->i', up, V.gradient(mesh, a, ok_edge)) + divergence(mesh, 6.0 * v_p * b[:, None],
                                                                                   ok_edge)
        at['upwell'].append(smooth_finite(mesh, np.where(inside & ~near, w, np.nan)))
        at['layer_speed_z'].append(np.where(inside, np.linalg.norm(u, axis=1) / 10.0, np.nan))
        at['shear_z'].append(np.where(inside, np.linalg.norm(-up + 6.0 * v_p * (1 - 2 * zeta), axis=1) / hh,
                                      np.nan))
        vec_z.append(np.where(inside[:, None], u, 0.0))
    smag = np.linalg.norm(s, axis=1)
    trend = -s * np.where(smag > 1.0, TREND_SPEED / np.maximum(smag, 1e-9), 0.0)[:, None]
    return flat, at, {'pressure': v_p, 'trend': trend, 'layer_z': vec_z}


def boundaries_at(topo, rm, time, max_deg=1.0):
    snap = pygplates.TopologicalSnapshot(topo, rm, float(time), 0)
    out = []
    for sec in snap.get_resolved_topological_sections():
        feat = sec.get_topological_section_feature()
        btype = TYPE_MAP.get(feat.get_feature_type(), 'other')
        polarity = feat.get_enumeration(pygplates.PropertyName.gpml_subduction_polarity, 'Unknown') \
            if btype == 'subduction' else None
        for sub in sec.get_shared_sub_segments():
            pts = sub.get_resolved_geometry().to_tessellated(np.radians(max_deg)).to_lat_lon_array()
            if len(pts) < 2:
                continue
            plates = sorted({t.get_feature().get_reconstruction_plate_id()
                             for t in sub.get_sharing_resolved_topologies()})
            prop = {'boundary_type': btype, 'plates': plates[:2]}
            if polarity in ('Left', 'Right'):
                prop['polarity'] = polarity
            out.append({'type': 'Feature', 'properties': prop,
                        'geometry': {'type': 'LineString', 'coordinates': np.round(pts[:, ::-1], 4).tolist()}})
    return {'type': 'FeatureCollection', 'features': out}


class Writer:
    """Writes each field, arrow set and land mask.  With sample=True it writes nothing and keeps the depth fields'
    values instead, from which levels() sets each depth's encoding and colour range (PER_DEPTH)."""

    def __init__(self, sample=False, levels=None):
        self.sample = sample
        self.seen = {k: [] for k in FIELDS}
        self.per_depth = levels or {}
        self.ages = {k: [] for k in list(FIELDS) + ['land']}
        self.vector_ages = {k: [] for k in VECTORS}

    def field(self, key, age, values):
        if self.sample:
            if key in PER_DEPTH:
                self.seen[key].append(np.asarray(values, float))
            return
        _, _, _, _, _, lo, hi, *_ = FIELDS[key]
        path = os.path.join(OUT, 'fields', key, '{:04d}.bin'.format(age))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if key in self.per_depth:
            lv = self.per_depth[key]
            np.array([encode(v, a, b) for v, a, b in zip(values, lv['lo'], lv['hi'])]).tofile(path)
        else:
            encode(values, lo, hi).tofile(path)
        self.ages[key].append(age)

    def levels(self):
        """Per depth: encoding [lo, hi] wide enough for the sampled 0.5-99.5th percentiles x 1.25, and the colour
        range (98th percentile of |value| about zero for signed fields, 99th percentile from zero otherwise)."""
        out = {}
        for key, seen in self.seen.items():
            if not seen:
                continue
            signed = FIELDS[key][5] < 0
            lv = {'lo': [], 'hi': [], 'min': [], 'max': []}
            for k in range(seen[0].shape[0]):
                x = np.concatenate([s[k][np.isfinite(s[k])] for s in seen])
                if not len(x):
                    lo, hi, cmin, cmax = FIELDS[key][5:9]
                elif signed:
                    m = 1.25 * max(float(np.max(np.abs(np.percentile(x, [0.5, 99.5])))), 1e-6)
                    c = max(float(np.percentile(np.abs(x), 98)), 1e-6)
                    lo, hi, cmin, cmax = -m, m, -c, c
                else:
                    lo, hi = 0.0, 1.25 * max(float(np.percentile(x, 99.5)), 1e-6)
                    cmin, cmax = 0.0, max(float(np.percentile(x, 99)), 1e-6)
                for name, v in zip(('lo', 'hi', 'min', 'max'), (lo, hi, cmin, cmax)):
                    lv[name].append(float('{:.4g}'.format(v)))
            if key in FIXED_COLOUR:
                lv['min'], lv['max'] = FIELDS[key][7], FIELDS[key][8]
            out[key] = lv
        return out

    def land(self, age, classes):
        if self.sample:
            return
        path = os.path.join(OUT, 'land', '{:04d}.bin'.format(age))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.asarray(classes, np.uint8).tofile(path)
        self.ages['land'].append(age)

    def vector(self, key, age, east, north):
        if self.sample:
            return
        path = os.path.join(OUT, 'vectors', key, '{:04d}.bin'.format(age))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.stack([np.nan_to_num(east), np.nan_to_num(north)], axis=-1).astype(np.float32).tofile(path)
        self.vector_ages[key].append(age)


def field_info(key, w):
    """run.json's entry for one field; offset, scale, min and max are lists (one per depth) for PER_DEPTH fields."""
    grp, label, units, gdeg, depth_set, lo, hi, cmin, cmax, cmap = FIELDS[key]
    info = {'id': key, 'group': grp, 'label': label, 'units': units, 'path': 'fields/' + key, 'grid': str(gdeg),
            'depth': depth_set, 'offset': lo, 'scale': (hi - lo) / 254.0, 'nodata': 255, 'min': cmin, 'max': cmax,
            'colormap': cmap, 'ages': sorted(w.ages[key], reverse=True)}
    if key in w.per_depth:
        lv = w.per_depth[key]
        info.update(offset=lv['lo'], scale=[(b - a) / 254.0 for a, b in zip(lv['lo'], lv['hi'])], min=lv['min'],
                    max=lv['max'])
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ages', type=int, nargs='*', default=AGES)
    args = ap.parse_args()
    ages = sorted(set(args.ages), reverse=True)

    mesh = SM.voronoi_mesh(40000)
    g05_lon, g05_lat = grid_nodes(0.5)
    LON05, LAT05 = np.meshgrid(g05_lon, g05_lat)
    node_cell = SM.nearest_cell(mesh, LON05.ravel(), LAT05.ravel())
    alon, alat, rows, counts = arrow_lattice(ARROW_DEG)
    arrow_cell = SM.nearest_cell(mesh, alon, alat)
    aE, aN = east_north(alon, alat)
    a_i = np.clip(np.floor(alat + 90).astype(int), 0, 179)            # the mantle's 1 deg cell under each arrow
    a_j = np.floor(alon + 180).astype(int) % 360

    def to05(values):
        return np.asarray(values)[node_cell].reshape(len(g05_lat), len(g05_lon))

    def en(vec):
        v = np.asarray(vec, float)[arrow_cell]
        return np.einsum('ij,ij->i', v, aE), np.einsum('ij,ij->i', v, aN)

    layer = dict(np.load(LAYER)) if os.path.exists(LAYER) else None
    layer_ages = [int(a) for a in layer['ages']] if layer is not None else []
    cfg = yaml.safe_load(open(CFG))
    recon = load_reconstruction(cfg)
    rm = pygplates.RotationModel(recon.rotation_files)
    topo = [pygplates.FeatureCollection(f) for f in recon.topology_files]
    model_dir = os.path.dirname(os.path.dirname(recon.continent_files[0]))   # gprm's Muller2022 folder
    coast = [pygplates.FeatureCollection(f) for f in glob.glob(os.path.join(model_dir, 'Coastlines', '*.gpmlz'))]
    assert coast, 'no coastlines in ' + model_dir
    law = cooling.get('gdh1')
    p = A.AsthenosphereParams(**LAYER_PARAMS)
    from slablet.gplates.plate_history import RealPlateHistory
    hist = RealPlateHistory(cfg)
    with xr.open_dataset(os.path.join(MANTLE, 'mantle_temp_0000Ma.nc')) as ds:
        depths = [float(d) for d in ds['depth'].values]
    frames = []

    def export_age(age, w):
        """Every field and arrow set at one age through the writer w; returns what was missing."""
        notes = []
        # --- seafloor ages (1 deg; 0.5 deg for the summed elevation) ---
        age05 = None
        if os.path.exists(AGEGRIDS.format(age)):
            g = xr.open_dataarray(AGEGRIDS.format(age))
            assert float(g.lon[0]) == -180 and float(g.lat[0]) == -90 and g.shape == (181, 360)
            age1 = g.values.astype(float)
            w.field('age', age, age1)
            w.field('age_depth', age, -cooling.depth_m(law, np.where(np.isfinite(age1), age1, 0.0))
                    + np.where(np.isfinite(age1), 0.0, np.nan))
            age05 = nodes1_to_05(age1)
        # --- the layer (mesh -> 0.5 deg) ---
        layer_depth = None
        i = layer_ages.index(age) if age in layer_ages else None
        f, info = forcing_from_step(mesh, hist.step(float(age)), p)
        if i is not None and layer['saved'][i]:
            h = layer['h'][i].astype(float)
            thin = (h < MIN_LAYER_KM) | layer['closed'][i]
            up, ul, ur = layer['u_plate'][i], layer['u_layer'][i], layer['u_rel'][i]
            layer_depth = to05(np.where(layer['closed'][i], np.nan, -1000.0 * layer['depth'][i]))
            w.field('layer_depth', age, layer_depth)
            w.field('h', age, to05(np.where(layer['closed'][i], np.nan, h)))
            w.field('layer_speed', age, to05(np.where(thin, np.nan, np.linalg.norm(ul, axis=1) / 10.0)))
            w.field('rise', age, to05(np.where(thin, np.nan, layer['w'][i])))
            cosang = np.einsum('ij,ij->i', ur, -up) / np.maximum(
                np.linalg.norm(ur, axis=1) * np.linalg.norm(up, axis=1), 1e-12)
            turn = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            w.field('turn', age, to05(np.where(thin | (np.linalg.norm(up, axis=1) < 1.0), np.nan, turn)))
            w.field('lithosphere', age, to05(layer['lithosphere'][i]))
            w.vector('plate', age, *en(up))
            w.vector('layer', age, *en(np.where(thin[:, None], 0.0, ul)))
            w.vector('chain', age, *en(np.where(thin[:, None], 0.0, ur)))
            assert (np.asarray(f.closed, bool) == layer['closed'][i]).all(), 'keels differ from the layer run'
            flat, at, vec = layer_profile(mesh, h, f, info, up, p)
            for key, values in flat.items():
                w.field(key, age, to05(values))
            for key, levels in at.items():
                w.field(key, age, np.array([to05(v) for v in levels]))
            w.vector('pressure', age, *en(vec['pressure']))
            w.vector('trend', age, *en(vec['trend']))
            ez = [en(v) for v in vec['layer_z']]
            w.vector('layer_z', age, np.array([e for e, _ in ez]), np.array([n for _, n in ez]))
            closed, continent = layer['closed'][i], layer['continent'][i]
        else:
            closed, continent = np.asarray(f.closed, bool), ~np.asarray(f.supply_mask, bool)
            notes.append('no layer')
        w.land(age, to05(np.where(closed, 2, np.where(continent, 1, 0))))
        # --- deep mantle and dynamic topography (1 deg) ---
        mt = os.path.join(MANTLE, 'mantle_temp_{:04d}Ma.nc'.format(age))
        dt = os.path.join(MANTLE, 'dyntopo_{:04d}Ma.nc'.format(age))
        if os.path.exists(mt):
            with xr.open_dataset(mt) as ds:
                assert [float(d) for d in ds['depth'].values] == depths
                w.field('dvs', age, cells_to_nodes(ds['dVs_native'].values, 1.0))
                w.field('dvs_tomo', age, cells_to_nodes(ds['dVs'].values, 1.0))
                w.field('pile', age, cells_to_nodes(ds['pile_thickness_km'].values, 1.0))
        else:
            notes.append('no mantle')
        if os.path.exists(dt):
            with xr.open_dataset(dt) as ds:
                dyn = ds['dyn_topo'].values
                w.field('dyntopo', age, cells_to_nodes(dyn, 1.0))
                w.field('geoid', age, cells_to_nodes(ds['geoid'].values, 1.0))
                w.field('cmb_topo', age, cells_to_nodes(ds['cmb_topo'].values / 1000.0, 1.0))
                w.field('ur', age, cells_to_nodes(ds['u_r'].values, 1.0))
                w.field('uh', age, cells_to_nodes(np.hypot(ds['u_theta'].values, ds['u_phi'].values), 1.0))
                tau = cells_to_nodes(np.hypot(ds['s_rt'].values, ds['s_rp'].values), 1.0)
                w.field('tau', age, np.log10(np.maximum(tau, 1e-3)))
                w.field('s_rr', age, cells_to_nodes(ds['s_rr'].values, 1.0))
                w.field('geoid_z', age, cells_to_nodes(ds['geoid_at_depth'].values, 1.0))
                north = -10.0 * ds['u_theta'].values[:, a_i, a_j]          # cm/yr, + south -> km/Myr, + north
                east = 10.0 * ds['u_phi'].values[:, a_i, a_j]
                w.vector('deep', age, east, north)
            if age05 is not None and layer_depth is not None:
                total = (-cooling.depth_m(law, np.where(np.isfinite(age05), age05, 0.0)) + layer_depth
                         + cells_to_nodes(dyn, 0.5))
                w.field('total_depth', age, np.where(np.isfinite(age05), total, np.nan))
        else:
            notes.append('no dynamic topography')
        # --- IPV (1 deg nodes already) ---
        ipv = os.path.join(IPV, 'fields_{:g}Ma.nc'.format(age))
        if os.path.exists(ipv):
            with xr.open_dataset(ipv) as ds:
                assert ds.sizes == {'lat': 181, 'lon': 360} and float(ds.lon[0]) == -180 and float(ds.lat[0]) == -90
                for key, var in IPV_FIELDS.items():
                    if var in ds:                 # exact zeros (most continents) left blank, so the land shows
                        a = ds[var].values
                        w.field(key, age, np.where(a == 0, np.nan, a) if key.startswith('vuln') else a)
        else:
            notes.append('no IPV')
        return notes

    sample = Writer(sample=True)
    for age in [a for a in SAMPLE_AGES if a in ages]:
        export_age(age, sample)
    w = Writer(levels=sample.levels())
    for age in ages:
        notes = export_age(age, w)
        # --- plate boundaries ---
        write_frame('boundaries', age, boundaries_at(topo, rm, age))
        write_frame('coastlines', age, coastlines_at(coast, rm, age))
        frames.append({'time': float(age), 'file': '{:04d}.geojson'.format(age)})
        print('{:3d} Ma{}'.format(age, ': ' + ', '.join(notes) if notes else ''), flush=True)

    for kind in ('boundaries', 'coastlines'):
        with open(os.path.join(OUT, kind, 'manifest.json'), 'w') as fh:
            json.dump({'frames': sorted(frames, key=lambda f: f['time'])}, fh, indent=1)
    grids = {}
    for deg in (0.5, 1):
        lo, la = grid_nodes(deg)
        grids[str(deg)] = {'deg': deg, 'nlon': len(lo), 'nlat': len(la)}
    run = {
        'label': 'Müller et al. (2022), optimised mantle frame',
        'ages': ages,
        'depth_sets': {'mantle': {'label': DEPTH_SETS['mantle'], 'km': depths or []},
                       'layer': {'label': DEPTH_SETS['layer'], 'km': LAYER_Z}},
        'grids': grids,
        'land': {'path': 'land', 'grid': '0.5', 'ages': w.ages['land'], 'classes': ['ocean', 'continent', 'keel']},
        'fields': [field_info(k, w) for k in FIELDS if w.ages[k]],
        'vectors': [{'id': k, 'group': grp, 'label': label, 'path': 'vectors/' + k, 'depth': has_depth,
                     'units': 'km/Myr', 'ages': sorted(w.vector_ages[k], reverse=True)}
                    for k, (grp, label, has_depth) in VECTORS.items() if w.vector_ages[k]],
        'arrow_lattice': {'spacing_deg': ARROW_DEG, 'row_lat': [round(float(x), 4) for x in rows],
                          'row_count': [int(n) for n in counts]},
        'sources': {'mantle': MANTLE, 'layer': LAYER, 'ipv': IPV, 'seafloor_ages': os.path.dirname(AGEGRIDS)},
    }
    with open(os.path.join(OUT, 'run.json'), 'w') as fh:
        json.dump(run, fh, indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
