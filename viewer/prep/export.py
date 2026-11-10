"""Export slablet's Earth layers on Müller et al. (2022) for the Earth viewer (viewer/index.html).

    conda run -n pygmt17 python viewer/prep/export.py [--ages 0 5 ...]          # from ~/GIT/slablet

Reads (run these first; a missing input leaves its layers out at those ages):
  runs_real/muller2022_refined_5myr/   viewer/prep/mantle_5myr.py: mantle_temp_<age>Ma.nc (shear-velocity anomaly,
                                       basal pile) and dyntopo_<age>Ma.nc (dynamic topography, deep flow)
  runs_real/layer_300Ma.npz            viewer/prep/layer_300.py: the asthenosphere layer and flow-driven volcanism
  runs_real/ipv_muller2022_300Ma/viewer_data/fields_<age>Ma.nc
                                       IPV (mantlememory) on Müller 2022: vulnerability and its parts
  runs_real/cache/ipv/agegrids_muller2022/   links to the published Müller 2022 seafloor ages, every Myr
and writes viewer/public/data/earth/ (gitignored: local only):

  run.json                    ages, depths, grids, fields, vectors, arrow lattice (read by src/earth/main.ts)
  fields/<id>/<AAAA>.bin      uint8 per grid node, value = offset + byte x scale, 255 no data (IPV's vulnerability
                              fields: also where exactly 0, mostly continents); a field with depth
                              holds every depth, shallowest first (depth-major).  AAAA = age in Ma, 4 digits.
  land/<AAAA>.bin             uint8 on the 0.5 deg grid: 0 ocean, 1 continent, 2 keel (the layer's own masks)
  vectors/<id>/<AAAA>.bin     float32, east and north (km/Myr) interleaved at every arrow-lattice point; with depth,
                              every depth, shallowest first
  boundaries/                 petrify BoundarySeries: ridges, trenches (with polarity), transforms, other
  coastlines/                 the same format: Müller 2022's coastlines (Merdith et al.) at each age, as 'other' lines

Grids use the Geode Archive layout: longitude fastest, -180 + i x 360 / nlon (no repeated seam), then latitude
ascending, -90 + j x 180 / (nlat - 1) (both poles).  Two grids: 0.5 deg (layer fields, from the 40,000-cell mesh by
the cell each node lies in; seafloor ages) and 1 deg (mantle, dynamic topography, IPV).  The mantle's 1 deg
cell-centred grids go to the nodes by bilinear interpolation, wrapping at the seam, with each pole the mean of the
row next to it.  Arrows sit on rows every 2 deg, about 2 deg apart along each row (equal area); each takes the
nearest mesh cell or mantle cell and is expressed in the arrow point's own east and north.  All in Müller 2022's
optimised mantle frame (anchor plate 0).
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

SLABLET = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('SLABLET_WORK', SLABLET)

from slablet import asthenosphere as A                       # noqa: E402
from slablet import cooling                                  # noqa: E402
from slablet import sphere_mesh as SM                        # noqa: E402
from slablet.asthenosphere_plates import forcing_from_step   # noqa: E402
from slablet.gplates.models import load as load_reconstruction   # noqa: E402

RUNS = os.path.join(SLABLET, 'runs_real')
MANTLE = os.path.join(RUNS, 'muller2022_refined_5myr')
LAYER = os.path.join(RUNS, 'layer_300Ma.npz')
IPV = os.path.join(RUNS, 'ipv_muller2022_300Ma', 'viewer_data')
AGEGRIDS = os.path.join(RUNS, 'cache', 'ipv', 'agegrids_muller2022',
                        'Muller_etal_2022_SE_AgeGrid-{:d}.nc')   # links to the published grids (README.txt there)
CFG = os.path.join(SLABLET, 'configs', 'muller2022_refined.yaml')
OUT = os.path.join(SLABLET, 'viewer', 'public', 'data', 'earth')
AGES = list(range(300, -1, -5))
ARROW_DEG = 2.0
MIN_LAYER_KM = 10.0            # layer fields blank where the layer is thinner than this (as flow_volcanism)

# id: (group, label, units, grid, depth?, encode lo, encode hi, colour min, colour max, colour map)
FIELDS = {
    'dyntopo': ('Surface', 'Dynamic topography (mantle below 300 km)', 'm', 1, False, -3000, 3000, -1500, 1500,
                'polar'),
    'layer_depth': ('Surface', 'Seafloor depth change from the layer (+ shallower)', 'm', 0.5, False, -1500, 1500,
                    -600, 600, 'polar'),
    'age_depth': ('Surface', 'Seafloor elevation from age (GDH1)', 'm', 0.5, False, -7000, -2000, -6500, -2500,
                  'ocean'),
    'total_depth': ('Surface', 'Seafloor elevation: age + layer + dynamic', 'm', 0.5, False, -9000, -1000,
                    -6500, -2500, 'ocean'),
    'age': ('Lithosphere', 'Seafloor age (Müller 2022 grids)', 'Myr', 0.5, False, 0, 254, 0, 200, 'age'),
    'lithosphere': ('Lithosphere', 'Lithosphere thickness (GDH1; continents 150 km)', 'km', 0.5, False, 0, 254,
                    0, 160, 'viridis'),
    'vuln': ('Lithosphere', 'Vulnerability (IPV): total', '', 1, False, 0, 1, 0, 0.6, 'magma'),
    'vuln_ssc': ('Lithosphere', 'Vulnerability: small-scale convection', '', 1, False, 0, 1, 0, 1, 'magma'),
    'vuln_stress': ('Lithosphere', 'Vulnerability: strain rate (plate boundaries)', '', 1, False, 0, 1, 0, 1,
                    'magma'),
    'vuln_shear': ('Lithosphere', 'Vulnerability: plate speed', '', 1, False, 0, 1, 0, 1, 'magma'),
    'vuln_edc': ('Lithosphere', 'Vulnerability: edge-driven convection', '', 1, False, 0, 1, 0, 1, 'magma'),
    'vuln_rt': ('Lithosphere', 'Vulnerability: keel drips', '', 1, False, 0, 1, 0, 1, 'magma'),
    'vuln_rift': ('Lithosphere', 'Vulnerability: rifting', '', 1, False, 0, 1, 0, 1, 'magma'),
    'keel_fraction': ('Lithosphere', 'Keel remaining (IPV drip fronts)', '', 1, False, 0, 1, 0, 1, 'viridis'),
    'h': ('Upper mantle', 'Layer thickness', 'km', 0.5, False, 0, 600, 0, 500, 'viridis'),
    'layer_speed': ('Upper mantle', 'Layer speed (depth mean)', 'cm/yr', 0.5, False, 0, 20, 0, 8, 'viridis'),
    'rise': ('Upper mantle', 'Layer rise rate along the plate base', 'mm/yr', 0.5, False, -5, 5, -1.5, 1.5,
             'polar'),
    'turn': ('Upper mantle', 'Chain direction vs plate motion', 'deg', 0.5, False, 0, 180, 0, 90, 'magma'),
    'dvs': ('Deep mantle', 'Shear-velocity anomaly', '%', 1, True, -6, 6, -2, 2, 'seis'),
    'ur': ('Deep mantle', 'Radial flow (up +)', 'cm/yr', 1, True, -5, 5, -1.5, 1.5, 'polar'),
    'pile': ('Deep mantle', 'Basal pile thickness', 'km', 1, False, 0, 1000, 0, 600, 'viridis'),
}
# id: (group, label, depth?)
VECTORS = {
    'plate': ('Surface', 'Plate velocity', False),
    'layer': ('Upper mantle', 'Layer flow (depth mean)', False),
    'chain': ('Upper mantle', 'Volcanic chain direction (layer minus plate; to young end)', False),
    'deep': ('Deep mantle', 'Deep flow', True),
}
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
    def __init__(self):
        self.ages = {k: [] for k in list(FIELDS) + ['land']}
        self.vector_ages = {k: [] for k in VECTORS}

    def field(self, key, age, values):
        _, _, _, _, _, lo, hi, *_ = FIELDS[key]
        path = os.path.join(OUT, 'fields', key, '{:04d}.bin'.format(age))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        encode(values, lo, hi).tofile(path)
        self.ages[key].append(age)

    def land(self, age, classes):
        path = os.path.join(OUT, 'land', '{:04d}.bin'.format(age))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.asarray(classes, np.uint8).tofile(path)
        self.ages['land'].append(age)

    def vector(self, key, age, east, north):
        path = os.path.join(OUT, 'vectors', key, '{:04d}.bin'.format(age))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.stack([np.nan_to_num(east), np.nan_to_num(north)], axis=-1).astype(np.float32).tofile(path)
        self.vector_ages[key].append(age)


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
    p = A.AsthenosphereParams(keel_mode='craton', supply_where='ocean')
    hist = None
    depths = None
    w = Writer()
    frames = []

    for age in ages:
        notes = []
        # --- seafloor ages (0.5 deg) ---
        age05 = None
        if os.path.exists(AGEGRIDS.format(age)):
            g = xr.open_dataarray(AGEGRIDS.format(age))
            assert abs(float(g.lon[0]) + 180) < 1e-6 and abs(float(g.lat[0]) + 90) < 1e-6 and g.shape == (721, 1441)
            age05 = g.values[::2, :-1:2].astype(float)
            w.field('age', age, age05)
            w.field('age_depth', age, -cooling.depth_m(law, np.where(np.isfinite(age05), age05, 0.0))
                    + np.where(np.isfinite(age05), 0.0, np.nan))
        # --- the layer (mesh -> 0.5 deg) ---
        layer_depth = None
        i = layer_ages.index(age) if age in layer_ages else None
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
            closed, continent = layer['closed'][i], layer['continent'][i]
        else:
            if hist is None:
                from slablet.gplates.plate_history import RealPlateHistory
                hist = RealPlateHistory(cfg)
            f, _ = forcing_from_step(mesh, hist.step(float(age)), p)
            closed, continent = np.asarray(f.closed, bool), ~np.asarray(f.supply_mask, bool)
            notes.append('no layer')
        w.land(age, to05(np.where(closed, 2, np.where(continent, 1, 0))))
        # --- deep mantle and dynamic topography (1 deg) ---
        mt = os.path.join(MANTLE, 'mantle_temp_{:04d}Ma.nc'.format(age))
        dt = os.path.join(MANTLE, 'dyntopo_{:04d}Ma.nc'.format(age))
        if os.path.exists(mt):
            with xr.open_dataset(mt) as ds:
                depths = [float(d) for d in ds['depth'].values]
                w.field('dvs', age, cells_to_nodes(ds['dVs_native'].values, 1.0))
                w.field('pile', age, cells_to_nodes(ds['pile_thickness_km'].values, 1.0))
        else:
            notes.append('no mantle')
        if os.path.exists(dt):
            with xr.open_dataset(dt) as ds:
                dyn = ds['dyn_topo'].values
                w.field('dyntopo', age, cells_to_nodes(dyn, 1.0))
                w.field('ur', age, cells_to_nodes(ds['u_r'].values, 1.0))
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
        'depths_km': depths or [],
        'grids': grids,
        'land': {'path': 'land', 'grid': '0.5', 'ages': w.ages['land'], 'classes': ['ocean', 'continent', 'keel']},
        'fields': [{'id': k, 'group': grp, 'label': label, 'units': units, 'path': 'fields/' + k,
                    'grid': str(gdeg), 'depth': has_depth, 'offset': lo, 'scale': (hi - lo) / 254.0, 'nodata': 255,
                    'min': cmin, 'max': cmax, 'colormap': cmap, 'ages': sorted(w.ages[k], reverse=True)}
                   for k, (grp, label, units, gdeg, has_depth, lo, hi, cmin, cmax, cmap) in FIELDS.items()
                   if w.ages[k]],
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
