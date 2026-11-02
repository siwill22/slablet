"""Drive slablet with a real GPlates reconstruction.

    conda run -n pygmt17 python -m slablet.gplates.run_mantle configs/muller2022.yaml

The plate motions are not computed: the reconstruction's subduction zones and their
convergence rates seed the slab tracers, which then sink, cool the CMB heat field and
give the dVs field, exactly as in a PlateDriver run (same mantle code, same parameter
names).  Time runs from start_ma down to end_ma (ages in Ma; present day = 0).

Config keys (see configs/muller2022.yaml):
    name, reconstruction (which model and files: see slablet/gplates/models.py; gprm,
    plate_model_manager or explicit files; or the original top-level model_dir, rotation_files,
    topology_files), anchor_plate_id, start_ma, end_ma, delta_time, output_every_myr,
    output_dir, cache_dir, initial_slabs (columns | none), mantle: {MantleParams keys},
    frame_lon_offset_deg, frame_lon_drift_deg_per_myr (optional, default 0: every trench sample
        is rotated about the spin axis by offset + drift x age degrees, east positive; a test of
        longitude errors in the frame, not a correction)
    frame_rotation (optional): {pole_lat, pole_lon, deg_per_myr, from_ma (default 0)}: every trench
        sample older than from_ma is rotated about the pole (right-hand rule, anticlockwise seen from
        above the pole) by deg_per_myr x (age - from_ma) degrees, with its down-dip azimuth and trench
        velocity; present-day trenches stay put.  A test of age-dependent frame errors such as
        unmodelled true polar wander, not a correction
    frame_africa (optional): {nodes: {age: [rx, ry, rz]}}: replaces the reconstruction's absolute frame.
        Africa's (701) absolute finite rotation at each node age is given as a rotation vector (degrees;
        direction = Euler pole as a unit vector in Earth-centred x, y, z; length = angle); 0 Ma is the
        identity, and rotations in between are interpolated (pygplates FiniteRotation.interpolate).  Every
        trench sample is moved by C(t) = Africa_new(t) Africa_file(t)^-1, a global frame change (the
        relative plate motions are untouched).  Ages beyond the oldest node are not allowed
    seafloor (optional, default false): read the cache of slablet/gplates/seafloor_tracker.py (run it first): each
        trench sample gets seafloor_age_myr (for the mantle.slab_age_* options), and sink_slabs gets the plate
        velocity grid (for mantle.slab_flow_top / slab_flow_bottom).  The velocities are in the reconstruction's
        own frame, so the flow cannot be combined with the frame_* options.  With mantle.lithosphere_dVs > 0
        the field also gets a lid from slablet/gplates/lithosphere.py (which needs the same cache)
Outputs in output_dir:
    mantle_temp_<age>Ma.nc   dVs (and cmb_heat or pile_thickness_km), with age_ma and the model as attributes
    steps.csv                one row per step: trench samples, trench length, slablets, seconds
    slabs_upper_mantle_<end>Ma.npz   the slablets above 700 km at the end (lon, lat, depth_km, age, weight)
    run_info.yaml            the config as run, with the full mantle parameters, the files the
                             reconstruction resolved to and the slablet version
"""
import argparse
import dataclasses
import os
import time

import numpy as np
import pandas as pd
import pygplates
import yaml

from slablet.paths import WORK

import slablet                                                    # noqa: E402
from slablet.gplates.models import load as load_reconstruction  # noqa: E402
from slablet.gplates.trench_samples import CachedTrenchSamples   # noqa: E402

KM_PER_DEG = np.radians(1.0) * 6371.0


def finite_rotation_from_vector(v_deg):
    """pygplates FiniteRotation from a rotation vector (x, y, z) in degrees."""
    v = np.asarray(v_deg, float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return pygplates.FiniteRotation.create_identity_rotation()
    return pygplates.FiniteRotation(pygplates.PointOnSphere(*(v / angle)), np.radians(angle))


def matrix_from_finite_rotation(r):
    """3x3 matrix of a pygplates FiniteRotation (columns: the rotated x, y, z unit vectors)."""
    return np.array([(r * pygplates.Vector3D(*e)).to_xyz() for e in np.eye(3)]).T


def africa_frame(nodes):
    """Africa's absolute rotation at any age from {age: rotation vector (deg)}, identity at 0 Ma."""
    ages = sorted([0.0] + [float(a) for a in nodes])
    rots = [pygplates.FiniteRotation.create_identity_rotation()] + [
        finite_rotation_from_vector(nodes[k]) for k in sorted(nodes, key=float)]

    def at(age):
        if age > ages[-1] + 1e-9:
            raise SystemExit('frame_africa: age {} is older than the oldest node {}'.format(age, ages[-1]))
        i = max(0, np.searchsorted(ages, age) - 1)
        i = min(i, len(ages) - 2)
        return pygplates.FiniteRotation.interpolate(rots[i], rots[i + 1], ages[i], ages[i + 1], age)
    return at


def rotation_matrix(pole_lat, pole_lon, angle_deg):
    """3x3 matrix rotating unit vectors by angle_deg about the pole (right-hand rule)."""
    k = slablet.lonlat_to_xyz(pole_lon, pole_lat)
    a = np.radians(angle_deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _east_north(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], axis=-1)
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], axis=-1)
    return east, north


def rotate_samples(s, R):
    """Trench samples rotated by matrix R: positions, and the down-dip azimuth and trench velocity
    (as vectors in the local east/north plane) where present."""
    xyz = slablet.lonlat_to_xyz(s['lon'], s['lat'])
    lon, lat = slablet.xyz_to_lonlat(xyz @ R.T)
    out = {**s, 'lon': lon, 'lat': lat}
    if 'downdip_azimuth_deg' in s:
        e0, n0 = _east_north(s['lon'], s['lat'])
        e1, n1 = _east_north(lon, lat)
        az = np.radians(np.nan_to_num(s['downdip_azimuth_deg']))
        d = (np.sin(az)[:, None] * e0 + np.cos(az)[:, None] * n0) @ R.T
        new_az = np.degrees(np.arctan2((d * e1).sum(-1), (d * n1).sum(-1))) % 360.0
        out['downdip_azimuth_deg'] = np.where(np.isfinite(s['downdip_azimuth_deg']), new_az, np.nan)
        v = (s['trench_east_km_per_myr'][:, None] * e0 + s['trench_north_km_per_myr'][:, None] * n0) @ R.T
        out['trench_east_km_per_myr'] = (v * e1).sum(-1)
        out['trench_north_km_per_myr'] = (v * n1).sum(-1)
    return out


def run(cfg, verbose=True):
    """Run the mantle model for a config dict; returns a summary dict."""
    p = slablet.MantleParams.from_config(cfg.get('mantle', {}))
    unknown = set(cfg.get('mantle', {})) - {f.name for f in dataclasses.fields(slablet.MantleParams)}
    if unknown:
        raise SystemExit('unknown mantle parameters: ' + ', '.join(sorted(unknown)))
    start, end = float(cfg['start_ma']), float(cfg['end_ma'])
    dt = float(cfg.get('delta_time', 1.0))
    every = float(cfg.get('output_every_myr', 10.0))
    out_dir = os.path.join(WORK, cfg['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    recon = load_reconstruction(cfg)
    if recon.frame == 'unknown' and verbose:
        print('{}: the anchor plate\'s frame is not recorded ({}); slablets assume it is fixed to the mantle'.format(
            recon.label, recon.frame_note))
    trenches = CachedTrenchSamples(recon, cfg['name'], os.path.join(WORK, cfg['cache_dir']),
                                   sample_deg=p.slab_sample_deg,
                                   anchor_plate_id=int(cfg.get('anchor_plate_id', 0)))

    with open(os.path.join(out_dir, 'run_info.yaml'), 'w') as f:
        yaml.safe_dump({**cfg, 'mantle': dataclasses.asdict(p), 'reconstruction_resolved': recon.summary(),
                        'slablet': slablet.version_info()}, f, sort_keys=False)

    def save(age):
        lith = None
        if p.lithosphere_dVs > 0:
            from slablet.gplates.lithosphere import thickness_km
            lith = thickness_km(cfg, age)
        ds = slablet.dvs_field(slabs, llsvp_state, p, lith)
        ds.attrs.update({'age_ma': age, 'reconstruction': cfg['name'],
                         'anchor_plate_id': int(cfg.get('anchor_plate_id', 0))})
        ds.to_netcdf(os.path.join(out_dir, 'mantle_temp_{:04.0f}Ma.nc'.format(age)),
                     encoding={v: {'zlib': True, 'complevel': 4} for v in ds.data_vars})

    lon_offset = float(cfg.get('frame_lon_offset_deg', 0.0))
    lon_drift = float(cfg.get('frame_lon_drift_deg_per_myr', 0.0))

    rot = cfg.get('frame_rotation')
    fa = cfg.get('frame_africa')
    if fa:
        africa_new = africa_frame({float(k): v for k, v in fa['nodes'].items()})
        file_model = pygplates.RotationModel(recon.rotation_files)
        anchor = int(cfg.get('anchor_plate_id', 0))

    seafloor = None
    if cfg.get('seafloor'):
        from slablet.gplates.seafloor_tracker import SeafloorCache, VEL_DEG
        seafloor = SeafloorCache(cfg, p.slab_sample_deg)
        uses_flow = p.slab_flow_top != 0 or p.slab_flow_bottom != 0
        if uses_flow and (lon_offset or lon_drift or rot or fa):
            raise SystemExit('the lower-mantle flow (slab_flow_*) is in the reconstruction frame: no frame_* options')
        vel_lats = np.arange(-90 + VEL_DEG / 2, 90, VEL_DEG)
        vel_lons = np.arange(-180 + VEL_DEG / 2, 180, VEL_DEG)

    def flow_at(age):
        if seafloor is None:
            return None
        rec = seafloor.at(age)
        return {'lats': vel_lats, 'lons': vel_lons, 'vel_e': rec['vel_e'], 'vel_n': rec['vel_n']}

    def samples_at(age):
        s = trenches.samples(age)
        if seafloor is not None:
            s = {**s, 'seafloor_age_myr': seafloor.at(age)['trench_age']}
        shift = lon_offset + lon_drift * age
        if shift:
            s = {**s, 'lon': ((s['lon'] + shift + 180.0) % 360.0) - 180.0}
        if fa and age > 0 and len(s['lon']):
            c = africa_new(age) * file_model.get_rotation(float(age), 701, anchor_plate_id=anchor).get_inverse()
            s = rotate_samples(s, matrix_from_finite_rotation(c))
        if rot:
            angle = float(rot['deg_per_myr']) * max(0.0, age - float(rot.get('from_ma', 0.0)))
            if angle and len(s['lon']):
                s = rotate_samples(s, rotation_matrix(float(rot['pole_lat']), float(rot['pole_lon']), angle))
        return s

    t_run = time.perf_counter()
    samples = samples_at(start)
    if cfg.get('initial_slabs', 'columns') == 'columns':
        slabs = slablet.initial_slab_columns(samples, dt, p)
    else:
        slabs = pd.DataFrame(columns=slablet.SLAB_COLUMNS)
    llsvp_state = slablet.new_llsvp_state(p)      # CMB heat, pile thickness or None (p.llsvp)
    rows = []
    ages = np.round(np.arange(start - dt, end - 1e-9, -dt), 6)
    for age in ages:
        t0 = time.perf_counter()
        samples = samples_at(age)
        slabs = slablet.seed_slabs(slablet.sink_slabs(slabs, dt, p, flow_at(age)), samples, dt, p)
        llsvp_state = slablet.update_llsvp(llsvp_state, slabs, dt, p)
        if abs((start - age) / every - round((start - age) / every)) < 1e-6 or age == ages[-1]:
            save(age)
        rows.append({'age_ma': age, 'n_trench_samples': len(samples['lon']),
                     'trench_km': len(samples['lon']) * p.slab_sample_deg * KM_PER_DEG,
                     'n_slablets': len(slabs), 'seconds': time.perf_counter() - t0})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'steps.csv'), index=False)
    um = slabs[slabs['depth_km'].astype(float) <= 700.0]          # for the check against Slab1.0
    np.savez_compressed(os.path.join(out_dir, 'slabs_upper_mantle_{:04.0f}Ma.npz'.format(ages[-1])),
                        **{c: um[c].values.astype(float) for c in ('lon', 'lat', 'depth_km', 'age', 'weight')})
    summary = {'seconds': time.perf_counter() - t_run, 'n_resolved': trenches.n_resolved,
               'trench_km': rows[-1]['trench_km'], 'n_slablets': len(slabs)}
    if verbose:
        print('{}: {:.0f} -> {:.0f} Ma in {:.1f} s ({} times resolved, the rest from the cache); '
              'present-day trench {:.0f} km; {} slablets'.format(
                  cfg['name'], start, end, summary['seconds'], summary['n_resolved'],
                  summary['trench_km'], summary['n_slablets']))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    args = ap.parse_args()
    with open(args.config) as f:
        run(yaml.safe_load(f))


if __name__ == '__main__':
    main()
