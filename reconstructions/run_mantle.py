"""Drive the simplified mantle model (mantle/) with a real GPlates reconstruction.

    conda run -n pygmt17 python reconstructions/run_mantle.py configs/real/muller2022.yaml

The plate motions are not computed: the reconstruction's subduction zones and their
convergence rates seed the slab tracers, which then sink, cool the CMB heat field and
give the dVs field, exactly as in a PlateDriver run (same mantle code, same parameter
names).  Time runs from start_ma down to end_ma (ages in Ma; present day = 0).

Config keys (see configs/real/muller2022.yaml):
    name, model_dir, rotation_files, topology_files (glob patterns relative to model_dir),
    anchor_plate_id, start_ma, end_ma, delta_time, output_every_myr,
    output_dir, cache_dir, initial_slabs (columns | none), mantle: {MantleParams keys},
    frame_lon_offset_deg, frame_lon_drift_deg_per_myr (optional, default 0: every trench sample
        is rotated about the spin axis by offset + drift x age degrees, east positive; a test of
        longitude errors in the frame, not a correction)
    frame_rotation (optional): {pole_lat, pole_lon, deg_per_myr, from_ma (default 0)}: every trench
        sample older than from_ma is rotated about the pole (right-hand rule, anticlockwise seen from
        above the pole) by deg_per_myr x (age - from_ma) degrees, with its down-dip azimuth and trench
        velocity; present-day trenches stay put.  A test of age-dependent frame errors such as
        unmodelled true polar wander, not a correction
Outputs in output_dir:
    mantle_temp_<age>Ma.nc   dVs (and cmb_heat or pile_thickness_km), with age_ma and the model as attributes
    steps.csv                one row per step: trench samples, trench length, slablets, seconds
    slabs_upper_mantle_<end>Ma.npz   the slablets above 700 km at the end (lon, lat, depth_km, age, weight)
    run_info.yaml            the config as run, with the full mantle parameters
"""
import argparse
import dataclasses
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import mantle                                                    # noqa: E402
from reconstructions.trench_samples import CachedTrenchSamples   # noqa: E402

KM_PER_DEG = np.radians(1.0) * 6371.0


def rotation_matrix(pole_lat, pole_lon, angle_deg):
    """3x3 matrix rotating unit vectors by angle_deg about the pole (right-hand rule)."""
    k = mantle.lonlat_to_xyz(pole_lon, pole_lat)
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
    xyz = mantle.lonlat_to_xyz(s['lon'], s['lat'])
    lon, lat = mantle.xyz_to_lonlat(xyz @ R.T)
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
    p = mantle.MantleParams.from_config(cfg.get('mantle', {}))
    unknown = set(cfg.get('mantle', {})) - {f.name for f in dataclasses.fields(mantle.MantleParams)}
    if unknown:
        raise SystemExit('unknown mantle parameters: ' + ', '.join(sorted(unknown)))
    start, end = float(cfg['start_ma']), float(cfg['end_ma'])
    dt = float(cfg.get('delta_time', 1.0))
    every = float(cfg.get('output_every_myr', 10.0))
    out_dir = os.path.join(REPO, cfg['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    trenches = CachedTrenchSamples(cfg['model_dir'], cfg['rotation_files'], cfg['topology_files'],
                                   cfg['name'], os.path.join(REPO, cfg['cache_dir']),
                                   sample_deg=p.slab_sample_deg,
                                   anchor_plate_id=int(cfg.get('anchor_plate_id', 0)))

    with open(os.path.join(out_dir, 'run_info.yaml'), 'w') as f:
        yaml.safe_dump({**cfg, 'mantle': dataclasses.asdict(p)}, f, sort_keys=False)

    def save(age):
        ds = mantle.dvs_field(slabs, llsvp_state, p)
        ds.attrs.update({'age_ma': age, 'reconstruction': cfg['name'],
                         'anchor_plate_id': int(cfg.get('anchor_plate_id', 0))})
        ds.to_netcdf(os.path.join(out_dir, 'mantle_temp_{:04.0f}Ma.nc'.format(age)),
                     encoding={v: {'zlib': True, 'complevel': 4} for v in ds.data_vars})

    lon_offset = float(cfg.get('frame_lon_offset_deg', 0.0))
    lon_drift = float(cfg.get('frame_lon_drift_deg_per_myr', 0.0))

    rot = cfg.get('frame_rotation')

    def samples_at(age):
        s = trenches.samples(age)
        shift = lon_offset + lon_drift * age
        if shift:
            s = {**s, 'lon': ((s['lon'] + shift + 180.0) % 360.0) - 180.0}
        if rot:
            angle = float(rot['deg_per_myr']) * max(0.0, age - float(rot.get('from_ma', 0.0)))
            if angle and len(s['lon']):
                s = rotate_samples(s, rotation_matrix(float(rot['pole_lat']), float(rot['pole_lon']), angle))
        return s

    t_run = time.perf_counter()
    samples = samples_at(start)
    if cfg.get('initial_slabs', 'columns') == 'columns':
        slabs = mantle.initial_slab_columns(samples, dt, p)
    else:
        slabs = pd.DataFrame(columns=mantle.SLAB_COLUMNS)
    llsvp_state = mantle.new_llsvp_state(p)      # CMB heat, pile thickness or None (p.llsvp)
    rows = []
    ages = np.round(np.arange(start - dt, end - 1e-9, -dt), 6)
    for age in ages:
        t0 = time.perf_counter()
        samples = samples_at(age)
        slabs = mantle.seed_slabs(mantle.sink_slabs(slabs, dt, p), samples, dt, p)
        llsvp_state = mantle.update_llsvp(llsvp_state, slabs, dt, p)
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
