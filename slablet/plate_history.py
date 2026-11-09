"""The plate-history contract: what a driver supplies to the response models at each step.

A driver is either a synthetic world (PlateDriver: self_driving_plates.plate_step) or a real
reconstruction (slablet.gplates.plate_history.RealPlateHistory).  At each step it gives a PlateStep
for the interval [time + delta_time, time], in a frame fixed to the mantle:

  plate_id_at(lon, lat)   plate identity at any points (int array; 0 where no plate)
  omega                   {plate id: rotation vector (3,), rad/Myr} over the step; the plate's velocity
                          at unit vector r is omega x r x RADIUS_KM.  Plates are rigid: deforming
                          networks count as rigid plates (decision 4 of
                          docs/plans/2026-11-08-cross-repo-reconciliation.md)
  trench                  trench samples, dict of equal-length arrays (TRENCH_KEYS):
                            lon, lat                      degrees
                            convergence_km_per_myr        > 0, normal to the trench
                            downdip_azimuth_deg           from the trench towards the overriding plate,
                                                          clockwise from north; NaN where unknown
                            trench_east_km_per_myr,       velocity of the trench itself
                            trench_north_km_per_myr
                            seafloor_age_myr              age of the subducting seafloor; NaN where unknown
  ridge                   ridge samples, dict of equal-length arrays (RIDGE_KEYS):
                            lon, lat; divergence_km_per_myr (> 0, full rate normal to the ridge);
                            length_km (ridge length the sample stands for)
  seafloor_age            xarray.DataArray (lat, lon), Myr, on a cell-centred grid (lon from -180 + d/2,
                          lat from -90 + d/2); NaN where there is no ocean or the age is unknown
  continent_at(lon, lat), craton_at(lon, lat)   bool arrays
  trench_lines            optional: subduction zones as a list of (n, 2) arrays of lon, lat (the
                          asthenosphere layer's walls need the lines, not just the samples); None if
                          the driver cannot give them

time is a label: Ma for a real reconstruction, elapsed model time for a synthetic world; only
differences matter.  Units: degrees, km, km/Myr (= mm/yr), rad/Myr, Myr; sphere radius RADIUS_KM.
Nothing here imports plate-model code; drivers build the queries.
"""
import dataclasses
from typing import Callable, Optional

import numpy as np
import xarray as xr

from slablet.core import SURFACE_RADIUS_KM as RADIUS_KM
from slablet.sphere import lonlat_to_xyz

TRENCH_KEYS = ('lon', 'lat', 'convergence_km_per_myr', 'downdip_azimuth_deg',
               'trench_east_km_per_myr', 'trench_north_km_per_myr', 'seafloor_age_myr')
RIDGE_KEYS = ('lon', 'lat', 'divergence_km_per_myr', 'length_km')


@dataclasses.dataclass
class PlateStep:
    time: float
    delta_time: float
    plate_id_at: Callable
    omega: dict
    trench: dict
    ridge: dict
    seafloor_age: xr.DataArray
    continent_at: Callable
    craton_at: Callable
    trench_lines: Optional[list] = None


def empty_trench():
    return {k: np.zeros(0) for k in TRENCH_KEYS}


def empty_ridge():
    return {k: np.zeros(0) for k in RIDGE_KEYS}


def cell_centred_grid(spacing_deg):
    """lons, lats of the global cell-centred grid of spacing_deg (lon -180 + d/2 ..., lat -90 + d/2 ...)."""
    n_lon, n_lat = int(round(360.0 / spacing_deg)), int(round(180.0 / spacing_deg))
    return (-180.0 + spacing_deg * (np.arange(n_lon) + 0.5),
            -90.0 + spacing_deg * (np.arange(n_lat) + 0.5))


def is_cell_centred(da):
    """True if a (lat, lon) DataArray covers the globe on a cell-centred grid."""
    if tuple(da.dims) != ('lat', 'lon'):
        return False
    lons, lats = da.lon.values, da.lat.values
    d = 360.0 / len(lons)
    want_lon, want_lat = cell_centred_grid(d)
    return (len(lats) == len(want_lat) and np.allclose(lons, want_lon, atol=1e-9)
            and np.allclose(lats, want_lat, atol=1e-9))


def on_grid(query, spacing_deg, name=None):
    """Evaluate a point query (plate_id_at, continent_at, ...) at the centres of a cell-centred grid."""
    lons, lats = cell_centred_grid(spacing_deg)
    lo, la = np.meshgrid(lons, lats)
    values = np.asarray(query(lo.ravel(), la.ravel())).reshape(lo.shape)
    return xr.DataArray(values, dims=('lat', 'lon'), coords={'lat': lats, 'lon': lons}, name=name)


def velocity_at(step, lon, lat):
    """East and north plate velocity (km/Myr) at points: omega x r of the plate there; NaN off plates
    or for a plate without a rotation."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    pid = np.asarray(step.plate_id_at(lon, lat))
    om = np.full(lon.shape + (3,), np.nan)
    for p in np.unique(pid):
        if int(p) in step.omega:
            om[pid == p] = step.omega[int(p)]
    r = lonlat_to_xyz(lon, lat)
    v = np.cross(om, r) * RADIUS_KM
    lo, la = np.radians(lon), np.radians(lat)
    east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], axis=-1)
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], axis=-1)
    return (v * east).sum(-1), (v * north).sum(-1)


def check(step):
    """Raise ValueError if a PlateStep breaks the contract; return it otherwise."""
    problems = []
    if not step.delta_time > 0:
        problems.append('delta_time must be > 0')
    for name, samples, keys, positive in (('trench', step.trench, TRENCH_KEYS, 'convergence_km_per_myr'),
                                          ('ridge', step.ridge, RIDGE_KEYS, 'divergence_km_per_myr')):
        missing = [k for k in keys if k not in samples]
        if missing:
            problems.append('{} samples lack {}'.format(name, ', '.join(missing)))
            continue
        n = {len(np.asarray(samples[k])) for k in keys}
        if len(n) != 1:
            problems.append('{} sample arrays differ in length: {}'.format(name, sorted(n)))
            continue
        lon, lat = np.asarray(samples['lon'], float), np.asarray(samples['lat'], float)
        if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
            problems.append('{} samples have non-finite positions'.format(name))
        elif (np.abs(lat) > 90).any() or (np.abs(lon) > 360).any():
            problems.append('{} samples have positions out of range'.format(name))
        if (np.asarray(samples[positive], float) <= 0).any():
            problems.append('{} samples need {} > 0'.format(name, positive))
    az = np.asarray(step.trench.get('downdip_azimuth_deg', []), float)
    if np.any(np.isfinite(az) & ((az < 0) | (az >= 360))):
        problems.append('downdip_azimuth_deg must be in [0, 360) or NaN')
    for pid, w in step.omega.items():
        if np.shape(w) != (3,) or not np.isfinite(w).all():
            problems.append('omega of plate {} is not a finite 3-vector'.format(pid))
            break
    if not isinstance(step.seafloor_age, xr.DataArray) or not is_cell_centred(step.seafloor_age):
        problems.append('seafloor_age must be a (lat, lon) DataArray on a global cell-centred grid')
    for name in ('plate_id_at', 'continent_at', 'craton_at'):
        if not callable(getattr(step, name)):
            problems.append('{} must be callable'.format(name))
    if step.trench_lines is not None:
        if any(np.ndim(line) != 2 or np.shape(line)[1] != 2 for line in step.trench_lines):
            problems.append('trench_lines must be (n, 2) arrays of lon, lat')
    if problems:
        raise ValueError('PlateStep at time {}: {}'.format(step.time, '; '.join(problems)))
    return step
