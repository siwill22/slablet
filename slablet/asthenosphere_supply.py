"""Where the asthenosphere layer's supply comes in: a weighted mixture of uniform, plume-fed and deep-upwelling parts
(decision 16 of docs/plans/2026-11-08-cross-repo-reconciliation.md).

    mantle = MantleRun('runs_real/muller2022_refined')       # a slablet mantle run, with dyntopo_NNNNMa.nc files
    share_at = mixture(mesh, {'uniform': 0.5, 'deep': 0.5}, mantle=mantle)
    for t, h, f, info, log in run_layer(step_at, mesh, p, start_ma=80, share_at=share_at): ...

The total supply still balances the sinks each step (asthenosphere.advance, supply 'balance'); only where it comes in
changes.  Each part is a share over the fed cells (open, and beneath ocean when supply_where is 'ocean'), summing to 1:
  uniform  in proportion to cell area (the baseline);
  plume    the same volume from every active plume site, spread over the fed cells within plume_radius_km of it
           (a site with none there feeds nothing);
  deep     in proportion to max(u_r, 0) x area, u_r the radial velocity at depth_km (the layer's base, about 400 km)
           from slablet's flow solution (slablet.dyntopo) of the mantle field at the saved ages, linear in time
           between them: a downwelling withholds supply.  Draining the layer above a downwelling is not built.
The mixture is the weighted sum of the parts' shares (weights normalised to sum to 1).  Plume sites are the mantle
run's own (fixed random sites active over thick pile, saved with each mantle_temp file; nearest saved age), so the same
plumes serve real reconstructions and synthetic worlds; hotspot_catalogue() gives present-day hotspots instead.
All in the mantle run's frame, which must be the layer run's.  No pygplates: this is shared physics.
"""
import glob
import os
import re

import numpy as np
import xarray as xr

from slablet.sphere import lonlat_to_xyz

PARTS = ('uniform', 'plume', 'deep')
PLUME_RADIUS_KM = 250.0
DEPTH_KM = 400.0


class MantleRun:
    """The saved state of a slablet mantle run that the supply needs: u_r at one depth from dyntopo_NNNNMa.nc and the
    active plume sites from mantle_temp_NNNNMa.nc, by age."""

    def __init__(self, folder, depth_km=DEPTH_KM):
        self.folder, self.depth_km = folder, float(depth_km)
        age = lambda f: int(re.search(r'_(\d+)Ma\.nc$', f).group(1))      # noqa: E731
        self.flow_files = {age(f): f for f in glob.glob(os.path.join(folder, 'dyntopo_*Ma.nc'))}
        self.field_files = {age(f): f for f in glob.glob(os.path.join(folder, 'mantle_temp_*Ma.nc'))}
        self._ur, self._plumes = {}, {}

    def _u_r_saved(self, age):
        if age not in self._ur:
            with xr.open_dataset(self.flow_files[age]) as ds:
                self._ur[age] = ds['u_r'].interp(depth=self.depth_km).load()
        return self._ur[age]

    def u_r(self, t):
        """Radial velocity (cm/yr, + up) at depth_km on the flow grid, linear in time between saved ages."""
        ages = np.array(sorted(self.flow_files))
        if not len(ages):
            raise FileNotFoundError('no dyntopo_*Ma.nc in {} (run slablet-dyntopo on its config)'.format(self.folder))
        if t < ages[0] or t > ages[-1]:
            raise ValueError('{} Ma is outside the saved flow ages {}-{} Ma'.format(t, ages[0], ages[-1]))
        hi = int(ages[np.searchsorted(ages, t)])
        lo = int(ages[max(np.searchsorted(ages, t, side='right') - 1, 0)])
        if lo == hi:
            return self._u_r_saved(lo)
        w = (t - lo) / (hi - lo)
        return (1 - w) * self._u_r_saved(lo) + w * self._u_r_saved(hi)

    def plumes(self, t):
        """Active plume sites (lon, lat) at the saved age nearest t."""
        ages = np.array(sorted(self.field_files))
        age = int(ages[np.argmin(np.abs(ages - t))])
        if age not in self._plumes:
            with xr.open_dataset(self.field_files[age]) as ds:
                self._plumes[age] = (ds['plume_lon'].values.astype(float), ds['plume_lat'].values.astype(float))
        return self._plumes[age]


def hotspot_catalogue(path=os.path.expanduser('~/.gmt/cache/hotspots.txt')):
    """Present-day hotspots (Müller, Royer & Lawver 1993; GMT's @hotspots.txt) as (lon, lat), for a real
    reconstruction in a mantle frame.  Pass as plume_sites=lambda t: hotspot_catalogue()."""
    lon, lat = [], []
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        c = line.split('\t')
        lon.append(float(c[0]))
        lat.append(float(c[1]))
    return np.array(lon), np.array(lat)


def _on_cells(grid, mesh):
    """Values of a cell-centred lat/lon grid at the mesh cell centres (the grid cell each centre lies in)."""
    lat, lon = grid['lat'].values, grid['lon'].values
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    i = np.clip(np.floor((mesh.lat - (lat[0] - dlat / 2)) / dlat).astype(int), 0, len(lat) - 1)
    j = np.floor(((mesh.lon - (lon[0] - dlon / 2)) % 360.0) / dlon).astype(int) % len(lon)
    return grid.values[i, j]


def uniform_share(mesh, fed):
    return np.where(fed, mesh.area, 0.0)


def plume_share(mesh, fed, lon, lat, radius_km=PLUME_RADIUS_KM):
    """Each site's equal volume spread by area over the fed cells within radius_km (great circle); sites with no fed
    cell there feed nothing.  Returns the share and the number of sites that feed."""
    share = np.zeros(mesh.n)
    unit = mesh.xyz / np.linalg.norm(mesh.xyz, axis=1)[:, None]
    used = 0
    for p in lonlat_to_xyz(np.asarray(lon, float), np.asarray(lat, float)):
        disc = fed & (mesh.radius_km * np.arccos(np.clip(unit @ p, -1.0, 1.0)) < radius_km)
        if disc.any():
            share[disc] += mesh.area[disc] / mesh.area[disc].sum()
            used += 1
    return share, used


def deep_share(mesh, fed, u_r_grid):
    return np.where(fed, np.maximum(_on_cells(u_r_grid, mesh), 0.0) * mesh.area, 0.0)


def mixture(mesh, weights, mantle=None, plume_sites=None, plume_radius_km=PLUME_RADIUS_KM):
    """A function (t, forcing) -> supply share per cell, for run_layer's share_at.  weights: {part: weight} over
    PARTS.  plume_sites(t) -> (lon, lat) defaults to the mantle run's plumes."""
    unknown = set(weights) - set(PARTS)
    if unknown:
        raise ValueError('unknown supply parts {}; choose from {}'.format(sorted(unknown), PARTS))
    w = {k: float(v) for k, v in weights.items() if v > 0}
    total = sum(w.values())
    if total <= 0:
        raise ValueError('supply weights must include a positive one')
    if ('deep' in w or ('plume' in w and plume_sites is None)) and mantle is None:
        raise ValueError('the deep and (default) plume parts need a MantleRun')
    sites = plume_sites if plume_sites is not None else (mantle.plumes if mantle is not None else None)

    def share_at(t, forcing):
        closed = np.zeros(mesh.n, bool) if forcing.closed is None else np.asarray(forcing.closed, bool)
        fed = ~closed if forcing.supply_mask is None else ~closed & np.asarray(forcing.supply_mask, bool)
        out = np.zeros(mesh.n)
        for part, weight in w.items():
            if part == 'uniform':
                s = uniform_share(mesh, fed)
            elif part == 'plume':
                s, _ = plume_share(mesh, fed, *sites(t), radius_km=plume_radius_km)
            else:
                s = deep_share(mesh, fed, mantle.u_r(t))
            if s.sum() > 0:
                out += weight / total * s / s.sum()
        return out
    return share_at
