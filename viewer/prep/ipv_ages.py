"""Seafloor-age grids for IPV from slablet's own seafloor tracker, every Myr, on Müller et al. (2022).

    conda run -n pygmt17 python viewer/prep/ipv_ages.py [oldest_ma]          # from ~/GIT/slablet

Reads runs_real/seafloor_ages_muller2022_1myr/t<t>.npz, written by
    python -m slablet.gplates.seafloor_tracker configs/muller2022_refined.yaml --grid-every 1 \
        --grids-only runs_real/seafloor_ages_muller2022_1myr
(seafloor only where the reconstruction made it, at its own diverging boundaries since 1000 Ma, and gone where
pygplates finds it subducted), and writes runs_real/cache/ipv/agegrids_muller2022_tracker/seafloor_age_<t>Ma.nc for
t = 0 ... oldest_ma (default 320), in IPV's layout: 1 deg nodes, lon -180 ... 179, lat -90 ... 90, NaN on continents.

- Continents are Müller 2022's continent polygons (the polygons slablet and IPV's keels use), so the age grid's blanks
  and the keels agree.  Tracked ages under them are dropped.
- Ages go from the tracker's 1 deg cell centres to the nodes as the mean of the (up to four) tracked cells around
  each node; each pole takes the mean of the row next to it.
- Ocean nodes with no tracked age nearby (a few %) take the nearest tracked ocean node's age (great-circle), so IPV
  never mistakes a gap in the tracking for a continent.
"""
import os
import sys
import warnings

import numpy as np
import xarray as xr
import yaml
from scipy.spatial import cKDTree

SLABLET = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('SLABLET_WORK', SLABLET)

from slablet.gplates.plate_history import RealPlateHistory   # noqa: E402

TRACKED = os.path.join(SLABLET, 'runs_real', 'seafloor_ages_muller2022_1myr')
OUT = os.path.join(SLABLET, 'runs_real', 'cache', 'ipv', 'agegrids_muller2022_tracker')


def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def cells_to_nodes(a):
    """(180, 360) cell-centred, lon -179.5 ... -> (181, 360) nodes, lon -180 ...: NaN-aware mean of the cells around."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        lon_pair = np.stack([np.roll(a, 1, axis=1), a])               # cells at node lon -0.5 and +0.5
        rows = np.nanmean(lon_pair, axis=0)                          # (180, 360) at node longitudes
        out = np.full((181, 360), np.nan)
        out[1:180] = np.nanmean(np.stack([rows[:-1], rows[1:]]), axis=0)
        out[0] = np.nanmean(a[0])
        out[180] = np.nanmean(a[-1])
    return out


if __name__ == '__main__':
    oldest = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    cfg = yaml.safe_load(open(os.path.join(SLABLET, 'configs', 'muller2022_refined.yaml')))
    hist = RealPlateHistory(cfg)
    lons, lats = -180.0 + np.arange(360), -90.0 + np.arange(181)
    LO, LA = np.meshgrid(lons, lats)
    xyz = unit(LO.ravel(), LA.ravel())
    os.makedirs(OUT, exist_ok=True)
    for t in range(oldest, -1, -1):
        with np.load(os.path.join(TRACKED, 't{:04d}.npz'.format(t))) as f:
            age = cells_to_nodes(f['age_grid'].astype(float))
        continent = np.asarray(hist._mask('continent', t)(LO, LA), bool)
        age[continent] = np.nan
        gap = ~continent & ~np.isfinite(age)
        known = ~continent & np.isfinite(age)
        if gap.any() and known.any():
            _, i = cKDTree(xyz[known.ravel()]).query(xyz[gap.ravel()])
            age[gap] = age[known][i]
        da = xr.DataArray(age.astype(np.float32), dims=('lat', 'lon'), coords={'lat': lats, 'lon': lons},
                          name='seafloor_age', attrs={'units': 'Myr', 'source': 'slablet seafloor tracker, Müller 2022'})
        da.to_netcdf(os.path.join(OUT, 'seafloor_age_{:d}Ma.nc'.format(t)))
        if t % 50 == 0:
            print('{:3d} Ma: ocean gaps filled {:.1f}% of ocean, oldest {:.0f} Myr'.format(
                t, 100 * gap.sum() / max((~continent).sum(), 1), np.nanmax(age)), flush=True)
    print('wrote', OUT)
