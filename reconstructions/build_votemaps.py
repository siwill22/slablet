"""Turn SubMachine vote-map depth slices (text, lon lat vote) into one NetCDF per folder.

    conda run -n pygmt17 python reconstructions/build_votemaps.py

A vote map (Shephard et al. 2017; SubMachine, Hosseini et al. 2018) counts, at each point, how
many tomography models are fast there: high vote = slab in many models.  So it scores the model's
slabs (fast regions); the slow scores against it mean nothing.  Inputs (0.5 deg, lon 0-359.5):
    ~/Data/SeismicTomography/votemaps/                 700-1500 km every 100, and 2000 km
    ~/Data/SeismicTomography/votemaps_allP_20220809/   750-2500 km every 250 (P-wave models)
Outputs: runs_real/votemaps/<folder name>.nc, variable `votes` (depth, lat, lon), lon -180..179.5.
"""
import glob
import os
import re

import numpy as np
import xarray as xr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.expanduser('~/Data/SeismicTomography')
FOLDERS = ['votemaps', 'votemaps_allP_20220809']


def build(folder):
    files = glob.glob(os.path.join(SRC, folder, 'SubMachine_votemap_depth_slice_*.txt'))
    depths = sorted(int(re.search(r'_(\d+)\.txt$', f).group(1)) for f in files)
    slices = []
    for z in depths:
        a = np.loadtxt(os.path.join(SRC, folder, 'SubMachine_votemap_depth_slice_{}.txt'.format(z)))
        lon = ((a[:, 0] + 180.0) % 360.0) - 180.0
        lons, lats = np.unique(lon), np.unique(a[:, 1])
        grid = np.full((len(lats), len(lons)), np.nan)
        grid[np.searchsorted(lats, a[:, 1]), np.searchsorted(lons, lon)] = a[:, 2]
        slices.append(grid)
    return xr.Dataset({'votes': (('depth', 'lat', 'lon'), np.array(slices))},
                      coords={'depth': np.array(depths, float), 'lat': lats, 'lon': lons},
                      attrs={'source': os.path.join(SRC, folder), 'note': 'number of models fast at each point'})


def main():
    out = os.path.join(REPO, 'runs_real', 'votemaps')
    os.makedirs(out, exist_ok=True)
    for folder in FOLDERS:
        ds = build(folder)
        path = os.path.join(out, folder + '.nc')
        ds.to_netcdf(path)
        print(path, dict(ds.sizes), 'depths', ds.depth.values.tolist(), 'max vote', float(ds.votes.max()))


if __name__ == '__main__':
    main()
