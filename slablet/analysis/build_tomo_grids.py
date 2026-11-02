"""The S-wave models other than REVEAL and SEMUCB-WM1 on the 5 deg mantle grid, as NetCDF the tomography scorer reads.

    conda run -n pygmt17 python -m slablet.analysis.build_tomo_grids

S40RTS, SMEAN, GypsumS and S362ANI (read and block-averaged exactly as for the slow vote map,
build_slow_votemap.model_gridded; each target depth 1000-2800 km takes the model's nearest depth), minus the
area-weighted layer mean.  Writes runs_real/tomo_grids/<name>.nc with vs_anomaly (depth, lat, lon); the
units differ between models, which does not matter to the scores (area-fraction regions and correlations).
"""
import os
import sys

import numpy as np
import xarray as xr

from slablet.paths import WORK
from slablet.core import mantle_grid                                    # noqa: E402
from slablet.analysis.build_slow_votemap import DEPTHS, SPACING, model_gridded   # noqa: E402

OUT = os.path.join(WORK, 'runs_real', 'tomo_grids')
INDEPENDENT = ['S40RTS', 'SMEAN', 'GypsumS', 'S362ANI']

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    lons_g, lats_g = mantle_grid(SPACING)
    w = np.cos(np.radians(lats_g))[:, None] * np.ones(len(lons_g))
    for name in INDEPENDENT:
        f = model_gridded(name, lons_g, lats_g)
        for k in range(len(DEPTHS)):
            ok = np.isfinite(f[k])
            f[k] -= (f[k][ok] * w[ok]).sum() / w[ok].sum()
        xr.Dataset({'vs_anomaly': (('depth', 'lat', 'lon'), f)},
                   coords={'depth': DEPTHS, 'lat': lats_g, 'lon': lons_g},
                   attrs={'model': name, 'source': 'build_slow_votemap.model_gridded, layer mean removed'}
                   ).to_netcdf(os.path.join(OUT, name + '.nc'))
        print(name, 'NaN cells:', int(np.isnan(f).sum()))
