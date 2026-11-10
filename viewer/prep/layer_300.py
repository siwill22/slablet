"""The asthenosphere layer (the stage 3 baseline set-up) from 300 Ma to the present, for the Earth viewer.

    conda run -n pygmt17 python viewer/prep/layer_300.py [start_ma]          # from ~/GIT/slablet

Müller et al. (2022) in its optimised mantle frame, Buck's values, trench walls, Merdith craton keels, uniform supply
beneath ocean, 40,000 cells, starting uniform at h_a.  Every 5 Myr it saves, per cell:
  h          layer thickness (km)
  depth      the layer's seafloor depth change (km, + deeper; AsthenosphereParams.depth_anomaly_km)
  u_layer    depth-mean layer velocity (km/Myr, x y z), from the layer's flux at that time
  u_plate    plate velocity (km/Myr, x y z)
  w, u_rel, lithosphere   flow-driven volcanism (slablet/asthenosphere_volcanism.py): rise rate along the plate's
             base (km/Myr), layer minus plate velocity (the chain direction), smoothed lithosphere thickness (km)
  closed, continent       keel cells, and continental cells (no supply)
  saved      False where nothing was saved (the start)
to runs_real/layer_300Ma.npz.
"""
import os
import sys
import time

import numpy as np
import yaml

SLABLET = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('SLABLET_WORK', SLABLET)

from slablet import asthenosphere as A                       # noqa: E402
from slablet import asthenosphere_volcanism as V             # noqa: E402
from slablet import sphere_mesh as SM                        # noqa: E402
from slablet.asthenosphere_plates import run_layer           # noqa: E402
from slablet.gplates.plate_history import RealPlateHistory   # noqa: E402

PARAMS = dict(delta_rho=10.0, h_a_km=300.0, viscosity_pa_s=1.3e19, lithosphere_km=100.0, trench_mode='wall',
              keel_mode='craton', supply_where='ocean', covered_mode='bulldoze')
EVERY_MYR = 5

if __name__ == '__main__':
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    cfg = yaml.safe_load(open(os.path.join(SLABLET, 'configs', 'muller2022_refined.yaml')))
    hist = RealPlateHistory(cfg)
    p = A.AsthenosphereParams(**PARAMS)
    mesh = SM.voronoi_mesh(40000)
    steps = {}

    def step_at(t):
        steps.clear()
        steps[t] = hist.step(t)
        return steps[t]

    ages = list(range(start - start % EVERY_MYR, -1, -EVERY_MYR))
    n, k = mesh.n, len(ages)
    out = {key: np.zeros((k, n), np.float32) for key in ('h', 'depth', 'w', 'lithosphere')}
    out.update({key: np.zeros((k, n, 3), np.float32) for key in ('u_layer', 'u_plate', 'u_rel')})
    out.update({key: np.zeros((k, n), bool) for key in ('closed', 'continent')})
    out['saved'] = np.zeros(k, bool)                          # False at the start (no flow yet)
    t0 = time.time()
    for t, h, f, info, log in run_layer(step_at, mesh, p, start_ma=start):
        if int(round(t)) not in ages:
            continue
        i = ages.index(int(round(t)))
        step = steps[t]
        u_layer = V.layer_velocity(mesh, h, f, p)
        vol = V.flow_volcanism(mesh, step, u_layer, h, forcing=f)
        out['h'][i] = h
        out['depth'][i] = p.depth_anomaly_km(h)
        out['u_layer'][i] = u_layer
        out['u_plate'][i] = u_layer - vol['u_rel']
        out['u_rel'][i] = vol['u_rel']
        out['w'][i] = vol['w']
        out['lithosphere'][i] = vol['lithosphere']
        out['closed'][i] = f.closed
        out['continent'][i] = ~np.asarray(f.supply_mask, bool)
        out['saved'][i] = True
        print('{:3.0f} Ma: {:.0f} s'.format(t, time.time() - t0), flush=True)
    path = os.path.join(SLABLET, 'runs_real', 'layer_{}Ma.npz'.format(start))
    np.savez_compressed(path, ages=np.array(ages), lon=mesh.lon, lat=mesh.lat, **out)
    print('wrote', path)
