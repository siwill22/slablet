"""Drive the asthenosphere layer (slablet/asthenosphere.py) with a GPlates reconstruction: the earlier entry point,
now a thin wrapper over the plate-history contract.

    setting = PlateSetting(cfg, mesh)               cfg: a run config (configs/*.yaml), mesh: sphere_mesh.voronoi_mesh
    forcing, info = setting.forcing_at(time_ma, params)
    fields = layer_fields(mesh, h, forcing, params)

The plates come from slablet.gplates.plate_history.RealPlateHistory (the config's anchor plate must be fixed to the
mantle; the mesosphere is fixed in that frame) and the forcing from slablet.asthenosphere_plates.forcing_from_step,
which holds the rules (plates, walls, trench sink, keels, supply, ridges).  New code should use those two directly,
and run_layer for the time loop.  On 2026-11-09 the contract's forcing was identical to this module's earlier
direct-from-pygplates version on Müller 2022 at 79, 50, 30, 20, 10 and 0 Ma (40,000 cells), once overlapping
plate polygons were resolved as pygplates does (highest plate ID first).  One difference remains for trench_mode
'sink': each trench sample now stands for the median sample spacing, not exactly 1 degree.
"""
from slablet.asthenosphere_plates import forcing_from_step, layer_fields   # noqa: F401  (layer_fields: old path)
from slablet.gplates.plate_history import RealPlateHistory


class PlateSetting:
    def __init__(self, cfg, mesh, exclude_craton_ids=()):
        if len(exclude_craton_ids):
            raise NotImplementedError('exclude_craton_ids is gone: the plate-history contract gives the craton query, '
                                      'not the polygons; use a reconstruction block with craton_files instead')
        self.cfg, self.mesh = cfg, mesh
        self.history = RealPlateHistory(cfg)

    def forcing_at(self, time_ma, p):
        """Forcing over [t + 1, t] Myr, and what went into it (for maps and checks)."""
        return forcing_from_step(self.mesh, self.history.step(float(time_ma)), p)
