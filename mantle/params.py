"""Every parameter of the mantle model, with its default, in one place.

Field names are the config keys (the `params:` block of a PlateDriver config, or the
`mantle:` block of a real-reconstruction config), so one block of keys means the same
thing in both streams.  The defaults are the values run_SDP.py fell back on before the
mantle code was moved here, so existing configs reproduce exactly.  Some of them differ
from what config.yaml sets (slab_max_sink_rate 200 here vs 400 there; slab_thermal_tau_myr
200 vs 4): a decision for later, not changed by the move.
"""
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class MantleParams:
    # slab tracers: seeding along trenches and sinking
    slab_sample_deg: float = 1.0            # spacing of slablets along trenches (great-circle degrees)
    slab_sink_rate_factor: float = 1.0      # upper-mantle sinking rate = factor × convergence (km/Myr)
    slab_slowing_factor: float = 4.0        # lower-mantle rate = upper-mantle rate / factor
    slab_max_sink_rate: float = 200.0       # km/Myr; cap on the upper-mantle rate
    upper_mantle_km: float = 670.0
    cmb_depth_km: float = 2890.0            # slablets are removed here
    initial_slab_depth_km: float = 1320.0   # depth of the slab columns built for the first trenches

    # 3-D dVs field
    mantle_grid_spacing_deg: float = 5.0
    mantle_depth_interval_km: float = 100.0
    slab_temp_anomaly_K: float = -500.0     # ΔT of a fresh unit-weight slablet
    slab_thermal_tau_myr: float = 200.0     # e-folding time of that ΔT
    slab_dVs_per_K: float = -0.001          # % per K
    slab_sigma_deg: float = 3.0             # horizontal smoothing of the slab field
    slab_sigma_depth_km: float = 150.0      # vertical smoothing of the slab field
    tomo_smooth_horiz_deg: float = 0.0      # optional blur of the final field (0 = off)
    tomo_smooth_depth_km: float = 0.0

    # CMB heat (LLSVP proxy)
    cmb_heating_rate: float = 0.02          # heat units per Myr everywhere
    cmb_slab_suppression: float = 0.05      # heat units removed per full-weight slablet per Myr at the CMB
    cmb_surface_coupling: float = 0.1       # fraction of that suppression from a slablet at the surface
    cmb_sigma_deg: float = 10.0             # smoothing per Myr; width scales with sqrt(dt)
    cmb_max_heat: float = 3.0
    cmb_dVs_scale: float = -0.5             # dVs % per heat unit
    cmb_decay_km: float = 400.0             # Gaussian decay of the warm anomaly upward from the CMB

    @classmethod
    def from_config(cls, params):
        """MantleParams from a dict of config keys; keys that are not mantle parameters
        are ignored, missing ones take the defaults above."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (params or {}).items() if k in names})
