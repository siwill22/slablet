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
    slab_lower_rate_gradient: float = 0.0   # lower-mantle rate x (1 + this x fraction of the way from
                                            # upper_mantle_km to the CMB): < 0 slower with depth (0: uniform)
    slab_max_sink_rate: float = 200.0       # km/Myr; cap on the upper-mantle rate
    slab_lower_rate_km_per_myr: float = 0.0 # > 0: lower-mantle sinking rate of every slablet, whatever its trench's
                                            # convergence (instead of its upper-mantle rate / slab_slowing_factor);
                                            # the gradient and the seafloor-age factor still apply (0: off)
    slab_lower_rate_exponent: float = 0.0   # with a fixed rate: x (the slablet's upper-mantle rate /
    slab_lower_rate_ref_km_per_myr: float = 25.0  # this)^exponent -- a curtain fed faster is denser and sinks
                                            # faster; 0: the same for every slab, 1: proportional, as without it
    # seafloor age at the trench (needs the optional trench-sample key seafloor_age_myr; off when ref is 0).
    # A = the age clipped to [slab_age_min_myr, slab_age_max_myr] (unknown: slab_age_default_myr), r = A / ref:
    slab_age_ref_myr: float = 0.0           # > 0: the age at which every factor below is 1 (0: off)
    slab_age_min_myr: float = 5.0
    slab_age_max_myr: float = 100.0         # older plates are about as thick (plate cooling model)
    slab_age_default_myr: float = 50.0
    slab_age_weight_exponent: float = 0.5   # weight x r^this: the thermal deficit of a plate grows as sqrt(age)
    slab_age_tau_exponent: float = 1.0      # slab_thermal_tau_myr x r^this: thicker slabs take longer to warm
                                            # (diffusion time ~ thickness^2 ~ age)
    slab_age_rate_exponent: float = 0.0     # lower-mantle sinking rate x r^this (thicker slabs sink faster)
    # sideways motion in the lower mantle with the plate above (needs a flow grid from the caller):
    slab_flow_top: float = 0.0              # fraction of the overlying plate's velocity at upper_mantle_km ...
    slab_flow_bottom: float = 0.0           # ... and at the CMB, linear in between; < 0: against the plate
                                            # (return flow).  Applies to lower-mantle slablets (0, 0: off)
    slab_flow_upper: float = 0.0            # the same for upper-mantle slablets, one fraction (0: off)
    upper_mantle_km: float = 670.0
    cmb_depth_km: float = 2890.0            # slablets are removed here (slab_at_cmb: remove)
    slab_at_cmb: str = 'remove'             # remove: delete slablets that reach the CMB;
                                            # rest: they stop there and stay until their thermal
                                            # anomaly falls below SLAB_REST_MIN_FRACTION
    initial_slab_depth_km: float = 1320.0   # depth of the slab columns built for the first trenches
    slab_cmb_tau_myr: float = 0.0           # > 0 (slab_at_cmb: rest): a slablet resting on the CMB also loses
                                            # weight as exp(-time resting / this), heated from below (0: off)
    # dipping slabs (need the optional trench-sample keys, see slablet/core.py; PlateDriver has none)
    slab_dip_deg: float = 90.0              # < 90: above upper_mantle_km a slablet also moves towards the
                                            # overriding plate at its sink rate / tan(dip) (90: vertical)
    slab_dip_shallow_deg: float = 90.0      # dip above slab_dip_shallow_km (when that is > 0), e.g. a shallow
    slab_dip_shallow_km: float = 0.0        # slab near the trench; below it slab_dip_deg (0: one dip)
    slab_trench_follow: bool = False        # True: above upper_mantle_km a slablet also moves with the
                                            # velocity its trench had when it was seeded (attached slab)
    slab_trench_follow_fraction: float = 1.0  # with slab_trench_follow: the fraction of that velocity (1: fully
                                            # attached; between 0 and 1: partly anchored in the mantle)
    slab_trench_follow_retreating_only: bool = False  # with slab_trench_follow: only slablets whose trench was
                                            # retreating (away from the overriding plate) faster than
    slab_trench_follow_retreat_km_per_myr: float = 0.0  # this when they were seeded follow it; the rest stay put
    # stalls: slab held at the base of the upper mantle before it enters the lower mantle.  Slab from one stall
    # patch (the stall site nearest its trench point x a window of seeding time) shares one stall time; each
    # slablet is held that long from when it arrives, fixed in the mantle frame, then sinks at its lower-mantle
    # rate.  Needs the time from the caller (seed_slabs time_myr).  Initial slab columns never stall.
    slab_stall_fraction: float = 0.0        # share of patches that stall (0: off; the rest go straight through)
    slab_stall_time_myr: float = 25.0       # stall time: the mean (exponential) or the time itself (fixed)
    slab_stall_form: str = 'exponential'    # exponential (a constant chance of breaking through per Myr) or fixed
    slab_stall_site_deg: float = 10.0       # spacing of the stall sites (a Fibonacci lattice, the same in every run)
    slab_stall_window_myr: float = 25.0     # length of a patch's window of seeding time
    slab_stall_seed: int = 0                # the random draws: one ensemble member per seed
    slab_stall_depth_km: float = 0.0        # > 0: hold slab at this depth, above or below upper_mantle_km (0: just
                                            # above upper_mantle_km, then released into the lower mantle)
    slab_stall_flat_factor: float = 0.0     # > 0: held slab moves down-dip (towards the overriding plate) along the
                                            # hold depth at this x its trench's convergence rate, lying flat (0: it
                                            # stays where it arrives).  Needs dipping slabs
    slab_stall_retreat_km_per_myr: float = -1.0  # >= 0: only slab from trenches retreating faster than this at
                                            # seeding (velocity away from the overriding plate, mantle frame) can
                                            # stall (< 0: any slab).  Needs the trench samples' dip keys
    slab_cmb_spread_km_per_myr: float = 0.0 # > 0: slablets within slab_cmb_spread_layer_km of the CMB also move
                                            # sideways, down the gradient of the basal flow potential ψ (as for
                                            # pile_drive: flow), at this rms speed over the sphere (0: off)
    slab_cmb_spread_layer_km: float = 300.0

    # 3-D dVs field
    mantle_grid_spacing_deg: float = 5.0
    mantle_depth_interval_km: float = 100.0
    slab_temp_anomaly_K: float = -500.0     # ΔT of a fresh unit-weight slablet
    slab_thermal_tau_myr: float = 200.0     # e-folding time of that ΔT
    slab_dVs_per_K: float = -0.001          # % per K
    slab_sigma_deg: float = 3.0             # horizontal smoothing of the slab field
    slab_sigma_depth_km: float = 150.0      # vertical smoothing of the slab field
    slab_sigma_lower_deg: float = 0.0       # > 0: horizontal smoothing of the slab field below upper_mantle_km
                                            # instead of slab_sigma_deg (slabs thicken and fold there; 0: same)
    slab_sigma_cmb_deg: float = 0.0         # > 0: slab width profile -- slab_sigma_deg above upper_mantle_km, then
                                            # slab_sigma_lower_deg (or slab_sigma_deg) just below it, growing
                                            # linearly to this at the CMB; also the slab load's smoothing (0: off)
    slab_sigma_depth_upper_km: float = -1.0 # >= 0 (with the profile): vertical smoothing above upper_mantle_km
                                            # (< 0: slab_sigma_depth_km everywhere)
    slab_dVs_max: float = 0.0               # > 0: the slab field saturates at this (%): max x tanh(field / max), before
                                            # the LLSVP, lid and blur are added -- a slab is at most as fast as its
                                            # coldest core, however many slablets share a cell (0: no cap)
    slab_dVs_max_depth_km: float = 1e9      # the cap applies above this depth only (upper_mantle_km: upper mantle)
    # slow mantle wedge and back-arc (needs the dip azimuth; off when wedge_dVs is 0): each upper-mantle slablet
    # deeper than wedge_top_km also marks the mantle wedge_offset_km towards the overriding plate at its depth
    wedge_dVs: float = 0.0                  # < 0: the anomaly there (%), reached where the shifted slab field is
    wedge_saturation: float = 1.0           # well above this (dVs x tanh(shifted field / this)) (0: none)
    wedge_offset_km: float = 200.0
    wedge_top_km: float = 50.0
    wedge_bottom_km: float = 670.0          # slablets deeper than this mark no wedge
    wedge_sigma_deg: float = 1.5            # horizontal smoothing of the wedge field
    tomo_smooth_horiz_deg: float = 0.0      # optional blur of the final field (0 = off)
    tomo_smooth_depth_km: float = 0.0
    mantle_field_spacing_deg: float = 0.0   # > 0: draw the dVs field on this finer grid (must divide
                                            # mantle_grid_spacing_deg); the pile and slab load stay on the
                                            # coarse grid and are interpolated up (0: the same grid)
    # thin hot boundary layer on the CMB (llsvp: pile): a second conserved layer drawn on top of the pile, pushed by
    # the local slab load (pile_drive: local) in the bottom tbl_slab_layer_km, so it gathers between slab pools
    tbl_thickness_km: float = 0.0           # > 0: its mean thickness (0: off)
    tbl_mobility_km_per_myr: float = 30.0
    tbl_slab_head_km: float = 10000.0
    tbl_slab_layer_km: float = 300.0
    tbl_dVs: float = -2.0                   # % inside it
    tbl_halo_km: float = 150.0              # and tbl_dVs x exp(-height above its top / this) above it
    # plumes: hot conduits above the thickest parts of the boundary layer (of the pile without one)
    plume_dVs: float = 0.0                  # < 0: anomaly on a conduit's axis, scaled by the layer's thickness there
                                            # relative to its maximum (0: none)
    plume_sigma_deg: float = 1.5            # conduit radius (Gaussian sigma, great-circle degrees)
    plume_source: str = 'maxima'            # maxima: the layer's local maxima within plume_spacing_deg; sites:
                                            # fixed random sites, one per plume_spacing_deg^2 on average (seed
                                            # plume_seed), each active while the layer under it is thicker than ...
    plume_spacing_deg: float = 15.0
    plume_seed: int = 0
    plume_min_fraction: float = 0.5         # ... this fraction of its maximum
    plume_top_km: float = 200.0             # conduits run from the hot layer up to this depth
    lithosphere_dVs: float = 0.0            # > 0 (with a thickness grid from the caller): a fast lid, +this % at the
                                            # surface falling linearly to 0 at the lithosphere's base (0: none)
    dvs_save_native: bool = False           # True: also write the field before the tomography blur, as dVs_native
    dvs_save_components: bool = False       # True: also write each source of that field (slabs, wedge, pile,
                                            # pile_halo, cmb_heat, tbl, plumes, lithosphere), each with zero mean
                                            # per layer, as dVs_native__<source>; they sum to dVs_native
    mantle_bin_area_normalise: bool = False  # True: divide each lon/lat cell's binned slab sum by the
                                             # cell's area relative to an equatorial cell, so a slab
                                             # counts the same at every latitude (False: a slab at 60°
                                             # gets half the area-integrated anomaly of one at the equator)
    resolution_independent: bool = False    # True: slab amounts are per 1 deg of trench and per 5 deg x 100 km
                                            # cell (the defaults), so slab_sample_deg, mantle_grid_spacing_deg,
                                            # mantle_field_spacing_deg and mantle_depth_interval_km change how
                                            # finely the model is drawn, not how strong slabs are (False: a slablet
                                            # has the same weight at any spacing and binned sums are per cell;
                                            # identical at the default spacings)

    # LLSVPs: heat = the CMB heat proxy below; pile = a conserved pile pushed by slabs
    # (slablet/pile.py); none = slab field only
    llsvp: str = 'heat'
    pile_mean_thickness_km: float = 150.0   # the pile's volume, as a layer spread over the whole CMB
    pile_mobility_km_per_myr: float = 100.0 # κ: volume flux per unit pile thickness and head gradient
    pile_slab_head_km: float = 10.0         # pressure head (km of pile) per unit slab load L
    pile_drive: str = 'local'               # local: head = pile_slab_head_km × L (the pile fills every
                                            # slab-free area); flow: head = pile_slab_head_km × ψ with
                                            # ∇²ψ = -(L - mean L) on the unit sphere, the potential of the
                                            # basal flow away from slabs (the pile collects where it converges)
    pile_flow_screen_deg: float = 0.0       # > 0: ψ solves the screened equation ∇²ψ - ψ / λ² = -(L - mean L),
                                            # λ this many great-circle degrees, so slabs push only within about λ
                                            # and the pile also fills the gaps between them (0: plain Poisson)
    pile_slab_layer_km: float = 300.0       # slablets this close to the CMB load it
    pile_dVs: float = -1.0                  # dVs % inside the pile
    pile_halo_km: float = 0.0               # > 0: warm layer above the pile top, decaying upward as
                                            # exp(-height above the top / this) (0: off)
    pile_halo_dVs: float = -1.0             # dVs % of that layer just above a pile at least pile_halo_h_km thick
    pile_halo_h_km: float = 300.0           # thinner pile: proportionally weaker layer above it

    # CMB heat (LLSVP proxy, llsvp: heat)
    cmb_heating_rate: float = 0.02          # heat units per Myr everywhere
    cmb_slab_suppression: float = 0.05      # heat units removed per full-weight slablet per Myr at the CMB
    cmb_surface_coupling: float = 0.1       # fraction of that suppression from a slablet at the surface
    cmb_sigma_deg: float = 10.0             # smoothing per Myr; width scales with sqrt(dt)
    cmb_max_heat: float = 3.0
    cmb_dVs_scale: float = -0.5             # dVs % per heat unit
    cmb_decay_km: float = 400.0             # Gaussian decay of the warm anomaly upward from the CMB
    cmb_suppression_thermal_decay: bool = False  # True: a slablet suppresses CMB heat in proportion to its
                                                 # remaining thermal anomaly exp(-age / slab_thermal_tau_myr),
                                                 # as in the dVs field (False: at full weight whatever its age)

    @classmethod
    def from_config(cls, params):
        """MantleParams from a dict of config keys; keys that are not mantle parameters
        are ignored, missing ones take the defaults above."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (params or {}).items() if k in names})
