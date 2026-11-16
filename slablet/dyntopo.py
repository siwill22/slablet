"""Dynamic topography, geoid and instantaneous mantle flow from a slablet mantle field.

The flow follows Hager & O'Connell (1981) and Richards & Hager (1984): incompressible, Newtonian flow in a
spherical shell with viscosity varying only with radius, driven by density anomalies, solved one spherical harmonic
degree at a time.  Self-gravitation is included as in HC (Becker, O'Neill & Steinberger), so that the surface
topography is measured from the geoid and the geoid includes the masses of the deflected surface and core-mantle
boundary.  The equations and boundary conditions are written out below; tests/test_dyntopo.py checks their limits
and scripts/check_against_hc.py compares the whole solution with HC.

Density (stage 1, see CONTEXT.md):
    drho = dlnrho_dlnvs x dVs / 100 x rho_PREM(depth)            (HC's -ds, with its default PREM scaling)
from the field `density_variable` (dVs), set to zero above cutoff_km.
The pile is treated like any slow region, i.e. as hot.  Each depth layer of the field is a thin sheet at its
depth carrying the mass of the slab of mantle between the midpoints to its neighbours (the top one up to the
surface, the bottom one down to the core-mantle boundary), as in HC.

Equations.  For degree l (L = l(l+1)), x = ln(r / a), and the poloidal velocity u_r = U Y, u_h = V grad_1 Y, the
state s = (U, V, R, S, P1, P2) holds, in units built from 1 kg/m3, g, a and the reference viscosity eta0:
    U, V      velocities / (g a^2 / eta0)
    R, S      r x (radial normal stress, shear stress) / (a g a), with the radial stress including rho0 x potential
              ("u3new" in HC): the self-gravitating part of the body force, -rho0 grad(phi), is a gradient when
              rho0 is constant and is absorbed into the pressure
    P1, P2    potential perturbation phi and r dphi/dr, / (4 pi G a^2)
Between levels, ds/dx = A s with eta = eta(r) / eta0:
    [ -2        L          0   0     ]
    [ -1        1          0   1/eta ]
    [ 12 eta   -6 L eta    1   L     ]      and for (P1, P2):   [ 0  1 ]
    [ -6 eta    2(2L-1)eta -1  -2    ]                          [ L -1 ]
whose solution over a layer of constant viscosity is the matrix exponential.  A density sheet drho dr at radius r
adds r drho dr (in the units above) to R and to P2.  At the core-mantle boundary: U = 0, S = 0 (free slip), and the
potential is regular in the core (P2 = l P1 below) and jumps by the mass of the CMB deflection; at the surface: U = 0,
S = 0 (free slip) or V = 0 (no slip), and the potential is regular above (P2 = -(l+1) P1) after the jump by the
mass of the surface deflection.  For l = 1 the potential at the CMB is set to zero instead of the surface condition,
as in HC (the centre of mass fixes the frame).  The state at every level is solved for at once ("multiple
shooting", as HC's default solver), with long intervals split so that no single step grows by more than 1e4.

Outputs (all in the mantle frame of the run, lon/lat on a sphere):
    dyn_topo      surface topography from the geoid, -sigma_rr / (drho_load g), m; water-loaded by default
                  (drho_load = rho_top - rho_water) or air-loaded
    geoid         m
    cmb_topo      core-mantle boundary topography from its equipotential, sigma_rr / ((rho_core - rho_mantle) g), m
    kernels       topography, CMB and geoid kernels per degree and depth (definitions in kernels())
    with volume: velocities (cm/yr), radial and shear stresses on horizontal planes (MPa) and the
                 geoid-equivalent potential at every layer depth, theta = colatitude (positive south) and
                 phi = longitude (positive east), as in HC (HC has no horizontal stresses, so neither has this)
"""
import argparse
import glob
import os
import re
from dataclasses import dataclass, fields, asdict

import numpy as np
import xarray as xr

from .core import current_dvs
from .sphharm import Grid

SECONDS_PER_YEAR = 3.1556926e7

# PREM density (Dziewonski & Anderson 1981), g/cm3 as polynomials in x = r / 6371 km, by region (top radius, km)
_PREM_RHO = [
    (1221.5, (13.0885, 0.0, -8.8381, 0.0)),
    (3480.0, (12.5815, -1.2638, -3.6426, -5.5281)),
    (5701.0, (7.9565, -6.4761, 5.5283, -3.0807)),
    (5771.0, (5.3197, -1.4836, 0.0, 0.0)),
    (5971.0, (11.2494, -8.0298, 0.0, 0.0)),
    (6151.0, (7.1089, -3.8045, 0.0, 0.0)),
    (6346.6, (2.6910, 0.6924, 0.0, 0.0)),
    (6356.0, (2.9000, 0.0, 0.0, 0.0)),
    (6368.0, (2.6000, 0.0, 0.0, 0.0)),
    (6371.0, (1.0200, 0.0, 0.0, 0.0)),
]

# Steinberger & Calderwood (2006, GJI 167, 1461): (radius of the bottom of each layer / a, viscosity Pa s), each
# holding up to the next, as distributed with ASPECT (data/material-model/steinberger/rad_viscosity-source-data.txt,
# which ASPECT's Steinberger material model attributes to that paper)
_SC2006 = [
    (0.5448, 0.186E+21),
    (0.5668, 0.354E+22),
    (0.5888, 0.332E+23),
    (0.6107, 0.885E+23),
    (0.6327, 0.112E+24),
    (0.6547, 0.111E+24),
    (0.6767, 0.101E+24),
    (0.6986, 0.875E+23),
    (0.7206, 0.731E+23),
    (0.7426, 0.586E+23),
    (0.7646, 0.451E+23),
    (0.7865, 0.332E+23),
    (0.8085, 0.235E+23),
    (0.8305, 0.159E+23),
    (0.8525, 0.103E+23),
    (0.8744, 0.624E+22),
    (0.8964, 0.651E+21),
    (0.9184, 0.431E+21),
    (0.9356, 0.432E+21),
    (0.9466, 0.290E+21),
    (0.9655, 0.241E+21),
    (0.9843, 0.264E+23),
]

VISCOSITY_PRESETS = {                       # [depth of the top of each layer (km), viscosity (Pa s)]
    'uniform': [[0.0, 1e21]],
    'two_layer': [[0.0, 1e21], [660.0, 3e22]],
    'four_layer': [[0.0, 1e22], [100.0, 1e21], [410.0, 1e21], [660.0, 3e22]],
    'steinberger_calderwood2006': [[6371.0 * (1 - top), eta] for top, eta in
                                   zip([r for r, _ in _SC2006[1:]] + [1.0], [e for _, e in _SC2006])],
}


def prem_density(depth_km, radius_km=6371.0):
    """PREM density (kg/m3) at the given depths."""
    r = radius_km - np.asarray(depth_km, float)
    out = np.empty_like(r)
    bottom = 0.0
    for top, (a0, a1, a2, a3) in _PREM_RHO:
        sel = (r >= bottom) & (r <= top)
        x = r[sel] / 6371.0
        out[sel] = 1000.0 * (a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3)
        bottom = top
    return out


@dataclass
class DynTopoParams:
    """The `dynamic_topography:` block of a run config.  Defaults are stage 1 (CONTEXT.md)."""
    density_variable: str = 'dVs'           # the field that drives the flow
    cutoff_km: float = 300.0                # no density anomalies above this depth (deep dynamic topography)
    dlnrho_dlnvs: float = 0.25              # d ln rho / d ln Vs, constant with depth
    scale_with_prem: bool = True            # drho = ... x PREM density at the depth (False: x rho_mantle)
    viscosity: object = 'two_layer'         # a preset name, [[top depth km, Pa s], ...], or {file: HC format}
    eta0: float = 1e21                      # reference viscosity (Pa s); results do not depend on it
    boundary: str = 'free_slip'             # surface: free_slip or no_slip (zero velocity)
    self_gravitation: bool = True
    loading: str = 'water'                  # dyn_topo: water- or air-loaded
    rho_top: float = 3380.0                 # density of the deflected surface layer (kg/m3)
    rho_water: float = 1030.0
    rho_mantle: float = 4448.8              # reference mantle density in the flow solution (HC's average)
    rho_core: float = 11601.01              # average core density (HC), for the potential below the CMB
    rho_cmb_mantle: float = 5566.0          # PREM densities either side of the CMB: CMB topography
    rho_cmb_core: float = 9903.0
    g: float = 10.0                         # gravity, constant with depth as in HC (m/s2)
    G: float = 6.6742e-11
    radius_km: float = 6371.0
    cmb_radius_km: float = 3480.0
    lmax: int = 30                          # degrees computed; comparisons truncate further (headline 1-3)
    volume: bool = True                     # velocities and stresses at every layer depth
    max_growth: float = 1e4                 # split intervals so no step grows by more than this
    after_run: bool = False                 # run_mantle: compute dynamic topography after the run (0 Ma only)

    @classmethod
    def from_config(cls, block):
        known = {f.name for f in fields(cls)}
        unknown = set(block or {}) - known
        if unknown:
            raise ValueError('unknown dynamic_topography keys: {}'.format(sorted(unknown)))
        return cls(**(block or {}))


def viscosity_layers(spec, radius_km=6371.0):
    """(top depths km, viscosities Pa s), shallowest first, from a preset name, a list, or {file: HC format}
    (non-dimensional radius of the bottom of each layer and its viscosity, deepest first)."""
    if isinstance(spec, str):
        if spec not in VISCOSITY_PRESETS:
            raise ValueError('unknown viscosity preset {!r}: {}'.format(spec, sorted(VISCOSITY_PRESETS)))
        spec = VISCOSITY_PRESETS[spec]
    if isinstance(spec, dict):
        rows = np.loadtxt(spec['file'], ndmin=2)
        bottoms_km = radius_km * (1 - rows[:, 0])
        order = np.argsort(bottoms_km)
        # each HC entry holds from its radius upward to the next entry
        tops = np.concatenate([[0.0], bottoms_km[order][:-1]])
        return tops, rows[order, 1]
    spec = sorted([(float(d), float(e)) for d, e in spec])
    if spec[0][0] != 0.0:
        raise ValueError('the first viscosity layer has to start at the surface (depth 0)')
    return np.array([d for d, _ in spec]), np.array([e for _, e in spec])


def _matrix(l, eta):
    L = l * (l + 1.0)
    A = np.zeros((6, 6))
    A[0, :4] = [-2.0, L, 0.0, 0.0]
    A[1, :4] = [-1.0, 1.0, 0.0, 1.0 / eta]
    A[2, :4] = [12.0 * eta, -6.0 * L * eta, 1.0, L]
    A[3, :4] = [-6.0 * eta, 2.0 * (2.0 * L - 1.0) * eta, -1.0, -2.0]
    A[4, 4:] = [0.0, 1.0]
    A[5, 4:] = [L, -1.0]
    return A


def sheet_thickness(r):
    """Thickness (non-dimensional) of the mantle each sheet at radii r (increasing) stands for: halfway to its
    neighbours, the top one up to the surface and the bottom one down to the core-mantle boundary (HC)."""
    r = np.asarray(r, float)
    mid = (r[1:] + r[:-1]) / 2
    top = np.concatenate([mid, [1.0]])
    bottom = np.concatenate([[np.nan], mid])
    return top, bottom


class FlowSolver:
    """Responses of the flow to unit density sheets, for every degree up to lmax.

    r_sheets: radii of the density sheets / a (increasing); the outputs are at the CMB, every sheet and the surface.
    """

    def __init__(self, r_sheets, dp):
        self.dp = dp
        a = dp.radius_km * 1e3
        self.c = dp.cmb_radius_km / dp.radius_km
        self.r_sheets = np.asarray(r_sheets, float)
        top, bottom = sheet_thickness(self.r_sheets)
        bottom[0] = self.c
        self.dr = top - bottom
        tops_km, etas = viscosity_layers(dp.viscosity, dp.radius_km)
        self.visc_r = 1.0 - tops_km / dp.radius_km               # radius of the top of each layer
        self.visc_eta = etas / dp.eta0
        self.gamma = 4 * np.pi * dp.G * a / dp.g                 # potential coupling per kg/m3
        self.r_out = np.concatenate([[self.c], self.r_sheets, [1.0]])
        self._solved = {}

    def eta_at(self, r):
        """Viscosity (relative) of the layer containing r; at an interface, the layer above, as in HC."""
        r = np.atleast_1d(r)
        bottoms = np.concatenate([self.visc_r[1:], [self.c - 1]])  # bottom radius of each layer
        out = np.empty_like(r)
        for i, ri in enumerate(r):
            k = np.nonzero(ri >= bottoms - 1e-12)[0]
            out[i] = self.visc_eta[k[0]] if len(k) else self.visc_eta[-1]
        return out

    def _levels(self, l):
        """Level radii from the CMB to the surface: sheets, viscosity interfaces, outputs, split for growth."""
        coarse = np.unique(np.concatenate([[self.c, 1.0], self.r_sheets,
                                           self.visc_r[(self.visc_r > self.c) & (self.visc_r < 1.0)]]))
        levels = [coarse[0]]
        for r1, r2 in zip(coarse[:-1], coarse[1:]):
            n = max(1, int(np.ceil((2 * l + 3) * np.log(r2 / r1) / np.log(self.dp.max_growth) - 1e-10)))
            levels.extend(r1 * (r2 / r1) ** (np.arange(1, n + 1) / n))
        levels = np.array(levels)
        levels[-1] = 1.0
        return levels

    def solve(self, l):
        """States at the output radii for a unit density (1 kg/m3) in each sheet: array [n_out, 6, n_sheets]."""
        if l in self._solved:
            return self._solved[l]
        from scipy.linalg import expm
        dp = self.dp
        r = self._levels(l)
        n = len(r)
        x = np.log(r)
        src = {int(np.argmin(np.abs(r - rs))): j for j, rs in enumerate(self.r_sheets)}
        out_idx = [int(np.argmin(np.abs(r - ro))) for ro in self.r_out]
        mid = (r[:-1] + r[1:]) / 2
        eta = self.eta_at(mid)
        nsys = 6 * n
        M = np.zeros((nsys, nsys))
        F = np.zeros((nsys, len(self.r_sheets)))
        row = 0
        rho0 = dp.rho_mantle
        sg = 1.0 if dp.self_gravitation else 0.0
        # CMB: U = 0, S = 0, P2 = l P1 + c sigma_c with c sigma_c = R - c gamma (rho_core - rho0) P1 (x sg)
        M[row, 0] = 1; row += 1
        M[row, 3] = 1; row += 1
        M[row, 5] = 1; M[row, 4] = -l + sg * self.c * self.gamma * (dp.rho_core - rho0); M[row, 2] = -1; row += 1
        if l == 1:
            M[row, 4] = 1; row += 1
        for k in range(n - 1):
            D = expm(_matrix(l, eta[k]) * (x[k + 1] - x[k]))
            M[row:row + 6, 6 * (k + 1):6 * (k + 2)] = np.eye(6)
            M[row:row + 6, 6 * k:6 * (k + 1)] = -D
            if k + 1 in src:
                j = src[k + 1]
                F[row + 2, j] = r[k + 1] * self.dr[j]
                F[row + 5, j] = r[k + 1] * self.dr[j]
            row += 6
        s = 6 * (n - 1)
        if 0 in src:                         # a sheet on the CMB level itself is not allowed
            raise ValueError('a density sheet coincides with the core-mantle boundary')
        M[row, s + 0] = 1; row += 1
        M[row, s + (3 if dp.boundary == 'free_slip' else 1)] = 1; row += 1
        if l != 1:
            # P2(above) = P2 + sigma_top = -(l+1) P1, with sigma_top = -R - gamma rho0 P1 (x sg)
            M[row, s + 4] = (l + 1) - sg * self.gamma * rho0; M[row, s + 5] = 1; M[row, s + 2] = -1; row += 1
        if row != nsys:
            raise RuntimeError('system is {} x {}'.format(row, nsys))
        sol = np.linalg.solve(M, F).reshape(n, 6, -1)
        out = sol[out_idx]
        self._solved[l] = out
        return out

    # conversions from the non-dimensional state to physical units ---------------------------------------
    def units(self):
        dp = self.dp
        a = dp.radius_km * 1e3
        return dict(
            stress=dp.g * a,                                     # Pa per unit R / r
            velocity=dp.g * a ** 2 / dp.eta0,                    # m/s per unit U, V (for 1 kg/m3)
            potential=4 * np.pi * dp.G * a ** 2,                 # m2/s2 per unit P1
        )

    def load_density(self):
        dp = self.dp
        return dp.rho_top - (dp.rho_water if dp.loading == 'water' else 0.0)

    def kernels(self):
        """Kernels per degree and sheet, for unit density in a sheet of the given thickness:
            topography  A_l(r) = h drho_load / (drho dr): -1 for a fully compensated load at the surface
            cmb         C_l(r) = h_cmb (rho_cmb_core - rho_cmb_mantle) / (drho dr)
            geoid       G_l(r) = N (2l+1) g / (4 pi G a drho dr): r^(l+2) for an uncompensated sheet
        """
        lmax = self.dp.lmax
        K = {k: np.full((lmax + 1, len(self.r_sheets)), np.nan) for k in ('topography', 'cmb', 'geoid')}
        a = self.dp.radius_km * 1e3
        for l in range(1, lmax + 1):
            s = self.solve(l)
            K['topography'][l] = -s[-1, 2] / self.dr
            K['cmb'][l] = s[0, 2] / self.c / self.dr
            K['geoid'][l] = -s[-1, 4] * (2 * l + 1) / self.dr
        return K


def density_anomaly(ds, dp):
    """drho (kg/m3) [depth, lat, lon] from the run's field, zero above the cutoff."""
    f = ds[dp.density_variable].values.astype(float)
    depth = ds['depth'].values.astype(float)
    rho = prem_density(depth, dp.radius_km) if dp.scale_with_prem else np.full(len(depth), dp.rho_mantle)
    drho = dp.dlnrho_dlnvs * f / 100.0 * rho[:, None, None]
    drho[depth < dp.cutoff_km] = 0.0
    return drho


def _state_coeffs(solver, dcoef):
    """States at the output radii as coefficients: [n_out, 6, 2, l, m] from density coefficients [sheet, 2, l, m]."""
    lmax = solver.dp.lmax
    n_out = len(solver.r_out)
    out = np.zeros((n_out, 6, 2, lmax + 1, lmax + 1))
    for l in range(1, lmax + 1):
        out[:, :, :, l, :] = np.einsum('osj,jcm->oscm', solver.solve(l), dcoef[:, :, l, :])
    return out


def compute(ds, dp, components=True):
    """Dynamic topography, geoid, CMB topography and (with dp.volume) the flow, as an xr.Dataset."""
    lats, lons = ds['lat'].values, ds['lon'].values
    depth = ds['depth'].values.astype(float)
    grid = Grid(lats, lons, dp.lmax)
    solver = FlowSolver(1.0 - depth[::-1] / dp.radius_km, dp)    # sheets deepest first
    u = solver.units()
    a = dp.radius_km * 1e3

    def surface_fields(drho):
        dcoef = grid.analyse(drho[::-1])                          # deepest first, as the solver
        st = _state_coeffs(solver, dcoef)
        c_topo = -st[-1, 2] * u['stress'] / (solver.load_density() * dp.g)
        c_geoid = -st[-1, 4] * u['potential'] / dp.g
        c_cmb = st[0, 2] / solver.c * u['stress'] / ((dp.rho_cmb_core - dp.rho_cmb_mantle) * dp.g)
        return st, dcoef, c_topo, c_geoid, c_cmb

    drho = density_anomaly(ds, dp)
    st, dcoef, c_topo, c_geoid, c_cmb = surface_fields(drho)
    coords = {'lat': lats, 'lon': lons}
    lm = {'cs': [0, 1], 'l': np.arange(dp.lmax + 1), 'm': np.arange(dp.lmax + 1)}
    out = xr.Dataset(coords=coords)
    load = 'water' if dp.loading == 'water' else 'air'
    for name, c, units, long in (
            ('dyn_topo', c_topo, 'm', 'dynamic topography from the geoid, {}-loaded'.format(load)),
            ('geoid', c_geoid, 'm', 'geoid'),
            ('cmb_topo', c_cmb, 'm', 'core-mantle boundary topography from its equipotential')):
        out[name] = xr.DataArray(grid.synth(c).astype(np.float32), dims=['lat', 'lon'],
                                 attrs={'units': units, 'long_name': long})
        out[name + '_coeffs'] = xr.DataArray(c, dims=['cs', 'l', 'm'], coords=lm,
                                             attrs={'units': units, 'long_name': long + ', 4-pi normalised real '
                                                    'spherical harmonic coefficients (pyshtools convention)'})
    # sources: any dVs component the run saved (dvs_save_components)
    if components:
        for var in sorted(v for v in ds.data_vars if v.startswith(dp.density_variable + '__')):
            name = var.split('__', 1)[1]
            sub = DynTopoParams(**{**asdict(dp), 'density_variable': var})
            _, _, ct, cg, cc = surface_fields(density_anomaly(ds, sub))
            for key, c in (('dyn_topo', ct), ('geoid', cg), ('cmb_topo', cc)):
                out['{}_coeffs__{}'.format(key, name)] = xr.DataArray(c, dims=['cs', 'l', 'm'], coords=lm,
                                                                      attrs={'units': 'm', 'source': name})
    K = solver.kernels()
    for name, k in K.items():
        out['kernel_' + name] = xr.DataArray(k[:, ::-1], dims=['l', 'depth'],
                                             coords={'l': np.arange(dp.lmax + 1), 'depth': depth},
                                             attrs={'long_name': solver.kernels.__doc__.split('\n')[0]})
    out['kernel_topography'].attrs['definition'] = 'h drho_load / (drho dr); -1: fully compensated at the surface'
    out['kernel_cmb'].attrs['definition'] = 'h_cmb (rho_core - rho_mantle at the CMB) / (drho dr)'
    out['kernel_geoid'].attrs['definition'] = 'N (2l+1) g / (4 pi G a drho dr); r^(l+2): uncompensated sheet'
    out['density_anomaly_rms'] = xr.DataArray(np.sqrt((drho ** 2 * np.cos(np.radians(lats))[None, :, None]).sum((1, 2))
                                                      / (np.cos(np.radians(lats)).sum() * len(lons))),
                                              dims=['depth'], coords={'depth': depth}, attrs={'units': 'kg/m3'})
    if dp.volume:
        _add_volume(out, st, solver, grid, depth, u)
    out.attrs.update({k: (int(v) if isinstance(v, bool) else v if isinstance(v, (int, float, str)) else str(v))
                      for k, v in asdict(dp).items()})
    out.attrs['method'] = 'Hager & O\'Connell (1981) propagators, self-gravitating as HC; slablet.dyntopo'
    return out


def _add_volume(out, st, solver, grid, depth, u):
    """Velocities (cm/yr), radial and shear stresses (MPa) and the geoid-equivalent potential at each layer depth."""
    dp = solver.dp
    s = st[1:-1][::-1]                                            # the sheets, shallowest first = depth order
    r = solver.r_sheets[::-1]
    cm_yr = 100 * SECONDS_PER_YEAR
    U = s[:, 0] * u['velocity']                                   # [depth, 2, l, m] m/s
    V = s[:, 1] * u['velocity']
    R = s[:, 2] * u['stress'] / r[:, None, None, None]            # radial stress (with rho0 phi), Pa
    S = s[:, 3] * u['stress'] / r[:, None, None, None]
    dims = ['depth', 'lat', 'lon']
    cd = {'depth': depth}
    gt, gp = grid.gradient(V)
    out['u_r'] = xr.DataArray((grid.synth(U) * cm_yr).astype(np.float32), dims=dims, coords=cd,
                              attrs={'units': 'cm/yr', 'long_name': 'radial velocity (up)'})
    out['u_theta'] = xr.DataArray((gt * cm_yr).astype(np.float32), dims=dims, coords=cd,
                                  attrs={'units': 'cm/yr', 'long_name': 'colatitude velocity (positive south)'})
    out['u_phi'] = xr.DataArray((gp * cm_yr).astype(np.float32), dims=dims, coords=cd,
                                attrs={'units': 'cm/yr', 'long_name': 'longitude velocity (positive east)'})
    st_, sp_ = grid.gradient(S)
    for name, f, long in (('s_rr', grid.synth(R), 'radial normal stress (incl. rho0 x potential, HC u3new)'),
                          ('s_rt', st_, 'shear stress r-theta'), ('s_rp', sp_, 'shear stress r-phi')):
        out[name] = xr.DataArray((f / 1e6).astype(np.float32), dims=dims, coords=cd,
                                 attrs={'units': 'MPa', 'long_name': long})
    out['geoid_at_depth'] = xr.DataArray((grid.synth(-s[:, 4] * u['potential'] / dp.g)).astype(np.float32),
                                         dims=dims, coords=cd,
                                         attrs={'units': 'm', 'long_name': '-potential / g at each depth'})


def run(cfg, ages=None, log=print):
    """Compute dynamic topography for a run: dyntopo_NNNNMa.nc beside each mantle_temp_NNNNMa.nc."""
    from .paths import WORK
    dp = DynTopoParams.from_config(cfg.get('dynamic_topography'))
    out_dir = os.path.join(WORK, cfg['output_dir'])
    files = sorted(glob.glob(os.path.join(out_dir, 'mantle_temp_*Ma.nc')))
    if ages is not None:
        want = {int(round(a)) for a in ages}
        files = [f for f in files if int(re.search(r'_(\d+)Ma\.nc$', f).group(1)) in want]
    if not files:
        raise SystemExit('no mantle_temp files for ages {} in {}'.format(ages, out_dir))
    written = []
    for f in files:
        with xr.open_dataset(f) as ds:
            ds = current_dvs(ds)                                # runs before 11 Nov 2026
            if dp.density_variable not in ds:
                raise SystemExit('{} has no {}'.format(f, dp.density_variable))
            res = compute(ds.load(), dp)
            res.attrs['age_ma'] = ds.attrs.get('age_ma', np.nan)
            res.attrs['source_file'] = os.path.basename(f)
        path = f.replace('mantle_temp_', 'dyntopo_')
        res.to_netcdf(path)
        written.append(path)
        log('  [dyntopo] {}: rms dynamic topography {:.0f} m, geoid {:.1f} m'.format(
            os.path.basename(path), float(np.sqrt((res['dyn_topo_coeffs'] ** 2).sum())),
            float(np.sqrt((res['geoid_coeffs'] ** 2).sum()))))
    return written


def main(argv=None):
    import yaml
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('config')
    ap.add_argument('--ages', type=float, nargs='*', default=[0.0], help='ages to compute (Ma); default 0')
    ap.add_argument('--all-ages', action='store_true', help='every saved age')
    args = ap.parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    run(cfg, None if args.all_ages else args.ages)


if __name__ == '__main__':
    main()
