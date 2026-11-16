"""Cooling laws: temperature, seafloor depth, heat flow and lithosphere thickness from one thermal age.

    law = cooling.get('gdh1')                       # the default; cooling.LAWS lists all of them
    cooling.depth_m(law, age_myr)                   # seafloor depth below sea level (m, positive down)
    cooling.heat_flow_mw_m2(law, age_myr)           # surface heat flow (mW/m^2)
    cooling.thickness_km(law, age_myr)              # depth of the lithosphere base, one isotherm (km)
    cooling.temperature_c(law, age_myr, z_km)       # the temperature field itself (deg C)

Within each law all four come from one temperature field (decision 11 of
docs/plans/2026-11-08-cross-repo-reconciliation.md): depth by isostasy, heat flow by conduction at the surface and
thickness as the depth of the isotherm ISOTHERM_FRACTION x the law's mantle temperature (0.9, the isotherm IPV's
2.32 sqrt(kappa t) uses).  Ages are in Myr, depths below the top of the plate in km.

    half_space  erf cooling with GDH1's properties and no base: 2600 + 365 sqrt(t) m
    gdh1        plate model, Stein & Stein (1992): 95 km, 1450 C           (the default)
    ps77        plate model, Parsons & Sclater (1977): 125 km, 1350 C
    crosby2007  plate model, Crosby (2007, PhD thesis; age2depth.c, as in pyBacktrack): 102 km, 1333 C, with its
                empirical depth term (a 300 m sinusoid about 94 Myr), the one term outside the temperature field
    rhcw18      Richards et al. (2018, 2020): finite-difference plate model with temperature- and
                pressure-dependent properties and a 7 km crust (property laws of Grose & Afonso 2013); Tp 1333 C,
                130 km, ridge 2500 m.  Read from the authors' own output (Tgrid, depth and hf files of
                github.com/freddrichards/RHCW18_Plate_Model) in $SLABLET_DATA/CoolingModels/RHCW18_Plate_Model
                ($SLABLET_DATA defaults to ~/Data); ages beyond its 300 Myr are held at 300 Myr.

The analytic laws switch to the half-space form when pi^2 kappa t / a^2 < 0.05, where the two differ by less than
exp(-pi^2 / 0.2) of the plate's amplitude, so the plate series never has to converge near the ridge.
No pygplates: this is shared physics, used by synthetic worlds and real reconstructions alike.
"""
import os
from dataclasses import dataclass

import numpy as np
from scipy.special import erf, erfinv

SECONDS_PER_MYR = 1e6 * 365.25 * 24 * 3600
ISOTHERM_FRACTION = 0.9
DEFAULT = 'gdh1'
_SWITCH = 0.05            # pi^2 kappa t / a^2 below which the plate series is replaced by the half-space form
_N_TERMS = 40             # plate-series terms; exp(-n^2 x) < 1e-30 for n = 40 at x >= 0.05


@dataclass(frozen=True)
class CoolingLaw:
    name: str
    kind: str                         # 'half_space' | 'plate' | 'table'
    ridge_depth_m: float
    mantle_temperature_c: float       # the base (plate) or far-field (half-space) temperature; Tp for 'table'
    plate_thickness_km: float = np.inf
    diffusivity_m2_s: float = np.nan
    expansivity_per_k: float = np.nan
    mantle_density: float = np.nan
    water_density: float = np.nan
    conductivity_w_m_k: float = np.nan
    crosby_bump: bool = False
    reference: str = ''


_GDH1 = dict(mantle_temperature_c=1450.0, diffusivity_m2_s=3.138 / (3330.0 * 1171.0), expansivity_per_k=3.1e-5,
             mantle_density=3330.0, water_density=1000.0, conductivity_w_m_k=3.138, ridge_depth_m=2600.0)

LAWS = {
    'half_space': CoolingLaw('half_space', 'half_space', reference='half-space cooling with the properties of GDH1 '
                             '(Stein & Stein 1992)', **_GDH1),
    'gdh1': CoolingLaw('gdh1', 'plate', plate_thickness_km=95.0, reference='Stein & Stein (1992)', **_GDH1),
    'ps77': CoolingLaw('ps77', 'plate', ridge_depth_m=2500.0, mantle_temperature_c=1350.0, plate_thickness_km=125.0,
                       diffusivity_m2_s=8.0e-7, expansivity_per_k=3.2e-5, mantle_density=3330.0, water_density=1000.0,
                       conductivity_w_m_k=3.138, reference='Parsons & Sclater (1977)'),
    # Crosby gives no conductivity; heat flow uses 3.138 W/m/K, the value of PS77 and GDH1
    'crosby2007': CoolingLaw('crosby2007', 'plate', ridge_depth_m=2600.0, mantle_temperature_c=1333.0,
                             plate_thickness_km=102.0, diffusivity_m2_s=7.8e-7, expansivity_per_k=3.2e-5,
                             mantle_density=3300.0, water_density=1030.0, conductivity_w_m_k=3.138, crosby_bump=True,
                             reference='Crosby (2007), age2depth.c as in pyBacktrack'),
    'rhcw18': CoolingLaw('rhcw18', 'table', ridge_depth_m=2500.0, mantle_temperature_c=1333.0,
                         plate_thickness_km=130.0,
                         reference='Richards et al. (2018, 2020), github.com/freddrichards/RHCW18_Plate_Model'),
}


def get(name=None):
    """The cooling law of this name (None: the default, GDH1)."""
    name = DEFAULT if name is None else name
    if name not in LAWS:
        raise ValueError('unknown cooling law {!r}; choose from {}'.format(name, ', '.join(LAWS)))
    return LAWS[name]


# ---- the analytic laws ------------------------------------------------------------------------------------------

def _x(law, t):
    """pi^2 kappa t / a^2: the plate's dimensionless age (0 for a half-space)."""
    if law.kind == 'half_space':
        return np.zeros_like(t)
    a = law.plate_thickness_km * 1e3
    return np.pi ** 2 * law.diffusivity_m2_s * t * SECONDS_PER_MYR / a ** 2


def _amplitude_m(law):
    """Depth change from fully hot to fully cooled, per metre of cooled layer: rho_m alpha Tm / (rho_m - rho_w)."""
    return law.mantle_density * law.expansivity_per_k * law.mantle_temperature_c / (law.mantle_density - law.water_density)


def _crosby_bump_m(t):
    return 300.0 * np.exp(-((t - 94.0) / 30.0) ** 2) * np.sin(t / 15.0 - 0.2)


def _ages(age_myr):
    t = np.asarray(age_myr, dtype=float)
    if np.any(t < 0):
        raise ValueError('ages must be non-negative')
    return t


def temperature_c(law, age_myr, z_km):
    """Temperature (deg C) at depth z_km below the top of the plate; age and z broadcast together."""
    t, z = np.broadcast_arrays(_ages(age_myr), np.asarray(z_km, dtype=float))
    if law.kind == 'table':
        return _table(law).temperature_c(t, z)
    tm = law.mantle_temperature_c
    with np.errstate(divide='ignore', invalid='ignore'):
        hs = tm * erf(z * 1e3 / (2.0 * np.sqrt(law.diffusivity_m2_s * t * SECONDS_PER_MYR)))
    hs = np.where(t > 0, hs, np.where(z > 0, tm, 0.0))
    if law.kind == 'half_space':
        return hs
    a = law.plate_thickness_km
    x = _x(law, t)
    n = np.arange(1, _N_TERMS + 1).reshape((-1,) + (1,) * t.ndim)
    series = (2.0 / np.pi) * np.sum(np.exp(-n ** 2 * x) / n * np.sin(n * np.pi * np.minimum(z, a) / a), axis=0)
    plate = tm * (np.minimum(z, a) / a + series)
    return np.where(z >= a, tm, np.where(x < _SWITCH, np.minimum(hs, tm), plate))


def depth_m(law, age_myr):
    """Seafloor depth below sea level (m, positive down) by isostasy of the temperature field."""
    t = _ages(age_myr)
    if law.kind == 'table':
        return _table(law).depth_m(t)
    hs = law.ridge_depth_m + 2.0 * _amplitude_m(law) * np.sqrt(law.diffusivity_m2_s * t * SECONDS_PER_MYR / np.pi)
    if law.kind == 'half_space':
        return hs
    a = law.plate_thickness_km * 1e3
    x = _x(law, t)
    n = np.arange(1, 2 * _N_TERMS, 2).reshape((-1,) + (1,) * t.ndim)
    plate = law.ridge_depth_m + _amplitude_m(law) * a * (0.5 - (4.0 / np.pi ** 2) * np.sum(np.exp(-n ** 2 * x) / n ** 2, axis=0))
    d = np.where(x < _SWITCH, hs, plate)
    if law.crosby_bump:
        d = d - _crosby_bump_m(t)
    return d


def heat_flow_mw_m2(law, age_myr):
    """Surface heat flow (mW/m^2): conductivity x the surface temperature gradient (infinite at age 0)."""
    t = _ages(age_myr)
    if law.kind == 'table':
        return _table(law).heat_flow_mw_m2(t)
    k, tm = law.conductivity_w_m_k, law.mantle_temperature_c
    with np.errstate(divide='ignore'):
        hs = 1e3 * k * tm / np.sqrt(np.pi * law.diffusivity_m2_s * t * SECONDS_PER_MYR)
    if law.kind == 'half_space':
        return hs
    a = law.plate_thickness_km * 1e3
    x = _x(law, t)
    n = np.arange(1, _N_TERMS + 1).reshape((-1,) + (1,) * t.ndim)
    plate = 1e3 * k * tm / a * (1.0 + 2.0 * np.sum(np.exp(-n ** 2 * x), axis=0))
    return np.where(x < _SWITCH, hs, plate)


def thickness_km(law, age_myr, isotherm_fraction=ISOTHERM_FRACTION):
    """Depth (km) of the isotherm isotherm_fraction x the law's mantle temperature: the lithosphere base."""
    t = _ages(age_myr)
    if law.kind == 'half_space':
        kt = law.diffusivity_m2_s * t * SECONDS_PER_MYR
        return 2.0 * erfinv(isotherm_fraction) * np.sqrt(kt) / 1e3
    ages, th = _thickness_table(law, isotherm_fraction)
    return np.interp(t, ages, th)


_THICKNESS = {}


def _thickness_table(law, fraction):
    """Isotherm depth on a fine age table (0-400 Myr, every 0.1 Myr), from the temperature on a 0.05 km grid."""
    key = (law, fraction)
    if key not in _THICKNESS:
        ages = np.arange(0.0, 400.05, 0.1)
        z = np.arange(0.0, law.plate_thickness_km + 1e-9, 0.05)
        temp = temperature_c(law, ages[:, None], z[None, :])
        target = fraction * law.mantle_temperature_c
        th = np.empty(len(ages))
        for i, row in enumerate(temp):
            j = int(np.argmax(row >= target))         # first node at or past the isotherm
            th[i] = 0.0 if j == 0 else z[j - 1] + (target - row[j - 1]) / (row[j] - row[j - 1]) * (z[j] - z[j - 1])
        _THICKNESS[key] = (ages, th)
    return _THICKNESS[key]


# ---- rhcw18: the authors' own output ----------------------------------------------------------------------------

def data_dir():
    from slablet.paths import DATA
    return os.path.join(DATA, 'CoolingModels',
                        'RHCW18_Plate_Model')


class _Table:
    """RHCW18's output, read once: Tgrid (age, depth below the plate top on nodes that contract with age, T),
    depth and hf against age.  Cached as an npz beside the files."""

    def __init__(self, folder, tp=1333, zp=130, zr=2500):
        cache = os.path.join(folder, 'slablet_cache_{}-{}-{}.npz'.format(tp, zp, zr))
        if not os.path.exists(cache):
            names = ['Tgrid-{}-{}.dat'.format(tp, zp), 'depth-{}-{}-{}.dat'.format(tp, zp, zr), 'hf-{}-{}.dat'.format(tp, zp)]
            missing = [n for n in names if not os.path.exists(os.path.join(folder, n))]
            if missing:
                raise FileNotFoundError('rhcw18 needs {} in {} (git clone '
                                        'https://github.com/freddrichards/RHCW18_Plate_Model there, or set '
                                        'SLABLET_DATA)'.format(', '.join(missing), folder))
            g = np.loadtxt(os.path.join(folder, names[0]))
            ages = np.unique(g[:, 0])
            nz = len(g) // len(ages)
            if nz * len(ages) != len(g):
                raise ValueError('{}: not a regular age x node grid'.format(names[0]))
            dep = np.loadtxt(os.path.join(folder, names[1]))
            hf = np.loadtxt(os.path.join(folder, names[2]))
            np.savez_compressed(cache + '.tmp.npz', age=ages, z=g[:, 1].reshape(len(ages), nz),
                                temp=g[:, 3].reshape(len(ages), nz), depth_age=dep[:, 0], depth=dep[:, 1],
                                hf_age=hf[:, 0], hf=hf[:, 1])
            os.replace(cache + '.tmp.npz', cache)
        with np.load(cache) as f:
            self.age, self.z, self.temp = f['age'], f['z'], f['temp']
            self.depth_age, self.depth, self.hf_age, self.hf = f['depth_age'], f['depth'], f['hf_age'], f['hf']

    def _clip(self, t):
        return np.minimum(t, self.age[-1])

    def depth_m(self, t):
        return np.interp(self._clip(t), self.depth_age, self.depth)

    def heat_flow_mw_m2(self, t):
        return 1e3 * np.interp(self._clip(t), self.hf_age, self.hf)

    def _profile(self, t):
        """Temperature on a fixed 0-130 km grid at age t, interpolated linearly between the two nearest ages."""
        t = float(min(t, self.age[-1]))
        i = int(np.clip(np.searchsorted(self.age, t) - 1, 0, len(self.age) - 2))
        w = (t - self.age[i]) / (self.age[i + 1] - self.age[i])
        zz = np.arange(0.0, 130.0 + 1e-9, 0.05)
        prof = [np.interp(zz, self.z[j], self.temp[j]) for j in (i, i + 1)]
        return zz, (1 - w) * prof[0] + w * prof[1]

    def temperature_c(self, t, z):
        out = np.empty(t.shape)
        for tu in np.unique(t):
            sel = t == tu
            zz, prof = self._profile(tu)
            out[sel] = np.interp(z[sel], zz, prof)
        return out


_TABLES = {}


def _table(law):
    if law.name not in _TABLES:
        _TABLES[law.name] = _Table(data_dir())
    return _TABLES[law.name]
