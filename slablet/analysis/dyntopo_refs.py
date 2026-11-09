"""Present-day references for slablet.dyntopo, and the measures used to compare with them.

All fields are returned as 4-pi normalised real spherical harmonic coefficients in metres (pyshtools convention,
c[cs, l, m]), on a sphere, lon/lat in the present-day frame.

References (paths default to this machine's ~/Data; set SLABLET_DATA to move them):
    steinberger(name)   present-day dynamic topography from Bernhard Steinberger's flow models driven by tomography
                        (2007/08 runs, packaged for the ICONS project by C. Heine).  smean_2mba, s20rts_4mba and
                        ngrand_3mba leave out the shallowest 2, 4 (to 200 km) and 3 (to 212.5 km) layers of the
                        tomography.  Units m; whether air- or water-loaded is not recorded (water-loaded is 1/0.685
                        times air-loaded with their densities), so only their pattern is used without that caveat.
    holdt_points()      oceanic residual depths, Holdt et al. (2022, JGR 127, e2022JB024391), Mendeley Data
                        10.17632/f79bw527cg.1: 'circles' are complete measurements, 'up' and 'down' triangles are
                        bounds; water-loaded, km in the file, m here.
    holdt_fit(lmax)     their spherical harmonic fit (degrees 1-10, 20, 30 or 40), as their grid.
    dunn_points()       Southern Ocean residual depths south of 50 S, Dunn et al. (2026, JGR 131, e2025JB033101),
                        Zenodo 10.5281/zenodo.17234276 (2,104 points; extends Holdt et al. there): symbol 'c' complete,
                        'i' (down triangles) upper limits, 't' (up triangles) lower limits; km in the file, m here.
    vd2020(dataset)     Valentine & Davies (2020, G3 21, e2020GC009240): the posterior mean of their Gaussian process
                        fit to the spot measurements of Davies et al. (2019), from their GitHub supplementary data.
    observed_geoid()    EGM2008 (Pavlis et al. 2012) with the hydrostatic shape of Chambat et al. (2010) removed,
                        to degree 31, as distributed with HC (geoid/egm2008-hc-geoid.chambat.31.ab).
"""
import os

import numpy as np
import xarray as xr

from slablet.sphharm import Grid

DATA = os.environ.get('SLABLET_DATA', os.path.expanduser('~/Data'))
HC_HOME = os.environ.get('HC_HOME', os.path.expanduser('~/GIT/hc'))
LATS = np.arange(-89.5, 90, 1.0)
LONS = np.arange(-179.5, 180, 1.0)
LMAX = 30

STEINBERGER = {
    'SMEAN': 'smean_2mba_dynto-0.nc',
    'S20RTS': 's20rts_4mba_dynto-0.nc',
    'NGRAND': 'ngrand_3mba_dynto-0.nc',
}

_grid = None


def grid():
    global _grid
    if _grid is None:
        _grid = Grid(LATS, LONS, LMAX)
    return _grid


def _on_grid(da, xname, yname):
    """A node grid (any longitude range) interpolated to the 1 deg cell centres."""
    lon = ((da[xname].values + 180.0) % 360.0) - 180.0
    da = da.assign_coords({xname: lon}).sortby(xname)
    da = da.isel({xname: np.unique(da[xname].values, return_index=True)[1]})
    da = xr.concat([da.isel({xname: [-1]}).assign_coords({xname: [da[xname].values[-1] - 360.0]}), da,
                    da.isel({xname: [0]}).assign_coords({xname: [da[xname].values[0] + 360.0]})], dim=xname)
    out = da.interp({xname: LONS, yname: LATS}).transpose(yname, xname).values
    if not np.isfinite(out).all():
        raise ValueError('the grid does not cover the 1 deg cell centres')
    return out


def steinberger(name):
    path = os.path.join(DATA, 'DynamicTopography', STEINBERGER[name])
    with xr.open_dataset(path) as ds:
        return grid().analyse(_on_grid(ds['z'].astype(float), 'x', 'y'))


def holdt_dir():
    return os.path.join(DATA, 'residual_topography', 'holdt2022', 'Residual depthtopographic measurements',
                        'Holdt_et_al_2022_SupportingInformation')


def holdt_points(kind='circles'):
    """(lon, lat, value m, error m) of the oceanic measurements of one kind: circles, up or down."""
    name = {'circles': 'circles', 'up': 'up_triangles', 'down': 'down_triangles'}[kind]
    d = np.loadtxt(os.path.join(holdt_dir(), 'oceanic_measurements_{}_1_deg.xyz'.format(name)))
    return d[:, 0], d[:, 1], d[:, 2] * 1e3, d[:, 4] * 1e3


def dunn_points(kinds=('c', 'i', 't')):
    """(lon, lat, value m, symbol) of Dunn et al. (2026) residual depths of the given symbols."""
    import pandas as pd
    d = pd.read_csv(os.path.join(DATA, 'residual_topography', 'dunn2026', 'ResidualDepths_SouthernOcean.csv')).dropna()
    d = d[d['Plotting_Symbol'].isin(kinds)]
    return (d['Longitude'].values, d['Latitude'].values, d['Residual_Depth_(km)'].values * 1e3,
            d['Plotting_Symbol'].values)


def holdt_fit(lmax=40):
    with xr.open_dataarray(os.path.join(holdt_dir(), 'degree_1_to_{}.grd'.format(lmax))) as g:
        return grid().analyse(_on_grid(g.astype(float) * 1e3, 'lon', 'lat'))


def vd2020(dataset='high_accuracy_spot'):
    """Valentine & Davies (2020) posterior mean; their basis is pyshtools' / sqrt(4 pi), cosine terms at m <= 0."""
    path = os.path.join(DATA, 'residual_topography', 'DynamicTopographyGP', 'paper_supplementary_data',
                        'sph_coeff_{}.dat'.format(dataset))
    rows = np.loadtxt(path, comments='#')
    lmax = int(rows[:, 0].max())
    c = np.zeros((2, lmax + 1, lmax + 1))
    for l, m, y, _ in rows:
        c[0 if m <= 0 else 1, int(l), abs(int(m))] = y * 1e3 / np.sqrt(4 * np.pi)
    return c


def observed_geoid():
    """Non-hydrostatic geoid (m); HC's harmonics are pyshtools' x (-1)^m / sqrt(4 pi)."""
    path = os.path.join(HC_HOME, 'geoid', 'egm2008-hc-geoid.chambat.31.ab')
    rows = [r.split() for r in open(path) if r.strip()]
    lmax = int(rows[0][0])
    c = np.zeros((2, lmax + 1, lmax + 1))
    i = 1
    for l in range(lmax + 1):
        for m in range(l + 1):
            c[0, l, m], c[1, l, m] = float(rows[i][0]), float(rows[i][1])
            i += 1
    return c * (-1.0) ** np.arange(lmax + 1) / np.sqrt(4 * np.pi)


# measures ------------------------------------------------------------------------------------------------------
def at_points(c, lon, lat, lmax):
    """The field of coefficients c, truncated at lmax, at the points."""
    import pyshtools as sh
    k = min(lmax, c.shape[-1] - 1)
    cc = np.zeros((2, k + 1, k + 1))
    cc[:, :k + 1, :k + 1] = c[:, :k + 1, :k + 1]
    cc[:, 0, 0] = 0.0
    return sh.SHCoeffs.from_array(cc).expand(lat=np.asarray(lat), lon=np.asarray(lon))


def point_correlation(c, lon, lat, values, lmax):
    """Pearson correlation between the field (degrees 1..lmax) and the measurements at their locations."""
    v = at_points(c, lon, lat, lmax)
    return float(np.corrcoef(v, values)[0, 1])


def rms(c, lmin, lmax):
    """rms of the field over degrees lmin..lmax (m)."""
    return float(np.sqrt((np.asarray(c)[:, lmin:lmax + 1] ** 2).sum()))


def geoid_scale(model, observed, lmin=2, lmax=8):
    """The single factor s minimising |s model - observed|^2 over degrees lmin..lmax, and the correlation there."""
    a = np.asarray(model)[:, lmin:lmax + 1, :lmax + 1]
    b = np.asarray(observed)[:, lmin:lmax + 1, :lmax + 1]
    s = float((a * b).sum() / (a * a).sum())
    r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    return s, r
