"""The asthenosphere layer's supply mixture (slablet/asthenosphere_supply.py and Forcing.supply_share): shares sum to
one over the fed cells, an uneven share moves the supply without changing its total, and a downwelling withholds."""
import numpy as np
import pytest
import xarray as xr

from slablet import asthenosphere as A
from slablet import asthenosphere_supply as S
from slablet import sphere_mesh as SM


@pytest.fixture(scope='module')
def mesh():
    return SM.voronoi_mesh(3000)


class FakeMantle:
    """u_r = +1 cm/yr in the northern hemisphere and -1 in the southern; one plume at 0E 0N."""

    def u_r(self, t):
        lat = np.arange(-89.5, 90, 1.0)
        lon = np.arange(-179.5, 180, 1.0)
        return xr.DataArray(np.where(lat[:, None] > 0, 1.0, -1.0) * np.ones(len(lon)), dims=('lat', 'lon'),
                            coords={'lat': lat, 'lon': lon})

    def plumes(self, t):
        return np.array([0.0]), np.array([0.0])


def _forcing(mesh, closed=None):
    z = np.zeros(len(mesh.edge_i))
    return A.Forcing(z, z, closed=closed, sink=np.zeros(mesh.n))


def test_each_part_and_the_mixture_sum_to_one(mesh):
    keel = mesh.lat > 70
    for weights in ({'uniform': 1}, {'deep': 1}, {'plume': 1}, {'uniform': 1, 'deep': 2, 'plume': 1}):
        s = S.mixture(mesh, weights, mantle=FakeMantle())(10.0, _forcing(mesh, keel))
        assert s.sum() == pytest.approx(1.0)
        assert np.all(s[keel] == 0)


def test_deep_part_withholds_above_downwellings(mesh):
    s = S.mixture(mesh, {'deep': 1}, mantle=FakeMantle())(0.0, _forcing(mesh))
    assert np.all(s[mesh.lat < 0] == 0) and np.all(s[mesh.lat > 0.6] > 0)


def test_plume_part_lies_within_its_radius(mesh):
    s = S.mixture(mesh, {'plume': 1}, mantle=FakeMantle())(0.0, _forcing(mesh))
    far = np.degrees(np.arccos(np.clip(mesh.xyz[:, 0] / np.linalg.norm(mesh.xyz, axis=1), -1, 1))) * \
        np.pi / 180 * mesh.radius_km > S.PLUME_RADIUS_KM
    assert np.all(s[far] == 0) and s.sum() == pytest.approx(1.0)


def test_unknown_parts_and_missing_mantle_are_refused(mesh):
    with pytest.raises(ValueError):
        S.mixture(mesh, {'slab': 1})
    with pytest.raises(ValueError):
        S.mixture(mesh, {'deep': 1})


def test_uneven_share_moves_the_supply_but_keeps_its_total(mesh):
    p = A.AsthenosphereParams(viscosity_pa_s=1e30)             # no pressure-driven flow: only the supply moves h
    sink = np.zeros(mesh.n)
    sink[np.argmin(np.abs(mesh.lat))] = 1e5                     # one ridge cell takes 1e5 km^3/Myr
    z = np.zeros(len(mesh.edge_i))
    h0 = np.full(mesh.n, 300.0)
    even = A.advance(mesh, h0, A.Forcing(z, z, sink=sink), p, 1.0)
    north = np.where(mesh.lat > 0, mesh.area, 0.0)
    uneven = A.advance(mesh, h0, A.Forcing(z, z, sink=sink, supply_share=north), p, 1.0)
    for h, log in (even, uneven):
        assert log.supplied == pytest.approx(1e5)
        assert (h * mesh.area).sum() == pytest.approx((h0 * mesh.area).sum())
    assert np.all(uneven[0][mesh.lat < -1] <= 300.0 + 1e-9)
    assert uneven[0][mesh.lat > 1].mean() > even[0][mesh.lat > 1].mean()
