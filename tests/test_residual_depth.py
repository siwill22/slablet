"""Tests for slablet.analysis.residual_depth: the sediment correction, the cell means and, where the data are on
disk, that a residual differs between cooling laws only by the laws' depths at the same ages."""
import os

import numpy as np
import pytest
import xarray as xr

from slablet import cooling
from slablet.analysis import residual_depth as rd


def test_sediment_correction():
    assert rd.sediment_correction_m(0.0) == 0.0
    # a thin layer has the surface porosity: (3330 - (0.66 x 1030 + 0.34 x 2647)) / 2300 of its thickness
    thin = (rd.MANTLE_DENSITY - (0.66 * 1030 + 0.34 * 2647)) / (rd.MANTLE_DENSITY - rd.WATER_DENSITY)
    assert rd.sediment_correction_m(1.0) == pytest.approx(thin, rel=1e-3)
    s = np.array([100.0, 500, 1000, 3000, 8000])
    c = rd.sediment_correction_m(s)
    assert np.all(np.diff(c) > 0) and np.all(c < s)
    # compaction makes each extra metre of a thick column denser, so it adds less depth than the first metre did
    assert np.all(np.diff(c / s) < 0)


def test_cell_means_are_area_means():
    lon = np.round(np.arange(100.0, 103.0 + 1e-9, 0.1), 6)
    lat = np.round(np.arange(-60.0, -57.0 + 1e-9, 0.1), 6)
    field = xr.DataArray(np.broadcast_to(lat[:, None], (len(lat), len(lon))).copy(), dims=('lat', 'lon'),
                         coords={'lat': lat, 'lon': lon})
    c = rd.cell_means(field, 1.0)
    assert list(c.lon.values) == [100.5, 101.5, 102.5] and list(c.lat.values) == [-59.5, -58.5, -57.5]
    nodes = lat[(lat >= -60) & (lat < -59 - 1e-9)]
    w = np.cos(np.radians(nodes))
    assert float(c.sel(lat=-59.5, lon=100.5)) == pytest.approx(np.sum(w * nodes) / np.sum(w))
    field[:12, :12] = np.nan                    # more than half of the first cell's nodes missing
    assert np.isnan(float(rd.cell_means(field, 1.0).sel(lat=-59.5, lon=100.5)))


def test_regions_crossing_the_dateline_are_refused():
    with pytest.raises(ValueError):
        rd._check_region((170.0, -170.0, -60.0, -40.0))


_have = all(os.path.exists(p) for k, p in rd._paths().items() if k in ('age', 'sediment'))


@pytest.mark.skipif(not _have, reason='Seton (2020) ages or GlobSed not under $SLABLET_DATA')
def test_laws_differ_only_by_their_depths():
    region = (120.0, 124.0, -52.0, -48.0)
    obs = rd.observations(region)
    a = rd.residual_depth(region, 'gdh1', obs=obs)
    b = rd.residual_depth(region, 'ps77', obs=obs)
    ok = np.isfinite(a.residual_m.values)
    assert ok.mean() > 0.9
    age = obs.age_myr.values[ok]
    want = cooling.depth_m(cooling.get('gdh1'), age) - cooling.depth_m(cooling.get('ps77'), age)
    assert np.allclose(a.residual_m.values[ok] - b.residual_m.values[ok], want)
    assert a.attrs['cooling_law'] == 'gdh1'
