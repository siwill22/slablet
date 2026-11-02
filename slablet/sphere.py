"""Spherical helpers used by the mantle model (and re-exported by self_driving_plates)."""
import numpy as np
from sklearn.neighbors import BallTree


def lonlat_to_xyz(lon, lat):
    """Unit vectors (..., 3) for lon/lat in degrees."""
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat_r) * np.cos(lon_r),
                     np.cos(lat_r) * np.sin(lon_r),
                     np.sin(lat_r)], axis=-1)


def xyz_to_lonlat(xyz):
    """lon, lat in degrees for unit vectors (..., 3)."""
    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    lat = np.degrees(np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return lon, lat


def move_along_azimuth(lon, lat, azimuth_deg, distance_km, radius_km):
    """lon, lat (degrees) after moving distance_km along the great circle that starts at
    azimuth_deg (clockwise from north), on a sphere of radius_km (scalar or per point)."""
    lat1, lon1 = np.radians(lat), np.radians(lon)
    az, d = np.radians(azimuth_deg), np.asarray(distance_km, float) / radius_km
    lat2 = np.arcsin(np.clip(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(az), -1.0, 1.0))
    lon2 = lon1 + np.arctan2(np.sin(az) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return (np.degrees(lon2) + 180.0) % 360.0 - 180.0, np.degrees(lat2)


def spherical_gaussian_smooth(field, lats, lons, sigma_deg):
    """Gaussian smoothing on the sphere of one 2-D field (nlat, nlon) or a stack
    (..., nlat, nlon), using great-circle distances and cell-area weights.

    Unlike scipy.ndimage.gaussian_filter on a lon/lat grid, the kernel has the same
    physical width everywhere, wraps across ±180 and passes smoothly over the poles.
    The sparse kernel is cached per grid/sigma.
    """
    if sigma_deg is None or sigma_deg <= 0:
        return field
    K = _spherical_kernel(tuple(np.round(lats, 6)), tuple(np.round(lons, 6)), float(sigma_deg))
    shape = field.shape
    flat = field.reshape(-1, shape[-2] * shape[-1])
    out = (K @ flat.T).T
    return out.reshape(shape)


_KERNEL_CACHE = {}


def _spherical_kernel(lats, lons, sigma_deg):
    key = (lats, lons, sigma_deg)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]
    from scipy.sparse import csr_matrix
    lon2d, lat2d = np.meshgrid(lons, lats)
    pts = np.radians(np.column_stack([lat2d.ravel(), lon2d.ravel()]))
    area = np.clip(np.cos(pts[:, 0]), 1e-6, None)
    tree = BallTree(pts, metric='haversine')   # radius query; built once per grid (cached)
    idx, dist = tree.query_radius(pts, r=np.radians(4 * sigma_deg), return_distance=True)
    rows = np.repeat(np.arange(len(pts)), [len(i) for i in idx])
    cols = np.concatenate(idx)
    w = np.exp(-0.5 * (np.degrees(np.concatenate(dist)) / sigma_deg) ** 2) * area[cols]
    K = csr_matrix((w, (rows, cols)), shape=(len(pts), len(pts)))
    row_sums = np.asarray(K.sum(axis=1)).ravel()
    K = csr_matrix(K.multiply(1.0 / row_sums[:, None]))
    _KERNEL_CACHE[key] = K
    return K
