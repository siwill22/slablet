"""Real spherical harmonics on slablet's cell-centred lon/lat grids.

Conventions are pyshtools' defaults: 4-pi normalised real harmonics without the Condon-Shortley phase, so that
the mean of Y_lm^2 over the sphere is 1, with coefficients stored as c[..., 0 (cos) or 1 (sin), l, m].

The grids are the mantle grids (cell centres, lat -89.5 ... 89.5 and lon -179.5 ... 179.5 at 1 deg). Analysis uses
Fejer's first quadrature rule in colatitude, whose nodes are exactly these cell centres; it is exact for a field of
degree lmax when the grid has more than 2 lmax latitudes. Synthesis also gives the gradient and the Hessian on the
unit sphere, which the flow solution needs for velocities and stresses.

Needs pyshtools (slablet[dyntopo]) for the Legendre functions, imported when a Grid is made.
"""
import numpy as np


def fejer_weights(n):
    """Fejer's first rule on the n colatitudes (2i+1) pi / 2n: weights w_i with sum_i w_i f(cos theta_i) =
    integral of f(x) dx over [-1, 1], exact for polynomials of degree < n."""
    theta = (2 * np.arange(n) + 1) * np.pi / (2 * n)
    k = np.arange(1, n // 2 + 1)
    s = (np.cos(2 * np.outer(theta, k)) / (4 * k ** 2 - 1)).sum(axis=1)
    return 2.0 / n * (1 - 2 * s)


class Grid:
    """Legendre functions and their colatitude derivatives on a cell-centred grid, for degrees 0 ... lmax."""

    def __init__(self, lats, lons, lmax):
        from pyshtools.legendre import PlmBar_d1
        self.lats, self.lons, self.lmax = np.asarray(lats, float), np.asarray(lons, float), int(lmax)
        nlat = len(self.lats)
        if nlat <= 2 * lmax:
            raise ValueError('{} latitudes are too few for degree {}'.format(nlat, lmax))
        spacing = 180.0 / nlat
        if not np.allclose(np.sort(self.lats), -90 + spacing * (np.arange(nlat) + 0.5)):
            raise ValueError('latitudes are not the cell centres of a regular grid')
        self.theta = np.radians(90.0 - self.lats)                     # colatitude
        self.phi = np.radians(self.lons)
        self.weights = fejer_weights(nlat)[np.argsort(np.argsort(self.theta))]
        n = lmax + 1
        self.P = np.zeros((nlat, n, n))
        self.dP = np.zeros((nlat, n, n))                              # dP / dtheta
        l_idx, m_idx = np.tril_indices(n)
        for i, th in enumerate(self.theta):
            p, dpz = PlmBar_d1(lmax, np.cos(th))
            k = l_idx * (l_idx + 1) // 2 + m_idx
            self.P[i, l_idx, m_idx] = p[k]
            self.dP[i, l_idx, m_idx] = -np.sin(th) * dpz[k]
        m = np.arange(n)
        self.cos = np.cos(np.outer(m, self.phi))                       # [m, lon]
        self.sin = np.sin(np.outer(m, self.phi))
        L = np.arange(n) * (np.arange(n) + 1.0)
        st, ct = np.sin(self.theta)[:, None, None], np.cos(self.theta)[:, None, None]
        # d2P / dtheta2 from the associated Legendre equation
        self.d2P = -ct / st * self.dP - (L[None, :, None] - (m[None, None, :] / st) ** 2) * self.P

    def analyse(self, f):
        """Coefficients c[..., 2, lmax+1, lmax+1] of fields f[..., lat, lon]."""
        f = np.asarray(f, float)
        dphi = 2 * np.pi / len(self.lons)
        fc = np.einsum('...ij,mj->...im', f, self.cos) * dphi        # [..., lat, m]
        fs = np.einsum('...ij,mj->...im', f, self.sin) * dphi
        w = self.weights[:, None, None] * self.P / (4 * np.pi)       # [lat, l, m]
        c = np.stack([np.einsum('ilm,...im->...lm', w, fc), np.einsum('ilm,...im->...lm', w, fs)], axis=-3)
        return np.tril(c)

    def _sum(self, c, P, kind='value'):
        """sum over l, m of P (a cos m phi + b sin m phi), or its phi derivative for kind 'dphi'."""
        a = np.einsum('ilm,...lm->...im', P, c[..., 0, :, :])
        b = np.einsum('ilm,...lm->...im', P, c[..., 1, :, :])
        if kind == 'dphi':
            m = np.arange(self.lmax + 1)
            return np.einsum('...im,mj->...ij', b * m, self.cos) - np.einsum('...im,mj->...ij', a * m, self.sin)
        if kind == 'dphi2':
            m2 = np.arange(self.lmax + 1) ** 2.0
            return -(np.einsum('...im,mj->...ij', a * m2, self.cos) + np.einsum('...im,mj->...ij', b * m2, self.sin))
        return np.einsum('...im,mj->...ij', a, self.cos) + np.einsum('...im,mj->...ij', b, self.sin)

    def synth(self, c):
        """The field f[..., lat, lon] of coefficients c[..., 2, l, m] (degrees above lmax are ignored)."""
        return self._sum(self._fit(c), self.P)

    def gradient(self, c):
        """Surface gradient on the unit sphere: (d/dtheta, 1/sin(theta) d/dphi), theta = colatitude (positive south)."""
        c = self._fit(c)
        st = np.sin(self.theta)[:, None]
        return self._sum(c, self.dP), self._sum(c, self.P, 'dphi') / st

    def hessian(self, c):
        """Covariant Hessian on the unit sphere in the (theta, phi) unit-vector frame: (H_tt, H_tp, H_pp).
        H_tt + H_pp is the surface Laplacian, -l(l+1) f."""
        c = self._fit(c)
        st, ct = np.sin(self.theta)[:, None], np.cos(self.theta)[:, None]
        f_t = self._sum(c, self.dP)
        f_p = self._sum(c, self.P, 'dphi')
        h_tt = self._sum(c, self.d2P)
        h_tp = self._sum(c, self.dP, 'dphi') / st - ct / st ** 2 * f_p
        h_pp = self._sum(c, self.P, 'dphi2') / st ** 2 + ct / st * f_t
        return h_tt, h_tp, h_pp

    def _fit(self, c):
        c = np.asarray(c, float)
        n = self.lmax + 1
        out = np.zeros(c.shape[:-2] + (n, n))
        k = min(n, c.shape[-1])
        out[..., :k, :k] = c[..., :k, :k]
        return out


def truncate(c, lmax):
    """Coefficients with every degree above lmax set to zero."""
    c = np.array(c, float)
    c[..., lmax + 1:, :] = 0.0
    return c


def degree_variance(c):
    """Variance of the field in each degree, [..., l] (4-pi normalisation: the sum of squares over m)."""
    return (np.asarray(c, float) ** 2).sum(axis=(-3, -1))


def degree_correlation(c1, c2):
    """Correlation of two fields in each degree, [..., l] (nan where either has no power)."""
    num = (np.asarray(c1) * np.asarray(c2)).sum(axis=(-3, -1))
    den = np.sqrt(degree_variance(c1) * degree_variance(c2))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def band_correlation(c1, c2, lmin, lmax):
    """Correlation of two fields over degrees lmin ... lmax together."""
    a, b = np.asarray(c1)[..., lmin:lmax + 1, :], np.asarray(c2)[..., lmin:lmax + 1, :]
    num = (a * b).sum(axis=(-3, -2, -1))
    return num / np.sqrt((a ** 2).sum(axis=(-3, -2, -1)) * (b ** 2).sum(axis=(-3, -2, -1)))
