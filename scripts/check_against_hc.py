"""Check slablet.dyntopo against HC (Becker, O'Neill & Steinberger; github.com/geodynamics/hc).

Runs HC and slablet.dyntopo on the same density field (a run's present-day dVs_native, below 300 km, d ln rho /
d ln Vs = 0.25 scaled with PREM), the same layering and the same viscosity, and compares, degree by degree up to
lmax = 31: the radial stress at the surface and at the CMB (which give dynamic topography), the geoid, the velocities
at two layers.  Pass: everything within 1 % per degree for degrees 1-8 (relative to the rms of
that degree).

    HC_HOME=~/GIT/hc python scripts/check_against_hc.py runs_real/muller2022_refined/mantle_temp_0000Ma.nc [--viscosity two_layer]

HC's spherical harmonics are pyshtools' 4-pi real harmonics x (-1)^m / sqrt(4 pi) (found with HC's sh_syn), and its
poloidal vector harmonics are grad_1 Y / sqrt(l(l+1)).
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np
import xarray as xr

from slablet import dyntopo
from slablet.sphharm import Grid

LMAX = 31


def to_hc(c):
    m = np.arange(c.shape[-1])
    return c * np.sqrt(4 * np.pi) * (-1.0) ** m


def from_hc(c):
    m = np.arange(c.shape[-1])
    return c * (-1.0) ** m / np.sqrt(4 * np.pi)


def write_dens(path, coeffs, depths):
    """HC long format, deepest layer first, values in % (HC multiplies by 0.01 x -ds x PREM density)."""
    n = len(depths)
    with open(path, 'w') as fh:
        for i, (c, z) in enumerate(zip(coeffs, depths)):
            hc = to_hc(c)
            fh.write('{} {} {:.6f} {} 1 0\n'.format(LMAX, i, z, n))
            for l in range(LMAX + 1):
                for m in range(l + 1):
                    fh.write('{:.10e} {:.10e}\n'.format(hc[0, l, m], hc[1, l, m]))


def read_sh(text, nset=1):
    """Coefficients [nset, 2, l, m] from HC's long-format output of one or more scalar expansions."""
    rows = [r.split() for r in text.strip().split('\n') if r.strip()]
    out, i = [], 0
    while i < len(rows):
        lmax = int(rows[i][0]); i += 1
        c = np.zeros((nset, 2, lmax + 1, lmax + 1))
        for l in range(lmax + 1):
            for m in range(l + 1):
                v = [float(x) for x in rows[i]]; i += 1
                for s in range(nset):
                    c[s, 0, l, m], c[s, 1, l, m] = v[2 * s], v[2 * s + 1]
        out.append(from_hc(c))
    return out


def hc(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit('{} failed:\n{}'.format(' '.join(cmd), r.stderr[-2000:]))
    return r.stdout


def per_degree_error(a, b, lmax=8):
    """max over degrees 1..lmax of |a - b| / rms(b) in that degree."""
    err = []
    for l in range(1, lmax + 1):
        num = np.sqrt(((a[..., l, :] - b[..., l, :]) ** 2).sum())
        den = np.sqrt((b[..., l, :] ** 2).sum())
        err.append(num / den if den > 0 else 0.0)
    return np.array(err)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mantle_file')
    ap.add_argument('--viscosity', default='two_layer')
    ap.add_argument('--hc-home', default=os.environ.get('HC_HOME', os.path.expanduser('~/GIT/hc')))
    args = ap.parse_args()
    bin_ = os.path.join(args.hc_home, 'bin')

    ds = xr.load_dataset(args.mantle_file)
    dp = dyntopo.DynTopoParams(viscosity=args.viscosity, lmax=LMAX, loading='air', rho_top=4448.8)
    depth = ds['depth'].values.astype(float)
    grid = Grid(ds['lat'].values, ds['lon'].values, LMAX)
    f = ds[dp.density_variable].values.astype(float)
    f[depth < dp.cutoff_km] = 0.0
    fcoef = grid.analyse(f)                                       # dVs % per layer, shallowest first

    work = tempfile.mkdtemp(prefix='hc_check_')
    write_dens(os.path.join(work, 'dens.ab'), fcoef[::-1], depth[::-1])
    tops, etas = dyntopo.viscosity_layers(dp.viscosity)
    bottoms_r = np.concatenate([1 - tops[1:] / dp.radius_km, [dp.cmb_radius_km / dp.radius_km]])
    with open(os.path.join(work, 'visc.dat'), 'w') as fh:            # deepest first: bottom radius, viscosity
        for r, e in sorted(zip(bottoms_r, etas)):
            fh.write('{:.8f} {:.6e}\n'.format(r, e))
    base = [os.path.join(bin_, 'hc'), '-fs', '-dens', 'dens.ab', '-ds', str(dp.dlnrho_dlnvs),
            '-vf', 'visc.dat', '-prem', os.path.join(args.hc_home, 'prem', 'prem.dat')]
    hc(base + ['-rtrac'], work)
    os.rename(os.path.join(work, 'rtrac.sol.bin'), os.path.join(work, 'rtrac.bin'))
    hc(base, work)                                                # velocities
    zlev = np.array([float(r.split()[1]) for r in hc([os.path.join(bin_, 'hc_extract_sh_layer'), 'rtrac.bin',
                                                      '1', '4'], work).strip().split('\n')])   # CMB first
    nlev = len(zlev)

    def layer(fname, k, mode, nset=1):
        return read_sh(hc([os.path.join(bin_, 'hc_extract_sh_layer'), fname, str(k), str(mode)], work), nset)[0]

    # slablet
    res = dyntopo.compute(ds, dp, components=False)
    solver = dyntopo.FlowSolver(1.0 - depth[::-1] / dp.radius_km, dp)
    drho = dyntopo.density_anomaly(ds, dp)
    st = dyntopo._state_coeffs(solver, grid.analyse(drho[::-1]))
    u = solver.units()
    print('HC levels (km):', np.round(zlev, 1)[:3], '...', np.round(zlev, 1)[-3:], '({} levels)'.format(nlev))

    rows = []
    # surface and CMB radial stress (MPa)
    srr_s_hc = layer('rtrac.bin', nlev, 1)[0]
    srr_c_hc = layer('rtrac.bin', 1, 1)[0]
    srr_s = st[-1, 2] * u['stress'] / 1e6
    srr_c = st[0, 2] / solver.c * u['stress'] / 1e6
    rows.append(('surface radial stress', per_degree_error(srr_s, srr_s_hc)))
    rows.append(('CMB radial stress', per_degree_error(srr_c, srr_c_hc)))
    geoid_hc = read_sh(open(os.path.join(work, 'geoid.ab')).read())[0][0]
    rows.append(('geoid', per_degree_error(res['geoid_coeffs'].values, geoid_hc)))
    # velocities at a mid-mantle layer and near the top
    for k in (nlev // 2, nlev - 3):
        z = zlev[k - 1]
        o = int(np.argmin(np.abs((1 - solver.r_out) * dp.radius_km - z)))
        ur_hc = layer('vel.sol.bin', k, 1)[0]
        cm_yr = 100 * dyntopo.SECONDS_PER_YEAR
        rows.append(('u_r at {:.0f} km'.format(z), per_degree_error(st[o, 0] * u['velocity'] * cm_yr, ur_hc)))
        # poloidal: HC's vector harmonics are grad_1 Y / sqrt(l(l+1)), so its coefficient is sqrt(l(l+1)) V
        pol_hc = layer('vel.sol.bin', k, 5)[0]
        L = np.arange(LMAX + 1) * (np.arange(LMAX + 1) + 1.0)
        pol = st[o, 1] * u['velocity'] * cm_yr * np.sqrt(L)[None, :, None]
        rows.append(('u_poloidal at {:.0f} km'.format(z), per_degree_error(pol, pol_hc)))
        srt_hc = layer('rtrac.bin', k, 5)[0]
        srt = st[o, 3] * u['stress'] / solver.r_out[o] / 1e6 * np.sqrt(L)[None, :, None]
        rows.append(('shear stress (poloidal) at {:.0f} km'.format(z), per_degree_error(srt, srt_hc)))
        srr_hc = layer('rtrac.bin', k, 1)[0]
        rows.append(('radial stress at {:.0f} km'.format(z),
                     per_degree_error(st[o, 2] * u['stress'] / solver.r_out[o] / 1e6, srr_hc)))

    print('\nmax relative error per degree 1-8 (pass < 0.01):')
    ok = True
    for name, err in rows:
        flag = 'ok' if err.max() < 0.01 else 'FAIL'
        ok &= err.max() < 0.01
        print('  {:55s} max {:.2e}  [{}]'.format(name, err.max(), flag), ' '.join('{:.1e}'.format(e) for e in err))
    print('\nwork dir:', work)
    print('PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
