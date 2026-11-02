"""Write a GPlates rotation file: Müller 2022 relative rotations, absolute frame from a fit_frame.py result.

    conda run -n pygmt17 python -m slablet.frame.write_frame_rotfile runs_real/frame_fit/pmag/best.yaml \\
        --out runs_real/frame_rotfiles/Muller2022_fitted_frame_coarse.rot

The fitted frame is the global correction run_mantle.py applies (frame_africa):
    C(t) = A_new(t) * A_M22(t)^-1
where A_new is Africa's (701) absolute rotation from the fit's nodes (identity at 0 Ma, interpolated
between nodes) and A_M22 is Africa's absolute rotation in the Müller 2022 optimised mantle frame.  Every
sequence whose fixed plate is 000 (Africa, and the plates the file attaches directly to the mantle, e.g.
the Panthalassic plates) is replaced by C(t) * R(t), sampled at its original times plus every 1 Myr
up to the oldest node; every other line (all relative rotations, comments) is copied unchanged.  Beyond
the oldest node C is held at its value there, so the frame is continuous and older ages keep Müller
2022's motion.  Spherical throughout (pygplates finite rotations).
"""
import argparse
import os

import numpy as np
import pygplates
import yaml

from slablet.paths import WORK
from slablet.gplates.run_mantle import africa_frame                 # noqa: E402

M22 = os.path.expanduser('~/Library/Caches/gprm/Muller2022/optimisation/1000_0_rotfile_MantleOpt.rot')


def parse(line):
    """(moving, time, rotation, fixed, comment) of a rotation line, or None for anything else."""
    body, _, comment = line.partition('!')
    f = body.split()
    if len(f) < 6:
        return None
    try:
        moving, t, lat, lon, ang, fixed = int(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), int(f[5])
    except ValueError:
        return None
    rot = (pygplates.FiniteRotation.create_identity_rotation() if ang == 0.0 else
           pygplates.FiniteRotation(pygplates.PointOnSphere(lat, lon), np.radians(ang)))
    return moving, t, rot, fixed, comment.strip()


def fmt(moving, t, rot, fixed, comment):
    if rot.represents_identity_rotation():
        lat, lon, ang = 90.0, 0.0, 0.0
    else:
        lat, lon, ang = rot.get_lat_lon_euler_pole_and_angle_degrees()
    return '{:d} {:.1f} {:.4f} {:.4f} {:.4f} {:03d} ! {}\n'.format(moving, t, lat, lon, ang, fixed, comment)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('best', help='best.yaml from fit_frame.py (its last stage is used)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model-file', default=M22, help='rotation file the fit was made against')
    args = ap.parse_args()
    b = yaml.safe_load(open(args.best))
    stage = [k for k in b if k.startswith('stage')][-1]
    nodes = {float(a): v for a, v in b[stage]['nodes'].items()}
    top = max(nodes)
    a_new = africa_frame(nodes)
    old = pygplates.RotationModel(args.model_file)

    def correction(t):
        t = min(t, top)
        return a_new(t) * old.get_rotation(t, 701, anchor_plate_id=0).get_inverse()

    lines = open(args.model_file, encoding='utf-8', errors='replace').readlines()
    tag = 'fitted frame ({}, {}, skill {:.3f})'.format(os.path.relpath(args.best, WORK), stage, b[stage]['skill'])
    out, i, n_seq = [], 0, 0
    while i < len(lines):
        p = parse(lines[i])
        if p is None or p[3] != 0 or p[0] in (0, 999):
            out.append(lines[i])
            i += 1
            continue
        block = []                                   # consecutive lines of one sequence fixed to 000
        while i < len(lines):
            q = parse(lines[i])
            if q is None or q[0] != p[0] or q[3] != 0:
                break
            block.append(q)
            i += 1
        n_seq += 1
        times = [q[1] for q in block]
        dense = [t for t in np.arange(np.ceil(times[0]), min(times[-1], top) + 1e-9, 1.0)]
        new_times = sorted(set(times) | set(float(t) for t in dense))
        for t in new_times:
            k = min(max(np.searchsorted(times, t, side='right') - 1, 0), len(times) - 2) if len(times) > 1 else 0
            if t in times:
                q = block[times.index(t)]
                r, comment = q[2], q[4]
            else:
                r = pygplates.FiniteRotation.interpolate(block[k][2], block[k + 1][2], times[k], times[k + 1], t)
                comment = ''
            note = '{} x [{}]'.format(tag, comment) if comment else '{} (1 Myr sample)'.format(tag)
            out.append(fmt(p[0], t, correction(t) * r, 0, note))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as fh:
        fh.write('999 0.0 0.0 0.0 0.0 999 ! Muller et al. (2022) relative rotations; absolute frame replaced by the '
                 'tomography-{} (slablet/frame/write_frame_rotfile.py); C held at its {:.0f} Ma value '
                 'beyond {:.0f} Ma\n'.format(tag, top, top))
        fh.writelines(out)
    print('{} sequences fixed to 000 rewritten; {} lines; written {}'.format(n_seq, len(out) + 1, args.out))

    # check: every plate's new absolute rotation = C(t) * old, and relative rotations are unchanged
    new = pygplates.RotationModel(args.out)
    worst_abs, worst_rel = 0.0, 0.0
    pts = [pygplates.PointOnSphere(la, lo) for la in (-60, 0, 60) for lo in (-120, 0, 120)]
    for pid in (701, 101, 201, 301, 501, 801, 901, 9000, 98041, 22013, 7300, 410):
        for t in np.arange(0.0, 421.0, 5.0):
            if old.get_rotation(float(t), pid, anchor_plate_id=0, use_identity_for_missing_plate_ids=False) is None:
                continue                             # plate has no rotation at this age in Müller 2022
            ra = new.get_rotation(float(t), pid, anchor_plate_id=0)
            rb = correction(float(t)) * old.get_rotation(float(t), pid, anchor_plate_id=0)
            rr_new = new.get_rotation(float(t), pid, anchor_plate_id=701)
            rr_old = old.get_rotation(float(t), pid, anchor_plate_id=701)
            for x in pts:
                worst_abs = max(worst_abs, pygplates.GeometryOnSphere.distance(ra * x, rb * x))
                worst_rel = max(worst_rel, pygplates.GeometryOnSphere.distance(rr_new * x, rr_old * x))
    print('check, 12 plates, 0-420 Ma: max misfit to C(t) x Muller 2022 {:.2f} km; max change of motion '
          'relative to Africa {:.2f} km'.format(np.degrees(worst_abs) * 111.195, np.degrees(worst_rel) * 111.195))
    for t in sorted(nodes):
        d = max(pygplates.GeometryOnSphere.distance(new.get_rotation(t, 701, anchor_plate_id=0) * x, a_new(t) * x)
                for x in pts)
        print('  Africa at node {:.0f} Ma: {:.3f} km from the fitted rotation'.format(t, np.degrees(d) * 111.195))


if __name__ == '__main__':
    main()
