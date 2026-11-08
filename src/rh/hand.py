"""Left hand in a relaxed pose: carpus, cupped palm, thenar/hypothenar pads, knuckles, extensor
tendons, five jointed digits with palmar pads, nail plates set in nail folds, palm creases.

Hand frame (skeleton.HAND): origin at the wrist joint, x radial (thumb side), y distal, z dorsal.
"""
import numpy as np

from .sdf import Capsule, Ellipsoid, RoundBox, RoundCone, Sphere, Tube, Union, norm, rot_from_axes
from .skeleton import HAND, rodrigues

H = HAND.p

DIGITS = {
    #          base (hand-local)              spread  lengths                      radii                                 flex (MCP, PIP, DIP)
    "index":  ((0.0290, 0.0880, 0.0020),    5.0, (0.0420, 0.0240, 0.0200), (0.0099, 0.0090, 0.0082, 0.0075), (10, 14, 8)),
    "middle": ((0.0085, 0.0920, 0.0025),    0.0, (0.0460, 0.0285, 0.0215), (0.0103, 0.0093, 0.0084, 0.0076), (11, 16, 9)),
    "ring":   ((-0.0120, 0.0885, 0.0015),  -5.0, (0.0435, 0.0270, 0.0205), (0.0097, 0.0088, 0.0080, 0.0072), (13, 18, 10)),
    "pinky":  ((-0.0310, 0.0800, -0.0005), -11.0, (0.0340, 0.0200, 0.0180), (0.0087, 0.0079, 0.0071, 0.0065), (15, 20, 11)),
}
THUMB = ((0.0235, 0.0230, -0.0090), (0.60, 0.72, -0.34), (0.55, -0.25, 0.80),
         (0.0410, 0.0310, 0.0260), (0.0128, 0.0110, 0.0100, 0.0090), (8, 12, 15))


def _chain(base, d, dorsal, lengths, flex):
    """Joint positions plus per-segment (direction, dorsal, lateral) frames."""
    d = norm(d)
    dorsal = norm(dorsal - d * np.dot(dorsal, d))
    lat = np.cross(d, dorsal)
    pts = [np.asarray(base, float)]
    frames = []
    for L, f in zip(lengths, flex):
        d = rodrigues(d, lat, -f)
        dorsal = rodrigues(dorsal, lat, -f)
        pts.append(pts[-1] + L * d)
        frames.append((d.copy(), dorsal.copy(), lat.copy()))
    return pts, frames


def digit(name, base, d, dorsal, lengths, radii, flex, bone_prefix):
    pts, frames = _chain(base, d, dorsal, lengths, flex)
    parts, nails, beds = [], [], []
    for i in range(3):
        a, b = pts[i], pts[i + 1]
        dd, dor, lat = frames[i]
        bone = f"{bone_prefix}_{i + 1:02d}_l"
        R = rot_from_axes(x=lat, y=dd)
        L = lengths[i]
        parts.append(RoundCone(a, b, radii[i], radii[i + 1], bone=bone))
        # palmar pad of the segment and a slightly wider-than-deep section
        parts.append(Ellipsoid(a + dd * 0.52 * L - dor * 0.30 * radii[i], (0.92 * radii[i], 0.42 * L, 0.62 * radii[i]), R, bone=bone))
        # dorsal knuckle over the proximal joint of each segment
        parts.append(Sphere(a + dor * 0.25 * radii[i], 0.80 * radii[i], bone=bone))
    # fingertip pulp
    dd, dor, lat = frames[2]
    tip = pts[3]
    R = rot_from_axes(x=lat, y=dd)
    parts.append(Ellipsoid(tip - dd * 0.004 - dor * 0.0022, (0.95 * radii[3], 0.0085, 0.80 * radii[3]), R,
                           bone=f"{bone_prefix}_03_l"))
    # nail plate on the distal segment
    L3, r3 = lengths[2], radii[3]
    nc = pts[2] + dd * (0.60 * L3) + dor * (0.86 * r3)
    nail_R = rot_from_axes(x=lat, y=dd)
    nails.append(Ellipsoid(nc, (0.80 * r3, 0.40 * L3, 0.0010), nail_R, bone=f"{bone_prefix}_03_l", tag="nail"))
    beds.append(Ellipsoid(nc - dor * 0.0004 - dd * 0.0006, (0.90 * r3, 0.46 * L3, 0.0016), nail_R))
    return Union(parts, k=0.0030), nails, beds, pts


def hand():
    R = HAND.R
    parts = [
        Ellipsoid(H(0, 0.018, 0.0), (0.0270, 0.0200, 0.0160), R),                     # carpus
        Ellipsoid(H(0.000, 0.056, 0.0), (0.0440, 0.0400, 0.0145), R),                 # palm
        RoundBox(H(0.000, 0.060, 0.001), (0.0395, 0.0300, 0.0108), R, rr=0.0090),
        Ellipsoid(H(0.0235, 0.0360, -0.0115), (0.0175, 0.0260, 0.0130), R),            # thenar eminence
        Ellipsoid(H(-0.0320, 0.0460, -0.0080), (0.0128, 0.0300, 0.0110), R),           # hypothenar eminence
    ]
    for name, (base, spread, L, r, flex) in DIGITS.items():
        b = H(*base)
        parts.append(Sphere(b + HAND.ez * 0.002, r[0] * 1.02))                          # metacarpal head
        parts.append(Capsule(H(base[0] * 0.45, 0.022, 0.0120), b + HAND.ez * 0.0080, 0.0021))  # extensor tendon
    digits, nails, beds = [], [], []
    for name, (base, spread, L, r, flex) in DIGITS.items():
        s = np.radians(spread)
        d = HAND.ey * np.cos(s) + HAND.ex * np.sin(s)
        node, n, bd, _ = digit(name, H(*base), d, HAND.ez, L, r, flex, name)
        digits.append(node); nails += n; beds += bd
    base, d, dor, L, r, flex = THUMB
    node, n, bd, _ = digit("thumb", H(*base), R @ np.array(d), R @ np.array(dor), L, r, flex, "thumb")
    digits.append(node); nails += n; beds += bd
    palm = Union(parts, k=0.008, bone="hand_l")
    return palm, digits, nails, beds


def palm_creases():
    z = -0.0142
    lines = [
        [H(-0.0360, 0.0745, z), H(-0.0150, 0.0790, z - 0.0004), H(0.0050, 0.0805, z - 0.0004), H(0.0170, 0.0835, z)],   # distal
        [H(-0.0200, 0.0610, z), H(0.0020, 0.0645, z - 0.0005), H(0.0260, 0.0690, z + 0.0005)],                        # proximal
        [H(0.0230, 0.0640, z), H(0.0120, 0.0450, z - 0.0010), H(0.0080, 0.0280, z - 0.0010), H(0.0050, 0.0150, z)],    # thenar
    ]
    return Union([Tube(l, 0.0008) for l in lines])


def build(prog, k_wrist=0.012):
    palm, digits, nails, beds = hand()
    left = Union([palm] + digits, k=0.0060)
    prog.add(left, k=k_wrist)
    prog.add(left.mirrored(), k=k_wrist)
    cr = palm_creases()
    prog.sub(cr, k=0.0012)
    prog.sub(cr.mirrored(), k=0.0012)
    for bed, nail in zip(beds, nails):
        prog.sub(bed, k=0.0004)
        prog.sub(bed.mirrored(), k=0.0004)
        prog.add(nail, k=0.0005)
        prog.add(nail.mirrored(), k=0.0005)
