"""Left foot: heel, talus, longitudinal arch, ball, five toes with toenails.

Authored with skeleton.foot_p(lateral, forward, up) measured from the ankle joint centre, which
sits 0.082 m above the ground (up = -0.082 is the floor).
"""
import numpy as np

from .sdf import Ellipsoid, RoundCone, Sphere, Union, norm, rot_from_axes
from .skeleton import FOOT, foot_p, rodrigues

GROUND = -0.082
TOES = {
    #       base (lat, fwd, up)            splay  lengths                   radii                              curl
    "big":  ((-0.0300, 0.1480, -0.0645),  -3.0, (0.0300, 0.0250),          (0.0130, 0.0118, 0.0105),          (4, 10)),
    "t2":   ((-0.0080, 0.1500, -0.0700),   1.0, (0.0200, 0.0110, 0.0100),  (0.0078, 0.0070, 0.0064, 0.0060),  (6, 22, 6)),
    "t3":   ((0.0100, 0.1430, -0.0710),    4.0, (0.0180, 0.0100, 0.0090),  (0.0074, 0.0067, 0.0061, 0.0057),  (8, 24, 6)),
    "t4":   ((0.0260, 0.1330, -0.0718),    7.0, (0.0160, 0.0090, 0.0085),  (0.0070, 0.0063, 0.0058, 0.0054),  (8, 26, 6)),
    "t5":   ((0.0395, 0.1200, -0.0722),   11.0, (0.0140, 0.0080, 0.0080),  (0.0066, 0.0060, 0.0055, 0.0052),  (8, 26, 6)),
}
UP = np.array([0.0, 0.0, 1.0])


def toe(name, base, splay, lengths, radii, curl):
    d = norm(rodrigues(FOOT.ey, UP, -splay))      # positive splay turns the toe laterally (+x side)
    if np.dot(d, FOOT.ex) * splay < 0:
        d = norm(rodrigues(FOOT.ey, UP, splay))
    d = norm(d + UP * 0.10)                        # toes rise slightly from the ball
    dorsal = norm(UP - d * np.dot(UP, d))
    lat = np.cross(d, dorsal)
    pts = [np.asarray(foot_p(*base), float)]
    parts = []
    for i, (L, c) in enumerate(zip(lengths, curl)):
        d = rodrigues(d, lat, -c)
        dorsal = rodrigues(dorsal, lat, -c)
        pts.append(pts[-1] + L * d)
        parts.append(RoundCone(pts[-2], pts[-1], radii[i], radii[i + 1]))
        R = rot_from_axes(x=lat, y=d)
        parts.append(Ellipsoid(pts[-2] + d * 0.5 * L - dorsal * 0.3 * radii[i], (0.95 * radii[i], 0.45 * L, 0.65 * radii[i]), R))
    R = rot_from_axes(x=lat, y=d)
    parts.append(Ellipsoid(pts[-1] - d * 0.002 - dorsal * 0.0015, (radii[-1], 0.0065, 0.8 * radii[-1]), R))   # toe pad
    L, r = lengths[-1], radii[-1]
    nc = pts[-2] + d * 0.62 * L + dorsal * 0.85 * r
    nail = Ellipsoid(nc, (0.78 * r, 0.42 * L if len(lengths) == 2 else 0.6 * L, 0.0009), R, tag="nail")
    bed = Ellipsoid(nc - dorsal * 0.0004, (0.88 * r, (0.48 if len(lengths) == 2 else 0.66) * L, 0.0014), R)
    return Union(parts, k=0.0030), nail, bed


def foot():
    R = FOOT.R
    parts = [
        Ellipsoid(foot_p(0.004, -0.032, -0.048), (0.030, 0.034, 0.034), R),        # heel (calcaneus + fat pad)
        Ellipsoid(foot_p(0.000, 0.000, -0.010), (0.032, 0.035, 0.030), R),         # talus / ankle
        Ellipsoid(foot_p(0.004, 0.055, -0.045), (0.040, 0.065, 0.030), R),         # midfoot
        Ellipsoid(foot_p(0.016, 0.060, -0.064), (0.026, 0.070, 0.016), R),         # lateral column on the floor
        Ellipsoid(foot_p(0.002, 0.118, -0.061), (0.047, 0.034, 0.020), R),         # forefoot
        Sphere(foot_p(-0.0290, 0.1370, -0.0675), 0.0145),                          # first metatarsal head
        Sphere(foot_p(-0.0060, 0.1390, -0.0700), 0.0115),
        Sphere(foot_p(0.0110, 0.1320, -0.0705), 0.0110),
        Sphere(foot_p(0.0260, 0.1230, -0.0710), 0.0105),
        Sphere(foot_p(0.0390, 0.1110, -0.0712), 0.0100),
        Ellipsoid(foot_p(0.000, 0.060, -0.020), (0.028, 0.060, 0.018), R),         # dorsum / extensor tendons
    ]
    toes, nails, beds = [], [], []
    for name, (base, splay, L, r, curl) in TOES.items():
        t, n, b = toe(name, base, splay, L, r, curl)
        toes.append(t); nails.append(n); beds.append(b)
    arch = Ellipsoid(foot_p(-0.040, 0.050, -0.090), (0.024, 0.055, 0.020), R)    # carve under the medial arch
    return Union(parts, k=0.012, bone="foot_l"), Union(toes, bone="ball_l"), arch, nails, beds


def build(prog, k_ankle=0.014):
    body, toes, arch, nails, beds = foot()
    left = Union([body, toes], k=0.0040)
    prog.add(left, k=k_ankle)
    prog.add(left.mirrored(), k=k_ankle)
    prog.sub(arch, k=0.010)
    prog.sub(arch.mirrored(), k=0.010)
    for bed, nail in zip(beds, nails):
        prog.sub(bed, k=0.0004)
        prog.sub(bed.mirrored(), k=0.0004)
        prog.add(nail, k=0.0005)
        prog.add(nail.mirrored(), k=0.0005)
