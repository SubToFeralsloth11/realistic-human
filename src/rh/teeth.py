"""Teeth, gums and tongue sculpted as SDFs (head-local placement behind the lips).

Crown sizes (mm, mesiodistal x labiolingual x height) follow adult averages; the arch is a
parabola through the incisal points. Upper teeth ride the head bone, the lower set and the tongue
ride the jaw bone. Each mesh carries a 'm_gum' mask for the shader.
"""
import numpy as np

from .head import O
from .sdf import Ellipsoid, Program, RoundBox, Tube, Union, rot_from_axes

UPPER = [("central", 8.5, 7.0, 10.0), ("lateral", 6.6, 6.0, 8.8), ("canine", 7.6, 8.0, 10.0),
         ("premolar1", 7.0, 9.0, 8.2), ("premolar2", 6.6, 9.0, 7.6), ("molar1", 10.2, 11.0, 7.0), ("molar2", 9.0, 10.5, 6.6)]
LOWER = [("central", 5.4, 6.0, 9.0), ("lateral", 5.9, 6.2, 9.2), ("canine", 6.9, 7.6, 10.4),
         ("premolar1", 7.0, 7.8, 8.4), ("premolar2", 7.1, 8.2, 8.0), ("molar1", 11.0, 10.4, 7.6), ("molar2", 10.4, 10.0, 7.0)]
STOMION_Z = -0.0716


def _arch(y0, k, total, n=400):
    """Points along y = y0 + k x^2 (x >= 0) parametrised by arc length up to `total` metres."""
    xs = np.linspace(0, 0.045, n)
    ys = y0 + k * xs ** 2
    seg = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0], np.cumsum(seg)])
    return lambda d: np.array([np.interp(d, s, xs), np.interp(d, s, ys)]), lambda d: np.interp(d, s, np.arctan2(np.gradient(ys), np.gradient(xs)))


def dental_arch(upper=True):
    table = UPPER if upper else LOWER
    y0 = -0.0262 if upper else -0.0242          # labial face of the central incisors (2 mm overjet)
    k = 60.0 if upper else 66.0
    pos, ang = _arch(y0, k, 0.06)
    crowns = []
    d = 0.0
    for name, w, l, h in table:
        w, l, h = w / 1000, l / 1000, h / 1000
        mid = d + w / 2
        x, y = pos(mid)
        a = ang(mid)
        tangent = np.array([np.cos(a), np.sin(a), 0.0])
        lingual = np.array([-np.sin(a), np.cos(a), 0.0])     # into the mouth
        if lingual[1] < 0:
            lingual = -lingual
        for sx in (1, -1):
            t = tangent * np.array([sx, 1, 1])
            lg = lingual * np.array([sx, 1, 1])
            c = np.array([sx * x, y, 0.0]) + lg * (l / 2)
            zc = STOMION_Z + (h / 2 - 0.0012 if upper else -h / 2 + 0.0002)
            if not upper:
                c = c + lg * 0.0015                            # lower incisors sit behind the upper ones
            R = rot_from_axes(x=t, y=lg)
            taper = 0.78 if name in ("central", "lateral") else 0.9
            crown = RoundBox(O + c + [0, 0, zc], (w / 2 * 0.98, l / 2 * taper, h / 2), R, rr=min(w, l) * 0.32, tag="tooth")
            crowns.append(crown)
        d += w * 1.005
    return crowns, pos, ang, y0


def gums(upper=True):
    crowns, pos, ang, y0 = dental_arch(upper)
    pts = []
    for dd in np.linspace(0, 0.050, 18):
        x, y = pos(dd)
        pts.append((x, y))
    zg = STOMION_Z + (0.0085 if upper else -0.0085)
    left = [O + [x, y + 0.0028, zg] for x, y in pts]
    tube = Tube([O + [-x, y + 0.0028, zg] for x, y in pts[:0:-1]] + left, 0.0042, tag="gum")
    return Union([tube], tag="gum")


def tongue():
    return Union([Ellipsoid(O + [0, 0.004, STOMION_Z - 0.0085], (0.022, 0.030, 0.0085), tag="gum"),
                  Ellipsoid(O + [0, -0.010, STOMION_Z - 0.0060], (0.017, 0.010, 0.0060), tag="gum")], k=0.008)


def program(upper=True):
    prog = Program()
    crowns, _, _, _ = dental_arch(upper)
    prog.add(gums(upper))
    prog.add(Union(crowns, k=0.0004), k=0.0012)
    if not upper:
        prog.add(tongue(), k=0.002)
    return prog
