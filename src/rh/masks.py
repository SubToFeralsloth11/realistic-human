"""Per-vertex region masks (0..1) used by the skin shader and the hair/stubble generators.

All regions are defined analytically in head-local or hand-local coordinates, so they follow the
anatomy instead of a texture layout.
"""
import numpy as np

from . import head
from .sdf import F, group_distance
from .skeleton import HAND

O = head.O


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def gauss(p, c, s):
    d = (p - np.asarray(c)) / np.asarray(s)
    return np.exp(-0.5 * np.einsum("ij,ij->i", d, d))


def _lip(q):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    ax = np.abs(x)
    cupid = np.exp(-((ax - 0.0055) / 0.0030) ** 2)
    zb = -0.0626 + 0.0010 * cupid - 0.0050 * (ax / 0.026) ** 2         # upper vermilion border
    zl = -0.0820 + 0.0080 * (ax / 0.026) ** 2                          # lower vermilion border
    e = 0.0006
    vert = smoothstep(zl - e, zl + e, z) * (1 - smoothstep(zb - e, zb + e, z))
    lat = 1 - smoothstep(0.0235, 0.0265, ax)
    front = 1 - smoothstep(-0.026 + 26 * x ** 2, -0.020 + 26 * x ** 2, y)
    return vert * lat * front


def _beard(q, world_z):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    ax = np.abs(x)
    # upper boundary across the cheek: low under the nose, rising to the sideburn at the ear
    zline = np.interp(ax, [0.0, 0.020, 0.030, 0.045, 0.058, 0.068], [-0.049, -0.050, -0.046, -0.034, -0.012, 0.010])
    top = 1 - smoothstep(zline - 0.004, zline + 0.004, z)
    ahead_of_ear = 1 - smoothstep(0.035, 0.055, y)
    neck_line = smoothstep(O[2] - 0.125, O[2] - 0.110, world_z)     # beard neckline under the jaw
    return top * ahead_of_ear * neck_line


def _scalp(q, world_z):
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    ax = np.abs(x)
    # front hairline with slight temple recession; behind the ears it drops to the nape
    zh = 0.072 + 0.012 * smoothstep(0.015, 0.045, ax) - 0.004 * smoothstep(0.045, 0.06, ax)
    front = smoothstep(zh - 0.004, zh + 0.004, z)
    # temple hairline descends in front of the ear into the sideburn (front edge ~2 cm before the tragus)
    side_line = np.where(y > 0.055, -0.060, 0.026 - 0.55 * np.clip(y - 0.015, 0, None))
    side = smoothstep(side_line - 0.004, side_line + 0.004, z)
    behind = smoothstep(-0.008, 0.012, y)
    nape = smoothstep(O[2] - 0.085, O[2] - 0.070, world_z)
    above_ear = np.maximum(front, side * behind) * nape
    return np.where(z > 0.10, 1.0, above_ear)


def _brows(q):
    out = np.zeros(len(q), F)
    for s in (1, -1):
        x = q[:, 0] * s
        t = (x - 0.010) / 0.042                                  # 0 at the medial end, 1 lateral
        zc = 0.0155 + 0.0040 * np.sin(np.clip(t, 0, 1) * np.pi * 0.85) - 0.002 * t
        half_h = 0.0042 * (1 - 0.55 * np.clip(t, 0, 1))
        inside = (1 - smoothstep(half_h * 0.6, half_h, np.abs(q[:, 2] - zc))) * smoothstep(-0.05, 0.0, t) * (1 - smoothstep(0.95, 1.05, t))
        out = np.maximum(out, inside * (q[:, 1] < 0.0))
    return out


def compute(verts, prog):
    """Dict name -> float32 per-vertex mask."""
    q = verts - O
    wz = verts[:, 2]
    m = {}
    m["m_lip"] = _lip(q).astype(F)
    m["m_beard"] = (_beard(q, wz) * (1 - m["m_lip"])).astype(F)
    m["m_scalp"] = _scalp(q, wz).astype(F)
    m["m_brow"] = _brows(q).astype(F)
    nose = gauss(q, (0, -0.040, -0.036), (0.012, 0.012, 0.012))
    cheeks = gauss(np.c_[np.abs(q[:, 0]), q[:, 1], q[:, 2]], (0.040, -0.016, -0.030), (0.014, 0.02, 0.014))
    ears_leaves = [l for l in prog.leaves() if l.tag == "ear"]
    ear = (1 - smoothstep(0.0005, 0.004, group_distance(ears_leaves, verts.astype(F)))) if ears_leaves else 0
    nail_leaves = [l for l in prog.leaves() if l.tag == "nail"]
    nail = (1 - smoothstep(0.0002, 0.0007, group_distance(nail_leaves, verts.astype(F)))) if nail_leaves else 0
    m["m_nail"] = np.asarray(nail, F) * np.ones(len(verts), F)
    mouth_leaves = [l for l in prog.leaves() if l.tag == "mouth"]
    mouth = (1 - smoothstep(0.0003, 0.0025, group_distance(mouth_leaves, verts.astype(F)))) if mouth_leaves else 0
    m["m_mouth"] = (np.asarray(mouth, F) * np.ones(len(verts), F)).astype(F)
    m["m_red"] = np.clip(0.8 * nose + 0.7 * cheeks + 0.45 * ear, 0, 1).astype(F)
    # eyelids: skin within 5 mm of either globe, outside the visible eyeball
    lid = np.zeros(len(verts), F)
    for key in ("eye_l", "eye_r"):
        r = np.linalg.norm(verts - head.LANDMARKS[key], axis=1)
        lid = np.maximum(lid, (1 - smoothstep(head.EYE_R + 0.004, head.EYE_R + 0.008, r)) * smoothstep(head.EYE_R, head.EYE_R + 0.0006, r))
    m["m_lid"] = lid.astype(F)
    m["m_oily"] = np.clip(gauss(q, (0, -0.022, 0.040), (0.022, 0.03, 0.02)) + gauss(q, (0, -0.035, -0.020), (0.010, 0.02, 0.022)), 0, 1).astype(F)
    # palms (both hands): hand-local z below the palm mid-plane
    palm = np.zeros(len(verts), F)
    for s in (1, -1):
        rel = (verts * [s, 1, 1]) - HAND.o
        loc = rel @ HAND.R
        inside = (np.abs(loc[:, 0]) < 0.06) & (loc[:, 1] > -0.01) & (loc[:, 1] < 0.20)
        palm = np.maximum(palm, inside * smoothstep(-0.002, -0.009, loc[:, 2]))
    m["m_palm"] = palm.astype(F)
    return m
