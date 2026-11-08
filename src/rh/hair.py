"""Hair strands grown on the skin surface: scalp (short textured crop, tapered sides), eyebrows,
eyelashes and stubble.

Roots are sampled on the final mesh with density from the region masks; strands are marched in
vectorised steps and kept a set distance off the skin with the body SDF (so short hair lies on
the scalp and longer hair builds volume instead of clipping). Clumps pull strand tips towards
guide strands; a smooth per-strand noise adds frizz.
Output per group: points (sum(n_pts), 3), counts (n_strands,), radius (sum(n_pts),).
"""
import numpy as np

from . import head as headmod
from .mesher import gradient
from .sdf import F, norm

O = headmod.O


def triangulate(faces, sizes):
    st = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    tris = [faces[st[sizes == 3, None] + np.arange(3)]]
    q = st[sizes == 4]
    tris.append(faces[q[:, None] + np.array([0, 1, 2])])
    tris.append(faces[q[:, None] + np.array([0, 2, 3])])
    return np.concatenate(tris)


def sample_roots(verts, tris, density, count, rng):
    """Area x density weighted points on the mesh; returns (points, barycentric-interpolated density)."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    dens = density[tris].mean(axis=1)
    p = area * dens
    if p.sum() <= 0:
        return np.zeros((0, 3), F), np.zeros(0, F)
    ti = rng.choice(len(tris), size=count, p=p / p.sum())
    r1, r2 = rng.random(count), rng.random(count)
    s = np.sqrt(r1)
    u, v, w = 1 - s, s * (1 - r2), s * r2
    pts = a[ti] * u[:, None] + b[ti] * v[:, None] + c[ti] * w[:, None]
    dval = density[tris[ti, 0]] * u + density[tris[ti, 1]] * v + density[tris[ti, 2]] * w
    return pts.astype(F), dval.astype(F)


def surface_normals(prog, P):
    g = gradient(prog, P, 0.0003)
    return (g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)).astype(F)


def tangent(v, nrm):
    t = v - nrm * np.einsum("ij,ij->i", v, nrm)[:, None]
    return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)


def grow(prog, roots, nrm, comb, length, lift, n_pts, offset0, volume, rng, curl=0.0, clump=0.0, frizz=0.0):
    """March strands. comb: unit tangent direction per strand; lift: start elevation (rad);
    offset0/volume: distance kept off the skin at the root / extra at the tip (m)."""
    n = len(roots)
    pts = np.zeros((n, n_pts, 3), F)
    pts[:, 0] = roots + nrm * 0.0002
    d = np.cos(lift)[:, None] * comb + np.sin(lift)[:, None] * nrm
    step = (length / (n_pts - 1))[:, None]
    phase = rng.random(n) * 6.28
    for k in range(1, n_pts):
        f = k / (n_pts - 1)
        p = pts[:, k - 1] + d * step
        # keep the strand above the skin by offset(f)
        want = offset0 + volume * f
        dist = prog(p, 0.02)
        under = dist < want
        if under.any():
            g = gradient(prog, p[under], 0.0004)
            g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
            p[under] += g * (want[under] - dist[under])[:, None] if np.ndim(want) else g * (want - dist[under])[:, None]
        if frizz > 0:
            p += frizz * np.stack([np.sin(phase + k * 1.7), np.cos(phase * 1.3 + k * 2.1), np.sin(phase * 0.7 + k * 1.1)], 1) * f
        nd = p - pts[:, k - 1]
        nd /= np.maximum(np.linalg.norm(nd, axis=1, keepdims=True), 1e-9)
        # relax towards the comb direction (hair falls along the style flow)
        d = nd * 0.75 + comb * 0.25
        if curl:
            d = d + curl * nrm * np.cos(f * 3.0)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pts[:, k] = p
    if clump > 0 and n > 50:
        from scipy.spatial import cKDTree
        guides = rng.choice(n, size=max(n // 25, 1), replace=False)
        gi = guides[cKDTree(roots[guides]).query(roots)[1]]
        w = clump * (np.linspace(0, 1, n_pts) ** 2)[None, :, None]
        pts = pts * (1 - w) + (pts[gi] - pts[gi][:, :1] + pts[:, :1]) * w * 0.4 + pts[gi] * w * 0.6
    return pts


def _pack(pts, r_root, r_tip):
    n, k, _ = pts.shape
    radius = np.linspace(r_root, r_tip, k)[None, :].repeat(n, 0)
    return pts.reshape(-1, 3).astype(F), np.full(n, k, np.int32), radius.reshape(-1).astype(F)


def scalp(prog, verts, tris, mask, rng, count=110000):
    # irregular hairline: thin out roots near the mask edge with noise so the front is not a ruler line
    jag = np.sin(verts[:, 0] * 900.0) * np.sin(verts[:, 0] * 2300.0 + 1.3) * 0.5 + 0.5
    edge = np.clip((mask - 0.25) / 0.6, 0, 1)
    roots, dens = sample_roots(verts, tris, np.clip(mask * (0.55 + 0.45 * jag) * (0.3 + 0.7 * edge), 0, 1), count, rng)
    nrm = surface_normals(prog, roots)
    q = roots - O
    # textured crop: ~32 mm on top brushed up and back, faded to 2-6 mm on the sides and nape
    top = np.clip((q[:, 2] - 0.050) / 0.030, 0, 1) * np.clip((0.072 - np.abs(q[:, 0])) / 0.012, 0, 1)
    length = 0.007 + 0.006 * np.clip((q[:, 2] - 0.0) / 0.05, 0, 1) + 0.024 * top
    length *= rng.uniform(0.75, 1.2, len(roots))
    front = O + [0.0, -0.035, 0.070]
    crown = O + [0.012, 0.100, 0.100]
    up_back = tangent(roots - front, nrm)                     # from the hairline up and back
    whorl = tangent(roots - crown, nrm)                        # radiating from the crown
    wf = np.clip((q[:, 1] - 0.03) / 0.07, 0, 1)[:, None]
    flow = tangent(up_back * (1 - wf) + whorl * wf + np.array([-0.15, 0.0, 0.0]) * top[:, None], nrm)
    down = tangent(np.tile([0.0, 0.3, -1.0], (len(roots), 1)), nrm)
    comb = tangent(flow * top[:, None] + down * (1 - top[:, None]), nrm)
    lift = np.radians(16 + 46 * top * (1 - 0.5 * wf[:, 0]) + rng.normal(0, 7, len(roots)))
    pts = grow(prog, roots, nrm, comb, length, lift, 10, 0.0005, 0.007 * top, rng, clump=0.55, frizz=0.0006)
    return _pack(pts, 0.000036, 0.000010)


def eyebrows(prog, verts, tris, mask, rng, count=1500):
    roots, dens = sample_roots(verts, tris, mask, count, rng)
    nrm = surface_normals(prog, roots)
    q = roots - O
    side = np.sign(q[:, 0])
    t = np.clip((np.abs(q[:, 0]) - 0.010) / 0.042, 0, 1)          # 0 medial .. 1 tail
    up = np.array([0.0, 0.0, 1.0])
    lat = np.stack([side, np.zeros_like(side), np.zeros_like(side)], 1)
    raw = up[None] * (1 - t[:, None]) ** 1.5 * 1.2 + lat * (0.4 + t[:, None]) - up[None] * 0.25 * t[:, None] ** 2
    comb = tangent(raw, nrm)
    length = (0.0070 + 0.0035 * np.sin(t * np.pi)) * rng.uniform(0.8, 1.2, len(roots))
    lift = np.radians(12 + rng.normal(0, 4, len(roots)))
    pts = grow(prog, roots, nrm, comb, length, lift, 6, 0.0002, 0.0008, rng, frizz=0.00015)
    return _pack(pts, 0.000045, 0.000015)


def stubble(prog, verts, tris, mask, rng, count=32000):
    roots, dens = sample_roots(verts, tris, mask, count, rng)
    nrm = surface_normals(prog, roots)
    comb = tangent(np.tile([0.0, -0.3, -1.0], (len(roots), 1)) + rng.normal(0, 0.35, (len(roots), 3)), nrm)
    length = rng.uniform(0.0007, 0.0017, len(roots))
    lift = np.radians(35 + rng.normal(0, 8, len(roots)))
    pts = grow(prog, roots, nrm, comb, length, lift, 3, 0.0001, 0.0001, rng)
    return _pack(pts, 0.000030, 0.000016)


def _margin(eye_side, upper, n):
    """Points along the upper or lower lid margin of one eye (same construction as head.eye_parts)."""
    E = headmod.LANDMARKS["eye_l" if eye_side > 0 else "eye_r"]
    R = headmod.LID_R

    def e(x, y, z):
        return E + np.array([eye_side * x, y, z])

    def on_shell(x, z, r=R):
        return e(x, -np.sqrt(r * r - x * x - z * z), z)

    cm, cl = e(*headmod.LID_CM), e(*headmod.LID_CL)
    apex = on_shell(*headmod.LID_TU) if upper else on_shell(*headmod.LID_TL)
    ctrl = 2 * apex - 0.5 * (cm + cl)
    t = np.linspace(0.06, 0.94, n)[:, None]
    p = (1 - t) ** 2 * cm + 2 * (1 - t) * t * ctrl + t ** 2 * cl
    v = p - E
    return E + v / np.linalg.norm(v, axis=1, keepdims=True) * (R - 0.0004), E


def eyelashes(rng):
    groups = []
    for s in (1, -1):
        for upper, n, L0 in ((True, 130, 0.0095), (False, 70, 0.0055)):
            base, E = _margin(s, upper, n)
            base = base + rng.normal(0, 0.00025, base.shape)
            out = base - E
            out /= np.linalg.norm(out, axis=1, keepdims=True)
            fwd = np.array([0.0, -1.0, 0.0])
            vert = np.array([0.0, 0.0, 1.0 if upper else -1.0])
            t = np.linspace(0, 1, n)
            L = L0 * (0.55 + 0.45 * np.sin(np.clip(t * 1.15, 0, 1) * np.pi)) * rng.uniform(0.85, 1.1, n)
            k = 7
            pts = np.zeros((n, k, 3), F)
            pts[:, 0] = base
            d = norm(fwd) * 0.6 + out * 0.6 + vert * 0.15
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            for i in range(1, k):
                f = i / (k - 1)
                d = d + vert[None] * (0.35 if upper else 0.2) * f
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                pts[:, i] = pts[:, i - 1] + d * (L / (k - 1))[:, None]
            groups.append(pts)
    pts = np.concatenate(groups)
    return _pack(pts, 0.000060, 0.000015)
