"""Clothes built from the body's signed distance field: t-shirt, five-pocket jeans, leather sneakers.

Every garment is derived from `build_body.make_program()` at build time, so it refits when the
body changes:
  * the body SDF is cached on a regular grid around each garment (trilinear lookups are ~100x
    cheaper than the full program and let the mesher run in seconds);
  * loose parts (shirt torso, sleeves, jeans seat and legs) are cylindrical height fields
    R(s, theta) measured by ray-marching sub-programs of the body (torso only, arms only, legs
    only), convex-hulled per section (fabric bridges concavities), then draped: going down the
    garment the radius may only shrink at a limited slope, so fabric falls from the chest, the
    shoulder blades, the seat and the knee instead of following the skin;
  * the outer garment surface is the smooth union of those height fields with the body offset by
    a small ease, minus fold displacements, clamped so the inner face always clears the skin;
  * the outer surface is meshed with the sparse surface-nets mesher as a closed solid, the cut
    caps (hem, neck, sleeve ends, waist) are deleted, boundary vertices are projected onto the
    exact cut curves, and the open sheet is thickened in Python (inner layer + rim).
Per-vertex attributes for the shaders (seam distances, arc length for dashed stitches,
chart coordinates for weave direction, fold height, wear masks, material masks) are computed
analytically from the same charts and seam polylines.
"""
import hashlib
import os
import time

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

from .mesher import surface_nets
from .sdf import F, Program, norm, smax, smin

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, "build", "cache")
UP = np.array([0.0, 0.0, 1.0])

ARM_BONES = ("upperarm", "lowerarm", "hand", "index", "middle", "ring", "pinky", "thumb")
LEG_BONES = ("thigh", "calf", "foot", "ball")


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------- body access
def _op_bones(node):
    return {l.bone for l in node.leaves() if l.bone}


def sub_program(prog, keep):
    """Program with only the ops whose leaf bones satisfy keep(set_of_bones); relief kept."""
    out = Program()
    for kind, node, k in prog.ops:
        if kind == "disp":
            out.ops.append((kind, node, k))
        elif kind == "add":
            if keep(_op_bones(node)):
                out.ops.append((kind, node, k))
        else:
            out.ops.append((kind, node, k))
    return out


def has_prefix(bones, prefixes):
    return any(b.startswith(prefixes) for b in bones)


def region_program(prog, lo, hi, band=0.06):
    """Ops (and relief bumps) that can influence distances inside [lo, hi] up to `band`."""
    lo = np.asarray(lo) - band
    hi = np.asarray(hi) + band
    out = Program()
    for kind, node, k in prog.ops:
        if kind == "disp":
            bumps = [b for b in node if np.all(b.hi >= lo) and np.all(b.lo <= hi)]
            if bumps:
                out.ops.append((kind, bumps, k))
        elif kind == "int" or (np.all(node.hi + k >= lo) and np.all(node.lo - k <= hi)):
            out.ops.append((kind, node, k))
    return out


def _source_hash(extra):
    h = hashlib.sha1(repr(extra).encode())
    src = os.path.join(ROOT, "src")
    for name in sorted(os.listdir(os.path.join(src, "rh"))):
        if name.endswith(".py") and name not in ("clothes.py", "mesher.py", "weights.py", "rig.py", "stitch.py",
                                                 "masks.py", "hair.py", "pose.py"):
            h.update(open(os.path.join(src, "rh", name), "rb").read())
    h.update(open(os.path.join(src, "build_body.py"), "rb").read())
    return h.hexdigest()[:12]


class GridField:
    """Signed distance sampled on a regular grid; trilinear lookup (clamped outside)."""

    def __init__(self, lo, h, vals):
        self.lo = np.asarray(lo, np.float32)
        self.h = float(h)
        self.vals = np.asarray(vals, np.float32)
        self.hi = self.lo + (np.array(self.vals.shape) - 1) * self.h

    def __call__(self, P, band=None):
        idx = (np.asarray(P, np.float32) - self.lo) / F(self.h)
        return ndimage.map_coordinates(self.vals, idx.T, order=1, mode="nearest").astype(F)

    def grad(self, P, eps=None):
        eps = eps or 0.5 * self.h
        g = np.zeros((len(P), 3), F)
        for a in range(3):
            d = np.zeros(3, F); d[a] = eps
            g[:, a] = (self(P + d) - self(P - d)) / (2 * eps)
        return g


def body_grid(prog, lo, hi, h, name, near=0.05, log=print):
    """Cache the body distance on a grid: exact within `near` of the skin, interpolated elsewhere."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    key = _source_hash((name, np.round(lo, 4).tolist(), np.round(hi, 4).tolist(), h, near))
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"body_{name}_{key}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return GridField(d["lo"], float(d["h"]), d["vals"])
    t0 = time.time()
    rp = region_program(prog, lo, hi, near + 0.02)
    n = np.ceil((hi - lo) / h).astype(int) + 1
    S = 4
    nc = (n - 1) // S + 2
    cx, cy, cz = (lo[i] + np.arange(nc[i]) * h * S for i in range(3))
    C = np.stack(np.meshgrid(cx, cy, cz, indexing="ij"), -1).reshape(-1, 3).astype(F)
    cv = np.concatenate([rp(C[i:i + 300000], 0.08) for i in range(0, len(C), 300000)]).reshape(nc)
    vals = np.empty(n, F)
    ix, iy = np.meshgrid(np.arange(n[0]), np.arange(n[1]), indexing="ij")
    n_exact = 0
    for k in range(n[2]):
        coords = np.stack([ix.ravel() / S, iy.ravel() / S, np.full(ix.size, k / S)])
        v = ndimage.map_coordinates(cv, coords, order=1, mode="nearest").astype(F)
        sel = np.abs(v) < near + 2 * h * S
        if sel.any():
            P = (lo + np.stack([ix.ravel()[sel], iy.ravel()[sel], np.full(sel.sum(), k)], 1) * h).astype(F)
            v[sel] = rp(P, near + 0.01)
            n_exact += int(sel.sum())
        vals[:, :, k] = v.reshape(n[0], n[1])
    np.savez(path, lo=lo, h=h, vals=vals)
    log(f"  body grid {name}: {tuple(n)} ({n_exact} exact) in {time.time() - t0:.0f}s")
    return GridField(lo, h, vals)


# ----------------------------------------------------------------------------- cylindrical charts
class Polar:
    """Cylindrical chart: axis from origin o along unit `ax`, theta measured from `ref` towards ax x ref.

    Height fields R[s, theta] live on a regular (s, theta) table.
    """

    def __init__(self, o, ax, ref, s0, s1, ds=0.002, nth=360):
        self.o = np.asarray(o, float)
        self.ax = norm(ax)
        r = np.asarray(ref, float)
        self.e1 = norm(r - self.ax * (r @ self.ax))
        self.e2 = np.cross(self.ax, self.e1)
        self.s = np.arange(s0, s1 + 1e-9, ds)
        self.ds = ds
        self.nth = nth
        self.th = -np.pi + np.arange(nth) * (2 * np.pi / nth)
        self.dth = 2 * np.pi / nth

    def coords(self, P):
        rel = np.asarray(P, np.float64) - self.o
        s = rel @ self.ax
        x = rel @ self.e1
        y = rel @ self.e2
        return s, np.arctan2(y, x), np.hypot(x, y)

    def world(self, s, th, r):
        s, th, r = np.broadcast_arrays(np.asarray(s, float), np.asarray(th, float), np.asarray(r, float))
        return (self.o + s[..., None] * self.ax + (r * np.cos(th))[..., None] * self.e1
                + (r * np.sin(th))[..., None] * self.e2)

    def dirs(self, th):
        th = np.asarray(th, float)
        return np.cos(th)[..., None] * self.e1 + np.sin(th)[..., None] * self.e2

    def _idx(self, s, th):
        i = (s - self.s[0]) / self.ds
        j = (np.mod(th + np.pi, 2 * np.pi)) / self.dth
        return np.stack([i, j])

    def lookup(self, T, s, th):
        Tp = np.concatenate([T, T[:, :1]], axis=1)
        return ndimage.map_coordinates(Tp, self._idx(s, th), order=1, mode="nearest")

    def field(self, R):
        """SDF-like function of the solid rho < R(s, theta)."""
        Rs = np.gradient(R, self.ds, axis=0)
        Rt = (np.roll(R, -1, 1) - np.roll(R, 1, 1)) / (2 * self.dth)
        tabs = np.stack([R, Rs, Rt]).astype(np.float32)

        def f(P):
            s, th, rho = self.coords(P)
            r, rs, rt = (self.lookup(t, s, th) for t in tabs)
            return ((rho - r) / np.sqrt(1 + (rt / np.maximum(rho, 0.02)) ** 2 + rs ** 2)).astype(F)
        return f


def march_radius(sdf, chart, s, th, r0, iters=80, tol=2e-4):
    """Outermost surface hit along rays from the chart axis, marching inwards from radius r0."""
    s = np.asarray(s, float).ravel(); th = np.asarray(th, float).ravel()
    r = np.full(len(s), float(r0))
    active = np.ones(len(s), bool)
    for _ in range(iters):
        if not active.any():
            break
        P = chart.world(s[active], th[active], r[active]).astype(F)
        d = sdf(P, 0.03).astype(float)
        d = np.minimum(d, 0.02)
        hit = d < tol
        idx = np.nonzero(active)[0]
        r[idx[~hit]] -= np.maximum(0.9 * d[~hit], 2e-4)
        active[idx[hit]] = False
        r = np.maximum(r, 0.0)
        active &= r > 0
    return r


def hull_radius(r, th):
    """Radius along each theta of the 2-D convex hull of the polar polygon (r, th)."""
    if (r > 1e-3).sum() < 8:
        return r
    pts = np.stack([r * np.cos(th), r * np.sin(th)], 1)[r > 1e-3]
    hull = ConvexHull(pts)
    out = np.full(len(th), np.inf)
    u = np.stack([np.cos(th), np.sin(th)], 1)
    for eq in hull.equations:          # n.x + c <= 0 inside
        n, c = eq[:2], eq[2]
        nu = u @ n
        ok = nu > 1e-9
        out[ok] = np.minimum(out[ok], -c / nu[ok])
    return np.where(np.isfinite(out), out, r)


def drape_down(base, slope, ds):
    """R[i] (rows ordered top -> bottom) = max(base[i], R[i-1] - slope*ds): fabric falls."""
    out = base.copy()
    for i in range(1, len(out)):
        out[i] = np.maximum(out[i], out[i - 1] - slope[i] * ds if np.ndim(slope) else out[i - 1] - slope * ds)
    return out


def periodic_smooth(T, sig_s, sig_t):
    return ndimage.gaussian_filter(T, (sig_s, sig_t), mode=("nearest", "wrap"))


# ----------------------------------------------------------------------------- mesh utilities
def numgrad(f, P, eps=4e-4):
    P = np.asarray(P, F)
    g = np.zeros_like(P)
    for a in range(3):
        d = np.zeros(3, F); d[a] = eps
        g[:, a] = (f(P + d) - f(P - d)) / (2 * eps)
    return g


def compact(V, Q, keep_face):
    Q = Q[keep_face]
    used = np.zeros(len(V), bool)
    used[Q.ravel()] = True
    remap = np.cumsum(used) - 1
    return V[used], remap[Q], used


def components(nv, Q):
    """Connected component label per vertex."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a = Q.ravel()
    b = np.roll(Q, -1, axis=1).ravel()
    m = coo_matrix((np.ones(len(a)), (a, b)), shape=(nv, nv))
    return connected_components(m, directed=False)[1]


def keep_largest(V, Q, min_frac=0.02):
    lab = components(len(V), Q)
    cnt = np.bincount(lab)
    good = cnt >= min_frac * cnt.max()
    return compact(V, Q, good[lab[Q[:, 0]]])[:2]


def boundary_edges(Q):
    a = Q.ravel()
    b = np.roll(Q, -1, axis=1).ravel()
    key = np.minimum(a, b).astype(np.int64) * (1 << 31) + np.maximum(a, b)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    single = cnt[inv] == 1
    return a[single], b[single]


def vertex_normals(V, Q):
    V = V.astype(np.float64)
    n = np.cross(V[Q[:, 2]] - V[Q[:, 0]], V[Q[:, 3]] - V[Q[:, 1]])
    vn = np.zeros_like(V)
    for c in range(4):
        np.add.at(vn, Q[:, c], n)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def laplacian_smooth(V, Q, mask, iters=3, lam=0.5):
    """Smooth only vertices where mask is true (e.g. boundary rings)."""
    a = Q.ravel(); b = np.roll(Q, -1, axis=1).ravel()
    for _ in range(iters):
        acc = np.zeros_like(V); cnt = np.zeros(len(V))
        np.add.at(acc, a, V[b]); np.add.at(cnt, a, 1)
        np.add.at(acc, b, V[a]); np.add.at(cnt, b, 1)
        avg = acc / np.maximum(cnt, 1)[:, None]
        V = np.where(mask[:, None], V + lam * (avg - V), V)
    return V


def sheet(outer, cut, lo, hi, h, log=print, project_iters=4):
    """Mesh {outer <= 0} n {cut <= 0} and keep only the `outer` surface: an open sheet.

    Returns (V, Q, boundary vertex mask). Boundary vertices are projected onto outer = cut = 0.
    """
    def solid(P, band):
        return np.maximum(outer(P), cut(P))
    V, Q = surface_nets(solid, lo, hi, h, levels=3, act=1.8, ev=3.0, project=1, log=lambda *a: None)
    V = V.astype(np.float64)
    C = V[Q].mean(axis=1).astype(F)
    on_outer = outer(C) >= cut(C)
    V, Q, _ = compact(V, Q, on_outer)
    V, Q = keep_largest(V, Q)
    ea, eb = boundary_edges(Q)
    bmask = np.zeros(len(V), bool)
    bmask[ea] = True
    bmask[eb] = True
    # boundary ring: snap onto the cut curve (alternate projections onto both surfaces)
    for _ in range(project_iters):
        P = V[bmask].astype(F)
        for f in (cut, outer):
            d = f(P); g = numgrad(f, P)
            P = P - (d / np.maximum((g * g).sum(1), 1e-9))[:, None] * g
        V[bmask] = P
    ring1 = bmask.copy()
    a = Q.ravel(); b = np.roll(Q, -1, axis=1).ravel()
    ring1[a[bmask[b]]] = True
    V = laplacian_smooth(V, Q, ring1 & ~bmask, iters=2)
    P = V[ring1 & ~bmask].astype(F)
    d = outer(P); g = numgrad(outer, P)
    V[ring1 & ~bmask] = P - (d / np.maximum((g * g).sum(1), 1e-9))[:, None] * g
    log(f"  sheet: {len(V)} verts {len(Q)} quads, {bmask.sum()} boundary verts")
    return V, Q, bmask


def thicken(V, Q, bmask, t):
    """Solidify an open sheet inwards by per-vertex thickness t (m). Returns verts, flat faces, sizes, side.

    side: 1 outer layer, 0 inner layer (for shading the fabric back)."""
    t = np.broadcast_to(np.asarray(t, float), (len(V),))
    n = vertex_normals(V, Q)
    inner = V - n * t[:, None]
    nv = len(V)
    ea, eb = boundary_edges(Q)
    rim = np.stack([eb, ea, ea + nv, eb + nv], 1)
    faces = np.concatenate([Q, Q[:, ::-1] + nv, rim])
    VV = np.concatenate([V, inner]).astype(np.float32)
    side = np.concatenate([np.ones(nv), np.zeros(nv)]).astype(np.float32)
    return VV, faces, side


class Part:
    """Mesh part with per-vertex attributes; parts are concatenated into one npz."""

    def __init__(self, V, faces, attrs=None):
        self.V = np.asarray(V, np.float32)
        self.faces = np.asarray(faces, np.int64)
        if self.faces.ndim == 1:
            raise ValueError("faces must be (n, k)")
        self.attrs = {k: np.broadcast_to(np.asarray(v, np.float32), (len(self.V),)).copy() for k, v in (attrs or {}).items()}


def merge(parts):
    keys = sorted({k for p in parts for k in p.attrs})
    V, flat, sizes, attrs = [], [], [], {k: [] for k in keys}
    off = 0
    for p in parts:
        V.append(p.V)
        flat.append((p.faces + off).ravel())
        sizes.append(np.full(len(p.faces), p.faces.shape[1], np.int32))
        for k in keys:
            attrs[k].append(p.attrs.get(k, np.zeros(len(p.V), np.float32)))
        off += len(p.V)
    out = {"verts": np.concatenate(V).astype(np.float32), "faces": np.concatenate(flat).astype(np.int32),
           "sizes": np.concatenate(sizes)}
    for k in keys:
        out[f"attr_{k}"] = np.concatenate(attrs[k]).astype(np.float32)
    return out


def solid_part(sdf, lo, hi, h, attrs_fn=None):
    """Closed solid detail (loops, button, rivets, laces) meshed directly."""
    V, Q = surface_nets(lambda P, band: sdf(P), lo, hi, h, levels=2, act=1.8, ev=3.0, project=1,
                        log=lambda *a: None)
    attrs = attrs_fn(V) if attrs_fn else {}
    return Part(V, Q, attrs)


# ----------------------------------------------------------------------------- seams / stitches
class Seams:
    """Polylines on a garment; per-vertex signed distance and arc length to the nearest one.

    Each polyline is densely resampled; sign is + on the left of the travel direction w.r.t. the
    surface normal (cross(tangent, normal) side)."""

    def __init__(self):
        self.P, self.T, self.S, self.K = [], [], [], []

    def add(self, pts, kind=0.0, step=0.0008, closed=False):
        pts = np.asarray(pts, float)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = np.concatenate([[0], np.cumsum(seg)])
        if L[-1] <= 0:
            return
        u = np.arange(0, L[-1], step)
        dense = np.stack([np.interp(u, L, pts[:, a]) for a in range(3)], 1)
        tan = np.gradient(dense, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
        self.P.append(dense); self.T.append(tan); self.S.append(u); self.K.append(np.full(len(u), kind))

    def attrs(self, V, N, maxd=0.03):
        """dist (signed, m), arc (m along the seam), kind; far vertices get dist=maxd."""
        nv = len(V)
        if not self.P:
            return np.full(nv, maxd), np.zeros(nv), np.zeros(nv)
        P = np.concatenate(self.P); T = np.concatenate(self.T)
        S = np.concatenate(self.S); K = np.concatenate(self.K)
        tree = cKDTree(P)
        d, i = tree.query(V, distance_upper_bound=maxd)
        far = ~np.isfinite(d)
        i = np.where(far, 0, i)
        side = np.einsum("ij,ij->i", np.cross(T[i], N), V - P[i])
        sd = np.where(far, maxd, np.sign(side + 1e-12) * d)
        return sd.astype(np.float32), np.where(far, 0, S[i]).astype(np.float32), np.where(far, 0, K[i]).astype(np.float32)


def project_curve(outer, pts, iters=6):
    """Project 3-D points onto outer = 0 along the field gradient."""
    P = np.asarray(pts, F)
    for _ in range(iters):
        d = outer(P); g = numgrad(outer, P)
        P = P - (d / np.maximum((g * g).sum(1), 1e-9))[:, None] * g
    return P.astype(float)


# ----------------------------------------------------------------------------- body context
class BodyCtx:
    """The body program plus cached distance grids and part-only sub-programs."""

    def __init__(self, prog, log=print):
        self.prog = prog
        self.log = log
        self.torso = sub_program(prog, lambda b: not has_prefix(b, ARM_BONES + LEG_BONES))
        self.arms = sub_program(prog, lambda b: has_prefix(b, ("upperarm", "lowerarm")))
        self.legs = sub_program(prog, lambda b: has_prefix(b, ("thigh", "calf")))
        self.lower = sub_program(prog, lambda b: not has_prefix(b, ARM_BONES))
        self.feet = sub_program(prog, lambda b: has_prefix(b, ("foot", "ball", "calf", "thigh")))
        self._grids = {}

    def grid(self, name, lo, hi, h=0.003):
        if name not in self._grids:
            self._grids[name] = body_grid(self.prog, lo, hi, h, name, log=self.log)
        return self._grids[name]


def measure(sub, chart, r0, lo=None, hi=None):
    """Section radii R[s, theta] of a sub-program around a chart axis."""
    if lo is not None:
        sub = region_program(sub, lo, hi, 0.04)
    S, TH = np.meshgrid(chart.s, chart.th, indexing="ij")
    r = march_radius(sub, chart, S, TH, r0)
    return r.reshape(S.shape)


def fold_noise(u, v, waves, seed=0):
    """Sum of tilted sinusoids in (u, v) metres; waves: (wavelength, amplitude, tilt)."""
    rng = np.random.default_rng(seed)
    out = np.zeros(np.broadcast(u, v).shape)
    tot = 0.0
    for lam, amp, tilt in waves:
        ph = rng.uniform(0, 2 * np.pi)
        drift = rng.uniform(-1, 1) * tilt
        out = out + amp * np.sin(2 * np.pi * (u + drift * v) / lam + ph)
        tot += amp
    return out / max(tot, 1e-9)


def angdiff(a, b):
    return np.mod(a - b + np.pi, 2 * np.pi) - np.pi


# ============================================================================= T-SHIRT
SHIRT = dict(
    hem_z=0.930,          # bottom hem height
    sleeve_end=0.165,     # sleeve hem, metres along the upper arm from the shoulder joint
    t=0.0011,             # single jersey (sheet thickness)
    t_hem=0.0019,         # folded hems
    t_collar=0.0024,      # double rib collar
    collar_w=0.024,
    clear=0.0015,         # min gap between the inner face and the skin
)
NECK_AXIS = np.array([0.0, 0.03, 0.0])


def neckline_z(phi):
    """Crew neckline height around the neck axis (phi = atan2(y, x)): low at the front."""
    return 1.480 + 0.0275 * np.sin(phi) - 0.0078 * np.cos(2 * phi)


def _neck_phi(P):
    return np.arctan2(P[:, 1] - NECK_AXIS[1], P[:, 0])


class Shirt:
    def __init__(self, ctx, jeans=None, log=print):
        from .skeleton import J, UPPERARM
        self.ctx = ctx
        self.log = log
        self.lo = np.array([-0.40, -0.17, 0.885]); self.hi = np.array([0.40, 0.21, 1.535])
        G = ctx.grid("shirt", [-0.42, -0.20, 0.80], [0.42, 0.22, 1.60])
        self.G = G
        t0 = time.time()
        # ---------------- torso: hull of the torso-only body + ease, draped from the chest/blades
        tc = Polar([0.0, 0.012, 0.0], UP, [1, 0, 0], 0.86, 1.44, 0.004, 360)
        rb = measure(ctx.torso, tc, 0.40, [-0.4, -0.25, 0.84], [0.4, 0.25, 1.46])
        rh = np.stack([hull_radius(rb[i], tc.th) for i in range(len(tc.s))])
        z = tc.s[:, None]
        back = smoothstep(-0.3, 1.0, np.sin(tc.th))[None, :]               # +y side
        e = np.interp(z, [0.86, 0.95, 1.05, 1.18, 1.30, 1.44], [0.016, 0.017, 0.014, 0.009, 0.006, 0.005])
        e = e + 0.004 * back * smoothstep(1.28, 1.05, z)
        base = rh + e
        R = drape_down(base[::-1], 0.10, tc.ds)[::-1]
        R = periodic_smooth(R, 1.5, 2.0)
        slack = np.maximum(R - (rb + 0.004), 0)
        u = (tc.th * 0.16)[None, :]
        pat = fold_noise(u + 0 * z, z + 0 * u, [(0.09, 1.0, 0.25), (0.13, 0.8, 0.35), (0.055, 0.45, 0.2)], seed=3)
        w = smoothstep(1.24, 1.02, z)
        self.t_fold = np.clip(0.55 * slack, 0, 0.007) * w * pat
        R = R + self.t_fold
        self.tc, self.t_R, self.t_rb = tc, R, rb
        self.f_torso = tc.field(R)
        # ---------------- sleeve (left; right by mirroring): arm hull + ease, sagging under gravity
        sc = Polar(J["upperarm_l"], UPPERARM.ey, UPPERARM.ez, -0.04, 0.23, 0.004, 240)
        ra = measure(ctx.arms, sc, 0.16, J["upperarm_l"] - 0.12, J["upperarm_l"] + 0.30)
        rah = np.stack([hull_radius(ra[i], sc.th) for i in range(len(sc.s))])
        down = -UP - sc.ax * (sc.ax @ -UP)
        th_down = np.arctan2(down @ sc.e2, down @ sc.e1)
        self.sleeve_th_down = th_down
        s = sc.s[:, None]
        base = rah + np.interp(s, [-0.04, 0.05, 0.12, 0.23], [0.006, 0.010, 0.012, 0.013])
        Rs = drape_down(base, 0.06, sc.ds)
        cosd = np.cos(angdiff(sc.th, th_down))[None, :]
        sag = np.interp(s, [-0.04, 0.03, 0.17, 0.23], [0.0, 0.006, 0.012, 0.012])
        Rs = np.maximum(ra + 0.004, Rs + sag * np.maximum(cosd, -0.3) - 0.002)
        Rs = periodic_smooth(Rs, 1.0, 1.5)
        under = smoothstep(0.2, 0.9, cosd) * smoothstep(-0.02, 0.02, s) * smoothstep(0.13, 0.05, s)
        bunch = np.sin(2 * np.pi * (s / 0.026 + 0.15 * angdiff(sc.th, th_down)[None, :])) * under
        slack_s = np.maximum(Rs - (ra + 0.004), 0)
        self.s_fold = np.clip(0.5 * slack_s, 0, 0.004) * bunch
        Rs = Rs + self.s_fold
        self.sc, self.s_R = sc, Rs
        self.f_sleeve_raw = sc.field(Rs)
        self.jeans = jeans
        self.log(f"  shirt charts in {time.time() - t0:.0f}s")

    # fields ------------------------------------------------------------------
    def f_sleeve(self, P):
        s, th, rho = self.sc.coords(P)
        f = self.f_sleeve_raw(P)
        f = smax(f, (s - (SHIRT["sleeve_end"] + 0.03)).astype(F), F(0.01))
        return smax(f, (0.0 - s).astype(F), F(0.03))

    def f_torso_c(self, P):
        z = P[:, 2]
        f = self.f_torso(P)
        f = smax(f, (z - 1.40).astype(F), F(0.05))
        return smax(f, (0.875 - z).astype(F), F(0.01))

    def ease_close(self, P):
        phi = _neck_phi(P)
        zn = neckline_z(phi)
        collar = smoothstep(0.035, 0.02, zn - P[:, 2])
        return (0.0046 - 0.0012 * collar).astype(F)

    def outer(self, P):
        P = np.asarray(P, F)
        Pm = P * np.array([-1, 1, 1], F)
        a = self.G(P) - self.ease_close(P)
        e = smin(a, self.f_torso_c(P), F(0.02))
        e = smin(e, smin(self.f_sleeve(P), self.f_sleeve(Pm), F(0.0)), F(0.03))
        # collar band stands proud of the body fabric
        zn = neckline_z(_neck_phi(P))
        band = smoothstep(-0.002, 0.001, zn - P[:, 2]) * smoothstep(SHIRT["collar_w"] + 0.002, SHIRT["collar_w"] - 0.001, zn - P[:, 2])
        e = e - (0.0009 * band).astype(F)
        # never closer to the skin than thickness + clearance
        e = np.minimum(e, self.G(P) - F(SHIRT["t_collar"] + SHIRT["clear"]))
        if self.jeans is not None:
            e = np.minimum(e, self.jeans.outer(P) - F(SHIRT["t_hem"] + 0.0015))
        return e.astype(F)

    def sleeve_cut_dist(self, P):
        """>0 inside the removed region beyond either sleeve hem."""
        out = np.full(len(P), -1.0, F)
        for sgn in (1, -1):
            Q = P * np.array([sgn, 1, 1], F)
            s, th, rho = self.sc.coords(Q)
            d = np.minimum(s - SHIRT["sleeve_end"], 0.13 - rho)
            out = np.maximum(out, d.astype(F))
        return out

    def hem_z(self, P):
        th = np.arctan2(P[:, 1] - 0.012, P[:, 0])
        return SHIRT["hem_z"] + 0.0025 * np.sin(3 * th + 0.7) + 0.0015 * np.sin(7 * th + 2.0)

    def cut(self, P):
        P = np.asarray(P, F)
        c = self.hem_z(P) - P[:, 2]
        c = np.maximum(c, P[:, 2] - neckline_z(_neck_phi(P)))
        c = np.maximum(c, self.sleeve_cut_dist(P))
        return c.astype(F)

    # mesh --------------------------------------------------------------------
    def build(self, h):
        t0 = time.time()
        V, Q, bmask = sheet(self.outer, self.cut, self.lo, self.hi, h, log=self.log)
        N = vertex_normals(V, Q)
        A = self.attributes(V, N)
        t = np.full(len(V), SHIRT["t"])
        t = np.where(A["hem"] < 0.022, SHIRT["t_hem"], t)
        t = np.where(A["collar"] < SHIRT["collar_w"], SHIRT["t_collar"], t)
        VV, faces, side = thicken(V, Q, bmask, t)
        attrs = {k: np.concatenate([v, v]) for k, v in A.items()}
        attrs["side"] = side
        self.log(f"  shirt built in {time.time() - t0:.0f}s")
        return Part(VV, faces, attrs)

    def attributes(self, V, N):
        P = V.astype(F)
        # chart coordinates: torso (u around, v up) or sleeve
        ft = np.abs(self.f_torso_c(P))
        s_l = self.sc.coords(P)[0]; s_r = self.sc.coords(P * [-1, 1, 1])[0]
        on_sleeve = (np.minimum(np.abs(self.f_sleeve(P)), np.abs(self.f_sleeve(P * np.array([-1, 1, 1], F)))) < ft) & (np.maximum(s_l, s_r) > 0.02)
        _, th, _ = self.tc.coords(P)
        u = th * 0.16
        v = P[:, 2].astype(float)
        right = P[:, 0] < 0
        Pl = np.where(right[:, None], P * [-1, 1, 1], P)
        ss, sth, _ = self.sc.coords(Pl)
        u = np.where(on_sleeve, sth * 0.055 * np.where(right, -1, 1), u)
        v = np.where(on_sleeve, 2.0 + ss, v)
        # distances to the cut edges
        hem = P[:, 2] - self.hem_z(P)
        sl = SHIRT["sleeve_end"] - ss
        hem = np.where(on_sleeve, sl, hem)
        neck = neckline_z(_neck_phi(P)) - P[:, 2]
        collar = np.where(neck < 0.06, neck, 1.0)
        # seams: shoulder, armhole, side seams + sleeve underarm seams
        seams = Seams()
        from .skeleton import J
        for sg in (1, -1):
            S = np.array([sg, 1, 1])
            a = np.array([0.072, 0.040, 1.50]); b = np.array([0.205, 0.022, 1.455])
            pts = np.linspace(a, b, 40)
            seams.add(project_curve(self.outer, pts * S), kind=0)
            th = np.linspace(-np.pi, np.pi, 160, endpoint=False)
            ring = self.sc.world(np.full_like(th, 0.035), th, 0.07)
            seams.add(project_curve(self.outer, ring * S), kind=1, closed=True)
            zz = np.linspace(self.hem_z(np.array([[0.18 * sg, 0.012, 0.93]]))[0] - 0.01, 1.28, 60)
            side = np.stack([np.full_like(zz, 0.20 * sg), np.full_like(zz, 0.012), zz], 1)
            seams.add(project_curve(self.outer, side), kind=0)
            ssv = np.linspace(0.05, SHIRT["sleeve_end"] + 0.01, 30)
            und = self.sc.world(ssv, np.full_like(ssv, self.sleeve_th_down), 0.09)
            seams.add(project_curve(self.outer, und * S), kind=0)
        sd, arc, kind = seams.attrs(V, N)
        fold = np.zeros(len(V))
        fs, fth, _ = self.tc.coords(P)
        fold = np.where(on_sleeve, 0, self.tc.lookup(self.t_fold, fs, fth))
        sfold = self.sc.lookup(self.s_fold, ss, sth)
        fold = np.where(on_sleeve, sfold, fold)
        return dict(u=u, v=v, hem=hem, collar=collar, seam=sd, seam_s=arc, seam_k=kind, fold=fold * 100)
