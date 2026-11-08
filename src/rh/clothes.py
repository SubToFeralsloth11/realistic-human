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

np.seterr(all="ignore")   # Accelerate's arm64 matmul raises spurious FP warnings; values are fine
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

    def blurred(self, sigma_m):
        """Low-passed copy: fabric bridges skin relief finer than ~sigma."""
        return GridField(self.lo, self.h, ndimage.gaussian_filter(self.vals, sigma_m / self.h))

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


ATTR_DEFAULTS = {"wb": 1.0, "hem": 1.0, "ptop": -1.0, "collar": 1.0, "seam": 0.03, "pk": 0.03, "side": 1.0}


def merge(parts):
    keys = sorted({k for p in parts for k in p.attrs})
    V, flat, sizes, attrs = [], [], [], {k: [] for k in keys}
    off = 0
    for p in parts:
        V.append(p.V)
        flat.append((p.faces + off).ravel())
        sizes.append(np.full(len(p.faces), p.faces.shape[1], np.int32))
        for k in keys:
            attrs[k].append(p.attrs.get(k, np.full(len(p.V), ATTR_DEFAULTS.get(k, 0.0), np.float32)))
        off += len(p.V)
    out = {"verts": np.concatenate(V).astype(np.float32), "faces": np.concatenate(flat).astype(np.int32),
           "sizes": np.concatenate(sizes)}
    for k in keys:
        out[f"attr_{k}"] = np.concatenate(attrs[k]).astype(np.float32)
    return out


def clearance(prog, V, grid=None):
    """Exact body distance at garment vertices that the grid says are within 5 mm of the skin."""
    V = np.asarray(V, F)
    d = grid(V) if grid is not None else np.full(len(V), -1.0, F)
    near = d < 0.005
    if near.any():
        lo, hi = V[near].min(0), V[near].max(0)
        rp = region_program(prog, lo, hi, 0.02)
        idx = np.nonzero(near)[0]
        for i in range(0, len(idx), 200000):
            j = idx[i:i + 200000]
            d[j] = rp(V[j], 0.01)
    return d


def solid_part(sdf, lo, hi, h, attrs_fn=None):
    """Closed solid detail (loops, button, rivets, laces) meshed directly."""
    V, Q = surface_nets(lambda P, band: sdf(P), lo, hi, h, levels=2, act=1.8, ev=3.0, project=1,
                        log=lambda *a: None)
    attrs = attrs_fn(V) if attrs_fn else {}
    return Part(V, Q, attrs)


# ----------------------------------------------------------------------------- seams / stitches
class Seams:
    """Polylines on a garment; per-vertex signed distance and arc length to the nearest one.

    Each polyline is densely resampled; the sign is + on the cross(tangent, normal) side. Each
    polyline carries the signed offsets (m) of up to two topstitch rows (0 = no row)."""

    def __init__(self):
        self.P, self.T, self.S, self.O = [], [], [], []

    def add(self, pts, o1=0.0, o2=0.0, step=0.0008, closed=False):
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
        self.P.append(dense); self.T.append(tan); self.S.append(u)
        self.O.append(np.tile([o1, o2], (len(u), 1)))

    def attrs(self, V, N, prefix, maxd=0.03):
        """{prefix: signed dist, prefix_s: arc length, prefix_o1/o2: row offsets}."""
        nv = len(V)
        if not self.P:
            z = np.zeros(nv, np.float32)
            return {prefix: z + maxd, f"{prefix}_s": z, f"{prefix}_o1": z, f"{prefix}_o2": z}
        P = np.concatenate(self.P); T = np.concatenate(self.T)
        S = np.concatenate(self.S); O = np.concatenate(self.O)
        d, i = cKDTree(P).query(V, distance_upper_bound=maxd)
        far = ~np.isfinite(d)
        i = np.where(far, 0, i)
        side = np.einsum("ij,ij->i", np.cross(T[i], N), V - P[i])
        sd = np.where(far, maxd, np.sign(side + 1e-12) * d)
        f = lambda x: np.where(far, 0, x).astype(np.float32)  # noqa: E731
        return {prefix: sd.astype(np.float32), f"{prefix}_s": f(S[i]), f"{prefix}_o1": f(O[i, 0]), f"{prefix}_o2": f(O[i, 1])}


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
        self.arms = sub_program(prog, lambda b: bool(b & {"upperarm_l", "lowerarm_l"}))   # left arm only
        self.legs = sub_program(prog, lambda b: bool(b & {"thigh_l", "calf_l"}))      # left leg only
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
    sleeve_end=0.185,     # sleeve hem, metres along the upper arm from the shoulder joint
    t=0.0011,             # single jersey (sheet thickness)
    t_hem=0.0019,         # folded hems
    t_collar=0.0024,      # double rib collar
    collar_w=0.024,
    clear=0.0015,         # min gap between the inner face and the skin
)
NECK_AXIS = np.array([0.0, 0.03, 0.0])
# crew-neck hole radius around the neck axis vs psi = atan2(y - 0.03, |x|): front, front-side,
# side, back-side, back. The neckline is where this vertical cylinder meets the shoulders/chest.
_HOLE_PSI = np.radians([-90, -45, 0, 45, 90])
_HOLE_R = np.array([0.094, 0.112, 0.121, 0.094, 0.077])


def neck_hole(P):
    """(rho - R_hole): < 0 inside the neck opening cylinder."""
    x = P[:, 0]; y = P[:, 1] - NECK_AXIS[1]
    psi = np.arctan2(y, np.abs(x))
    R = _pchip(_HOLE_PSI, _HOLE_R)(psi)
    return np.hypot(x, y) - R


def _neck_phi(P):
    return np.arctan2(P[:, 1] - NECK_AXIS[1], P[:, 0])


def hole_r(phi):
    return _pchip(_HOLE_PSI, _HOLE_R)(np.arctan2(np.sin(phi), np.abs(np.cos(phi))))


class Shirt:
    def __init__(self, ctx, jeans=None, log=print):
        from .skeleton import J, UPPERARM
        self.ctx = ctx
        self.log = log
        self.lo = np.array([-0.40, -0.17, 0.885]); self.hi = np.array([0.40, 0.21, 1.535])
        G = ctx.grid("shirt", [-0.42, -0.20, 0.80], [0.42, 0.22, 1.60])
        self.G = G
        self.Gs = G.blurred(0.005)
        t0 = time.time()
        # ---------------- torso: hull of the torso-only body + ease, draped from the chest/blades
        tc = Polar([0.0, 0.012, 0.0], UP, [1, 0, 0], 0.86, 1.44, 0.004, 360)
        rb = measure(ctx.torso, tc, 0.40, [-0.4, -0.25, 0.84], [0.4, 0.25, 1.46])
        rh = np.stack([hull_radius(rb[i], tc.th) for i in range(len(tc.s))])
        z = tc.s[:, None]
        back = smoothstep(-0.3, 1.0, np.sin(tc.th))[None, :]               # +y side
        e = np.interp(z, [0.86, 0.95, 1.05, 1.18, 1.30, 1.44], [0.022, 0.022, 0.017, 0.010, 0.006, 0.005])
        e = e + 0.004 * back * smoothstep(1.28, 1.05, z)
        base = rh + e
        # fabric falls from the chest and the shoulder blades (front/back), but the side seams
        # follow the flanks in to the waist; a uniform slope kept the shoulder-girdle width all
        # the way down and turned the shirt into a tent that swallowed the hanging arms
        side = np.cos(tc.th) ** 2
        slope = np.tile(0.07 + 0.50 * side, (len(tc.s), 1))
        R = drape_down(base[::-1], slope, tc.ds)[::-1]
        R = periodic_smooth(R, 1.5, 2.0)
        slack = np.maximum(R - (rb + 0.004), 0)
        u = (tc.th * 0.16)[None, :]
        pat = fold_noise(u + 0 * z, z + 0 * u, [(0.09, 1.0, 0.25), (0.13, 0.8, 0.35), (0.055, 0.45, 0.2)], seed=3)
        w = smoothstep(1.26, 1.02, z)
        pat = np.sign(pat) * np.abs(pat) ** 0.75
        amp = np.clip(0.8 * slack, 0, 0.015) * w
        self.t_fold = np.where(pat > 0, amp * pat, 0.45 * amp * pat)      # ridges stand out, valleys stay off the skin
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
        sag = np.interp(s, [-0.04, 0.03, 0.17, 0.23], [0.0, 0.003, 0.007, 0.007])
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
        # ---------------- crew collar: a rib band standing up on a cylinder around the neck base.
        # It starts where the cylinder meets the shirt on the shoulders/chest and rises 17 mm.
        phi = np.linspace(-np.pi, np.pi, 181)
        Rc = hole_r(phi) + 0.0025
        zz = np.arange(1.40, 1.62, 0.001)
        Pq = np.stack([np.repeat(Rc * np.cos(phi), len(zz)), np.repeat(NECK_AXIS[1] + Rc * np.sin(phi), len(zz)),
                       np.tile(zz, len(phi))], 1).astype(F)
        a = (self.Gs(Pq) - F(0.0034)).reshape(len(phi), len(zz))
        first = np.argmax(a >= 0, axis=1)            # first height (going up) where the cylinder leaves the shirt
        self.collar_phi, self.collar_top = phi, ndimage.gaussian_filter1d(zz[first] + 0.016, 3, mode="wrap")
        self.log(f"  collar top z: front {np.interp(-np.pi / 2, phi, self.collar_top):.3f} side "
                 f"{np.interp(0.0, phi, self.collar_top):.3f} back {np.interp(np.pi / 2, phi, self.collar_top):.3f}")
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

    ARMPIT_FOLDS = [  # (start near the armpit, end on the chest/back, height, half-width)
        ((0.185, -0.035, 1.325), (0.120, -0.118, 1.215), 0.0032, 0.007),
        ((0.190, -0.020, 1.300), (0.145, -0.112, 1.150), 0.0026, 0.008),
        ((0.190, 0.040, 1.320), (0.125, 0.125, 1.215), 0.0030, 0.007),
        ((0.195, 0.030, 1.295), (0.150, 0.118, 1.160), 0.0024, 0.008),
        ((0.205, -0.010, 1.330), (0.270, -0.040, 1.300), 0.0022, 0.006),
    ]

    def armpit_folds(self, P):
        out = np.zeros(len(P), F)
        for sg in (1, -1):
            Q = P * np.array([sg, 1, 1], F)
            for a, b, A, w in self.ARMPIT_FOLDS:
                a = np.asarray(a, F); b = np.asarray(b, F)
                ab = b - a
                t = np.clip(((Q - a) @ ab) / (ab @ ab), 0, 1)
                d = np.linalg.norm(Q - (a + t[:, None] * ab), axis=1)
                taper = np.sin(np.pi * np.clip(t * 1.15 + 0.05, 0, 1)) * (1 - 0.5 * t)
                out = out + F(A) * np.exp(-(d / w) ** 2) * taper
        return out

    def ease_close(self, P):
        collar = smoothstep(0.03, 0.012, neck_hole(P)) * (P[:, 2] > 1.40)
        return (0.0046 - 0.0010 * collar).astype(F)

    def outer(self, P):
        P = np.asarray(P, F)
        Pm = P * np.array([-1, 1, 1], F)
        a = self.Gs(P) - self.ease_close(P)
        e = smin(a, self.f_torso_c(P), F(0.02))
        e = smin(e, smin(self.f_sleeve(P), self.f_sleeve(Pm), F(0.0)), F(0.018))
        e = e - self.armpit_folds(P)
        # collar band: cylinder wall around the neck, blended into the shoulders
        cyl = np.maximum(neck_hole(P) - 0.0025, np.maximum(P[:, 2] - self.neck_top(P) - 0.01, 1.40 - P[:, 2]))
        e = smin(e, cyl.astype(F), F(0.006))
        # never closer to the skin than thickness + clearance
        e = np.minimum(e, self.G(P) - F(SHIRT["t_collar"] + SHIRT["clear"]))
        if self.jeans is not None:
            low = P[:, 2] < 1.075
            if low.any():
                e[low] = np.minimum(e[low], self.jeans.clad(P[low]) - F(SHIRT["t_hem"] + 0.0015))
        return e.astype(F)

    def sleeve_cut_dist(self, P):
        """>0 inside the removed region beyond either sleeve hem."""
        out = np.full(len(P), -1.0, F)
        for sgn in (1, -1):
            Q = P * np.array([sgn, 1, 1], F)
            s, th, rho = self.sc.coords(Q)
            d = np.minimum(s - SHIRT["sleeve_end"], 0.088 - rho)
            out = np.maximum(out, d.astype(F))
        return out

    def neck_top(self, P):
        return np.interp(_neck_phi(P), self.collar_phi, self.collar_top)

    def neckline(self, n=720):
        """3-D neckline (top edge of the collar) on the outer surface."""
        phi = np.linspace(-np.pi, np.pi, n, endpoint=False)
        R = hole_r(phi) + 0.0025
        P = np.stack([R * np.cos(phi), NECK_AXIS[1] + R * np.sin(phi), np.interp(phi, self.collar_phi, self.collar_top)], 1)
        return project_curve(self.outer, P)

    def collar_dist(self, V):
        if not hasattr(self, "_neck"):
            self._neck = self.neckline()
        d, _ = cKDTree(self._neck).query(V)
        return np.where(V[:, 2] > 1.38, d, 1.0)

    def hem_z(self, P):
        th = np.arctan2(P[:, 1] - 0.012, P[:, 0])
        return SHIRT["hem_z"] + 0.0025 * np.sin(3 * th + 0.7) + 0.0015 * np.sin(7 * th + 2.0)

    def cut(self, P):
        P = np.asarray(P, F)
        c = self.hem_z(P) - P[:, 2]
        c = np.maximum(c, np.minimum(P[:, 2] - self.neck_top(P), 0.03 - neck_hole(P)))
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
        collar = self.collar_dist(V)
        # seams: shoulder, armhole, side seams + sleeve underarm seams
        seams = Seams()
        from .skeleton import J
        for sg in (1, -1):
            S = np.array([sg, 1, 1])
            a = np.array([0.072, 0.040, 1.50]); b = np.array([0.205, 0.022, 1.455])
            pts = np.linspace(a, b, 40)
            seams.add(project_curve(self.outer, pts * S))
            th = np.linspace(-np.pi, np.pi, 160, endpoint=False)
            ring = self.sc.world(np.full_like(th, 0.035), th, 0.07)
            seams.add(project_curve(self.outer, ring * S), o1=0.0055 * sg, closed=True)
            zz = np.linspace(self.hem_z(np.array([[0.18 * sg, 0.012, 0.93]]))[0] - 0.01, 1.28, 60)
            side = np.stack([np.full_like(zz, 0.20 * sg), np.full_like(zz, 0.012), zz], 1)
            seams.add(project_curve(self.outer, side))
            ssv = np.linspace(0.05, SHIRT["sleeve_end"] + 0.01, 30)
            und = self.sc.world(ssv, np.full_like(ssv, self.sleeve_th_down), 0.09)
            seams.add(project_curve(self.outer, und * S))
        sa = seams.attrs(V, N, "seam")
        fold = np.zeros(len(V))
        fs, fth, _ = self.tc.coords(P)
        fold = np.where(on_sleeve, 0, self.tc.lookup(self.t_fold, fs, fth))
        sfold = self.sc.lookup(self.s_fold, ss, sth)
        fold = np.where(on_sleeve, sfold, fold)
        rib = _neck_phi(P) * 0.075
        return dict(u=u, v=v, hem=hem, collar=collar, rib=rib, fold=fold * 100, **sa)


# ============================================================================= SNEAKERS
SOLE_LIFT = 0.012          # outsole bottom at z = -SOLE_LIFT (body frame); the scene lifts the root


def _pchip(xs, ys):
    from scipy.interpolate import PchipInterpolator
    f = PchipInterpolator(np.asarray(xs, float), np.asarray(ys, float), extrapolate=True)
    lo, hi = xs[0], xs[-1]
    return lambda x: f(np.clip(x, lo, hi))


class Shoe:
    """Low-top leather court sneaker around the left foot (the right one is mirrored).

    Foot-local coordinates: lat (+ lateral), fwd (towards the toes) from the ankle centre, world z.
    """
    F0, F1 = -0.082, 0.238      # heel back / toe tip of the upper
    ZAX = 0.018                 # loft axis height (widest part of the upper)

    def __init__(self, ctx, log=print):
        from .sdf import Loft
        from .skeleton import FOOT, J
        self.log = log
        self.o = J["foot_l"].copy()
        self.ex, self.ey = FOOT.ex.copy(), FOOT.ey.copy()
        corners = np.array([self.o + a * self.ex + b * self.ey for a in (-0.09, 0.09) for b in (-0.12, 0.27)])
        self.lo = np.r_[corners[:, :2].min(0), -0.02]; self.hi = np.r_[corners[:, :2].max(0), 0.13]
        self.G = ctx.grid("foot", self.lo - 0.01, self.hi + [0.01, 0.01, 0.06], 0.002)
        L = self.F1 - self.F0
        fw = [-0.082, -0.065, -0.035, 0.000, 0.040, 0.080, 0.120, 0.150, 0.180, 0.205, 0.226, 0.238]
        latp = [0.022, 0.033, 0.039, 0.043, 0.049, 0.055, 0.059, 0.057, 0.047, 0.033, 0.018, 0.006]
        latn = [0.022, 0.030, 0.035, 0.039, 0.043, 0.047, 0.055, 0.057, 0.055, 0.049, 0.038, 0.024]
        top = [0.112, 0.114, 0.112, 0.108, 0.102, 0.089, 0.063, 0.054, 0.048, 0.043, 0.036, 0.028]
        self.top_f = _pchip(fw, top)
        self.latp_f = _pchip(fw, latp); self.latn_f = _pchip(fw, latn)
        a = self.o + self.F0 * self.ey + (self.ZAX - self.o[2]) * UP
        b = self.o + self.F1 * self.ey + (self.ZAX - self.o[2]) * UP
        sec = [((f - self.F0) / L, p, n, t - self.ZAX, self.ZAX - 0.002, 2.5 if f < 0.1 else 2.3)
               for f, p, n, t in zip(fw, latp, latn, top)]
        self.upper_loft = Loft(a, b, self.ex, sec, round0=0.010, round1=0.012)
        e = 0.0045
        zs = 0.006
        sec_s = [((f - self.F0) / L, p + e, n + e, 0.018 + (0.004 if f > 0.17 else 0.0), 0.018, 4.0)
                 for f, p, n in zip(fw, latp, latn)]
        a2 = a + (zs - self.ZAX) * UP - 0.0035 * self.ey
        b2 = b + (zs - self.ZAX) * UP + 0.0035 * self.ey
        self.sole_loft = Loft(a2, b2, self.ex, sec_s, round0=0.006, round1=0.006)
        # lacing: slot centre line / half-width and eyelets
        self.lace_c = _pchip([0.02, 0.06, 0.12], [-0.006, -0.008, -0.012])
        self.slot_w = _pchip([0.02, 0.06, 0.115], [0.0125, 0.011, 0.0085])
        self.eyelet_f = np.linspace(0.104, 0.036, 6)
        self.eyelets = self._eyelets()
        log(f"  shoe setup: {len(self.eyelets)} eyelets")

    # coordinates ---------------------------------------------------------------
    def local(self, P):
        rel = np.asarray(P, np.float64) - self.o
        return rel @ self.ex, rel @ self.ey, np.asarray(P, np.float64)[:, 2]

    def world(self, lat, fwd, z):
        lat, fwd, z = np.broadcast_arrays(np.asarray(lat, float), np.asarray(fwd, float), np.asarray(z, float))
        return self.o * [1, 1, 0] + lat[..., None] * self.ex + fwd[..., None] * self.ey + z[..., None] * UP

    def spring(self, fwd):
        return 0.009 * smoothstep(0.17, 0.25, fwd) ** 1.5 + 0.004 * smoothstep(-0.06, -0.09, fwd)

    def collar_z(self, fwd):
        return _pchip([-0.09, -0.07, -0.04, -0.01, 0.015, 0.035], [0.091, 0.086, 0.074, 0.066, 0.068, 0.074])(fwd)

    # fields --------------------------------------------------------------------
    def _sprung(self, P):
        lat, fwd, z = self.local(P)
        return (np.asarray(P, F) - (self.spring(fwd)[:, None] * UP).astype(F)).astype(F), lat, fwd, z

    def cavity(self, P):
        return self.G(P) - F(0.0018)

    def f_sole(self, P):
        Ps, lat, fwd, z = self._sprung(P)
        d = self.sole_loft._eval(Ps, 0.05)
        zb = Ps[:, 2]
        d = smax(d, (-SOLE_LIFT - zb).astype(F), F(0.003))
        # foxing groove around the sidewall
        gz = np.abs(zb - 0.0045)
        d = smax(d, (-(np.maximum(gz - 0.0006, -d - 0.0007))).astype(F), F(0.0004))
        # outsole tread: herringbone chevrons on the bottom 2 mm, inside a 4 mm margin
        p = 0.0048
        lm = np.mod(lat, 0.016) - 0.008
        q = (fwd + np.abs(lm)) / p
        g2 = np.abs(q - np.round(q)) * p * 0.707 - 0.0007
        g2 = np.where((fwd > 0.07) & (fwd < 0.10), 0.01, g2)          # flex zone: plain
        flex = np.abs(fwd - 0.085) - 0.0012                            # flex groove
        g2 = np.minimum(g2, flex)
        margin = -d - 0.004
        carve = np.maximum(np.maximum(g2, zb + SOLE_LIFT - 0.0018), -margin)
        return smax(d, (-carve).astype(F), F(0.0003)).astype(F)

    def f_upper_raw(self, P):
        Ps, lat, fwd, z = self._sprung(P)
        d = self.upper_loft._eval(Ps, 0.05)
        # overlays stand proud by 0.7 mm: toe cap, eyestays, heel counter
        d = d - F(0.0007) * self.overlay(lat, fwd, z).astype(F)
        # padded collar roll
        cz = self.collar_z(fwd)
        roll = smoothstep(0.022, 0.004, cz - z) * smoothstep(0.035, 0.015, fwd)
        d = d - F(0.0028) * roll.astype(F)
        return d

    def overlay(self, lat, fwd, z):
        toe = smoothstep(-0.0005, 0.0005, fwd - self.toecap_f(lat))
        es = self.eyestay(lat, fwd, z)
        heel = smoothstep(-0.0005, 0.0005, self.heel_f(z) - fwd)
        return np.maximum(np.maximum(toe, es), heel)

    def toecap_f(self, lat):
        x = (lat + 0.012) / 0.05
        return 0.168 - 0.035 * x * x

    def heel_f(self, z):
        return -0.008 - 0.42 * np.clip(z - 0.02, 0, None)

    def eyestay(self, lat, fwd, z):
        c = self.lace_c(fwd); w = self.slot_w(fwd)
        band = smoothstep(w + 0.0185, w + 0.0175, np.abs(lat - c))
        return band * smoothstep(0.122, 0.118, fwd) * smoothstep(0.010, 0.015, fwd) * smoothstep(0.045, 0.06, z)

    def _eyelets(self):
        pts = []
        for f in self.eyelet_f:
            for sgn in (-1, 1):
                lat = self.lace_c(f) + sgn * (self.slot_w(f) + 0.0075)
                pts.append((lat, f))
        P = self.world([p[0] for p in pts], [p[1] for p in pts], 0.2)
        # drop onto the upper top surface
        for _ in range(60):
            d = self.f_upper_raw(P.astype(F)).astype(float)
            P[:, 2] -= np.maximum(d, 2e-4) * (d > 1e-4)
        return P

    def f_slot(self, P):
        """Removal: the lacing gap above the tongue (>0 outside the gap)."""
        lat, fwd, z = self.local(P)
        a = np.abs(lat - self.lace_c(fwd)) - self.slot_w(fwd)
        b = np.maximum(0.02 - fwd, fwd - 0.118)
        return np.maximum(a, b).astype(F)

    def f_tongue(self, P):
        lat, fwd, z = self.local(P)
        g = self.G(P)
        shell = np.abs(g - F(0.0048)) - F(0.0028)
        reg = np.maximum(np.abs(lat - self.lace_c(fwd)) - (self.slot_w(fwd) + 0.012), np.maximum(-0.012 - fwd, fwd - 0.122))
        topz = 0.104 + 0.004 * np.cos(np.clip((lat - self.lace_c(fwd)) / 0.03, -1.5, 1.5))
        reg = smax(reg.astype(F), (z - topz).astype(F), F(0.006))
        return smax(shell, reg.astype(F), F(0.002))

    def f_upper(self, P):
        P = np.asarray(P, F)
        lat, fwd, z = self.local(P)
        d = self.f_upper_raw(P)
        g = self.G(P) - F(0.0045)
        fit = np.where(fwd > 0.024, np.maximum(g, (z - self.top_f(fwd) + 0.004).astype(F)), g)
        d = np.minimum(d, fit)                               # the foot always fits inside
        d = smax(d, -self.cavity(P), F(0.0008))
        # collar opening behind the throat, lacing gap in front of it
        d = smax(d, np.where(fwd < 0.028, z - self.collar_z(fwd), -1.0).astype(F), F(0.002))
        d = smax(d, -self.f_slot(P), F(0.0015))
        # punched eyelet holes (vertical cylinders)
        for e in self.eyelets:
            near = np.linalg.norm(P - e.astype(F), axis=1) < 0.012
            if near.any():
                hole = F(0.0021) - np.linalg.norm(P[near, :2] - e[:2].astype(F), axis=1)
                d[near] = smax(d[near], hole, F(0.0004))
        # heel pull tab: a curved leather tongue standing above the collar at the back
        sh = np.abs(self.f_upper_raw(P) + F(0.0004)) - F(0.0017)
        lat_c = lat + 0.002
        reg = np.maximum(np.maximum(np.abs(lat_c) - 0.0115, fwd - (self.F0 + 0.03)), z - (self.collar_z(fwd) + 0.010))
        reg = np.maximum(reg, self.collar_z(fwd) - 0.03 - z)
        tab = np.maximum(sh, reg.astype(F))
        return smin(d, tab.astype(F), F(0.002))

    def solid(self, P):
        P = np.asarray(P, F)
        d = smin(self.f_upper(P), self.f_sole(P), F(0.0015))
        d = smax(d, -self.cavity(P), F(0.0005))
        d = np.minimum(d, self.f_tongue(P))
        return d.astype(F)

    # helpers ---------------------------------------------------------------------
    def march_to(self, f, C, D, r0=0.12, iters=120):
        """Outermost point of {f < 0} along rays C + r * D (r decreasing from r0)."""
        C = np.asarray(C, float); D = np.asarray(D, float)
        D = D / np.linalg.norm(D, axis=-1, keepdims=True)
        r = np.full(len(C), r0)
        for _ in range(iters):
            d = f((C + r[:, None] * D).astype(F)).astype(float)
            step = np.where(d > 1e-4, np.maximum(0.8 * np.minimum(d, 0.01), 1e-4), 0.0)
            r -= step
            if not (step > 0).any():
                break
        return C + r[:, None] * D

    def surface_top(self, lat, fwd, f=None):
        f = f or self.f_upper_raw
        lat = np.atleast_1d(np.asarray(lat, float)); fwd = np.broadcast_to(np.atleast_1d(fwd), lat.shape)
        C = self.world(lat, fwd, 0.0)
        return self.march_to(f, C, np.tile(UP, (len(C), 1)), r0=0.16)

    def lace_tubes(self, r=0.0019):
        from .sdf import Tube
        E = self.eyelets.astype(float)
        med = E[0::2]; lat = E[1::2]
        tubes = []
        mid_z = []
        for f in self.eyelet_f:
            mid_z.append(self.march_to(lambda P: self.solid(P), self.world(self.lace_c(f), f, 0.0)[None], UP[None], 0.16)[0, 2])

        def lift(p, h):
            return p + UP * h

        def seg(a, b, zmid, over):
            m = 0.5 * (a + b)
            m[2] = zmid + r * (2.2 if over else 1.0)
            pts = [lift(a, -0.003), lift(a, r * 0.9), 0.5 * (lift(a, r) + m), m, 0.5 * (lift(b, r) + m), lift(b, r * 0.9), lift(b, -0.003)]
            return Tube(pts, r, samples=6)
        tubes.append(seg(med[0], lat[0], mid_z[0], False))
        for i in range(len(med) - 1):
            zm = 0.5 * (mid_z[i] + mid_z[i + 1])
            tubes.append(seg(med[i], lat[i + 1], zm, i % 2 == 0))
            tubes.append(seg(lat[i], med[i + 1], zm, i % 2 == 1))
        # bow on top of the last pair
        f = self.eyelet_f[-1] + 0.008
        K = self.world(self.lace_c(f), f, mid_z[-1] + 0.006)
        ex, ey = self.ex, self.ey
        for sgn in (1, -1):
            K = self._push_out([K], r)[0]
            tubes.append(Tube([lift(med[-1] if sgn < 0 else lat[-1], r), K], r, samples=4))
            loop = [K, K + sgn * 0.016 * ex + 0.004 * ey + 0.006 * UP, K + sgn * 0.034 * ex + 0.002 * ey - 0.004 * UP,
                    K + sgn * 0.030 * ex - 0.010 * ey - 0.008 * UP, K + sgn * 0.012 * ex - 0.008 * ey + 0.0, K]
            tubes.append(Tube(self._push_out(loop, r), r, samples=8))
            end = [K, K + sgn * 0.008 * ex + 0.012 * ey + 0.0, K + sgn * 0.020 * ex + 0.034 * ey - 0.006 * UP,
                   K + sgn * 0.026 * ex + 0.046 * ey - 0.006 * UP]
            tubes.append(Tube(self._push_out(end, r), r, samples=8))
            tip = self._push_out([end[-1]], r)[0]
        knot = self._push_out([self.world(self.lace_c(f), f, mid_z[-1] + 0.0065)], 0.0034)[0]
        from .sdf import Ellipsoid
        tubes.append(Ellipsoid(knot, (0.0052, 0.0042, 0.0034)))
        return tubes

    def _push_out(self, pts, r):
        from .sdf import catmull_rom
        P = catmull_rom(np.asarray(pts, float), 6)
        for _ in range(30):
            d = np.minimum(self.solid(P.astype(F)), self.G(P.astype(F))).astype(float)
            bad = d < r + 0.0006
            if not bad.any():
                break
            P[bad, 2] += (r + 0.0008 - d[bad])
        # re-thin to control points
        return P[::3] if len(P) > 6 else P

    def seams(self, V, N):
        S = Seams()
        up_f = self.f_upper_raw
        # toe cap edge (rows on the cap side)
        lat = np.linspace(-0.075, 0.075, 90)
        cap = []
        for a in np.linspace(-np.pi * 0.05, np.pi * 1.05, 120):
            cap.append(a)
        alph = np.linspace(-0.05 * np.pi, 1.05 * np.pi, 120)
        C = self.world(-0.006, self.toecap_f(np.array([-0.006]))[0], 0.020)
        pts = []
        for a in alph:
            d = -np.cos(a) * self.ex + np.sin(a) * UP
            p = self.march_to(up_f, C[None], d[None])[0]
            for _ in range(3):
                l_, f_, z_ = self.local(p[None])
                c2 = self.world(-0.006, self.toecap_f(l_)[0], 0.020)
                p = self.march_to(up_f, c2[None], d[None])[0]
            pts.append(p)
        pts = np.array(pts)
        pts = pts[pts[:, 2] > 0.024]
        toward = self.world(-0.006, 0.23, 0.04)
        self._add(S, pts, toward, 0.0016, 0.0036)
        # eyestay outer edges
        for sgn in (-1, 1):
            fw = np.linspace(0.016, 0.119, 60)
            la = self.lace_c(fw) + sgn * (self.slot_w(fw) + 0.018)
            pts = np.array([self.surface_top(l, f)[0] for l, f in zip(la, fw)])
            self._add(S, pts, self.world(self.lace_c(0.07), 0.07, 0.09), 0.0016, 0.0036)
            # slot edge single row
            la2 = self.lace_c(fw) + sgn * (self.slot_w(fw) + 0.0005)
            pts2 = np.array([self.surface_top(l, f)[0] for l, f in zip(la2, fw)])
            self._add(S, pts2, self.world(self.lace_c(0.07) + sgn * 0.03, 0.07, 0.09), 0.0022, 0.0)
        # heel counter edges (rows on the counter side) and collar row
        for sgn in (-1, 1):
            zz = np.linspace(0.024, 0.08, 40)
            fw = self.heel_f(zz)
            keep = zz < self.collar_z(fw) - 0.002
            C = self.world(-0.004, fw[keep], zz[keep])
            pts = self.march_to(up_f, C, np.tile(sgn * self.ex, (len(C), 1)))
            self._add(S, pts, self.world(sgn * 0.03, -0.08, 0.05), 0.0016, 0.0036)
        beta = np.linspace(0.02, np.pi - 0.02, 120)
        zc = []
        C0 = self.world(-0.004, -0.02, 0.0)
        pts = []
        for b in beta:
            d = np.cos(b) * self.ex - np.sin(b) * self.ey
            z = 0.075
            for _ in range(3):
                p = self.march_to(self.f_upper, (C0 + UP * (z - 0.005))[None], d[None])[0]
                z = self.collar_z(self.local(p[None])[1])[0]
            pts.append(p)
        pts = np.array(pts)
        self._add(S, pts, pts.mean(0) - UP * 0.03, 0.0045, 0.0)
        # cupsole stitch around the sidewall
        alph = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        C = self.world(-0.004, 0.075, 0.0)
        pts = []
        for a in alph:
            d = np.cos(a) * self.ex + np.sin(a) * self.ey
            f = 0.075 + 0.16 * np.sin(a)
            p = self.march_to(self.f_sole, (self.world(-0.004, 0.075, 0.0) + UP * (0.0145 + self.spring(np.array([f]))[0]))[None], d[None], r0=0.25)[0]
            pts.append(p)
        S.add(np.array(pts), o1=1e-4, closed=True)
        return S.attrs(V, N, "seam", maxd=0.02)

    def _add(self, S, pts, toward, o1, o2):
        if len(pts) < 3:
            return
        g = numgrad(self.solid, pts.astype(F)).astype(float)
        n = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        m = len(pts) // 2
        t = pts[min(m + 1, len(pts) - 1)] - pts[max(m - 1, 0)]
        sgn = np.sign(np.dot(np.cross(t, n[m]), np.asarray(toward) - pts[m])) or 1.0
        S.add(pts, o1=sgn * o1, o2=sgn * o2)

    # mesh ------------------------------------------------------------------------
    def build(self, h):
        t0 = time.time()
        lo, hi = self.lo, self.hi
        V, Q = surface_nets(lambda P, band: self.solid(P), lo, hi, h, levels=3, act=1.8, ev=3.0, project=1,
                            log=lambda *a: None)
        V, Q = keep_largest(V.astype(np.float64), Q, 0.05)
        N = vertex_normals(V, Q)
        P = V.astype(F)
        lat, fwd, z = self.local(P)
        fs, fu = self.f_sole(P), self.f_upper(P)
        sole = smoothstep(0.0004, -0.0004, fs - fu)
        cav = self.cavity(P)
        lining = smoothstep(0.0016, 0.0006, cav) * (1 - sole)
        tong = self.f_tongue(P)
        dmin = np.full(len(P), 1.0)
        for e in self.eyelets:
            dmin = np.minimum(dmin, np.linalg.norm(V[:, :2] - e[:2], axis=1) + 2.0 * np.abs(V[:, 2] - e[2]))
        eyelet = smoothstep(0.0046, 0.0038, dmin) * smoothstep(0.0019, 0.0025, dmin) * (1 - lining)
        tab = (np.abs(lat + 0.002) < 0.0125) & (fwd < self.F0 + 0.03) & (z > self.collar_z(fwd) - 0.028) & (cav > 0.0012)
        crease = smoothstep(0.095, 0.115, fwd) * smoothstep(0.172, 0.158, fwd) * smoothstep(0.035, 0.045, z) * (1 - sole)
        zs = z - self.spring(fwd)
        tread = smoothstep(-SOLE_LIFT + 0.0022, -SOLE_LIFT + 0.0008, zs) * sole
        A = dict(sole=sole, lining=lining, eyelet=eyelet, accent=tab.astype(float) * (1 - lining), crease=crease,
                 tread=tread, cu=lat, cv=fwd, tongue=smoothstep(0.0008, -0.0008, tong - np.minimum(fu, fs)))
        A.update(self.seams(V, N))
        upper = Part(V, Q, {**A, "lace": 0.0})
        self.log(f"  shoe body: {len(V)} verts ({time.time() - t0:.0f}s)")
        # laces
        from .sdf import Union
        lace = Union(self.lace_tubes())
        lv, lq = surface_nets(lambda P, band: lace._eval(np.asarray(P, F), band) if True else None,
                              lace.lo - 0.003, lace.hi + 0.003, h * 0.7, levels=2, act=1.8, ev=3.0, project=1,
                              log=lambda *a: None)
        laces = Part(lv, lq, {"lace": 1.0, "cu": self.local(lv)[0], "cv": self.local(lv)[1]})
        self.log(f"  laces: {len(lv)} verts ({time.time() - t0:.0f}s)")
        return [upper, laces]


def mirror_part(p):
    V = p.V * np.array([-1, 1, 1], np.float32)
    return Part(V, p.faces[:, ::-1], dict(p.attrs))


def build_shoes(ctx, h, log=print):
    shoe = Shoe(ctx, log=log)
    left = shoe.build(h)
    return shoe, left + [mirror_part(p) for p in left]


# ============================================================================= JEANS
JEANS = dict(
    t=0.0012, t_wb=0.0026, t_hem=0.0030, clear=0.0015,
    wb_h=0.040,                     # waistband height
    scoop_a=0.088, scoop_v=0.100,   # front pocket mouth: quarter ellipse (arc from side seam, depth below band)
    facing_r=0.011,                 # how far the facing sits below the panel at the mouth
    bag_depth=0.034,                # pocket bag interior target (clamped by the skin)
    bulge=0.019,                    # extra room over the front pockets (bag + pistol)
)
R_REF = 0.165                       # metres per radian around the pelvis (arc-length proxy)
PELVIS_O = np.array([0.0, 0.012, 0.0])


def jeans_top_z(th):
    """Waistband top: mid-rise, slightly higher at the back (th = 0 at centre back)."""
    return 1.0155 + 0.0085 * np.cos(th)


def _rbox2(a, v, ca, cv, ha, hv, r):
    qa = np.abs(a - ca) - (ha - r); qv = np.abs(v - cv) - (hv - r)
    return np.hypot(np.maximum(qa, 0), np.maximum(qv, 0)) + np.minimum(np.maximum(qa, qv), 0) - r


class Jeans:
    def __init__(self, ctx, shoe=None, log=print):
        from .skeleton import J
        self.ctx, self.shoe, self.log = ctx, shoe, log
        self.lo = np.array([-0.26, -0.17, -0.005]); self.hi = np.array([0.26, 0.20, 1.05])
        self.G = ctx.grid("jeans", [-0.30, -0.22, -0.02], [0.30, 0.24, 1.10])
        self.Gs = self.G.blurred(0.004)
        t0 = time.time()
        # ---------------- pelvis / seat: hull of the lower body, draped off the seat
        pc = Polar(PELVIS_O, UP, [0, 1, 0], 0.76, 1.08, 0.004, 360)
        rb = measure(ctx.lower, pc, 0.40, [-0.32, -0.25, 0.74], [0.32, 0.26, 1.10])
        rh = np.stack([hull_radius(rb[i], pc.th) for i in range(len(pc.s))])
        z = pc.s[:, None]
        e = np.interp(z, [0.76, 0.86, 0.95, 1.08], [0.009, 0.007, 0.0045, 0.0035]) + 0 * pc.th[None, :]
        Rp = drape_down((rh + e)[::-1], 0.30, pc.ds)[::-1]
        self.pc, self.p_R = pc, periodic_smooth(Rp, 1.0, 1.5)
        self.f_pelvis_raw = pc.field(self.p_R)
        # ---------------- legs: hull of the leg + ease, straight below the knee, stacked at the ankle
        lc = Polar(J["thigh_l"], J["foot_l"] - J["thigh_l"], [1, 0, 0], -0.06, 0.90, 0.004, 240)
        rl = measure(ctx.legs, lc, 0.20, J["foot_l"] * [1, 1, 0] - [0.15, 0.15, 0.0], J["thigh_l"] + [0.15, 0.15, 0.10])
        rlh = np.stack([hull_radius(rl[i], lc.th) for i in range(len(lc.s))])
        s = lc.s[:, None]
        self.s_knee = 0.43
        e = np.interp(s, [-0.06, 0.05, 0.18, 0.32, 0.43, 0.9], [0.008, 0.011, 0.016, 0.017, 0.017, 0.017])
        base = rlh + e
        ik = np.searchsorted(lc.s, self.s_knee)
        straight = base[ik][None, :] + 0.001 - 0.008 * np.clip((s - self.s_knee) / 0.45, 0, 1)
        base = np.where(s > self.s_knee, np.maximum(base, straight), base)
        Rl = drape_down(base, 0.20, lc.ds)
        Rl = periodic_smooth(Rl, 1.5, 1.5)
        slack = np.maximum(Rl - (rl + 0.004), 0)
        th = lc.th[None, :]
        front = np.cos(angdiff(th, np.pi / 2))          # +1 at the front of the leg
        rng = np.random.default_rng(7)
        ph = rng.uniform(0, 2 * np.pi, 4)
        # stacked folds above the hem: a few warped, broken rings, bulkier towards the shoe
        stack = np.zeros_like(Rl)
        for k, (sc_, lam, amp) in enumerate(((0.835, 0.030, 0.0042), (0.800, 0.034, 0.0036), (0.765, 0.036, 0.0030), (0.728, 0.04, 0.0020))):
            p1, p2, p3 = rng.uniform(0, 2 * np.pi, 3)
            centre = sc_ + 0.010 * np.sin(th + p1) + 0.004 * np.sin(2 * th + p2)
            prof = np.exp(-((s - centre) / (0.30 * lam)) ** 2) - 0.35 * np.exp(-((s - centre - 0.5 * lam) / (0.25 * lam)) ** 2)
            stack = stack + amp * prof * (0.55 + 0.45 * np.sin(th + p3))
        stack = stack + 0.0015 * smoothstep(0.70, 0.86, s)
        # whiskers at the top front, honeycomb behind the knee
        u = th * 0.075
        wh = np.sin(2 * np.pi * (u * 0.8 + s * 0.6) / 0.024 + ph[2])
        whisk = 0.0016 * wh * smoothstep(0.3, 0.8, front) * smoothstep(0.0, 0.04, s) * smoothstep(0.17, 0.08, s)
        hc = np.sin(2 * np.pi * (s / 0.019 + 0.05 * np.sin(3 * th + ph[3])))
        honey = 0.0022 * np.maximum(hc, -0.4) * smoothstep(0.2, 0.75, -front) * np.exp(-((s - 0.455) / 0.035) ** 2)
        folds = stack + np.minimum(whisk + honey, 0.6 * slack + 0.0005)
        self.l_fold = folds
        self.lc, self.l_R = lc, Rl + folds
        self.f_leg_raw = lc.field(self.l_R)
        self.log(f"  jeans charts in {time.time() - t0:.0f}s")

    # chart coordinates --------------------------------------------------------
    def pel(self, P):
        """(theta from centre back towards his right, arc from CB, v below the waistband seam)."""
        s, th, rho = self.pc.coords(P)
        v = (jeans_top_z(th) - JEANS["wb_h"]) - np.asarray(P)[:, 2]
        return th, np.abs(th) * R_REF, v

    def pocket_av(self, P):
        """Front pocket coordinates: a = arc from the side seam towards the fly, v below the band."""
        th, ab, v = self.pel(P)
        return (np.abs(th) - np.pi / 2) * R_REF, v, np.sign(th)

    def scoop(self, a, v):
        A, B = JEANS["scoop_a"], JEANS["scoop_v"]
        k = np.sqrt((a / A) ** 2 + (v / B) ** 2)
        g = np.sqrt((a / A ** 2) ** 2 + (v / B ** 2) ** 2) / np.maximum(k, 1e-6)
        return (k - 1) / np.maximum(g, 1e-6)          # < 0 inside the facing corner

    def facing_r(self, a, v):
        taper = smoothstep(0.0, 0.03, v) * smoothstep(0.0, 0.025, a)
        cd = self.scoop(a, v)
        return JEANS["facing_r"] * taper * smoothstep(-0.07, -0.01, cd) * smoothstep(0.0003, -0.0009, cd) * (a > -0.01) * (v > -0.01)

    def bag_outline(self, a, v):
        return _rbox2(a, v, 0.066, 0.108, 0.074, 0.120, 0.04)

    # fields ---------------------------------------------------------------------
    def f_leg(self, P):
        s, th, rho = self.lc.coords(P)
        f = self.f_leg_raw(P)
        f = smax(f, (-0.05 - s).astype(F), F(0.03))
        return smax(f, (s - 0.89).astype(F), F(0.005))

    def f_pelvis(self, P):
        z = P[:, 2]
        f = self.f_pelvis_raw(P)
        f = smax(f, (0.84 - z).astype(F), F(0.03))
        return smax(f, (z - 1.07).astype(F), F(0.005))

    def bulge(self, P):
        a, v, sg = self.pocket_av(P)
        w = np.exp(-((a - 0.055) / 0.055) ** 2 - ((v - 0.10) / 0.085) ** 2)
        return JEANS["bulge"] * w

    def nominal(self, P):
        """Outer surface without pocket mouths / overlays."""
        P = np.asarray(P, F)
        Pm = P * np.array([-1, 1, 1], F)
        e = smin(self.Gs(P) - F(0.0035), self.f_pelvis(P), F(0.012))
        e = smin(e, np.minimum(self.f_leg(P), self.f_leg(Pm)), F(0.014))
        e = e - self.bulge(P).astype(F)
        return np.minimum(e, self.G(P) - F(JEANS["t_wb"] + JEANS["clear"]))

    def overlays(self, P):
        """Outward offsets: waistband, fly overlap, panel edge gape; inward: facing."""
        th, ab, v = self.pel(P)
        z = P[:, 2]
        up = z > 0.70
        wb = smoothstep(0.0005, -0.0005, v) * up
        cf = (np.pi - np.abs(th)) * R_REF * -np.sign(th)       # + towards his left from the centre front
        fly = smoothstep(-0.0006, 0.0004, cf) * smoothstep(0.042, 0.039, cf) * smoothstep(0.150, 0.140, v) * (v > -0.001)
        a, vv, sg = self.pocket_av(P)
        cd = self.scoop(a, vv)
        r = self.facing_r(a, vv)
        taper = smoothstep(0.0, 0.03, vv) * smoothstep(0.0, 0.025, a)
        gape = 0.0016 * taper * smoothstep(0.03, 0.0, cd) * (cd > -0.001)
        return 0.0008 * wb + 0.0010 * fly + gape - r

    def outer(self, P):
        P = np.asarray(P, F)
        e = self.nominal(P) - self.overlays(P).astype(F)
        e = np.minimum(e, self.G(P) - F(JEANS["t"] + JEANS["clear"]))
        # the hem rests on the shoe: near the ankle the fabric wraps over the shoe surface
        low = P[:, 2] < 0.16
        if low.any() and self.shoe is not None:
            Q = P[low]
            rho = np.minimum(self.lc.coords(Q)[2], self.lc.coords(Q * np.array([-1, 1, 1], F))[2])
            w = np.where(rho < 0.085, self.shoe_d(Q) - F(JEANS["t_hem"] + 0.0006), 1.0).astype(F)
            e[low] = smin(e[low], w, F(0.004))
        return e.astype(F)

    def hem_z(self, P):
        """Leg hem height: resting on the laces at the front, on the heel counter at the back."""
        sg = np.where(P[:, 0] >= 0, 1.0, -1.0).astype(F)
        s, th, rho = self.lc.coords(P * np.stack([sg, np.ones_like(sg), np.ones_like(sg)], 1))
        front = np.maximum(np.cos(th - np.pi / 2), 0) ** 1.5
        back = np.maximum(np.cos(th + np.pi / 2), 0) ** 2
        return 0.050 + 0.046 * front + 0.040 * back + 0.003 * np.sin(3 * th + 1.0)

    def shoe_d(self, P):
        out = np.full(len(P), 1.0, F)
        if self.shoe is None:
            return out
        low = P[:, 2] < 0.16
        if low.any():
            Q = P[low]
            out[low] = np.minimum(self.shoe.solid(Q), self.shoe.solid(Q * np.array([-1, 1, 1], F)))
        return out

    def slit(self, P):
        a, v, sg = self.pocket_av(P)
        cd = self.scoop(a, v)
        strip = 0.0011 - np.abs(cd + 0.0005)
        return np.where((a > 0.003) & (v > 0.003) & (P[:, 2] > 0.8), strip, -1.0)

    def cut(self, P):
        P = np.asarray(P, F)
        th, ab, v = self.pel(P)
        c = P[:, 2] - jeans_top_z(th)
        low = P[:, 2] < 0.3
        c = np.maximum(c, np.where(low, self.hem_z(P) - P[:, 2], -1.0))
        c = np.maximum(c, 0.0004 - self.shoe_d(P))
        c = np.maximum(c, self.slit(P))
        return c.astype(F)

    # surface helpers ----------------------------------------------------------------
    def surf_point(self, th, z, f=None):
        """Point on a surface at pelvis-chart angle th and height z (marching in from outside)."""
        f = f or self.outer
        th = np.atleast_1d(np.asarray(th, float)); z = np.broadcast_to(np.atleast_1d(z), th.shape).astype(float)
        C = self.pc.world(z, th, 0.0)
        D = self.pc.dirs(th)
        out = np.empty_like(C)
        for i in range(len(C)):
            r = 0.30
            for _ in range(200):
                d = float(f((C[i] + r * D[i])[None].astype(F))[0])
                if d < 1e-4:
                    break
                r -= max(0.8 * min(d, 0.01), 1e-4)
            out[i] = C[i] + r * D[i]
        return out

    def normal(self, P, f=None):
        g = numgrad(f or self.outer, np.asarray(P, F)).astype(float)
        return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)

    def av_point(self, a, v, side, f=None):
        """Front pocket chart (a, v) on side (+1 his right, -1 his left) -> surface point."""
        th = side * (np.pi / 2 + np.asarray(a, float) / R_REF)
        z = jeans_top_z(th) - JEANS["wb_h"] - np.asarray(v, float)
        return self.surf_point(th, z, f)

    def back_av_point(self, ab, v, side, f=None):
        th = side * np.asarray(ab, float) / R_REF
        z = jeans_top_z(th) - JEANS["wb_h"] - np.asarray(v, float)
        return self.surf_point(th, z, f)

    # parts ---------------------------------------------------------------------------
    def yoke_v(self, ab):
        return 0.072 - 0.040 * np.clip(ab / (np.pi / 2 * R_REF), 0, 1)

    def back_pocket(self, ab, v):
        """2-D pentagon SDF of a back patch pocket (ab from CB, v below the band)."""
        top = v - (0.088 - 0.010 * (ab - 0.025) / 0.14)
        d = -top
        d = np.maximum(d, 0.024 - ab)
        d = np.maximum(d, ab - 0.164)
        # pointed bottom: two slanted edges meeting at (0.094, 0.262)
        for sgn in (1, -1):
            n = np.array([0.06 * sgn, 1.0]); n /= np.linalg.norm(n)
            p0 = np.array([0.094, 0.262])
            d = np.maximum(d, (ab - p0[0]) * n[0] * 0 + ((ab - p0[0]) * (0.36 * sgn) + (v - p0[1])) / np.hypot(0.36, 1))
        return d

    def seams_main(self, V, N):
        from .skeleton import J
        S = Seams(); K = Seams()
        # out- and inseams of both legs
        for sg in (1, -1):
            zz = np.linspace(jeans_top_z(np.pi / 2) - JEANS["wb_h"], 0.03, 160)
            ax = np.array([np.interp(z, [J["foot_l"][2], J["thigh_l"][2], 1.1], [J["foot_l"][0], J["thigh_l"][0], J["thigh_l"][0]]) for z in zz])
            ay = np.array([np.interp(z, [J["foot_l"][2], J["thigh_l"][2]], [J["foot_l"][1], J["thigh_l"][1]]) for z in zz])
            ax = np.where(zz > 0.88, 0.0, ax)
            C = np.stack([ax * sg, ay, zz], 1)
            pts = np.array([self._march(C[i], np.array([sg, 0, 0])) for i in range(len(C))])
            pts = pts[self.shoe_d(pts.astype(F)) > 0.004]
            self._addrows(S, pts, pts.mean(0) + [0, 0.1, 0], 0.0015, 0.0075)
            zi = np.linspace(0.78, 0.03, 120)
            C = np.stack([np.interp(zi, [J["foot_l"][2], J["thigh_l"][2]], [J["foot_l"][0], J["thigh_l"][0]]) * sg,
                          np.interp(zi, [J["foot_l"][2], J["thigh_l"][2]], [J["foot_l"][1], J["thigh_l"][1]]), zi], 1)
            pts = np.array([self._march(C[i], np.array([-sg, 0, 0]), r0=abs(C[i, 0]) - 0.001) for i in range(len(C))])
            pts = pts[self.shoe_d(pts.astype(F)) > 0.004]
            self._addrows(S, pts, pts.mean(0) + [0, -0.1, 0], 0.0015, 0.0075)
        # centre back seam (yoke to crotch), crotch seam below the fly, yoke
        zz = np.linspace(jeans_top_z(0) - JEANS["wb_h"] - self.yoke_v(0), 0.80, 60)
        pts = np.array([self._march(np.array([0, 0.012, z]), np.array([0, 1.0, 0])) for z in zz])
        self._addrows(S, pts, pts.mean(0) + [-0.1, 0, 0], 0.0015, 0.0075)
        zz = np.linspace(jeans_top_z(np.pi) - JEANS["wb_h"] - 0.150, 0.80, 30)
        pts = np.array([self._march(np.array([0, 0.012, z]), np.array([0, -1.0, 0])) for z in zz])
        self._addrows(S, pts, pts.mean(0) + [-0.1, 0, 0], 0.0015, 0.0075)
        ab = np.linspace(-np.pi / 2 * R_REF, np.pi / 2 * R_REF, 160)
        th = ab / R_REF
        pts = self.surf_point(th, jeans_top_z(th) - JEANS["wb_h"] - self.yoke_v(np.abs(ab)))
        self._addrows(S, pts, pts.mean(0) - [0, 0, 0.1], 0.0015, 0.0075)
        # pocket mouths (rows on the panel side) and the J-stitch fly
        for side in (1, -1):
            t = np.linspace(0, np.pi / 2, 90)
            a = JEANS["scoop_a"] * np.cos(t); v = JEANS["scoop_v"] * np.sin(t)
            pts = self.av_point(a + 0.0006, v + 0.0006, side)
            self._addrows(K, pts, self.av_point(0.09, 0.1, side)[0], 0.0022, 0.0080)
        cf_th = -np.pi + 0.036 / R_REF
        vj = np.linspace(0.0, 0.112, 40)
        ang = np.linspace(0, np.pi / 2, 24)
        dj = np.r_[np.full(len(vj), 0.036), 0.036 * np.cos(ang)]
        vv = np.r_[vj, 0.112 + 0.036 * np.sin(ang)]
        th = -np.pi + dj / R_REF
        pts = self.surf_point(th, jeans_top_z(th) - JEANS["wb_h"] - vv)
        self._addrows(K, pts, self.surf_point(np.array([-np.pi + 0.01]), 0.92)[0], 1e-4, 0.006)
        return {**S.attrs(V, N, "seam"), **K.attrs(V, N, "pk")}

    def _march(self, C, D, r0=0.30):
        r = r0
        for _ in range(200):
            d = float(self.outer((C + r * D)[None].astype(F))[0])
            if d < 1e-4:
                break
            r -= max(0.8 * min(d, 0.01), 1e-4)
        return C + r * D

    def _addrows(self, S, pts, toward, o1, o2):
        pts = np.asarray(pts, float)
        if len(pts) < 3:
            return
        n = self.normal(pts)
        m = len(pts) // 2
        t = pts[min(m + 1, len(pts) - 1)] - pts[max(m - 1, 0)]
        sgn = np.sign(np.dot(np.cross(t, n[m]), np.asarray(toward) - pts[m])) or 1.0
        S.add(pts, o1=sgn * o1, o2=sgn * o2)

    def wear(self, P):
        P = np.asarray(P, F)
        th, ab, v = self.pel(P)
        out = np.zeros(len(P))
        # whiskers: rays fanning out from the crotch over the top of the thighs (front)
        cfa = (np.pi - np.abs(th)) * R_REF                       # arc from the centre front
        rng = np.random.default_rng(11)
        o = np.array([0.025, 0.205])
        for ang in np.radians([8, 16, 25, 33, 42]) + rng.uniform(-0.04, 0.04, 5):
            d = np.array([np.cos(ang), -np.sin(ang)])
            rel = np.stack([cfa - o[0], v - o[1]], 1)
            along = rel @ d
            perp = np.abs(rel @ np.array([-d[1], d[0]]))
            L = rng.uniform(0.08, 0.12)
            out = np.maximum(out, np.exp(-(perp / 0.0035) ** 2) * smoothstep(0.0, 0.03, along) * smoothstep(L, L * 0.5, along) * 0.9)
        out *= (P[:, 2] > 0.70) & (P[:, 2] < 0.98)
        # broad fading: thigh fronts, knees, seat
        for sg in (1, -1):
            s, lth, rho = self.lc.coords(P * np.array([sg, 1, 1], F))
            front = np.cos(angdiff(lth, np.pi / 2))
            out = np.maximum(out, 0.45 * smoothstep(0.2, 0.9, front) * np.exp(-((s - 0.18) / 0.12) ** 2) * (P[:, 0] * sg > 0))
            out = np.maximum(out, 0.40 * smoothstep(0.3, 0.9, front) * np.exp(-((s - 0.44) / 0.05) ** 2) * (P[:, 0] * sg > 0))
            hc = np.abs(np.sin(2 * np.pi * (s / 0.019 + 0.25 * np.sin(2.0 * lth + 0.7 * sg))))
            out = np.maximum(out, 0.32 * smoothstep(0.8, 0.97, hc) * smoothstep(0.2, 0.75, -front) * np.exp(-((s - 0.455) / 0.03) ** 2) * (P[:, 0] * sg > 0))
        seat = np.exp(-((v - 0.16) / 0.06) ** 2) * smoothstep(0.5, 0.95, np.cos(th)) * np.exp(-((np.abs(ab) - 0.08) / 0.07) ** 2)
        out = np.maximum(out, 0.40 * seat)
        return np.clip(out, 0, 1)

    def build(self, h):
        t0 = time.time()
        parts = []
        V, Q, bmask = sheet(self.outer, self.cut, self.lo, self.hi, h, log=self.log)
        N = vertex_normals(V, Q)
        P = V.astype(F)
        th, ab, v = self.pel(P)
        wb = jeans_top_z(th) - V[:, 2]
        hem_b = np.nonzero(bmask & (V[:, 2] < 0.2))[0]
        hem = cKDTree(V[hem_b]).query(V)[0] if len(hem_b) else np.full(len(V), 1.0)
        t = np.full(len(V), JEANS["t"])
        t = np.where(wb < JEANS["wb_h"] + 0.001, JEANS["t_wb"], t)
        t = np.where(hem < 0.016, JEANS["t_hem"], t)
        A = self.surface_attrs(V, N, P)
        A.update(self.seams_main(V, N))
        A.update(wb=np.where(V[:, 2] > 0.85, wb, 1.0), hem=hem)
        VV, faces, side = thicken(V, Q, bmask, t)
        parts.append(Part(VV, faces, {**{k: np.concatenate([x, x]) for k, x in A.items()}, "side": side}))
        self.log(f"  jeans shell: {len(V)} verts ({time.time() - t0:.0f}s)")
        parts += self.pocket_bags(h)
        parts += self.back_pockets(h)
        parts += self.hardware(h)
        self.log(f"  jeans built in {time.time() - t0:.0f}s")
        return parts

    def surface_attrs(self, V, N, P):
        th, ab, v = self.pel(P)
        u = th * R_REF
        vv = V[:, 2].astype(float)
        legs = V[:, 2] < 0.84
        sg = np.where(V[:, 0] >= 0, 1.0, -1.0)
        s, lth, rho = self.lc.coords(P * np.stack([sg, np.ones(len(P)), np.ones(len(P))], 1).astype(F))
        u = np.where(legs, lth * 0.075 * sg + 2.0, u)
        vv = np.where(legs, -s, vv)
        lf = self.lc.lookup(self.l_fold, s, lth)
        fold = np.where(legs, lf, 0.0)
        return dict(u=u, v=vv, fold=fold * 100, wear=self.wear(P))

    def pocket_bags(self, h):
        parts = []
        for side in (1, -1):
            def bag_field(P, side=side):
                P = np.asarray(P, F)
                a, v, sg = self.pocket_av(P)
                r = self.facing_r(a, v)
                prof = smoothstep(0.0, 0.035, -self.bag_outline(a, v))
                D = JEANS["t"] + 0.0012 + np.maximum(JEANS["bag_depth"] * prof, r + JEANS["t"] + 0.0025)
                return self.nominal(P) + D.astype(F)

            def bag_cut(P, side=side):
                P = np.asarray(P, F)
                a, v, sg = self.pocket_av(P)
                return np.where(sg == side, self.bag_outline(a, v), 1.0).astype(F)
            c = self.av_point(0.066, 0.108, side)[0]
            lo = c - 0.16; hi = c + 0.16
            V, Q, bmask = sheet(bag_field, bag_cut, lo, hi, h, log=self.log)
            # where the thigh leaves no room the bag lies flat against it: drop those faces
            ok = self.G(V[Q].mean(axis=1).astype(F)) > 0.0042
            V, Q, used = compact(V, Q, ok)
            ea, eb = boundary_edges(Q)
            bmask = np.zeros(len(V), bool); bmask[ea] = True; bmask[eb] = True
            VV, faces, sd = thicken(V, Q, bmask, 0.0010)
            parts.append(Part(VV, faces, {"bag": 1.0, "side": sd, "u": 0.0, "v": 0.0}))
            if side == 1:
                self._bag_field = bag_field
        return parts

    def back_pockets(self, h):
        parts = []
        for side in (1, -1):
            def pf(P):
                return (self.outer(P) - F(0.0013)).astype(F)

            def pcut(P, side=side):
                P = np.asarray(P, F)
                th, ab, v = self.pel(P)
                return np.where((np.sign(th) == side) & (np.cos(th) > 0), self.back_pocket(ab, v), 1.0).astype(F)
            c = self.back_av_point(0.094, 0.17, side)[0]
            V, Q, bmask = sheet(pf, pcut, c - 0.13, c + 0.13, h * 0.8, log=self.log)
            N = vertex_normals(V, Q)
            th, ab, v = self.pel(V.astype(F))
            K = Seams()
            # outline: double rows inside the edge; top hem rows; arcuate double arc
            outline = self._pocket_outline(side)
            self._addrows(K, outline, c, 0.0018, 0.0068)
            arc_a = np.linspace(0.034, 0.154, 60)
            for k, dv in enumerate((0.0, 0.007)):
                av_ = 0.20 - 0.045 * np.sin(np.pi * (arc_a - 0.034) / 0.12) ** 0.8 + dv
                pts = self.back_av_point(arc_a, av_, side, pf)
                K.add(pts, o1=1e-4)
            A = K.attrs(V, N, "pk")
            top = v - (0.088 - 0.010 * (ab - 0.025) / 0.14)
            A["wb"] = np.full(len(V), 1.0)
            A["hem"] = np.full(len(V), 1.0)
            A["ptop"] = top
            A["u"] = th * R_REF; A["v"] = V[:, 2]
            A["wear"] = 0.25 + 0.3 * smoothstep(0.004, 0.0, np.abs(self.back_pocket(ab, v)))
            VV, faces, sd = thicken(V, Q, bmask, 0.0011)
            parts.append(Part(VV, faces, {**{k: np.concatenate([x, x]) for k, x in A.items()}, "side": sd}))
        return parts

    def _pocket_outline(self, side):
        ts = np.linspace(0, 1, 200)
        corners = np.array([[0.025, 0.088], [0.164, 0.078], [0.164, 0.235], [0.094, 0.262], [0.025, 0.235], [0.025, 0.088]])
        seg = np.linalg.norm(np.diff(corners, axis=0), axis=1)
        L = np.r_[0, np.cumsum(seg)]
        u = ts * L[-1]
        a = np.interp(u, L, corners[:, 0]); v = np.interp(u, L, corners[:, 1])
        return self.back_av_point(a, v, side, lambda P: (self.outer(P) - F(0.0013)).astype(F))

    def hw_nodes(self):
        """Belt loops (3 boxes each) and metal (button, rivets) as SDF nodes; cached."""
        if hasattr(self, "_hw"):
            return self._hw
        from .sdf import Capsule, Ellipsoid, RoundBox, Union, rot_from_axes
        loops = []
        for th in np.radians([-152, 152, -98, 98, -32, 32, 0.0]):
            zc = jeans_top_z(th) - JEANS["wb_h"] * 0.5
            p0 = self.surf_point(np.array([th]), zc)[0]
            n = self.normal(p0[None])[0]
            tv = np.cross(UP, n); tv /= np.linalg.norm(tv)
            up = np.cross(n, tv)
            R = rot_from_axes(x=tv, y=up)
            hl = JEANS["wb_h"] * 0.5 + 0.004
            boxes = [RoundBox(p0 + n * 0.0030, (0.0052, hl, 0.0009), R, rr=0.0007)]
            for e in (1, -1):
                boxes.append(RoundBox(p0 + up * e * (hl - 0.0016) + n * 0.0016, (0.0052, 0.0019, 0.0017), R, rr=0.0007))
            loops.append(Union(boxes))
        metal = []
        p0 = self.surf_point(np.array([np.pi]), jeans_top_z(np.pi) - 0.0195)[0]
        n = self.normal(p0[None])[0]
        R = rot_from_axes(x=np.cross(UP, n), y=UP)
        metal.append(Union([Capsule(p0 + n * 0.0012, p0 + n * 0.0030, 0.0072),
                            Ellipsoid(p0 + n * 0.0034, (0.0068, 0.0068, 0.0022), R)]))
        rivets = []
        for side in (1, -1):
            for a, v in ((JEANS["scoop_a"] + 0.002, 0.004), (0.004, JEANS["scoop_v"] + 0.004)):
                p = self.av_point(a, v, side)[0]
                nn = self.normal(p[None])[0]
                Rr = rot_from_axes(x=np.cross(UP, nn), y=UP)
                metal.append(Ellipsoid(p + nn * 0.0004, (0.0036, 0.0036, 0.0020), Rr))
                rivets.append(p)
        self._hw = (loops, metal, rivets)
        return self._hw

    def clad(self, P):
        """Outer envelope including loops and hardware (what the shirt must clear)."""
        P = np.asarray(P, F)
        e = self.outer(P)
        loops, metal, _ = self.hw_nodes()
        from .sdf import evaluate
        for n in loops + metal:
            e = np.minimum(e, evaluate(n, P, 0.005))
        return e

    def hardware(self, h):
        parts = []
        loops, metal, self.rivets = self.hw_nodes()
        for u in loops:
            def f(P, u=u):
                return u._eval(np.asarray(P, F), 0.01)
            prt = solid_part(f, u.lo - 0.003, u.hi + 0.003, h * 0.45)
            parts.append(Part(prt.V, prt.faces, {"side": 1.0, "u": prt.V[:, 0] * 1.0, "v": prt.V[:, 2] * 1.0,
                                                  "wear": 0.35}))
        for m in metal:
            def f(P, m=m):
                return m._eval(np.asarray(P, F), 0.01)
            prt = solid_part(f, m.lo - 0.002, m.hi + 0.002, h * 0.3)
            parts.append(Part(prt.V, prt.faces, {"metal": 1.0, "side": 1.0}))
        # leather label on the back waistband (his right)
        def lf(P):
            return (self.outer(P) - F(0.0012)).astype(F)

        def lcut(P):
            th, ab, v = self.pel(np.asarray(P, F))
            return np.where(np.sign(th) > 0, _rbox2(ab, v, 0.085, -0.020, 0.035, 0.0135, 0.002), 1.0).astype(F)
        c = self.back_av_point(0.085, -0.02, 1)[0]
        V, Q, bm = sheet(lf, lcut, c - 0.06, c + 0.06, h * 0.5, log=self.log)
        VV, faces, sd = thicken(V, Q, bm, 0.0012)
        parts.append(Part(VV, faces, {"label": 1.0, "side": 1.0}))
        return parts

    def pocket_info(self, lift=0.0):
        """Right front pocket: opening centre / direction, depth direction, bag interior centre."""
        side = 1
        t = np.linspace(0, np.pi / 2, 41)
        a = JEANS["scoop_a"] * np.cos(t); v = JEANS["scoop_v"] * np.sin(t)
        edge = self.av_point(a + 0.0006, v + 0.0006, side)
        mid = edge[20]
        tan = edge[22] - edge[18]; tan /= np.linalg.norm(tan)
        n_out = self.normal(mid[None])[0]
        ac, vc = 0.066, 0.115
        sp = self.av_point(ac, vc, side, self.nominal)[0]
        nn = self.normal(sp[None], self.nominal)[0]
        # bag back wall depth along -n
        r = 0.0
        while float(self._bag_field((sp - nn * r)[None].astype(F))[0]) > 0 and r < 0.08:
            r += 0.0002
        inner = JEANS["t"]
        centre = sp - nn * (inner + 0.5 * (r - inner))
        depth_dir = centre - mid
        depth_dir -= n_out * np.dot(depth_dir, n_out) * 0.0
        depth_dir /= np.linalg.norm(depth_dir)
        gap_pt = self.av_point(JEANS["scoop_a"] * 0.6, JEANS["scoop_v"] * 0.8, side, self.nominal)[0]
        return dict(
            side="right (-X)",
            opening_centre=mid.tolist(), opening_dir=tan.tolist(), outward_normal=n_out.tolist(),
            depth_dir=depth_dir.tolist(), bag_centre=centre.tolist(),
            opening_length=float(np.linalg.norm(np.diff(edge, axis=0), axis=1).sum()),
            mouth_gap=float(JEANS["facing_r"] - JEANS["t"]),
            bag_interior_thickness_at_centre=float(r - inner),
            bag_size_wh=[0.148, 0.240],
            rivets=[p.tolist() for p in self.hw_nodes()[2][:2]],
            note="body-frame metres (+Z up, faces -Y); add sole lift %.3f to z in the final scene" % SOLE_LIFT,
        )
