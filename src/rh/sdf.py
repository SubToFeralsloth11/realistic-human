"""Signed distance fields: primitives and smooth CSG nodes evaluated on (n, 3) float32 point arrays.

Units are metres. Blender axes: +Z up, the character faces -Y, its left side is +X.
Every node knows its axis-aligned bounds, so evaluation skips points that are too far away to
change the result (see `prepare` and `evaluate`). Leaves can carry a `bone` (used for skin
weights) and a `tag` (used for material masks).
"""
import numpy as np

BIG = np.float32(1e3)
F = np.float32


def v3(x):
    return np.asarray(x, dtype=np.float64).reshape(3)


def norm(v):
    v = v3(v)
    return v / np.linalg.norm(v)


def rot_from_axes(x=None, y=None, z=None):
    """3x3 matrix whose columns are the local x, y, z axes in world space.

    Give two axes; the third is completed (right-handed) and the second re-orthogonalised.
    """
    if x is not None and y is not None:
        x = norm(x); z = norm(np.cross(x, y)); y = np.cross(z, x)
    elif y is not None and z is not None:
        y = norm(y); x = norm(np.cross(y, z)); z = np.cross(x, y)
    elif x is not None and z is not None:
        x = norm(x); y = norm(np.cross(z, x)); z = np.cross(x, y)
    else:
        raise ValueError("need two axes")
    return np.stack([x, y, z], axis=1)


def rot_euler(rx=0.0, ry=0.0, rz=0.0):
    """XYZ Euler angles in degrees -> 3x3 rotation (applied X, then Y, then Z)."""
    a, b, c = np.radians([rx, ry, rz])
    X = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Y = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Z = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Z @ Y @ X


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * (k * 0.25)


def smax(a, b, k):
    if k <= 0:
        return np.maximum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + h * h * (k * 0.25)


def _rowlen(v):
    return np.sqrt(np.einsum("ij,ij->i", v, v))


class Node:
    bone = None
    tag = None
    margin = 0.0

    def leaves(self):
        yield self

    def mirrored(self):
        """Copy reflected through the YZ plane (left <-> right)."""
        raise NotImplementedError

    def _eval(self, P, band):
        raise NotImplementedError


def _mirror_rot(R):
    S = np.diag([-1.0, 1.0, 1.0])
    return S @ R @ S


def _mirror_name(s):
    if s is None:
        return None
    for a, b in (("_l", "_r"), ("_r", "_l")):
        if s.endswith(a):
            return s[: -len(a)] + b
    return s


class Leaf(Node):
    def __init__(self, bone=None, tag=None):
        self.bone = bone
        self.tag = tag

    def _meta(self, other):
        other.bone = _mirror_name(self.bone)
        other.tag = self.tag
        return other


class Sphere(Leaf):
    def __init__(self, c, r, **kw):
        super().__init__(**kw)
        self.c = v3(c); self.r = float(r)
        self.lo = self.c - self.r; self.hi = self.c + self.r

    def _eval(self, P, band):
        return _rowlen(P - self.c.astype(F)) - F(self.r)

    def mirrored(self):
        return self._meta(Sphere(self.c * [-1, 1, 1], self.r))


class Ellipsoid(Leaf):
    """Ellipsoid with semi-axes `r` along the columns of rotation `R` (approximate distance)."""

    def __init__(self, c, r, R=None, **kw):
        super().__init__(**kw)
        self.c = v3(c); self.r = v3(r); self.R = np.eye(3) if R is None else np.asarray(R, float)
        ext = np.sqrt(((self.R * self.r[None, :]) ** 2).sum(axis=1))
        self.lo = self.c - ext; self.hi = self.c + ext

    def _eval(self, P, band):
        q = (P - self.c.astype(F)) @ self.R.astype(F)
        k0 = _rowlen(q / self.r.astype(F))
        k1 = _rowlen(q / (self.r ** 2).astype(F))
        k1 = np.maximum(k1, F(1e-9))
        return k0 * (k0 - 1.0) / k1

    def mirrored(self):
        return self._meta(Ellipsoid(self.c * [-1, 1, 1], self.r, _mirror_rot(self.R)))


class RoundCone(Leaf):
    """Exact distance to the convex hull of two spheres (a, ra) and (b, rb)."""

    def __init__(self, a, b, ra, rb=None, **kw):
        super().__init__(**kw)
        self.a = v3(a); self.b = v3(b); self.ra = float(ra); self.rb = float(ra if rb is None else rb)
        m = max(self.ra, self.rb)
        self.lo = np.minimum(self.a, self.b) - m; self.hi = np.maximum(self.a, self.b) + m

    def _eval(self, P, band):
        a = self.a.astype(F); ba = (self.b - self.a).astype(F)
        r1, r2 = F(self.ra), F(self.rb)
        l2 = F(np.dot(ba, ba)); rr = r1 - r2; a2 = l2 - rr * rr; il2 = F(1.0) / l2
        pa = P - a
        y = pa @ ba
        z = y - l2
        x2v = pa * l2 - y[:, None] * ba[None, :]
        x2 = np.einsum("ij,ij->i", x2v, x2v)
        y2 = y * y * l2
        z2 = z * z * l2
        k = np.sign(rr) * rr * rr * x2
        d3 = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - r1
        d1 = np.sqrt(x2 + z2) * il2 - r2
        d2 = np.sqrt(x2 + y2) * il2 - r1
        out = np.where(np.sign(y) * a2 * y2 < k, d2, d3)
        return np.where(np.sign(z) * a2 * z2 > k, d1, out)

    def mirrored(self):
        S = np.array([-1, 1, 1])
        return self._meta(RoundCone(self.a * S, self.b * S, self.ra, self.rb))


def Capsule(a, b, r, **kw):
    return RoundCone(a, b, r, r, **kw)


class RoundBox(Leaf):
    def __init__(self, c, half, R=None, rr=0.0, **kw):
        super().__init__(**kw)
        self.c = v3(c); self.half = v3(half); self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.rr = float(rr)
        ext = np.abs(self.R) @ self.half
        self.lo = self.c - ext; self.hi = self.c + ext

    def _eval(self, P, band):
        q = np.abs((P - self.c.astype(F)) @ self.R.astype(F)) - (self.half - self.rr).astype(F)
        out = _rowlen(np.maximum(q, 0)) + np.minimum(q.max(axis=1), 0)
        return out - F(self.rr)

    def mirrored(self):
        return self._meta(RoundBox(self.c * [-1, 1, 1], self.half, _mirror_rot(self.R), self.rr))

def _pchip_table(ts, values, n=513):
    from scipy.interpolate import PchipInterpolator
    grid = np.linspace(0.0, 1.0, n)
    return grid, np.stack([PchipInterpolator(ts, v)(grid) for v in np.asarray(values, float).T], axis=1)


class Loft(Leaf):
    """Generalised cylinder along segment a->b with asymmetric superelliptic cross-sections.

    The section frame: u along `side` (made perpendicular to the axis), v = axis x u.
    sections: rows (t, u_pos, u_neg, v_pos, v_neg, n, cu, cv): half extents on each side of the
    section centre (cu, cv), superellipse exponent n (2 = ellipse, larger = boxier). t in [0, 1].
    Ends are cut flat, or rounded with smooth-max radii round0 / round1 (metres).
    """

    def __init__(self, a, b, side, sections, round0=0.0, round1=0.0, **kw):
        super().__init__(**kw)
        self.round0, self.round1 = float(round0), float(round1)
        self.a = v3(a); self.b = v3(b)
        ax = self.b - self.a
        self.L = float(np.linalg.norm(ax))
        self.e_ax = ax / self.L
        s = v3(side)
        self.e_u = norm(s - self.e_ax * (s @ self.e_ax))
        self.e_v = np.cross(self.e_ax, self.e_u)
        sec = np.asarray(sections, float)
        if sec.shape[1] == 6:
            sec = np.hstack([sec, np.zeros((len(sec), 2))])
        self.sections = sec
        self.grid, self.table = _pchip_table(sec[:, 0], sec[:, 1:])
        ext = max(np.abs(sec[:, 1:5]).max() + np.abs(sec[:, 6:8]).max(), 1e-3)
        self.lo = np.minimum(self.a, self.b) - ext; self.hi = np.maximum(self.a, self.b) + ext

    def params(self, t):
        idx = np.clip(t, 0, 1) * (len(self.grid) - 1)
        i0 = np.minimum(idx.astype(np.int64), len(self.grid) - 2)
        w = (idx - i0)[:, None].astype(F)
        return self.table[i0].astype(F) * (1 - w) + self.table[i0 + 1].astype(F) * w

    def _eval(self, P, band):
        rel = P - self.a.astype(F)
        t = rel @ (self.e_ax / self.L).astype(F)
        pr = self.params(t)
        u = rel @ self.e_u.astype(F) - pr[:, 5]
        v = rel @ self.e_v.astype(F) - pr[:, 6]
        a = np.where(u >= 0, pr[:, 0], pr[:, 1])
        b = np.where(v >= 0, pr[:, 2], pr[:, 3])
        n = pr[:, 4]
        qu = np.abs(u) / a + F(1e-9)
        qv = np.abs(v) / b + F(1e-9)
        rho = (qu ** n + qv ** n) ** (1.0 / n)
        gu = (qu / rho) ** (n - 1) / a
        gv = (qv / rho) ** (n - 1) / b
        g = np.sqrt(gu * gu + gv * gv)
        d2 = (rho - 1.0) / np.maximum(g, F(1e-6))
        d = smax(d2, -t * self.L, self.round0)
        return smax(d, (t - 1.0) * self.L, self.round1)

    def mirrored(self):
        S = np.array([-1.0, 1.0, 1.0])
        sec = self.sections.copy()
        # mirroring flips the handedness of (u, v): swap the v sides and negate the v centre
        sec[:, [3, 4]] = sec[:, [4, 3]]
        sec[:, 7] *= -1
        return self._meta(Loft(self.a * S, self.b * S, self.e_u * S, sec, self.round0, self.round1))


class HeightField(Leaf):
    """Solid behind the surface y = f(x, z) (the front is -y), symmetric in x.

    xs (>= 0, ascending), zs (ascending) and F[len(zs), len(xs)] give f relative to `origin`.
    A bicubic spline through the table is resampled at `step`; evaluation is bilinear.
    Bounded by lo/hi (world): use it inside an Intersect with a closed shape.
    """

    def __init__(self, origin, xs, zs, Fv, lo, hi, step=0.0004, **kw):
        from scipy.interpolate import RectBivariateSpline
        super().__init__(**kw)
        self.o = v3(origin)
        xs = np.asarray(xs, float); zs = np.asarray(zs, float); Fv = np.asarray(Fv, float)
        xx = np.concatenate([-xs[:0:-1], xs])
        FF = np.concatenate([Fv[:, :0:-1], Fv], axis=1)
        spl = RectBivariateSpline(zs, xx, FF, kx=3, ky=3)
        self.gx = np.arange(xx[0], xx[-1] + 1e-9, step)
        self.gz = np.arange(zs[0], zs[-1] + 1e-9, step)
        self.f = spl(self.gz, self.gx).astype(F)
        self.fz = spl(self.gz, self.gx, dx=1).astype(F)
        self.fx = spl(self.gz, self.gx, dy=1).astype(F)
        self.step = step
        self.lo = v3(lo); self.hi = v3(hi)

    def _bilinear(self, tab, iz, ix, wz, wx):
        return (tab[iz, ix] * (1 - wz) * (1 - wx) + tab[iz + 1, ix] * wz * (1 - wx)
                + tab[iz, ix + 1] * (1 - wz) * wx + tab[iz + 1, ix + 1] * wz * wx)

    def _eval(self, P, band):
        rel = P - self.o.astype(F)
        fx_ = (rel[:, 0] - self.gx[0]) / self.step
        fz_ = (rel[:, 2] - self.gz[0]) / self.step
        ix = np.clip(fx_.astype(np.int64), 0, len(self.gx) - 2)
        iz = np.clip(fz_.astype(np.int64), 0, len(self.gz) - 2)
        wx = np.clip(fx_ - ix, 0, 1).astype(F)
        wz = np.clip(fz_ - iz, 0, 1).astype(F)
        f = self._bilinear(self.f, iz, ix, wz, wx)
        gx = self._bilinear(self.fx, iz, ix, wz, wx)
        gz = self._bilinear(self.fz, iz, ix, wz, wx)
        return (f - rel[:, 1]) / np.sqrt(1 + gx * gx + gz * gz)

    def mirrored(self):
        return self   # symmetric by construction


class Bump:
    """Smooth compact displacement: amplitude A (m, + pushes the surface out) inside an ellipsoid."""

    bone = None
    tag = None

    def __init__(self, c, r, A, R=None, power=3, tag=None):
        self.c = v3(c); self.r = v3(r); self.A = float(A); self.power = power
        self.R = np.eye(3) if R is None else np.asarray(R, float)
        self.tag = tag
        ext = np.sqrt(((self.R * self.r[None, :]) ** 2).sum(axis=1))
        self.lo = self.c - ext; self.hi = self.c + ext

    def leaves(self):
        return iter(())

    def mirrored(self):
        return Bump(self.c * [-1, 1, 1], self.r, self.A, _mirror_rot(self.R), self.power, self.tag)

    def disp(self, P):
        q = (P - self.c.astype(F)) @ self.R.astype(F) / self.r.astype(F)
        w = np.maximum(1 - np.einsum("ij,ij->i", q, q), 0)
        return F(self.A) * w ** self.power


def bump_tube(points, r_across, r_along, A, tag=None):
    """Chain of overlapping bumps along a polyline (folds, creases, ridges) with ~constant height A."""
    pts = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1).max()
    P = catmull_rom(pts, max(2, int(np.ceil(seg / (0.4 * r_along)))))
    step = np.linalg.norm(np.diff(P, axis=0), axis=1).mean()
    each = A / ((32.0 / 35.0) * r_along / step)
    out = []
    for i in range(len(P)):
        d = norm(P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)])
        R = rot_from_axes(y=d, z=[0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
        out.append(Bump(P[i], (r_across, r_along, r_across), each, R, tag=tag))
    return out



class Plane(Leaf):
    """Half-space n.p <= d (inside). Unbounded: only use as the second operand of Intersect/Subtract."""

    def __init__(self, n, d, **kw):
        super().__init__(**kw)
        self.n = norm(n); self.d = float(d)
        self.lo = np.full(3, -np.inf); self.hi = np.full(3, np.inf)

    def _eval(self, P, band):
        return P @ self.n.astype(F) - F(self.d)

    def mirrored(self):
        return self._meta(Plane(self.n * [-1, 1, 1], self.d))


def catmull_rom(points, samples=6):
    """Dense polyline through control points (centripetal-ish uniform Catmull-Rom)."""
    P = np.asarray(points, float)
    if len(P) < 3:
        return P
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for t in np.linspace(0, 1, samples, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1])
    return np.array(out)


def Tube(points, radii, samples=6, k=0.0, **kw):
    """Smooth tube through control points; radii are per control point (interpolated)."""
    P = np.asarray(points, float)
    r = np.broadcast_to(np.asarray(radii, float), (len(P),))
    dense = catmull_rom(P, samples)
    rd = np.interp(np.linspace(0, len(P) - 1, len(dense)), np.arange(len(P)), r)
    segs = [RoundCone(dense[i], dense[i + 1], rd[i], rd[i + 1], **kw) for i in range(len(dense) - 1)]
    return Union(segs, k=k)


class Union(Node):
    def __init__(self, children, k=0.0, bone=None, tag=None):
        self.children = [c for c in children if c is not None]
        self.k = float(k)
        if bone is not None or tag is not None:
            for leaf in self.leaves():
                if bone is not None and leaf.bone is None:
                    leaf.bone = bone
                if tag is not None and leaf.tag is None:
                    leaf.tag = tag
        self.lo = np.min([c.lo for c in self.children], axis=0)
        self.hi = np.max([c.hi for c in self.children], axis=0)

    def leaves(self):
        for c in self.children:
            yield from c.leaves()

    def mirrored(self):
        return Union([c.mirrored() for c in self.children], k=self.k)

    def _eval(self, P, band):
        d = None
        for c in self.children:
            dc = evaluate(c, P, band)
            d = dc if d is None else smin(d, dc, self.k)
        return d


class Subtract(Node):
    """a minus b, smoothed by k."""

    def __init__(self, a, b, k=0.0):
        self.a, self.b, self.k = a, b, float(k)
        self.lo, self.hi = a.lo, a.hi

    def leaves(self):
        yield from self.a.leaves()
        yield from self.b.leaves()

    def mirrored(self):
        return Subtract(self.a.mirrored(), self.b.mirrored(), self.k)

    def _eval(self, P, band):
        da = evaluate(self.a, P, band)
        db = evaluate(self.b, P, band)
        return smax(da, -db, self.k)


class Intersect(Node):
    def __init__(self, a, b, k=0.0):
        self.a, self.b, self.k = a, b, float(k)
        self.lo = np.maximum(a.lo, b.lo); self.hi = np.minimum(a.hi, b.hi)

    def leaves(self):
        yield from self.a.leaves()
        yield from self.b.leaves()

    def mirrored(self):
        return Intersect(self.a.mirrored(), self.b.mirrored(), self.k)

    def _eval(self, P, band):
        return smax(evaluate(self.a, P, band), evaluate(self.b, P, band), self.k)


class Shell(Node):
    """Hollow skin of thickness 2t around the child's surface."""

    def __init__(self, child, t):
        self.child, self.t = child, float(t)
        self.lo, self.hi = child.lo - t, child.hi + t

    def leaves(self):
        yield from self.child.leaves()

    def mirrored(self):
        return Shell(self.child.mirrored(), self.t)

    def _eval(self, P, band):
        return np.abs(evaluate(self.child, P, band + self.t)) - F(self.t)


class Offset(Node):
    """Grow (r > 0) or shrink (r < 0) the child."""

    def __init__(self, child, r):
        self.child, self.r = child, float(r)
        self.lo, self.hi = child.lo - max(r, 0), child.hi + max(r, 0)

    def leaves(self):
        yield from self.child.leaves()

    def mirrored(self):
        return Offset(self.child.mirrored(), self.r)

    def _eval(self, P, band):
        return evaluate(self.child, P, band + max(self.r, 0.0)) - F(self.r)


class Program:
    """A sequence of operations applied to a running distance: the body is built in order.

    ops: list of (kind, node, k) with kind in {"add", "sub", "int"}.
    """

    def __init__(self):
        self.ops = []

    def add(self, node, k=0.0):
        self.ops.append(("add", node, float(k))); return node

    def sub(self, node, k=0.0):
        self.ops.append(("sub", node, float(k))); return node

    def inter(self, node, k=0.0):
        self.ops.append(("int", node, float(k))); return node

    def disp(self, bumps):
        """Displace the surface built so far by a list of Bumps (sculpting relief)."""
        bumps = list(bumps)
        self.ops.append(("disp", bumps, 0.0)); return bumps

    def leaves(self):
        for kind, n, _ in self.ops:
            if kind != "disp":
                yield from n.leaves()

    def bounds(self):
        adds = [n for kind, n, _ in self.ops if kind == "add"]
        return np.min([n.lo for n in adds], axis=0), np.max([n.hi for n in adds], axis=0)

    def __call__(self, P, band):
        P = np.asarray(P, dtype=F)
        d = np.full(len(P), BIG, dtype=F)
        # numpy's Accelerate matmul raises spurious FP warnings on arm64; values are fine.
        with np.errstate(all="ignore"):
            for kind, node, k in self.ops:
                if kind == "disp":
                    for b in node:
                        m = np.all((P >= b.lo.astype(F)) & (P <= b.hi.astype(F)), axis=1)
                        if m.any():
                            d[m] -= b.disp(P[m])
                    continue
                dn = evaluate(node, P, band + k)
                if kind == "add":
                    d = smin(d, dn, k)
                elif kind == "sub":
                    d = smax(d, -dn, k)
                else:
                    d = smax(d, dn, k)
        return d


def evaluate(node, P, band):
    """Distance from node for every row of P; rows outside the node's padded bounds get BIG."""
    margin = band + getattr(node, "k", 0.0)
    lo = (node.lo - margin).astype(F); hi = (node.hi + margin).astype(F)
    m = np.all((P >= lo) & (P <= hi), axis=1)
    n = int(m.sum())
    if n == 0:
        return np.full(len(P), BIG, dtype=F)
    if n == len(P):
        return node._eval(P, margin).astype(F, copy=False)
    out = np.full(len(P), BIG, dtype=F)
    out[m] = node._eval(P[m], margin)
    return out


def group_distance(leaves, P, band=0.05):
    """Hard union of the given leaves (no smoothing): used for bone weights and tag masks."""
    d = np.full(len(P), BIG, dtype=F)
    for leaf in leaves:
        d = np.minimum(d, evaluate(leaf, P, band))
    return d
