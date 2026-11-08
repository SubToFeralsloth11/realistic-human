"""Join meshes of different resolution along cut planes (neck, wrists, ankles).

Each mesh is cut by a plane n.p = d; the fine part keeps the n.p > d side, the coarse body keeps
the other side. The two open boundary loops are snapped onto the plane and zipped together with
a strip of triangles ordered by angle around the cut's centroid.
Faces are stored as a flat index array plus per-face sizes (quads and triangles mixed).
"""
import numpy as np


class Mesh:
    def __init__(self, verts, faces, sizes, attrs=None):
        self.verts = np.asarray(verts, np.float32)
        self.faces = np.asarray(faces, np.int64)       # flat
        self.sizes = np.asarray(sizes, np.int64)
        self.attrs = {k: np.asarray(v, np.float32) for k, v in (attrs or {}).items()}

    @classmethod
    def from_quads(cls, verts, quads, attrs=None):
        q = np.asarray(quads, np.int64)
        return cls(verts, q.ravel(), np.full(len(q), 4), attrs)

    def starts(self):
        return np.concatenate([[0], np.cumsum(self.sizes)[:-1]])


def cut(mesh, n, d, keep_positive, box=None):
    """Keep faces entirely on one side of the plane; returns a compacted Mesh.

    With box=(lo, hi) only vertices inside the box are subject to the cut (others are kept),
    so cutting one ankle does not also cut the other.
    """
    n = np.asarray(n, float)
    side = mesh.verts @ n - d
    ok_v = side > 0 if keep_positive else side <= 0
    if box is not None:
        inside = np.all((mesh.verts >= box[0]) & (mesh.verts <= box[1]), axis=1)
        ok_v |= ~inside
    st = mesh.starts()
    face_ok = np.logical_and.reduceat(ok_v[mesh.faces], st)
    keep_idx = np.repeat(face_ok, mesh.sizes)
    faces = mesh.faces[keep_idx]
    sizes = mesh.sizes[face_ok]
    used = np.zeros(len(mesh.verts), bool)
    used[faces] = True
    remap = np.cumsum(used) - 1
    return Mesh(mesh.verts[used], remap[faces], sizes, {k: v[used] for k, v in mesh.attrs.items()})


def boundary_loops(mesh):
    """Ordered vertex loops along edges used by exactly one face."""
    st = mesh.starts()
    pos_in_face = np.arange(len(mesh.faces)) - np.repeat(st, mesh.sizes)
    last = pos_in_face == np.repeat(mesh.sizes - 1, mesh.sizes)
    nxt_idx = np.where(last, np.repeat(st, mesh.sizes), np.arange(len(mesh.faces)) + 1)
    a = mesh.faces
    b = mesh.faces[nxt_idx]
    key = np.minimum(a, b) * (len(mesh.verts) + 1) + np.maximum(a, b)
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    single = cnt[inv] == 1
    ea, eb = a[single], b[single]           # directed boundary edges (face winding)
    succ = dict(zip(ea.tolist(), eb.tolist()))
    loops, seen = [], set()
    for start in ea.tolist():
        if start in seen:
            continue
        loop, v = [], start
        while v not in seen and v in succ:
            seen.add(v)
            loop.append(v)
            v = succ[v]
        if len(loop) > 2:
            loops.append(np.array(loop))
    return loops


def _main_loop(mesh, n, d, near=None):
    loops = boundary_loops(mesh)
    if not loops:
        raise ValueError("no boundary after cut")
    on_plane = [l for l in loops if np.abs(mesh.verts[l] @ n - d).mean() < 0.01]
    if not on_plane:
        raise ValueError("no boundary loop on the cut plane")
    if near is None:
        return max(on_plane, key=len)
    return min(on_plane, key=lambda l: np.linalg.norm(mesh.verts[l].mean(axis=0) - near))


def zip_meshes(fine, coarse, n, d, box=None):
    """fine keeps n.p > d, coarse keeps n.p <= d (inside `box` only); returns the joined Mesh."""
    n = np.asarray(n, float) / np.linalg.norm(n)
    A = cut(fine, n, d, True)
    B = cut(coarse, n, d, False, box)
    la = _main_loop(A, n, d)
    lb = _main_loop(B, n, d, near=A.verts[la].mean(axis=0))
    # snap the two rims onto the plane
    for M, L in ((A, la), (B, lb)):
        p = M.verts[L]
        M.verts[L] = p - np.outer(p @ n - d, n)
    off = len(A.verts)
    verts = np.vstack([A.verts, B.verts])
    pa, pb = verts[la], verts[lb + off]
    c = pa.mean(axis=0)
    e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)

    def ang(p):
        r = p - c
        return np.arctan2(r @ e2, r @ e1)

    def ordered(loop, base):
        """Loop in boundary order, turned to run with increasing angle; unwrapped angles."""
        t = np.unwrap(ang(verts[loop + base]))
        if t[-1] < t[0]:
            loop, t = loop[::-1], np.unwrap(ang(verts[loop[::-1] + base]))
        return loop + base, t

    ia, ta = ordered(la, 0)
    ib, tb = ordered(lb, off)
    # start B at the vertex nearest in angle to A's first vertex, aligned to A's angle branch
    j0 = int(np.argmin(np.abs(np.angle(np.exp(1j * (tb - ta[0]))))))
    ib = np.roll(ib, -j0)
    tb = np.unwrap(ang(verts[ib]))
    tb += 2 * np.pi * np.round((ta[0] - tb[0]) / (2 * np.pi))
    ta_ext = np.append(ta, ta[0] + 2 * np.pi)
    tb_ext = np.append(tb, tb[0] + 2 * np.pi)
    tris = []
    i = j = 0
    na, nb = len(ia), len(ib)
    while i < na or j < nb:
        a0, b0 = ia[i % na], ib[j % nb]
        adv_a = j >= nb or (i < na and ta_ext[i + 1] <= tb_ext[j + 1])
        if adv_a:
            tris.append((a0, ia[(i + 1) % na], b0))
            i += 1
        else:
            tris.append((a0, ib[(j + 1) % nb], b0))
            j += 1
    tris = np.array(tris, np.int64)
    # orient the strip against A's own boundary edges (each rim edge must be used once each way)
    a_dir = set(zip(la[:-1].tolist(), la[1:].tolist())) | {(int(la[-1]), int(la[0]))}
    a_edges = [(int(t[0]), int(t[1])) for t in tris if t[0] < off and t[1] < off]
    if sum(e in a_dir for e in a_edges) > len(a_edges) / 2:
        tris = tris[:, ::-1]
    faces = np.concatenate([A.faces, B.faces + off, tris.ravel()])
    sizes = np.concatenate([A.sizes, B.sizes, np.full(len(tris), 3)])
    attrs = {k: np.concatenate([A.attrs[k], B.attrs[k]]) for k in A.attrs if k in B.attrs}
    return Mesh(verts, faces, sizes, attrs)
