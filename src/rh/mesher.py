"""Sparse, multi-level surface nets: signed distance field -> watertight all-quad mesh.

The field is sampled on a coarse lattice, then only cells that can contain the surface are split
in two along each axis, `levels` times. Child samples far from the surface reuse trilinear
interpolation of their parent; samples near it are evaluated exactly. The domain is processed in
z slabs so memory stays bounded at 1 mm resolution over a whole body.
"""
import time

import numpy as np

SHIFT = 21
MASK = (1 << SHIFT) - 1

# Cube corners: index = a + 2b + 4c for offsets (a, b, c).
CORNERS = np.array([[a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)], dtype=np.int64)
EDGES = np.array([(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)])
CHILD27 = np.array([[a, b, c] for a in (0, 1, 2) for b in (0, 1, 2) for c in (0, 1, 2)], dtype=np.int64)
# Cells around an edge owned by cell (i, j, k), ordered so the quad faces +axis.
RING = {
    0: np.array([[0, 0, 0], [0, -1, 0], [0, -1, -1], [0, 0, -1]]),   # x edge: (dy, dz) ccw in (y, z)
    1: np.array([[0, 0, 0], [0, 0, -1], [-1, 0, -1], [-1, 0, 0]]),   # y edge: (dz, dx) ccw in (z, x)
    2: np.array([[0, 0, 0], [-1, 0, 0], [-1, -1, 0], [0, -1, 0]]),   # z edge: (dx, dy) ccw in (x, y)
}


def pack(ijk):
    ijk = np.asarray(ijk, dtype=np.int64)
    return (ijk[..., 0] << (2 * SHIFT)) | (ijk[..., 1] << SHIFT) | ijk[..., 2]


def unpack(keys):
    keys = np.asarray(keys, dtype=np.int64)
    return np.stack([(keys >> (2 * SHIFT)) & MASK, (keys >> SHIFT) & MASK, keys & MASK], axis=-1)


class Store:
    """Sorted key -> value table for lattice samples."""

    def __init__(self, keys, vals):
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.vals = vals[order]

    def get(self, keys):
        idx = np.searchsorted(self.keys, keys)
        idx = np.minimum(idx, len(self.keys) - 1)
        ok = self.keys[idx] == keys
        if not ok.all():
            raise KeyError(f"{(~ok).sum()} lattice samples missing")
        return self.vals[idx]


def surface_nets(sdf, lo, hi, h, levels=3, slab_cells=12, act=1.3, ev=2.5, project=1, log=print):
    """Mesh the zero level set of `sdf(P, band)` inside [lo, hi] at final spacing h.

    Returns (verts float32 (V, 3), quads int64 (Q, 4)).
    """
    t0 = time.time()
    L = levels
    S = 1 << L
    H = h * S
    lo = np.asarray(lo, float) - 2 * H
    hi = np.asarray(hi, float) + 2 * H
    n0 = np.ceil((hi - lo) / H).astype(np.int64)  # coarse cells per axis
    if np.any(n0 * S >= (1 << SHIFT)):
        raise ValueError("domain too large for key packing")
    lo32 = lo.astype(np.float32)

    def world(ijk_final):
        return (lo + ijk_final * h).astype(np.float32)

    vkeys_all, vpos_all, quads_all = [], [], []
    n_exact = 0
    for K0 in range(0, int(n0[2]), slab_cells):
        K1 = min(K0 + slab_cells, int(n0[2]))
        kc0 = max(K0 - 1, 0)
        # ---- coarse level: dense samples over the slab
        I, J, K = np.meshgrid(np.arange(n0[0] + 1), np.arange(n0[1] + 1), np.arange(kc0, K1 + 1), indexing="ij")
        cp = np.stack([I.ravel(), J.ravel(), K.ravel()], 1)
        cv = sdf(world(cp * S), act * H + 2 * H)
        n_exact += len(cp)
        grid = cv.reshape(I.shape)
        a = np.abs(grid)
        mn = np.minimum.reduce([a[x:x + a.shape[0] - 1, y:y + a.shape[1] - 1, z:z + a.shape[2] - 1]
                                for x in (0, 1) for y in (0, 1) for z in (0, 1)])
        ci = np.argwhere(mn <= act * H)
        cells = ci + np.array([0, 0, kc0])
        store = Store(pack(cp * S), cv)
        # ---- refine
        for lev in range(1, L + 1):
            step = S >> lev          # final-lattice units per sample at this level
            hl = H / (1 << lev)
            if len(cells) == 0:
                break
            child = (cells[:, None, :] * 2 + CHILD27[None, :, :]).reshape(-1, 3)
            ck = np.unique(pack(child * step))
            cijk = unpack(ck) // step                      # level-lev lattice coords
            odd = cijk & 1
            base = cijk >> 1                               # parent lattice coords (floor)
            vals = np.zeros(len(ck), np.float32)
            cnt = np.zeros(len(ck), np.float32)
            pstep = step * 2
            for off in CORNERS:
                use = np.all((off == 0) | (odd == 1), axis=1)   # only corners spanned by odd axes
                if not use.any():
                    continue
                pk = pack((base[use] + off) * pstep)
                vals[use] += store.get(pk)
                cnt[use] += 1
            vals /= cnt
            near = np.abs(vals) < ev * hl
            if near.any():
                vals[near] = sdf(world(cijk[near] * step), ev * hl + 2 * hl)
                n_exact += int(near.sum())
            store = Store(ck, vals)
            child_cells = (cells[:, None, :] * 2 + CORNERS[None, :, :]).reshape(-1, 3)
            child_cells = unpack(np.unique(pack(child_cells)))
            corner_v = np.stack([store.get(pack((child_cells + o) * step)) for o in CORNERS], 1)
            if lev < L:
                keep = np.abs(corner_v).min(axis=1) <= act * hl
                cells = child_cells[keep]
            else:
                inside = corner_v < 0
                surf = inside.any(axis=1) & ~inside.all(axis=1)
                cells = child_cells[surf]
                cvals = corner_v[surf]
        if len(cells) == 0:
            continue
        # ---- surface nets on the finest level (step = 1)
        v0 = cvals[:, EDGES[:, 0]]
        v1 = cvals[:, EDGES[:, 1]]
        cross = (v0 < 0) != (v1 < 0)
        t = np.where(cross, v0 / np.where(cross, v0 - v1, 1.0), 0.0)
        p = CORNERS[EDGES[:, 0]][None] + t[..., None] * (CORNERS[EDGES[:, 1]] - CORNERS[EDGES[:, 0]])[None]
        local = (p * cross[..., None]).sum(1) / cross.sum(1, keepdims=True)
        vpos = (lo + (cells + local) * h).astype(np.float32)
        vkeys = pack(cells)
        order = np.argsort(vkeys)
        vkeys, vpos, cells_s, cv_s = vkeys[order], vpos[order], cells[order], cvals[order]
        own = (cells_s[:, 2] >= K0 * S) & (cells_s[:, 2] < K1 * S)
        quads = []
        for axis, (ca, cb) in enumerate(((0, 1), (0, 2), (0, 4))):
            s0 = cv_s[:, ca] < 0
            s1 = cv_s[:, cb] < 0
            sel = own & (s0 != s1)
            if not sel.any():
                continue
            base = cells_s[sel]
            ring = base[:, None, :] + RING[axis][None]
            rk = pack(ring)
            idx = np.searchsorted(vkeys, rk)
            idx = np.minimum(idx, len(vkeys) - 1)
            ok = (vkeys[idx] == rk).all(axis=1)
            q = rk[ok]
            flip = ~s0[sel][ok]
            q[flip] = q[flip][:, ::-1]
            quads.append(q)
        if quads:
            quads_all.append(np.concatenate(quads))
        vkeys_all.append(vkeys)
        vpos_all.append(vpos)
        log(f"  slab {K0:4d}-{K1:4d}: {len(vkeys):8d} verts  ({time.time() - t0:6.1f}s, exact samples {n_exact})")

    vkeys = np.concatenate(vkeys_all)
    vpos = np.concatenate(vpos_all)
    uk, first = np.unique(vkeys, return_index=True)
    vpos = vpos[first]
    qk = np.concatenate(quads_all)
    quads = np.searchsorted(uk, qk)
    # drop vertices no quad uses
    used = np.zeros(len(uk), bool)
    used[quads.ravel()] = True
    remap = np.cumsum(used) - 1
    vpos = vpos[used]
    quads = remap[quads]
    for _ in range(project):
        vpos = project_to_surface(sdf, vpos, h)
    log(f"  mesh: {len(vpos)} verts, {len(quads)} quads, {n_exact} exact samples, {time.time() - t0:.1f}s")
    return vpos, quads


def gradient(sdf, P, eps):
    P = np.asarray(P, np.float32)
    g = np.zeros_like(P)
    for ax in range(3):
        d = np.zeros(3, np.float32); d[ax] = eps
        g[:, ax] = (sdf(P + d, 4 * eps) - sdf(P - d, 4 * eps)) / (2 * eps)
    return g


def project_to_surface(sdf, P, h, chunk=400_000):
    """One Newton step towards the zero level set, limited to half a cell."""
    out = np.empty_like(P)
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk]
        d = sdf(p, 2 * h)
        g = gradient(sdf, p, 0.25 * h)
        g2 = np.maximum(np.einsum("ij,ij->i", g, g), 1e-12)
        step = (d / g2)[:, None] * g
        ln = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, (0.5 * h) / np.maximum(ln, 1e-12))
        out[s:s + chunk] = p - step
    return out
