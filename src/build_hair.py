#!/usr/bin/env python3
"""Grow hair on a meshed body: python3 src/build_hair.py [mesh.npz] [--out build/hair.npz] [--scale 1.0]"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_body import make_program  # noqa: E402
from rh import hair  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", nargs="?", default=os.path.join(ROOT, "build", "human.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "hair.npz"))
    ap.add_argument("--scale", type=float, default=1.0, help="strand count multiplier")
    a = ap.parse_args()
    t0 = time.time()
    m = np.load(a.mesh)
    verts = m["verts"].astype(np.float32)
    if "faces" in m.files:
        tris = hair.triangulate(m["faces"], m["sizes"])
    else:
        q = m["quads"]
        tris = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    prog = make_program()
    rng = np.random.default_rng(7)
    out = {}
    groups = {
        "scalp": lambda: hair.scalp(prog, verts, tris, m["attr_m_scalp"], rng, int(90000 * a.scale)),
        "brows": lambda: hair.eyebrows(prog, verts, tris, m["attr_m_brow"], rng, int(1500 * a.scale)),
        "stubble": lambda: hair.stubble(prog, verts, tris, m["attr_m_beard"], rng, int(45000 * a.scale)),
        "lashes": lambda: hair.eyelashes(rng),
    }
    for name, fn in groups.items():
        pts, counts, radius = fn()
        out[f"{name}_points"], out[f"{name}_counts"], out[f"{name}_radius"] = pts, counts, radius
        print(f"[{time.time() - t0:6.1f}s] {name}: {len(counts)} strands, {len(pts)} points", flush=True)
    np.savez_compressed(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
