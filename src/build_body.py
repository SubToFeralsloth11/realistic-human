#!/usr/bin/env python3
"""Assemble the anatomical SDF and mesh it.

  python3 src/build_body.py [--h 0.002] [--region body|head|hand|foot|torso] [--out build/body.npz]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rh import body, ear, foot, hand, head  # noqa: E402
from rh.mesher import surface_nets  # noqa: E402
from rh.sdf import Program  # noqa: E402
from rh.skeleton import HEAD, HAND  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGIONS = {
    "head": (HEAD.o + [-0.105, -0.075, -0.20], HEAD.o + [0.105, 0.185, 0.13]),
    "hand": (HAND.o + [-0.10, -0.12, -0.15], HAND.o + [0.16, 0.10, 0.07]),
    "foot": (np.array([0.0, -0.20, -0.01]), np.array([0.22, 0.10, 0.20])),
    "torso": (np.array([-0.30, -0.20, 0.80]), np.array([0.30, 0.20, 1.62])),
}


def make_program():
    prog = Program()
    body.build(prog)
    hand.build(prog)
    foot.build(prog)
    head.build(prog)
    ear.build(prog)
    return prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.002)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--region", default="body")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = time.time()
    prog = make_program()
    if a.region == "body":
        lo, hi = prog.bounds()
    else:
        lo, hi = REGIONS[a.region]
    print(f"{sum(1 for _ in prog.leaves())} primitives, {len(prog.ops)} ops; region {a.region} {np.round(lo, 3)} {np.round(hi, 3)}")
    V, Q = surface_nets(prog, lo, hi, a.h, levels=a.levels, act=1.8, ev=3.0)
    out = a.out or os.path.join(ROOT, "build", f"{a.region}_{int(round(a.h * 1e4))}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, verts=V, quads=Q)
    print(f"wrote {out}: {len(V)} verts {len(Q)} quads in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
