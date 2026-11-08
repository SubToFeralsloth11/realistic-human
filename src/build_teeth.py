#!/usr/bin/env python3
"""Mesh the upper and lower dentition: python3 src/build_teeth.py [--h 0.0003]"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rh import teeth  # noqa: E402
from rh.mesher import surface_nets  # noqa: E402
from rh.sdf import F, group_distance  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.0003)
    a = ap.parse_args()
    for name, upper in (("teeth_upper", True), ("teeth_lower", False)):
        prog = teeth.program(upper)
        lo, hi = prog.bounds()
        V, Q = surface_nets(prog, lo, hi, a.h, levels=3, act=1.8, ev=3.0, log=lambda *_: None)
        gum_leaves = [l for l in prog.leaves() if l.tag == "gum"]
        tooth_leaves = [l for l in prog.leaves() if l.tag == "tooth"]
        dg = group_distance(gum_leaves, V.astype(F))
        dt = group_distance(tooth_leaves, V.astype(F))
        m_gum = np.clip((dt - dg) / 0.0006 * 0.5 + 0.5, 0, 1).astype(np.float32)
        out = os.path.join(ROOT, "build", name + ".npz")
        np.savez(out, verts=V, quads=Q, attr_m_gum=m_gum)
        print(f"wrote {out}: {len(V)} verts")


if __name__ == "__main__":
    main()
