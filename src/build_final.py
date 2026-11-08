#!/usr/bin/env python3
"""Final body mesh: adaptive resolution, region masks and skin weights.

  python3 src/build_final.py [--body 0.0025] [--head 0.0008] [--hand 0.0009] [--foot 0.0015]

The whole body is meshed coarsely, then the head (cut at the neck), both hands (cut above the
wrists) and both feet (cut above the ankles) are re-meshed finely and zipped in.
Writes build/human.npz: verts, faces, sizes, attr_* masks, bone names, top-4 weights.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_body import make_program  # noqa: E402
from rh import head, masks, rig, weights  # noqa: E402
from rh.mesher import surface_nets  # noqa: E402
from rh.skeleton import HAND, HEAD, J  # noqa: E402
from rh.stitch import Mesh, zip_meshes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def mesh_region(prog, lo, hi, h, levels=3):
    V, Q = surface_nets(prog, lo, hi, h, levels=levels, act=1.8, ev=3.0, log=lambda *_: None)
    return Mesh.from_quads(V, Q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", type=float, default=0.0025)
    ap.add_argument("--head", type=float, default=0.0008)
    ap.add_argument("--hand", type=float, default=0.0009)
    ap.add_argument("--foot", type=float, default=0.0015)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "human.npz"))
    a = ap.parse_args()
    t0 = time.time()
    prog = make_program()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    lo, hi = prog.bounds()
    M = mesh_region(prog, lo, hi, a.body, levels=2)
    log(f"body {len(M.verts)} verts")
    # head: cut through the neck
    hlo, hhi = HEAD.o + [-0.105, -0.075, -0.20], HEAD.o + [0.105, 0.185, 0.13]
    Hm = mesh_region(prog, hlo, hhi, a.head)
    M = zip_meshes(Hm, M, (0, 0, 1), 1.53, box=(hlo - 0.01, hhi + 0.01))
    log(f"+head {len(Hm.verts)} -> {len(M.verts)} verts")
    del Hm
    # hands: cut perpendicular to the forearm, 35 mm above the wrist joint
    for s in (1.0, -1.0):
        S = np.array([s, 1.0, 1.0])
        gh, wr = J["upperarm_l"] * S, J["hand_l"] * S
        n = (wr - gh) / np.linalg.norm(wr - gh)
        d = float(n @ (wr - 0.035 * n))
        c = HAND.o * S
        blo = np.minimum(c + [-0.10, -0.13, -0.16], c + [0.16 * s, 0.10, 0.08])
        bhi = np.maximum(c + [-0.10 * s, -0.13, -0.16], c + [0.17 * s, 0.10, 0.08])
        Hd = mesh_region(prog, blo, bhi, a.hand)
        M = zip_meshes(Hd, M, n, d, box=(blo - 0.01, bhi + 0.01))
        log(f"+hand {s:+.0f} {len(Hd.verts)} -> {len(M.verts)} verts")
    # feet: cut horizontally above the malleoli
    for s in (1.0, -1.0):
        flo = np.array([0.0 if s > 0 else -0.22, -0.20, -0.005])
        fhi = np.array([0.22 if s > 0 else 0.0, 0.11, 0.20])
        Ft = mesh_region(prog, flo, fhi, a.foot)
        M = zip_meshes(Ft, M, (0, 0, -1), -0.15, box=(flo - 0.01, fhi + 0.01))
        log(f"+foot {s:+.0f} {len(Ft.verts)} -> {len(M.verts)} verts")
    M.attrs = masks.compute(M.verts, prog)
    log("masks")
    bones = rig.all_bones()
    names, idx, w = weights.compute(M.verts, M.faces, M.sizes, prog, bones)
    log("weights")
    attrs = {f"attr_{k}": v for k, v in M.attrs.items()}
    np.savez(a.out, verts=M.verts, faces=M.faces, sizes=M.sizes, bone_names=np.array(names),
             w_idx=idx, w_val=w, **attrs)
    with open(os.path.join(ROOT, "build", "rig.json"), "w") as f:
        json.dump({k: {"head": v[0].tolist(), "tail": v[1].tolist(), "parent": v[2], "z": v[3].tolist()}
                   for k, v in bones.items()}, f, indent=1)
    with open(os.path.join(ROOT, "build", "landmarks.json"), "w") as f:
        json.dump({"eye_l": head.LANDMARKS["eye_l"].tolist(), "eye_r": head.LANDMARKS["eye_r"].tolist(),
                   "eye_radius": head.EYE_R}, f, indent=1)
    tris = int((M.sizes == 3).sum())
    log(f"wrote {a.out}: {len(M.verts)} verts, {len(M.sizes)} faces ({tris} seam triangles)")


if __name__ == "__main__":
    main()
