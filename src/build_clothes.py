#!/usr/bin/env python3
"""Build the t-shirt, jeans and sneakers from the current body SDF.

  python3 src/build_clothes.py [--h 0.0018] [--only shoes,jeans,shirt]

Writes build/shirt.npz, build/jeans.npz, build/shoes.npz (verts, faces, sizes, attr_*) and
build/pocket.json (right front pocket opening and bag, body-frame coordinates).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_body  # noqa: E402
from rh import clothes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")


def save(name, parts):
    data = clothes.merge(parts)
    path = os.path.join(BUILD, f"{name}.npz")
    np.savez_compressed(path, **data)
    print(f"wrote {path}: {len(data['verts'])} verts, {len(data['sizes'])} faces, "
          f"attrs {[k[5:] for k in data if k.startswith('attr_')]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.0018)
    ap.add_argument("--only", default="shoes,jeans,shirt")
    a = ap.parse_args()
    only = set(a.only.split(","))
    t = time.time()
    ctx = clothes.BodyCtx(build_body.make_program())
    os.makedirs(BUILD, exist_ok=True)
    if "shirt" in only:
        shirt = clothes.Shirt(ctx)
        save("shirt", [shirt.build(a.h)])
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
