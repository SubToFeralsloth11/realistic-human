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


def save(name, parts, ctx=None, grid=None):
    data = clothes.merge(parts)
    if ctx is not None:
        d = clothes.clearance(ctx.prog, data["verts"], grid)
        print(f"  {name}: skin clearance min {d.min() * 1000:.2f} mm, {(d < 0).sum()} verts inside the body, "
              f"{(d < 0.0005).sum()} within 0.5 mm")
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
    if "shoes" in only:
        shoe, parts = clothes.build_shoes(ctx, a.h * 0.6)
        save("shoes", parts, ctx, shoe.G)
    jeans = None
    if "jeans" in only or "shirt" in only:
        shoe = clothes.Shoe(ctx) if "shoes" not in only else shoe
        jeans = clothes.Jeans(ctx, shoe=shoe)
    if "jeans" in only:
        save("jeans", jeans.build(a.h), ctx, jeans.G)
        info = jeans.pocket_info()
        with open(os.path.join(BUILD, "pocket.json"), "w") as f:
            json.dump(info, f, indent=1)
        print(json.dumps(info, indent=1))
    if "shirt" in only:
        shirt = clothes.Shirt(ctx, jeans=jeans)
        save("shirt", [shirt.build(a.h)], ctx, shirt.G)
    print(f"done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
