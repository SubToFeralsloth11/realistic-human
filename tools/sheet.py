#!/usr/bin/env python3
"""Tile images into one contact sheet: sheet.py out.jpg [--h 700] [--cols N] img1 img2 ..."""
import argparse
import math

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--h", type=int, default=700)
    ap.add_argument("--cols", type=int, default=0)
    a = ap.parse_args()
    ims = [Image.open(p).convert("RGB") for p in a.images]
    ims = [im.resize((round(im.width * a.h / im.height), a.h), Image.LANCZOS) for im in ims]
    cols = a.cols or len(ims)
    rows = math.ceil(len(ims) / cols)
    w = max(im.width for im in ims)
    sheet = Image.new("RGB", (w * cols, a.h * rows), (20, 20, 22))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * w, (i // cols) * a.h))
    sheet.save(a.out, quality=88)


main()
