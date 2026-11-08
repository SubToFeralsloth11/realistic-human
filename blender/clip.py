"""Render one animation clip of the dressed, rigged character with EEVEE.

blender -b --factory-startup --python blender/clip.py -- <clip> <out.mp4> [--mesh build/human_dev.npz]
        [--res 1280x720] [--samples 48] [--frames a:b] [--still N out.png]
Clips are defined in src/rh/motions.py (CLIPS registry).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix  # noqa: E402

import anim  # noqa: E402  (also puts src/ on sys.path)
import character  # noqa: E402
import common as C  # noqa: E402
import hair  # noqa: E402
from rh import motions  # noqa: E402
from rh.pose import Skeleton  # noqa: E402


def opt(args, name, default):
    return args[args.index(name) + 1] if name in args else default


def main():
    a = C.args()
    clip, out = a[0], a[1]
    mesh = opt(a, "--mesh", os.path.join(C.BUILD, "human_dev.npz"))
    res = tuple(int(v) for v in opt(a, "--res", "1280x720").split("x"))
    samples = int(opt(a, "--samples", "48"))
    C.reset_scene()
    rig, body, eyes = character.build_character(mesh)
    hp = os.path.join(C.BUILD, "hair.npz")
    if os.path.exists(hp):
        hair.load_hair(hp, parent=rig, parent_bone="head", groups=hair.EEVEE_GROUPS)
    try:
        import clothes  # noqa: F401  (built by the wardrobe pipeline)
        clothes.load_clothes(rig=rig, body=body, ratio=0.3)
    except Exception as ex:  # clothes are optional while they are being built
        print("no clothes:", ex)
    skel = Skeleton(os.path.join(C.BUILD, "rig.json"))
    frames_data, cam_spec = motions.CLIPS[clip](skel)
    n = len(frames_data)
    fr = np.arange(1, n + 1)
    anim.key_rig(rig, fr, [f.quats for f in frames_data], [f.root_loc for f in frames_data], name=clip)
    targets = [f.extra.get("cam", f.extra.get("pelvis", np.array([0, 0, 1.0]))) for f in frames_data]
    offset = cam_spec.get("offset", (2.6, -3.4, 0.35))
    anim.outdoor_stage(sun_elev=34.0, sun_az=anim.key_light_azimuth(offset), strength=0.85)
    anim.follow_camera("Cam", fr, targets, offset, lens=cam_spec.get("lens", 40), smooth=cam_spec.get("smooth", 10))
    anim.setup_eevee(res=res, samples=samples)
    rng = opt(a, "--frames", None)
    f0, f1 = (int(x) for x in rng.split(":")) if rng else (1, n)
    if "--stills" in a:                      # --stills 1,9,17 <prefix>
        prefix = a[a.index("--stills") + 2]
        for k in [int(x) for x in opt(a, "--stills", "1").split(",")]:
            bpy.context.scene.frame_set(k)
            C.render(f"{prefix}_{k:03d}.png")
        return
    anim.render_mp4(out, f0, f1)


main()
