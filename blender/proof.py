"""Proof stills for several clips in one Blender session.

blender -b --factory-startup --python blender/proof.py -- <out_prefix> clip:f1,f2,... [clip:...] [--mesh m.npz]
Each clip gets its own action; stills are EEVEE 960x540 with a camera that frames the pelvis.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402

import anim  # noqa: E402
import character  # noqa: E402
import common as C  # noqa: E402
import hair  # noqa: E402
from rh import motions  # noqa: E402
from rh.pose import Skeleton  # noqa: E402


def main():
    a = C.args()
    prefix = a[0]
    mesh = a[a.index("--mesh") + 1] if "--mesh" in a else os.path.join(C.BUILD, "human_dev.npz")
    specs = [x for x in a[1:] if ":" in x and not x.startswith("--")]
    C.reset_scene()
    rig, body, eyes = character.build_character(mesh)
    hp = os.path.join(C.BUILD, "hair.npz")
    if os.path.exists(hp):
        hair.load_hair(hp, parent=rig, parent_bone="head", groups=hair.EEVEE_GROUPS)
    try:
        import clothes
        clothes.load_clothes(rig=rig, body=body, ratio=0.3)
    except Exception as ex:
        print("no clothes:", ex)
    anim.outdoor_stage()
    anim.setup_eevee(res=(960, 540), samples=24, blur=False)
    skel = Skeleton(os.path.join(C.BUILD, "rig.json"))
    for spec in specs:
        clip, frames = spec.split(":")
        data, cam = motions.CLIPS[clip](skel)
        n = len(data)
        fr = np.arange(1, n + 1)
        anim.key_rig(rig, fr, [f.quats for f in data], [f.root_loc for f in data], name=clip)
        targets = [f.extra.get("cam", f.extra.get("pelvis", np.array([0, 0, 1.0]))) for f in data]
        if "Cam" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["Cam"])
        anim.follow_camera("Cam", fr, targets, cam.get("offset", (2.6, -3.4, 0.35)), lens=cam.get("lens", 40),
                           smooth=cam.get("smooth", 10))
        for k in [int(x) for x in frames.split(",")]:
            bpy.context.scene.frame_set(min(k, n))
            C.render(f"{prefix}_{clip}_{k:03d}.png")


main()
