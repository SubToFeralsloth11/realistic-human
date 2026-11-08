"""Cycles look-dev portrait of the head: skin shader, eyes, three-point softbox lighting.

blender -b --factory-startup --python blender/portrait.py -- <mesh.npz> <out_prefix> [yaws] [samples]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import common as C  # noqa: E402
import eye  # noqa: E402
import hair  # noqa: E402
import skin  # noqa: E402


def lights(target):
    t = Vector(target)
    C.area_light("Key", t + Vector((-1.0, -1.25, 0.55)), t, 55, 1.2, (1.0, 0.94, 0.86))
    C.area_light("Fill", t + Vector((1.4, -0.9, 0.05)), t, 6, 1.0, (0.86, 0.92, 1.0))
    C.area_light("Rim", t + Vector((0.6, 1.2, 0.7)), t, 45, 0.5, (1.0, 0.96, 0.9))
    C.area_light("Top", t + Vector((0.0, 0.2, 1.4)), t, 6, 1.5, (1.0, 1.0, 1.0))


def main():
    a = C.args()
    npz, out = a[0], a[1]
    yaws = [float(y) for y in (a[2] if len(a) > 2 else "0,35,90").split(",")]
    samples = int(a[3]) if len(a) > 3 else 256
    C.reset_scene()
    body = C.load_npz_mesh(npz, "Body")
    body.data.materials.append(skin.skin_material())
    lm = json.load(open(os.path.join(C.BUILD, "landmarks.json")))
    inner_mat, cornea_mat = eye.iris_material(), eye.cornea_material()
    for key in ("eye_l", "eye_r"):
        eye.build_eye(key, lm[key], lm["eye_radius"], mats={"inner": inner_mat, "cornea": cornea_mat})
    hair_path = os.path.join(C.BUILD, "hair.npz")
    if os.path.exists(hair_path):
        hair.load_hair(hair_path)
    scene = C.setup_cycles(samples=samples, res=(1200, 1500))
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 12
    scene.cycles.glossy_bounces = 6
    C.world_color((0.018, 0.018, 0.02))
    target = (0.0, -0.075, 1.655)
    lights(target)
    cam = C.add_camera((0, -1.0, 1.66), target, lens=105)
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = 5.6
    for yaw in yaws:
        C.orbit_camera(cam, target, 1.05, yaw, 3.0)
        focus = Vector(lm["eye_l"] if yaw >= 0 else lm["eye_r"])   # camera at +x sees his left eye first
        cam.data.dof.focus_distance = (cam.location - focus).length
        C.render(f"{out}_{int(yaw):03d}.png")


main()
