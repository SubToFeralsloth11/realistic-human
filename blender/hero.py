"""Hero stills (Cycles) and a turntable (EEVEE) of the finished, dressed character.

blender -b --factory-startup --python blender/hero.py -- <out_dir> [--mesh build/human.npz]
        [--shots full,portrait,eyes,aim,hand] [--samples 256] [--turntable 96]
"""
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import anim  # noqa: E402
import character  # noqa: E402
import common as C  # noqa: E402
import hair  # noqa: E402
from rh import gunplay, motions  # noqa: E402
from rh.pose import Skeleton  # noqa: E402


def opt(a, name, default):
    return a[a.index(name) + 1] if name in a else default


def cyclorama(color=(0.055, 0.055, 0.058)):
    """Seamless studio backdrop: floor curving up into a back wall."""
    import bmesh
    me = bpy.data.meshes.new("Cyc")
    bm = bmesh.new()
    prof = [(0.0, y) for y in np.linspace(-6.0, 1.2, 8)] + \
           [(1.5 - 1.5 * math.cos(a), 1.2 + 1.5 * math.sin(a)) for a in np.linspace(0.1, math.pi / 2, 10)] + \
           [(z, 2.7) for z in np.linspace(1.6, 6.0, 5)]
    rows = []
    for x in np.linspace(-7.0, 7.0, 15):
        rows.append([bm.verts.new((x, y, z)) for z, y in prof])
    for r0, r1 in zip(rows[:-1], rows[1:]):
        for i in range(len(prof) - 1):
            bm.faces.new((r0[i], r1[i], r1[i + 1], r0[i + 1]))
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("Cyclorama", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.modifiers.new("sub", "SUBSURF").levels = 2
    me.shade_smooth()
    mat = C.clay_material(color, 0.75, "Backdrop")
    ob.data.materials.append(mat)
    return ob


def studio(target):
    """Three-point softbox light at the key:fill ratio of the look-dev portrait (~8:1); a bright
    fill and a light backdrop flattened the skin to pale pink."""
    t = Vector(target)
    C.area_light("Key", t + Vector((-1.9, -2.3, 1.2)), t, 230, 1.8, (1.0, 0.94, 0.86))
    C.area_light("Fill", t + Vector((2.4, -1.8, 0.2)), t, 16, 2.0, (0.86, 0.92, 1.0))
    C.area_light("Rim", t + Vector((1.4, 2.0, 1.4)), t, 180, 0.8, (1.0, 0.96, 0.9))
    C.area_light("Rim2", t + Vector((-1.6, 1.8, 1.0)), t, 70, 0.8, (0.92, 0.96, 1.0))
    C.world_color((0.012, 0.012, 0.014), 1.0)


def pose_rig(rig, frame, lid_drop=8.0):
    """Key one pose. The upper lids come down `lid_drop` degrees (about their local X, the closing
    axis) so they cover the top 1-2 mm of the iris like relaxed lids do; fully open lids stare."""
    quats = dict(frame.quats)
    c, s = math.cos(math.radians(-lid_drop) / 2), math.sin(math.radians(-lid_drop) / 2)
    for side in ("l", "r"):
        b = f"lid_upper_{side}"
        if b in quats:
            w, x, y, z = quats[b]
            quats[b] = (w * c - x * s, w * s + x * c, y * c + z * s, z * c - y * s)
    anim.key_rig(rig, [1], [quats], [frame.root_loc], name="hero_pose")
    bpy.context.scene.frame_set(1)


def main():
    a = C.args()
    out = a[0]
    os.makedirs(out, exist_ok=True)
    mesh = opt(a, "--mesh", os.path.join(C.BUILD, "human.npz"))
    shots = opt(a, "--shots", "full,portrait,eyes,aim,hand").split(",")
    samples = int(opt(a, "--samples", "256"))
    C.reset_scene()
    rig, body, eyes = character.build_character(mesh)
    hp = os.path.join(C.BUILD, "hair.npz")
    if os.path.exists(hp):
        hair.load_hair(hp, parent=rig, parent_bone="head")
    try:
        import clothes
        clothes.load_clothes(rig=rig, body=body, ratio=0.6)
    except Exception as ex:
        print("no clothes:", ex)
    skel = Skeleton(os.path.join(C.BUILD, "rig.json"))
    relaxed = motions.idle(skel, 2.0)[40]
    cyclorama()
    studio((0.0, 0.0, 1.0))
    scene = C.setup_cycles(samples=samples, res=(1200, 1500))
    scene.cycles.max_bounces = 10
    scene.cycles.transmission_bounces = 10
    scene.cycles.use_adaptive_sampling = True
    cam = C.add_camera((0, -4, 1.2), (0, 0, 1.0), lens=50)
    for shot in shots:
        if shot not in ("full", "portrait", "eyes", "aim", "hand"):
            continue
        if shot in ("aim", "hand"):
            import pistol
            if "Pistol" not in bpy.data.objects:
                parts = pistol.build_pistol()
            seq = gunplay.pistol_sequence(skel, C.ROOT)
            fr = seq[int(2.6 * motions.FPS)]
            pose_rig(rig, fr)
            bpy.data.objects["Pistol"].matrix_world = Matrix(fr.extra["gun"].tolist())
        else:
            pose_rig(rig, relaxed)
        if shot == "full":
            scene.render.resolution_x, scene.render.resolution_y = 1200, 1600
            cam.data.lens = 55
            C.orbit_camera(cam, (0.0, 0.0, 0.95), 3.7, 28, 9)     # camera near eye height, not under the chin
            cam.data.dof.use_dof = False
        elif shot == "portrait":
            scene.render.resolution_x, scene.render.resolution_y = 1200, 1500
            cam.data.lens = 105
            C.orbit_camera(cam, (0.0, -0.06, 1.63), 1.15, 22, 3)
            cam.data.dof.use_dof = True
            cam.data.dof.aperture_fstop = 4.0
            cam.data.dof.focus_distance = (cam.location - rig.matrix_world @ rig.pose.bones["eye_l"].head).length
        elif shot == "eyes":
            scene.render.resolution_x, scene.render.resolution_y = 1600, 900
            cam.data.lens = 135
            C.orbit_camera(cam, (0.0, -0.075, 1.683), 0.42, 12, 2)
            cam.data.dof.use_dof = True
            cam.data.dof.aperture_fstop = 11.0                    # f/5.6 left ~2 mm in focus at 42 cm
            cam.data.dof.focus_distance = (cam.location - rig.matrix_world @ rig.pose.bones["eye_l"].head).length - 0.010
        elif shot == "aim":
            scene.render.resolution_x, scene.render.resolution_y = 1600, 1000
            cam.data.lens = 50
            C.orbit_camera(cam, (0.0, -0.25, 1.30), 2.6, 52, 4)
            cam.data.dof.use_dof = False
        elif shot == "hand":
            scene.render.resolution_x, scene.render.resolution_y = 1600, 1000
            cam.data.lens = 100
            g = bpy.data.objects["Pistol"].matrix_world.translation
            C.orbit_camera(cam, g, 0.75, 70, 10)
            cam.data.dof.use_dof = True
            cam.data.dof.aperture_fstop = 5.6
            cam.data.dof.focus_distance = (cam.location - g).length
        C.render(os.path.join(out, f"hero_{shot}.png"))
    nturn = int(opt(a, "--turntable", "0"))
    if nturn:
        # the character spins on a turntable in front of a fixed camera (an orbiting camera ends
        # up behind the cyclorama wall for the back half)
        pose_rig(rig, relaxed)
        for ob in bpy.data.objects:
            if ob.name.startswith("Pistol") or (ob.parent and ob.parent.name == "Pistol"):
                ob.hide_render = True
            if "stubble" in ob.name.lower():                   # EEVEE draws stubble too thick
                ob.hide_render = True
        spin = bpy.data.objects.new("Turntable", None)
        bpy.context.scene.collection.objects.link(spin)
        mw = rig.matrix_world.copy()
        rig.parent = spin
        rig.matrix_world = mw
        frames = np.arange(1, nturn + 1)
        anim.key_channels(spin, frames, {("rotation_euler", 2): [2 * math.pi * (k - 1) / nturn for k in frames]})
        anim.setup_eevee(res=(1080, 1350), samples=64, blur=False)
        cam.data.lens = 50
        cam.data.dof.use_dof = False
        C.orbit_camera(cam, (0.0, 0.0, 0.92), 4.0, 0, 2)
        anim.render_mp4(os.path.join(out, "turntable.mp4"), 1, nturn)


main()
