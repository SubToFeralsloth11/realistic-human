"""Clay renders of a meshed .npz from several angles.

blender -b --factory-startup --python blender/clay.py -- <mesh.npz> <out_prefix> <preset> [wb|cycles] [views]
presets: body, head, face, hand_l, foot_l, torso; views: comma list of yaw angles (default per preset)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

PRESETS = {
    #           target              lens  dist  res          yaws
    "body":   ((0.0, 0.0, 0.93),     50, 2.9, (1200, 1500), "0,35,90,180"),
    "torso":  ((0.0, 0.0, 1.28),     70, 1.9, (1200, 1500), "0,35,90,180"),
    "head":   ((0.0, -0.03, 1.665),  100, 0.95, (1200, 1500), "0,30,60,90"),
    "face":   ((0.0, -0.06, 1.66),   135, 0.75, (1500, 1200), "0,25,55,90"),
    "hand_l": ((0.62, -0.08, 0.98),  90, 0.75, (1500, 1200), "0,60,120,200"),
    "test":   ((0.03, 0.0, 0.12),    50, 0.8, (1200, 1200), "0,30,90"),
    "foot_l": ((0.11, -0.06, 0.07),  80, 0.95, (1500, 1200), "20,70,140,260"),
}


def main():
    a = C.args()
    npz, out, preset = a[0], a[1], a[2]
    engine = a[3] if len(a) > 3 else "wb"
    target, lens, dist, res, yaws = PRESETS[preset]
    if len(a) > 4:
        yaws = a[4]
    C.reset_scene()
    obj = C.load_npz_mesh(npz, "Body")
    lm_path = os.path.join(C.BUILD, "landmarks.json")
    if os.path.exists(lm_path):
        import json
        lm = json.load(open(lm_path))
        for key in ("eye_l", "eye_r"):
            C.bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=lm["eye_radius"], location=lm[key])
            C.bpy.ops.object.shade_smooth()
    if engine == "cycles":
        C.setup_cycles(samples=96, res=res)
        obj.data.materials.append(C.clay_material())
        C.world_color((0.035, 0.035, 0.04))
        C.studio_lights(target=(0, -0.2, target[2]), scale=1.0 if preset in ("body", "torso") else 0.6)
        C.ground()
    else:
        C.setup_workbench(res=res)
        C.world_color((0.08, 0.08, 0.09))
        C.bpy.context.scene.world.color = (0.08, 0.08, 0.09)
    cam = C.add_camera((0, -dist, target[2]), target, lens=lens)
    for yaw in [float(y) for y in yaws.split(",")]:
        pitch = 4.0 if preset in ("head", "face") else 2.0
        C.orbit_camera(cam, target, dist, yaw, pitch)
        C.render(f"{out}_{preset}_{int(yaw):03d}.png")


main()
