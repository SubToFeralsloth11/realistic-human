"""Render the pistol sequence (draw, shoot, reload, put back) with every prop keyed.

blender -b --factory-startup --python blender/pistol_clip.py -- <out_dir> [--mesh m.npz] [--res 1280x720]
        [--samples 48] [--stills 30,80,...]
Writes <out_dir>/pistol_full.mp4 and the four segments (draw/shoot/reload/holster) cut from it.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix  # noqa: E402

import anim  # noqa: E402  (puts src/ on sys.path)
import character  # noqa: E402
import common as C  # noqa: E402
import hair  # noqa: E402
import pistol  # noqa: E402
from rh import gunplay  # noqa: E402
from rh.pose import Skeleton  # noqa: E402


def opt(a, name, default):
    return a[a.index(name) + 1] if name in a else default


def unparent(ob):
    mw = ob.matrix_world.copy()
    ob.parent = None
    ob.matrix_world = mw


def pocket_bulge(jeans, skel, data, frames):
    """Shape key that pushes the denim out over the pocketed pistol, keyed by how far the gun
    is from its seat in the pocket. Computed in rest space (garments carry no sole lift)."""
    origin, _, R = gunplay.load_pocket(C.ROOT)
    origin = origin - np.array([0.0, 0.0, gunplay.LIFT])
    me = jeans.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    mw = np.array(jeans.matrix_world, float)
    V = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]  # rest-space positions
    q = (V - origin) @ R                         # gun-local coords: x outward, y muzzle, z sight
    # silhouette of the slide + grip in the gun's y-z plane, 14 mm soft edge
    dy = np.maximum(np.maximum(-0.073 - q[:, 1], q[:, 1] - 0.112), 0)
    dz = np.maximum(np.maximum(-0.095 - q[:, 2], q[:, 2] - 0.040), 0)
    inside = np.clip(1 - np.hypot(dy, dz) / 0.014, 0, 1)
    inside = inside * inside * (3 - 2 * inside)
    nrm = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("normal", nrm)
    facing = (nrm.reshape(-1, 3) @ mw[:3, :3].T) @ R[:, 0] > 0.2
    push = np.clip(0.0165 - q[:, 0], 0, 0.02) * inside * facing * (q[:, 0] > -0.02)
    if not push.any():
        return
    out = np.linalg.solve(mw[:3, :3], R[:, 0])
    if me.shape_keys is None:
        jeans.shape_key_add(name="Basis", from_mix=False)
    sk = jeans.shape_key_add(name="GunInPocket", from_mix=False)
    new = co.reshape(-1, 3) + np.outer(push, out).astype(np.float32)
    sk.data.foreach_set("co", new.ravel())
    seat = gunplay.inv(gunplay.bone_M(*skel.fk({}, np.array([0.0, 0.0, gunplay.LIFT])), "thigh_r")) @ \
        gunplay.M(R, origin + np.array([0.0, 0.0, gunplay.LIFT]))
    vals = []
    for f in data:
        rel = gunplay.inv(gunplay.bone_M(f.W, f.H, "thigh_r")) @ f.extra["gun"]
        vals.append(float(np.clip(1 - np.linalg.norm(rel[:3, 3] - seat[:3, 3]) / 0.10, 0, 1)))
    cb = anim._channelbag(me.shape_keys, "pocket_bulge")
    anim._fcurve(cb, 'key_blocks["GunInPocket"].value', 0, np.asarray(frames, np.float32), np.asarray(vals, np.float32))
    print("BULGE verts", int((push > 0).sum()), "max mm", round(1000 * float(push.max()), 1))


def main():
    a = C.args()
    out_dir = a[0]
    os.makedirs(out_dir, exist_ok=True)
    mesh = opt(a, "--mesh", os.path.join(C.BUILD, "human.npz"))
    res = tuple(int(v) for v in opt(a, "--res", "1280x720").split("x"))
    C.reset_scene()
    rig, body, eyes = character.build_character(mesh)
    hp = os.path.join(C.BUILD, "hair.npz")
    if os.path.exists(hp):
        hair.load_hair(hp, parent=rig, parent_bone="head")
    garments = {}
    try:
        import clothes
        garments = clothes.load_clothes(rig=rig, body=body)
    except Exception as ex:
        print("no clothes:", ex)
    parts = pistol.build_pistol()
    skel = Skeleton(os.path.join(C.BUILD, "rig.json"))
    data = gunplay.pistol_sequence(skel, C.ROOT)
    n = len(data)
    fr = np.arange(1, n + 1)
    anim.key_rig(rig, fr, [f.quats for f in data], [f.root_loc for f in data], name="pistol")
    if "Jeans" in garments:
        pocket_bulge(garments["Jeans"], skel, data, fr)
    # gun root in world space
    anim.key_object(parts["root"], fr, [Matrix(f.extra["gun"].tolist()) for f in data], name="gun")
    # local mechanisms (relative to their rest local transforms)
    slide, trig, rel, stop = parts["slide"], parts["trigger"], parts["mag_release"], parts["slide_stop"]
    base = {ob.name: (ob.location.copy(), ob.rotation_euler.copy()) for ob in (slide, trig, rel, stop, parts["barrel"])}
    anim.key_channels(slide, fr, {("location", 1): [base[slide.name][0][1] + f.extra["slide"] for f in data]})
    lock = [1.0 if f.extra["slide"] <= -0.0279 else 0.0 for f in data]
    bar = parts["barrel"]
    anim.key_channels(bar, fr, {("location", 1): [base[bar.name][0][1] - 0.003 * k for k in lock],
                                ("rotation_euler", 0): [base[bar.name][1][0] + np.radians(5.0) * k for k in lock]})
    anim.key_channels(trig, fr, {("rotation_euler", 0): [base[trig.name][1][0] + np.radians(f.extra["trigger"]) for f in data]})
    anim.key_channels(rel, fr, {("location", 0): [base[rel.name][0][0] + f.extra["mag_release"] for f in data]})
    anim.key_channels(stop, fr, {("rotation_euler", 0): [base[stop.name][1][0] + np.radians(f.extra["slide_stop"]) for f in data]})
    # muzzle flash: visible on shot frames only, random roll, slightly random size
    flash = parts["muzzle_flash"]
    flash.hide_render = flash.hide_viewport = False
    rng = np.random.default_rng(5)
    on = [1.0 if f.extra["flash"] else 0.0 for f in data]
    anim.key_channels(flash, fr, {("scale", 0): [o * rng.uniform(0.9, 1.2) for o in on],
                                  ("scale", 1): [o * rng.uniform(0.9, 1.3) for o in on],
                                  ("scale", 2): [o * rng.uniform(0.9, 1.2) for o in on],
                                  ("rotation_euler", 1): [rng.uniform(0, 6.28) for _ in on]}, constant=True)
    # magazines: the spent one and the spare, both in world space
    mag1 = parts["magazine"]
    unparent(mag1)
    anim.key_object(mag1, fr, [Matrix(f.extra["mag1"].tolist()) for f in data], name="mag1")
    mag2 = mag1.copy()
    bpy.context.scene.collection.objects.link(mag2)
    mag2.name = "Magazine_Spare"
    mag2.animation_data_clear()
    anim.key_object(mag2, fr, [Matrix(f.extra["mag2"].tolist()) for f in data], name="mag2")
    # casings
    cas = parts["casing"]
    for k in range(len(gunplay.SHOTS)):
        c = cas.copy()
        bpy.context.scene.collection.objects.link(c)
        c.parent = None
        c.hide_render = c.hide_viewport = False
        mats, vis = [], []
        for f in data:
            m = f.extra["casings"][k]
            vis.append(m is not None)
            mats.append(Matrix(m.tolist()) if m is not None else Matrix.Identity(4))
        anim.key_object(c, fr, mats, name=f"casing{k}", visible=vis)
    offset = (1.55, -2.75, 0.12)
    anim.outdoor_stage(sun_elev=34.0, sun_az=anim.key_light_azimuth(offset), strength=0.85)
    targets = [f.extra["cam"] for f in data]
    anim.follow_camera("Cam", fr, targets, offset, lens=40, smooth=40)
    anim.setup_eevee(res=res, samples=int(opt(a, "--samples", "48")))
    if "--stills" in a:
        for k in [int(x) for x in opt(a, "--stills", "1").split(",")]:
            bpy.context.scene.frame_set(k)
            C.render(os.path.join(out_dir, f"pistol_{k:03d}.png"))
        return
    full = os.path.join(out_dir, "pistol_full.mp4")
    anim.render_mp4(full, 1, n)
    for name, (t0, t1) in gunplay.SEGMENTS.items():
        seg = os.path.join(out_dir, f"pistol_{name}.mp4")
        subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}",
                        "-i", full, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", seg],
                       check=True)
        print("MP4", seg)


main()
