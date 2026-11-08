"""Assemble the rigged character in the current Blender scene.

build_character() loads build/human.npz (mesh + masks + top-4 weights) and build/rig.json, makes
the armature, vertex groups, armature modifier, skin material and the two eyes on their bones.
Run directly to save build/character.blend:
  blender -b --factory-startup --python blender/character.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import common as C  # noqa: E402
import eye  # noqa: E402
import skin  # noqa: E402


def build_armature(rig, name="Rig"):
    arm = bpy.data.armatures.new(name)
    ob = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    for bname, d in rig.items():
        b = arm.edit_bones.new(bname)
        b.head = Vector(d["head"])
        b.tail = Vector(d["tail"])
        b.align_roll(Vector(d["z"]))
        eb[bname] = b
    for bname, d in rig.items():
        if d["parent"]:
            child, parent = eb[bname], eb[d["parent"]]
            child.parent = parent
            child.use_connect = (child.head - parent.tail).length < 1e-5
    for bname in ("root", "eye_l", "eye_r", "jaw"):
        if bname in eb:
            eb[bname].use_deform = bname not in ("root", "eye_l", "eye_r")
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.display_type = "STICK"
    ob.show_in_front = True
    # forearm twist: half of the hand's roll about the forearm axis
    for s in ("l", "r"):
        pb = ob.pose.bones[f"lowerarm_twist_01_{s}"]
        c = pb.constraints.new("COPY_ROTATION")
        c.target = ob
        c.subtarget = f"hand_{s}"
        c.use_x = c.use_z = False
        c.target_space = c.owner_space = "LOCAL"
        c.mix_mode = "ADD"
        c.influence = 0.5
    return ob


def assign_weights(obj, names, idx, val):
    names = [str(n) for n in names]
    groups = {}
    for slot in range(idx.shape[1]):
        w = val[:, slot]
        ok = w > 1e-3
        for bid in np.unique(idx[ok, slot]):
            sel = ok & (idx[:, slot] == bid)
            vi = np.nonzero(sel)[0]
            groups.setdefault(int(bid), []).append((vi, w[sel]))
    for bid, parts in groups.items():
        vg = obj.vertex_groups.get(names[bid]) or obj.vertex_groups.new(name=names[bid])
        vi = np.concatenate([p[0] for p in parts])
        wv = np.concatenate([p[1] for p in parts])
        q = np.round(wv * 512).astype(np.int32)
        for level in np.unique(q):
            vg.add(vi[q == level].tolist(), float(level) / 512.0, "REPLACE")


def build_character(npz=None, with_eyes=True, material=True):
    npz = npz or os.path.join(C.BUILD, "human.npz")
    rig = json.load(open(os.path.join(C.BUILD, "rig.json")))
    data = np.load(npz)
    body = C.load_npz_mesh(npz, "Body")
    assign_weights(body, data["bone_names"], data["w_idx"], data["w_val"])
    if material:
        body.data.materials.append(skin.skin_material())
    rig_ob = build_armature(rig)
    body.parent = rig_ob
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = rig_ob
    mod.use_deform_preserve_volume = False
    eyes = []
    if with_eyes:
        lm = json.load(open(os.path.join(C.BUILD, "landmarks.json")))
        inner_mat, cornea_mat = eye.iris_material(), eye.cornea_material()
        for key in ("eye_l", "eye_r"):
            parts = eye.build_eye(key, lm[key], lm["eye_radius"], mats={"inner": inner_mat, "cornea": cornea_mat})
            for ob in parts:
                mw = ob.matrix_world.copy()
                ob.parent = rig_ob
                ob.parent_type = "BONE"
                ob.parent_bone = key
                ob.matrix_world = mw
            eyes += parts
    return rig_ob, body, eyes


if __name__ == "__main__":
    C.reset_scene()
    build_character()
    out = os.path.join(C.BUILD, "character.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print("SAVED", out)
