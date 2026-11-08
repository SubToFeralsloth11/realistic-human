"""Hair curves objects from build/hair.npz with Principled Hair BSDF (Chiang) materials."""
import os

import bpy
import numpy as np

STYLE = {
    #          melanin redness  roughness radial  random-color
    "scalp":   (0.90, 0.12, 0.32, 0.30, 0.12),
    "brows":   (0.90, 0.22, 0.34, 0.30, 0.10),
    "stubble": (0.86, 0.22, 0.40, 0.35, 0.18),
    "lashes":  (0.97, 0.15, 0.25, 0.30, 0.05),
}


def hair_material(name, melanin, redness, rough, radial, rand):
    mat = bpy.data.materials.new("Hair_" + name)
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    h = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    h.parametrization = "MELANIN"
    h.inputs["Melanin"].default_value = melanin
    h.inputs["Melanin Redness"].default_value = redness
    h.inputs["Roughness"].default_value = rough
    h.inputs["Radial Roughness"].default_value = radial
    h.inputs["Random Color"].default_value = rand
    h.inputs["Random Roughness"].default_value = 0.15
    h.inputs["Coat"].default_value = 0.0
    h.inputs["IOR"].default_value = 1.55
    # per-strand randomness for the Random input
    info = nt.nodes.new("ShaderNodeHairInfo")
    nt.links.new(info.outputs["Random"], h.inputs["Random"])
    out.target = "CYCLES"
    nt.links.new(h.outputs[0], out.inputs[0])
    # EEVEE: its Hair BSDF ignores the fibre colour on curves (dark hair renders blond), so the
    # EEVEE output is a Principled sheen over the equivalent melanin albedo with per-strand jitter
    tint = _melanin_rgb(melanin, redness)
    ev = nt.nodes.new("ShaderNodeOutputMaterial")
    ev.target = "EEVEE"
    pb = nt.nodes.new("ShaderNodeBsdfPrincipled")
    jitter = nt.nodes.new("ShaderNodeMapRange")
    jitter.inputs["To Min"].default_value = 1 - rand
    jitter.inputs["To Max"].default_value = 1 + rand
    nt.links.new(info.outputs["Random"], jitter.inputs["Value"])
    col = nt.nodes.new("ShaderNodeVectorMath")
    col.operation = "SCALE"
    col.inputs[0].default_value = [c * 0.6 for c in tint]
    nt.links.new(jitter.outputs["Result"], col.inputs["Scale"])
    nt.links.new(col.outputs["Vector"], pb.inputs["Base Color"])
    pb.inputs["Roughness"].default_value = 0.42
    pb.inputs["Specular IOR Level"].default_value = 0.35
    nt.links.new(pb.outputs[0], ev.inputs[0])
    return mat


def _melanin_rgb(melanin, redness):
    """Approximate albedo of a hair fibre with the given melanin (0..1) and redness (0..1)."""
    eu = melanin * (1 - redness)
    pheo = melanin * redness
    base = np.array([0.95, 0.80, 0.55]) * (1 - eu) ** 2.2 + np.array([0.02, 0.012, 0.008]) * eu
    red = np.array([0.55, 0.20, 0.08])
    col = base * (1 - 0.6 * pheo) + red * 0.6 * pheo * (1 - eu)
    return tuple(float(c) for c in np.clip(col * 1.6, 0.01, 1.0))


def curves_object(name, points, counts, radius):
    cv = bpy.data.hair_curves.new(name)
    cv.add_curves(counts.tolist())
    cv.position_data.foreach_set("vector", points.astype(np.float32).ravel())
    rad = cv.attributes.get("radius") or cv.attributes.new("radius", "FLOAT", "POINT")
    rad.data.foreach_set("value", radius.astype(np.float32))
    ob = bpy.data.objects.new(name, cv)
    bpy.context.scene.collection.objects.link(ob)
    return ob


LASH_BONES = {"lash_ul": "lid_upper_l", "lash_ll": "lid_lower_l", "lash_ur": "lid_upper_r", "lash_lr": "lid_lower_r"}


def load_hair(path, parent=None, parent_bone="head", groups=None):
    """One Curves object per group in the npz; lashes ride their eyelid bones when the rig has them."""
    data = np.load(path)
    names = sorted({k[:-len("_points")] for k in data.files if k.endswith("_points")})
    objs, mats = {}, {}
    for g in names:
        if groups and g not in groups:
            continue
        style_key = "lashes" if g.startswith("lash") else g
        if style_key not in mats:
            mats[style_key] = hair_material(style_key, *STYLE[style_key])
        ob = curves_object("Hair_" + g, data[f"{g}_points"], data[f"{g}_counts"], data[f"{g}_radius"])
        ob.data.materials.append(mats[style_key])
        if parent is not None:
            bone = LASH_BONES.get(g, parent_bone)
            if bone not in parent.data.bones:
                bone = parent_bone
            mw = ob.matrix_world.copy()
            ob.parent = parent
            ob.parent_type = "BONE"
            ob.parent_bone = bone
            ob.matrix_world = mw
        objs[g] = ob
    return objs
