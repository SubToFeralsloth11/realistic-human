"""Hair curves objects from build/hair.npz with Principled Hair BSDF (Chiang) materials."""
import os

import bpy
import numpy as np

STYLE = {
    #          melanin redness  roughness radial  random-color
    "scalp":   (0.86, 0.40, 0.30, 0.30, 0.12),
    "brows":   (0.90, 0.35, 0.34, 0.30, 0.10),
    "stubble": (0.88, 0.38, 0.40, 0.35, 0.15),
    "lashes":  (0.97, 0.20, 0.25, 0.30, 0.05),
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
    nt.links.new(h.outputs[0], out.inputs[0])
    return mat


def curves_object(name, points, counts, radius):
    cv = bpy.data.hair_curves.new(name)
    cv.add_curves(counts.tolist())
    cv.position_data.foreach_set("vector", points.astype(np.float32).ravel())
    rad = cv.attributes.get("radius") or cv.attributes.new("radius", "FLOAT", "POINT")
    rad.data.foreach_set("value", radius.astype(np.float32))
    ob = bpy.data.objects.new(name, cv)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def load_hair(path, parent=None, parent_bone="head", groups=None):
    data = np.load(path)
    objs = {}
    for g, style in STYLE.items():
        if groups and g not in groups:
            continue
        if f"{g}_points" not in data.files:
            continue
        ob = curves_object("Hair_" + g, data[f"{g}_points"], data[f"{g}_counts"], data[f"{g}_radius"])
        ob.data.materials.append(hair_material(g, *style))
        if parent is not None:
            mw = ob.matrix_world.copy()
            ob.parent = parent
            ob.parent_type = "BONE"
            ob.parent_bone = parent_bone
            ob.matrix_world = mw
        objs[g] = ob
    return objs
