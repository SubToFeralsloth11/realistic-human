"""Teeth / gums / tongue objects (from build/teeth_*.npz) with enamel and wet mucosa shading."""
import os

import bpy

import common as C
from skin import G


def dentition_material():
    mat = bpy.data.materials.new("Dentition")
    nt = mat.node_tree
    nt.nodes.clear()
    g = G(nt)
    out = g.n("ShaderNodeOutputMaterial", (800, 0))
    p = g.n("ShaderNodeBsdfPrincipled", (500, 0))
    g.link(p.outputs[0], out.inputs[0])
    gum = g.attr("m_gum", (-400, 0))
    tc = g.n("ShaderNodeTexCoord", (-600, -200))
    noise = g.n("ShaderNodeTexNoise", (-400, -200), Scale=300.0, Detail=4.0)
    g.link(tc.outputs["Object"], noise.inputs["Vector"])
    enamel = g.mix(g.math("MULTIPLY", noise.outputs["Fac"], 0.25), (0.80, 0.76, 0.66), (0.70, 0.64, 0.52))
    col = g.mix(gum, enamel, (0.50, 0.14, 0.13))
    g.link(col, p.inputs["Base Color"])
    g.link(g.math("ADD", 0.16, g.math("MULTIPLY", gum, 0.10)), p.inputs["Roughness"])
    p.inputs["Subsurface Weight"].default_value = 0.6
    p.inputs["Subsurface Radius"].default_value = (1.0, 0.5, 0.35)
    p.inputs["Subsurface Scale"].default_value = 0.0012
    p.inputs["Coat Weight"].default_value = 0.35
    p.inputs["Coat Roughness"].default_value = 0.08
    return mat


def load_teeth(rig=None):
    mat = dentition_material()
    objs = []
    for name, bone in (("teeth_upper", "head"), ("teeth_lower", "jaw")):
        path = os.path.join(C.BUILD, name + ".npz")
        if not os.path.exists(path):
            continue
        ob = C.load_npz_mesh(path, name)
        ob.data.materials.append(mat)
        if rig is not None:
            mw = ob.matrix_world.copy()
            ob.parent = rig
            ob.parent_type = "BONE"
            ob.parent_bone = bone
            ob.matrix_world = mw
        objs.append(ob)
    return objs
