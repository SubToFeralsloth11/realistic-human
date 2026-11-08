"""Procedural skin shader (Cycles random-walk skin SSS; EEVEE compatible).

Layers, all generated from object-space noise (no image files):
  albedo   base tone + low-frequency mottling + sparse freckles; region tints from mesh attributes
           (m_red: cheeks/nose/ears/knuckles, m_lip, m_beard shadow, m_lid, m_nail, m_palm)
  rough    0.45 base, oilier T-zone (m_oily), smoother lips, glossy nails
  normal   pores (Voronoi F1 ~0.45 mm), micro furrows (distance-to-edge cells ~3 mm), fine noise
Attributes that are missing on a mesh read as 0, so the shader works on any mesh.
"""
import bpy

SKIN_BASE = (0.47, 0.285, 0.205)      # linear albedo, light-medium skin

class G:
    """Tiny node-graph helper."""

    def __init__(self, nt):
        self.nt = nt
        self.x = 0

    def n(self, kind, loc=None, **props):
        node = self.nt.nodes.new(kind)
        if loc is not None:
            node.location = loc
        for k, v in props.items():
            if k in node.inputs:
                node.inputs[k].default_value = v
            else:
                setattr(node, k, v)
        return node

    def link(self, a, b):
        self.nt.links.new(a, b)

    def math(self, op, a, b=None, loc=None, clamp=False):
        m = self.n("ShaderNodeMath", loc, operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                self.link(v, m.inputs[i])
        return m.outputs[0]

    def mix(self, fac, a, b, blend="MIX", loc=None):
        m = self.n("ShaderNodeMix", loc, data_type="RGBA", blend_type=blend)
        for sock, v in ((m.inputs[0], fac), (m.inputs[6], a), (m.inputs[7], b)):
            if isinstance(v, (int, float)):
                sock.default_value = v
            elif isinstance(v, tuple):
                sock.default_value = (*v, 1.0) if len(v) == 3 else v
            else:
                self.link(v, sock)
        return m.outputs[2]

    def attr(self, name, loc=None):
        a = self.n("ShaderNodeAttribute", loc, attribute_name=name, attribute_type="GEOMETRY")
        return a.outputs["Fac"]


def skin_material(name="Skin", base=SKIN_BASE):
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    nt.nodes.clear()
    g = G(nt)
    out = g.n("ShaderNodeOutputMaterial", (1800, 0))
    bsdf = g.n("ShaderNodeBsdfPrincipled", (1500, 0))
    bsdf.subsurface_method = "RANDOM_WALK_SKIN"
    g.link(bsdf.outputs[0], out.inputs[0])
    tc = g.n("ShaderNodeTexCoord", (-1600, 0))
    pos = tc.outputs["Object"]

    # ---------------- albedo
    mott = g.n("ShaderNodeTexNoise", (-1200, 400), Scale=14.0, Detail=4.0, Roughness=0.55)
    g.link(pos, mott.inputs["Vector"])
    mott2 = g.n("ShaderNodeTexNoise", (-1200, 200), Scale=55.0, Detail=3.0, Roughness=0.5)
    g.link(pos, mott2.inputs["Vector"])
    mot = g.math("MULTIPLY", g.math("SUBTRACT", mott.outputs["Fac"], 0.5), 0.9)
    mot2 = g.math("MULTIPLY", g.math("SUBTRACT", mott2.outputs["Fac"], 0.5), 0.5)
    hue = g.math("ADD", mot, mot2)
    col = g.mix(g.math("ADD", hue, 0.5, clamp=True), (base[0] * 1.03, base[1] * 0.93, base[2] * 0.95),
                (base[0] * 0.97, base[1] * 1.04, base[2] * 1.05))
    # freckles / tiny moles: sparse dark voronoi cells
    fr = g.n("ShaderNodeTexVoronoi", (-1200, 0), feature="F1", Scale=420.0, Randomness=1.0)
    g.link(pos, fr.inputs["Vector"])
    frs = g.n("ShaderNodeSeparateColor", (-1000, 0))
    g.link(fr.outputs["Color"], frs.inputs[0])
    sel = g.math("GREATER_THAN", frs.outputs[0], 0.965)
    spot = g.math("LESS_THAN", fr.outputs["Distance"], 0.22)
    frk = g.math("MULTIPLY", sel, spot)
    col = g.mix(g.math("MULTIPLY", frk, 0.45), col, (base[0] * 0.55, base[1] * 0.42, base[2] * 0.35))
    # region tints
    red = g.attr("m_red", (-600, -200))
    col = g.mix(g.math("MULTIPLY", red, 0.55), col, (base[0] * 1.05, base[1] * 0.70, base[2] * 0.72))
    lid = g.attr("m_lid", (-600, -300))
    col = g.mix(g.math("MULTIPLY", lid, 0.45), col, (base[0] * 0.86, base[1] * 0.70, base[2] * 0.72))
    beard = g.attr("m_beard", (-600, -400))
    col = g.mix(g.math("MULTIPLY", beard, 0.38), col, (base[0] * 0.66, base[1] * 0.62, base[2] * 0.68))
    lip = g.attr("m_lip", (-600, -500))
    col = g.mix(g.math("MULTIPLY", lip, 0.9), col, (0.40, 0.13, 0.11))
    palm = g.attr("m_palm", (-600, -600))
    col = g.mix(g.math("MULTIPLY", palm, 0.6), col, (base[0] * 1.02, base[1] * 0.86, base[2] * 0.80))
    nail = g.attr("m_nail", (-600, -700))
    col = g.mix(nail, col, (0.62, 0.42, 0.38))
    g.link(col, bsdf.inputs["Base Color"])

    # ---------------- subsurface
    bsdf.inputs["Subsurface Weight"].default_value = 1.0
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.45, 0.28)
    bsdf.inputs["Subsurface Scale"].default_value = 0.0045
    bsdf.inputs["Subsurface IOR"].default_value = 1.40
    bsdf.inputs["Subsurface Anisotropy"].default_value = 0.8

    # ---------------- micro normal
    pores = g.n("ShaderNodeTexVoronoi", (-1200, -900), feature="F1", Scale=2200.0, Randomness=0.9)
    g.link(pos, pores.inputs["Vector"])
    pore_h = g.math("POWER", g.math("MINIMUM", g.math("MULTIPLY", pores.outputs["Distance"], 2.2), 1.0), 0.5)
    cells = g.n("ShaderNodeTexVoronoi", (-1200, -1100), feature="DISTANCE_TO_EDGE", Scale=320.0, Randomness=1.0)
    g.link(pos, cells.inputs["Vector"])
    furrow = g.math("MINIMUM", g.math("MULTIPLY", cells.outputs["Distance"], 6.0), 1.0)
    fine = g.n("ShaderNodeTexNoise", (-1200, -1300), Scale=1800.0, Detail=3.0)
    g.link(pos, fine.inputs["Vector"])
    h = g.math("ADD", g.math("MULTIPLY", pore_h, 0.55), g.math("MULTIPLY", furrow, 0.30))
    h = g.math("ADD", h, g.math("MULTIPLY", fine.outputs["Fac"], 0.15))
    smooth = g.math("MAXIMUM", g.math("MAXIMUM", lip, nail), 0.0)
    strength = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, g.math("MULTIPLY", smooth, 0.8)), 0.55)
    bump = g.n("ShaderNodeBump", (1100, -600), Distance=0.00012)
    g.link(strength, bump.inputs["Strength"])
    g.link(h, bump.inputs["Height"])
    g.link(bump.outputs[0], bsdf.inputs["Normal"])

    # ---------------- roughness
    oily = g.attr("m_oily", (600, -300))
    rough = g.math("SUBTRACT", 0.47, g.math("MULTIPLY", oily, 0.13))
    rough = g.math("ADD", rough, g.math("MULTIPLY", g.math("SUBTRACT", pore_h, 0.5), 0.12))
    rough = g.math("SUBTRACT", rough, g.math("MULTIPLY", lip, 0.10))
    rough = g.math("SUBTRACT", rough, g.math("MULTIPLY", nail, 0.28))
    g.link(rough, bsdf.inputs["Roughness"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.5
    bsdf.inputs["IOR"].default_value = 1.40
    bsdf.inputs["Coat Weight"].default_value = 0.08
    bsdf.inputs["Coat Roughness"].default_value = 0.28
    bsdf.inputs["Sheen Weight"].default_value = 0.08      # vellus hair peach fuzz
    bsdf.inputs["Sheen Roughness"].default_value = 0.4
    return mat
