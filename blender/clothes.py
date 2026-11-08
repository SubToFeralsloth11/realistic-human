"""Garments: load build/{shirt,jeans,shoes}.npz, build procedural cloth / leather materials.

load_clothes(rig=None, body=None) -> {"Shirt": obj, "Jeans": obj, "Shoes": obj}
  With rig and body: every garment is parented to the armature with an Armature modifier and gets
  the body's vertex-group weights through a Data Transfer modifier (nearest face, interpolated)
  that is applied once.

Preview (Cycles): blender -b --factory-startup --python blender/clothes.py -- preview <out_prefix> [samples]
  renders <out_prefix>_{front,34,back,pocket,shoe}.png of the dressed body (build/body_40.npz).
The shoes are modelled in body coordinates with the outsole at z = -SOLE_LIFT; final scenes lift
the character root by SOLE_LIFT so the soles stand on z = 0 (the preview does the same).

All patterns come from per-vertex attributes written by src/rh/clothes.py:
  u, v        chart coordinates in metres (weave / knit direction)
  seam*,pk*   signed distance (m) to the nearest seam polyline, arc length along it (seam_s) and the
              two stitch-row offsets st1/st2 of that seam (0 = no row) -> dashed topstitching
  hem, collar distance to hems / neckline; fold = fold height (cm, + = ridge); wear = fading mask
  side        1 on the outer face of the fabric, 0 on the back face
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import common as C  # noqa: E402
from skin import G  # noqa: E402

SOLE_LIFT = 0.012


# ----------------------------------------------------------------------------- node helpers
class N(G):
    """G plus a few vector/value conveniences."""

    def val(self, v):
        if isinstance(v, (int, float)):
            n = self.n("ShaderNodeValue")
            n.outputs[0].default_value = v
            return n.outputs[0]
        return v

    def lerp(self, f, a, b):
        """Scalar a + f * (b - a); a, b may be numbers or sockets."""
        return self.add(a, self.mul(f, self.sub(b, a))) if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) \
            else self.add(a, self.mul(f, b - a))

    def add(self, a, b): return self.math("ADD", a, b)
    def sub(self, a, b): return self.math("SUBTRACT", a, b)
    def mul(self, a, b): return self.math("MULTIPLY", a, b)
    def mx(self, a, b): return self.math("MAXIMUM", a, b)
    def mn(self, a, b): return self.math("MINIMUM", a, b)
    def absv(self, a): return self.math("ABSOLUTE", a)
    def frac(self, a): return self.math("FRACT", a)
    def sin(self, a): return self.math("SINE", a)

    def smooth(self, e0, e1, x):
        """smoothstep(e0, e1, x) (works for e0 > e1 as a falling edge)."""
        m = self.n("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        m.inputs["From Min"].default_value = e0
        m.inputs["From Max"].default_value = e1
        self.link(x, m.inputs["Value"])
        return m.outputs["Result"]

    def combine(self, x, y, z=0.0):
        c = self.n("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            if isinstance(v, (int, float)):
                c.inputs[i].default_value = v
            else:
                self.link(v, c.inputs[i])
        return c.outputs[0]

    def noise(self, vec, scale, detail=2.0, rough=0.5, dim="3D", w=None):
        n = self.n("ShaderNodeTexNoise", noise_dimensions=dim, Scale=scale, Detail=detail, Roughness=rough)
        if vec is not None:
            self.link(vec, n.inputs["Vector"])
        return n.outputs["Fac"]

    def obj(self):
        if not hasattr(self, "_tc"):
            self._tc = self.n("ShaderNodeTexCoord")
        return self._tc.outputs["Object"]

    def line(self, d, offset, half_w):
        """1 where |d - offset| < half_w (anti-aliased by a soft edge)."""
        x = self.absv(self.sub(d, offset))
        return self.smooth(half_w, half_w * 0.55, x)

    def dashes(self, s, period, duty):
        return self.smooth(duty + 0.04, duty - 0.04, self.frac(self.mul(s, 1.0 / period)))

    def stitch_rows(self, prefix, period, duty, half_w):
        """Mask of dashed topstitch rows for an attribute family (prefix, prefix_s, st1, st2)."""
        d = self.attr(prefix)
        s = self.attr(f"{prefix}_s")
        o1 = self.attr(f"{prefix}_o1")
        o2 = self.attr(f"{prefix}_o2")
        row = self.mx(self.mul(self.line(d, o1, half_w), self.math("GREATER_THAN", self.absv(o1), 1e-5)),
                      self.mul(self.line(d, o2, half_w), self.math("GREATER_THAN", self.absv(o2), 1e-5)))
        # per-stitch rounded profile: thread lies in short raised dashes
        dash = self.dashes(s, period, duty)
        return self.mul(row, dash), d

    def bsdf(self, target="ALL"):
        out = self.n("ShaderNodeOutputMaterial", target=target)
        b = self.n("ShaderNodeBsdfPrincipled")
        self.link(b.outputs[0], out.inputs[0])
        return b


def _new_mat(name):
    mat = bpy.data.materials.new(name)
    mat.node_tree.nodes.clear()
    return mat, N(mat.node_tree)


# ----------------------------------------------------------------------------- materials
def jersey_material(color=(0.115, 0.118, 0.122), name="CottonJersey"):
    """Heather cotton jersey: fibre-speckled colour, knit loops, coverstitched hems, rib collar."""
    mat, g = _new_mat(name)
    b = g.bsdf()
    u, v = g.attr("u"), g.attr("v")
    side = g.attr("side")
    # knit: wales (columns, ~0.75 mm) of V loops, courses ~0.6 mm
    kx = g.mul(u, 1.0 / 0.00075)
    ky = g.mul(v, 1.0 / 0.0006)
    lean = g.mul(g.absv(g.sub(g.frac(kx), 0.5)), 0.9)
    loop = g.absv(g.sin(g.mul(g.add(ky, lean), math.pi)))
    col_wale = g.absv(g.sin(g.mul(kx, math.pi)))
    knit = g.mul(g.math("POWER", col_wale, 0.6), g.add(0.55, g.mul(loop, 0.45)))
    # heather: dark/light fibres streaking along the courses
    pos = g.obj()
    fib = g.noise(g.n("ShaderNodeVectorMath", operation="MULTIPLY").outputs[0], 1.0)
    vm = fib.node.inputs["Vector"].links[0].from_node
    g.link(pos, vm.inputs[0])
    vm.inputs[1].default_value = (900.0, 900.0, 3200.0)
    fib.node.inputs["Detail"].default_value = 3.0
    heather = g.smooth(0.35, 0.72, fib)
    mott = g.noise(pos, 38.0, 3.0, 0.6)
    base = g.mix(heather, tuple(c * 0.70 for c in color), tuple(min(1.0, c * 1.55) for c in color))
    base = g.mix(g.mul(g.sub(mott, 0.5), 0.35), base, tuple(c * 0.8 for c in color), "MULTIPLY")
    # rib collar: 1x1 rib running across the band
    collar = g.attr("collar")
    in_collar = g.smooth(0.0262, 0.0245, collar)
    rib_u = g.attr("rib")
    rib = g.absv(g.sin(g.mul(rib_u, math.pi / 0.0011)))
    # coverstitch rows: hems (two rows ~6 mm apart) and the collar seam
    hem = g.attr("hem")
    cover = g.mx(g.line(hem, 0.0165, 0.00035), g.line(hem, 0.0225, 0.00035))
    cover = g.mx(cover, g.line(collar, 0.0262, 0.00035))
    cover = g.mx(cover, g.line(collar, 0.0300, 0.00035))
    cover = g.mul(cover, g.dashes(g.add(g.mul(u, 1.0), g.mul(v, 0.0)), 0.0026, 0.78))
    seam_rows, sd = g.stitch_rows("seam", 0.0028, 0.8, 0.0003)
    stitch = g.mx(cover, seam_rows)
    seam_line = g.line(sd, 0.0, 0.0005)
    # hem fold edge: slight dark crease where the folded hem ends
    hem_edge = g.line(hem, 0.0245, 0.0007)
    fold = g.attr("fold")
    shade = g.add(g.mul(seam_line, 0.35), g.mul(hem_edge, 0.2))
    col = g.mix(shade, base, tuple(c * 0.55 for c in color))
    col = g.mix(g.mul(stitch, 0.6), col, tuple(min(1, c * 1.15) for c in color))
    col = g.mix(g.mul(g.sub(1.0, side), 0.25), col, tuple(c * 0.85 for c in color))
    g.link(col, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.88
    b.inputs["Sheen Weight"].default_value = 0.55
    b.inputs["Sheen Roughness"].default_value = 0.35
    b.inputs["Sheen Tint"].default_value = (0.9, 0.9, 0.9, 1)
    b.inputs["Specular IOR Level"].default_value = 0.3
    # height: knit (body) or rib (collar), seams sunk, stitches proud, hem allowance raised
    h = g.lerp(in_collar, knit, rib)
    h = g.add(g.mul(h, 0.5), g.mul(stitch, 0.6))
    h = g.sub(h, g.mul(seam_line, 0.8))
    h = g.add(h, g.mul(g.smooth(0.0255, 0.0235, hem), 0.4))
    h = g.add(h, g.mul(fold, 0.0))
    bump = g.n("ShaderNodeBump", Strength=0.35, Distance=0.00045)
    g.link(h, bump.inputs["Height"])
    g.link(bump.outputs[0], b.inputs["Normal"])
    return mat


def denim_material(name="Denim"):
    """Indigo 3x1 right-hand twill with slub streaks, whiskers / honeycombs, orange-tan topstitching."""
    mat, g = _new_mat(name)
    b = g.bsdf("CYCLES")
    u, v = g.attr("u"), g.attr("v")
    side = g.attr("side")
    wear = g.attr("wear")
    fold = g.attr("fold")
    metal = g.attr("metal")
    bag = g.attr("bag")
    leather = g.attr("label")
    # twill: diagonal wales ~0.75 mm apart at ~62 degrees
    tw = g.frac(g.mul(g.add(g.mul(u, 0.88), g.mul(v, 0.47)), 1.0 / 0.00075))
    wale = g.smooth(0.0, 0.5, g.absv(g.sub(tw, 0.5)))
    # slub: warp yarn thickness varies along the warp (v) -> vertical streaks
    sl = g.combine(g.mul(u, 1400.0), g.mul(v, 45.0), 0.0)
    slub = g.noise(sl, 1.0, 4.0, 0.6)
    weft = g.combine(g.mul(u, 70.0), g.mul(v, 900.0), 0.0)
    wnoise = g.noise(weft, 1.0, 2.0, 0.5)
    indigo = (0.014, 0.024, 0.062)
    light = (0.036, 0.058, 0.118)
    faded = (0.12, 0.17, 0.27)
    white = (0.30, 0.33, 0.38)
    c = g.mix(g.add(g.mul(wale, 0.40), g.mul(g.sub(slub, 0.5), 0.8)), indigo, light)
    c = g.mix(g.mul(g.smooth(0.55, 0.85, wnoise), 0.08), c, white)
    st1, sd = g.stitch_rows("seam", 0.0034, 0.78, 0.00042)
    st2, pd = g.stitch_rows("pk", 0.0032, 0.78, 0.00042)
    wbd, hem, ptop = g.attr("wb"), g.attr("hem"), g.attr("ptop")
    rows = g.mx(g.line(wbd, 0.0022, 0.00042), g.line(wbd, 0.0378, 0.00042))
    rows = g.mx(rows, g.line(hem, 0.0135, 0.00045))
    rows = g.mx(rows, g.mx(g.line(ptop, 0.0045, 0.00042), g.line(ptop, 0.0105, 0.00042)))
    rows = g.mul(rows, g.dashes(g.add(u, g.mul(v, 0.0)), 0.0034, 0.78))
    stitch = g.mx(g.mx(st1, st2), rows)
    seam_line = g.mx(g.line(sd, 0.0, 0.0006), g.line(pd, 0.0, 0.0006))
    seam_line = g.mx(seam_line, g.line(wbd, 0.0405, 0.0006))
    edge = g.mx(g.smooth(0.004, 0.0, wbd), g.smooth(0.006, 0.0, hem))
    # wear: whiskers, knees, seat, thigh tops; fold ridges abrade lighter
    ridge = g.smooth(0.05, 0.6, fold)
    wear_n = g.noise(g.combine(g.mul(u, 200.0), g.mul(v, 60.0), 0.0), 1.0, 3.0, 0.65)
    wtot = g.mx(wear, g.mul(ridge, 0.75))
    wtot = g.mx(wtot, g.mul(edge, 0.7))
    wtot = g.mul(wtot, g.add(0.55, g.mul(wear_n, 0.9)))
    c = g.mix(g.math("MINIMUM", g.mul(wtot, 0.75), 0.75), c, faded)
    # topstitching (orange-tan) for both seam families
    # seam allowances / felled seams show as raised, abraded ridges
    allow = g.mul(g.smooth(0.0085, 0.006, g.absv(sd)), g.smooth(0.0, 0.0012, g.absv(sd)))
    c = g.mix(g.mul(allow, 0.35), c, faded)
    c = g.mix(g.mul(seam_line, 0.6), c, (0.008, 0.012, 0.03))
    thread = (0.62, 0.33, 0.085)
    c = g.mix(stitch, c, thread)
    c = g.mix(g.mul(g.sub(1.0, side), 1.0), c, (0.30, 0.33, 0.40))
    c = g.mix(bag, c, (0.62, 0.60, 0.55))
    c = g.mix(leather, c, (0.30, 0.17, 0.08))
    c = g.mix(metal, c, (0.80, 0.52, 0.28))
    g.link(c, b.inputs["Base Color"])
    rough = g.lerp(stitch, 0.82, 0.6)
    rough = g.lerp(metal, rough, 0.28)
    g.link(rough, b.inputs["Roughness"])
    g.link(metal, b.inputs["Metallic"])
    b.inputs["Sheen Weight"].default_value = 0.12
    b.inputs["Sheen Roughness"].default_value = 0.4
    h = g.add(g.mul(wale, 0.35), g.mul(g.sub(slub, 0.5), 0.25))
    h = g.add(h, g.mul(stitch, 1.0))
    h = g.sub(h, g.mul(seam_line, 0.8))
    h = g.add(h, g.mul(allow, 0.5))
    h = g.mul(h, g.sub(1.0, metal))
    bump = g.n("ShaderNodeBump", Strength=0.45, Distance=0.0005)
    g.link(h, bump.inputs["Height"])
    g.link(bump.outputs[0], b.inputs["Normal"])
    _denim_eevee(g)
    return mat


def _denim_eevee(g):
    """EEVEE has a small per-material vertex-attribute budget (the full tree reads 19): a lean
    twill + wear + seams tree with 9 attributes, used only by EEVEE."""
    b = g.bsdf("EEVEE")
    u, v, side, wear, fold = g.attr("u"), g.attr("v"), g.attr("side"), g.attr("wear"), g.attr("fold")
    metal, bag, label, sd = g.attr("metal"), g.attr("bag"), g.attr("label"), g.attr("seam")
    tw = g.frac(g.mul(g.add(g.mul(u, 0.88), g.mul(v, 0.47)), 1.0 / 0.00075))
    wale = g.smooth(0.0, 0.5, g.absv(g.sub(tw, 0.5)))
    slub = g.noise(g.combine(g.mul(u, 1400.0), g.mul(v, 45.0), 0.0), 1.0, 3.0, 0.6)
    c = g.mix(g.add(g.mul(wale, 0.40), g.mul(g.sub(slub, 0.5), 0.8)), (0.014, 0.024, 0.062), (0.036, 0.058, 0.118))
    wtot = g.mx(wear, g.mul(g.smooth(0.05, 0.6, fold), 0.75))
    c = g.mix(g.math("MINIMUM", g.mul(wtot, 0.7), 0.75), c, (0.12, 0.17, 0.27))
    row = g.mx(g.line(sd, 0.0015, 0.0005), g.line(sd, -0.0015, 0.0005))
    c = g.mix(g.mul(row, 0.8), c, (0.62, 0.33, 0.085))
    c = g.mix(g.sub(1.0, side), c, (0.30, 0.33, 0.40))
    c = g.mix(bag, c, (0.62, 0.60, 0.55))
    c = g.mix(label, c, (0.30, 0.17, 0.08))
    c = g.mix(metal, c, (0.80, 0.52, 0.28))
    g.link(c, b.inputs["Base Color"])
    g.link(g.lerp(metal, 0.82, 0.28), b.inputs["Roughness"])
    g.link(metal, b.inputs["Metallic"])
    bump = g.n("ShaderNodeBump", Strength=0.35, Distance=0.0005)
    g.link(g.add(g.mul(wale, 0.35), g.mul(row, 0.6)), bump.inputs["Height"])
    g.link(bump.outputs[0], b.inputs["Normal"])


def sneaker_material(name="Sneaker"):
    """Off-white leather upper, white rubber cupsole, cotton laces, textile lining, stitching."""
    mat, g = _new_mat(name)
    b = g.bsdf()
    pos = g.obj()
    sole = g.attr("sole")
    lace = g.attr("lace")
    lining = g.attr("lining")
    eyelet = g.attr("eyelet")
    accent = g.attr("accent")
    crease = g.attr("crease")
    tread = g.attr("tread")
    # leather: pebble grain (voronoi cells ~0.5 mm) + soft mottling
    vor = g.n("ShaderNodeTexVoronoi", feature="DISTANCE_TO_EDGE", Scale=1900.0, Randomness=1.0)
    g.link(pos, vor.inputs["Vector"])
    grain = g.smooth(0.0, 0.12, vor.outputs["Distance"])
    mott = g.noise(pos, 60.0, 3.0, 0.6)
    leather = g.mix(g.mul(g.sub(mott, 0.5), 0.5), (0.78, 0.76, 0.71), (0.70, 0.68, 0.63))
    # toe-box creases: wavy lines across the vamp
    cr_n = g.noise(pos, 140.0, 2.0, 0.5)
    vc = g.combine(g.mul(g.attr("cu"), 60.0), g.mul(g.attr("cv"), 380.0), 0.0)
    cr = g.noise(vc, 1.0, 3.0, 0.55)
    creases = g.mul(g.smooth(0.56, 0.62, cr), crease)
    leather = g.mix(g.mul(creases, 0.25), leather, (0.55, 0.53, 0.49))
    rubber = g.mix(g.mul(g.sub(g.noise(pos, 300.0, 2.0, 0.5), 0.5), 0.2), (0.86, 0.85, 0.80), (0.80, 0.79, 0.74))
    rubber = g.mix(tread, rubber, (0.60, 0.58, 0.54))
    # laces: woven flat cotton -> diagonal weave along the lace
    lw = g.noise(pos, 1800.0, 2.0, 0.5)
    lace_c = g.mix(g.mul(lw, 0.3), (0.86, 0.85, 0.82), (0.70, 0.69, 0.66))
    lining_c = (0.62, 0.60, 0.56)
    accent_c = (0.035, 0.12, 0.075)
    st, sd = g.stitch_rows("seam", 0.0028, 0.76, 0.00034)
    seam_line = g.line(sd, 0.0, 0.0005)
    c = g.mix(sole, leather, rubber)
    c = g.mix(accent, c, accent_c)
    c = g.mix(lining, c, lining_c)
    c = g.mix(g.mul(eyelet, 0.7), c, (0.45, 0.44, 0.41))
    c = g.mix(g.mul(seam_line, 0.4), c, (0.35, 0.34, 0.31))
    c = g.mix(g.mul(st, 0.85), c, (0.60, 0.58, 0.54))
    c = g.mix(lace, c, lace_c)
    g.link(c, b.inputs["Base Color"])
    rough = g.lerp(sole, 0.42, 0.62)
    rough = g.lerp(lace, rough, 0.9)
    rough = g.lerp(lining, rough, 0.95)
    rough = g.add(rough, g.mul(grain, 0.06))
    g.link(rough, b.inputs["Roughness"])
    b.inputs["Coat Weight"].default_value = 0.0
    b.inputs["Specular IOR Level"].default_value = 0.45
    h = g.mul(grain, g.mul(g.sub(1.0, sole), g.sub(1.0, lace)))
    h = g.mul(h, 0.25)
    h = g.sub(h, g.mul(creases, 0.6))
    h = g.add(h, g.mul(st, 0.8))
    h = g.sub(h, g.mul(seam_line, 0.6))
    h = g.add(h, g.mul(g.mul(lace, lw), 0.5))
    bump = g.n("ShaderNodeBump", Strength=0.4, Distance=0.0004)
    g.link(h, bump.inputs["Height"])
    g.link(bump.outputs[0], b.inputs["Normal"])
    return mat


# ----------------------------------------------------------------------------- loading / rigging
FILES = {"Shirt": ("shirt.npz", jersey_material), "Jeans": ("jeans.npz", denim_material),
         "Shoes": ("shoes.npz", sneaker_material)}


def bind_to_rig(ob, rig, body, proxy=None):
    """Parent to the armature and copy the body's skin weights (nearest face, interpolated)."""
    for vg in body.vertex_groups:
        if ob.vertex_groups.get(vg.name) is None:
            ob.vertex_groups.new(name=vg.name)
    dt = ob.modifiers.new("SkinWeights", "DATA_TRANSFER")
    dt.object = body
    dt.use_object_transform = True
    dt.use_vert_data = True
    dt.data_types_verts = {"VGROUP_WEIGHTS"}
    dt.vert_mapping = "POLYINTERP_NEAREST"
    dt.layers_vgroup_select_src = "ALL"
    dt.layers_vgroup_select_dst = "NAME"
    with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
        bpy.ops.object.modifier_apply(modifier=dt.name)
    mw = ob.matrix_world.copy()
    ob.parent = rig
    ob.matrix_world = mw
    arm = ob.modifiers.new("Armature", "ARMATURE")
    arm.object = rig
    arm.use_deform_preserve_volume = False
    # skinning shears the modelled folds where the shoulder bends hard (aiming): relax the shirt's
    # deformation against its rest shape, then push out any single-layer cloth the posed body
    # pokes through. The jeans are left alone: their pocket bags and fly are extra layers that a
    # shrinkwrap would press onto the outer denim.
    if ob.name == "Shirt":
        cs = ob.modifiers.new("Relax", "CORRECTIVE_SMOOTH")
        cs.rest_source = "ORCO"
        cs.smooth_type = "SIMPLE"
        cs.factor = 0.5
        cs.iterations = 6
        cs.use_only_smooth = False
    if ob.name in ("Shirt", "Shoes"):
        sw = ob.modifiers.new("KeepOutside", "SHRINKWRAP")
        sw.target = proxy or body
        sw.wrap_method = "NEAREST_SURFACEPOINT"
        sw.wrap_mode = "OUTSIDE_SURFACE"
        sw.offset = 0.0015


def decimate(ob, ratio):
    """Collapse-decimate a garment in place (attributes are interpolated). The full garments are
    ~2M vertices; animation previews on an 8 GB machine swap with them, and at 720p a third of
    the vertices carries the same folds."""
    if ratio is None or ratio >= 1.0:
        return
    mod = ob.modifiers.new("Decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob]):
        bpy.ops.object.modifier_apply(modifier=mod.name)


def collision_proxy(body, ratio=0.12):
    """Skinned low-res copy of the body for the garments' shrinkwrap: a 1M-vertex target means a
    1M-vertex BVH rebuilt every frame. Never seen by camera or rays; a bounding box in the viewport."""
    proxy = body.copy()
    proxy.data = body.data.copy()
    proxy.name = "BodyCollisionProxy"
    bpy.context.scene.collection.objects.link(proxy)
    mod = proxy.modifiers.new("Decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    with bpy.context.temp_override(object=proxy, active_object=proxy, selected_objects=[proxy]):
        bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)
        bpy.ops.object.modifier_apply(modifier=mod.name)
    for attr in ("visible_camera", "visible_shadow", "visible_diffuse", "visible_glossy",
                 "visible_transmission", "visible_volume_scatter"):
        setattr(proxy, attr, False)
    proxy.display_type = "BOUNDS"
    proxy.hide_select = True
    return proxy


def load_clothes(rig=None, body=None, ratio=None):
    """ratio: optional decimation ratio (float, or {garment name: float})."""
    objs = {}
    proxy = collision_proxy(body) if rig is not None and body is not None else None
    for name, (fname, matfn) in FILES.items():
        path = os.path.join(C.BUILD, fname)
        if not os.path.exists(path):
            print("missing", path)
            continue
        ob = C.load_npz_mesh(path, name)
        decimate(ob, ratio.get(name) if isinstance(ratio, dict) else ratio)
        ob.data.materials.append(matfn())
        if proxy is not None:
            bind_to_rig(ob, rig, body, proxy)
        objs[name] = ob
    return objs


# ----------------------------------------------------------------------------- preview
def _pocket():
    p = os.path.join(C.BUILD, "pocket.json")
    return json.load(open(p)) if os.path.exists(p) else None


def preview(prefix, samples=96):
    import skin
    C.reset_scene()
    body = C.load_npz_mesh(os.path.join(C.BUILD, "body_40.npz"), "Body")
    body.data.materials.append(skin.skin_material())
    objs = load_clothes()
    lift = bpy.data.objects.new("Lift", None)
    bpy.context.scene.collection.objects.link(lift)
    lift.location = (0, 0, SOLE_LIFT)
    for ob in [body, *objs.values()]:
        ob.parent = lift
    C.setup_cycles(samples=samples, res=(1100, 1500))
    bpy.context.scene.view_settings.exposure = -1.5
    C.world_color((0.30, 0.30, 0.32), 0.35)
    C.studio_lights(target=(0, -0.2, 1.0))
    C.ground(color=(0.25, 0.25, 0.26))
    cam = C.add_camera((0, -3, 1), (0, 0, 0.93), lens=50)
    for tag, yaw in (("front", 0), ("34", 35), ("back", 180)):
        C.orbit_camera(cam, (0, 0, 0.93), 2.9, yaw, 2.0)
        C.render(f"{prefix}_{tag}.png")
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = 1400, 1100
    pk = _pocket()
    tgt = Vector(pk["opening_centre"]) + Vector((0, 0, SOLE_LIFT)) if pk else Vector((-0.15, -0.08, 0.95))
    cam.data.lens = 85
    C.orbit_camera(cam, tgt + Vector((0, 0, -0.02)), 0.62, -62, 6)
    C.render(f"{prefix}_pocket.png")
    C.orbit_camera(cam, Vector((-0.11, -0.06, 0.07)), 0.75, -40, 18)
    C.render(f"{prefix}_shoe.png")


if __name__ == "__main__":
    a = C.args()
    if a and a[0] == "preview":
        preview(a[1], int(a[2]) if len(a) > 2 else 96)
