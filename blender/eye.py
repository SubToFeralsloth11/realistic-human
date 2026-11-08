"""Procedural human eye: inner globe (sclera, recessed iris, pupil) and outer cornea / tear-film shell.

Anatomy (adult): globe radius 12.2 mm, limbus radius 5.8 mm, corneal curvature radius 7.8 mm,
anterior chamber depth ~2.6 mm, pupil radius 1.9 mm in daylight. The iris shader is built from
polar noise (radial stroma fibres, crypts, collarette, limbal ring) so it has no texture files.
"""
import math

import bmesh
import bpy
from mathutils import Matrix, Vector

LIMBUS = 0.0058
PUPIL = 0.0019
CORNEA_R = 0.0078


def _globe_mesh(name, r):
    """UV sphere with its pole on local +Z; the front cap is pressed into a slightly domed iris."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=96, v_segments=64, radius=r)
    zl = math.sqrt(r * r - LIMBUS * LIMBUS)
    for v in bm.verts:
        rho = math.hypot(v.co.x, v.co.y)
        if v.co.z > 0 and rho < LIMBUS:
            t = rho / LIMBUS
            v.co.z = zl + 0.00035 * (1 - t * t)     # iris surface, gently domed by the lens
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return me


def _cornea_mesh(name, r):
    """Outer shell: corneal cap in front, a 0.06 mm tear film over the sclera elsewhere."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=96, v_segments=64, radius=r + 0.00006)
    zl = math.sqrt(r * r - LIMBUS * LIMBUS)
    zc = zl - math.sqrt(CORNEA_R ** 2 - LIMBUS ** 2)
    for v in bm.verts:
        d = v.co.normalized()
        # intersect the ray from the globe centre with the corneal sphere (centre (0, 0, zc))
        b = d.z * zc
        disc = b * b - (zc * zc - CORNEA_R ** 2)
        if d.z > 0 and disc > 0:
            tc = b + math.sqrt(disc)
            if tc > v.co.length:
                v.co = d * tc
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return me


def _node(nt, kind, loc, **inputs):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, val in inputs.items():
        n.inputs[k].default_value = val
    return n


def iris_material(name="EyeInner", iris_a=(0.035, 0.022, 0.009), iris_b=(0.13, 0.085, 0.030)):
    """Sclera + iris + pupil shader driven by object-space position (pole on +Z)."""
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    nt.nodes.clear()
    L = nt.links
    out = _node(nt, "ShaderNodeOutputMaterial", (1400, 0))
    bsdf = _node(nt, "ShaderNodeBsdfPrincipled", (1100, 0))
    L.new(bsdf.outputs[0], out.inputs[0])
    tc = _node(nt, "ShaderNodeTexCoord", (-1400, 0))
    sep = _node(nt, "ShaderNodeSeparateXYZ", (-1200, 0))
    L.new(tc.outputs["Object"], sep.inputs[0])
    # rho = sqrt(x^2 + y^2) in mm, phi = atan2(y, x)
    xx = _node(nt, "ShaderNodeMath", (-1000, 120)); xx.operation = "MULTIPLY"
    L.new(sep.outputs[0], xx.inputs[0]); L.new(sep.outputs[0], xx.inputs[1])
    yy = _node(nt, "ShaderNodeMath", (-1000, 0)); yy.operation = "MULTIPLY"
    L.new(sep.outputs[1], yy.inputs[0]); L.new(sep.outputs[1], yy.inputs[1])
    s = _node(nt, "ShaderNodeMath", (-850, 60)); s.operation = "ADD"
    L.new(xx.outputs[0], s.inputs[0]); L.new(yy.outputs[0], s.inputs[1])
    rho = _node(nt, "ShaderNodeMath", (-700, 60)); rho.operation = "SQRT"
    L.new(s.outputs[0], rho.inputs[0])
    mm = _node(nt, "ShaderNodeMath", (-550, 60)); mm.operation = "MULTIPLY"; mm.inputs[1].default_value = 1000.0
    L.new(rho.outputs[0], mm.inputs[0])
    phi = _node(nt, "ShaderNodeMath", (-850, -120)); phi.operation = "ARCTAN2"
    L.new(sep.outputs[1], phi.inputs[0]); L.new(sep.outputs[0], phi.inputs[1])
    front = _node(nt, "ShaderNodeMath", (-850, -260)); front.operation = "GREATER_THAN"; front.inputs[1].default_value = 0.0
    L.new(sep.outputs[2], front.inputs[0])
    # polar texture coordinate: (phi * 6, rho_mm, 0) -> radial fibres
    comb = _node(nt, "ShaderNodeCombineXYZ", (-400, -60))
    phs = _node(nt, "ShaderNodeMath", (-550, -120)); phs.operation = "MULTIPLY"; phs.inputs[1].default_value = 7.0
    L.new(phi.outputs[0], phs.inputs[0])
    L.new(phs.outputs[0], comb.inputs[0]); L.new(mm.outputs[0], comb.inputs[1])
    fib = _node(nt, "ShaderNodeTexNoise", (-200, -60))
    fib.inputs["Scale"].default_value = 1.0
    fib.inputs["Detail"].default_value = 8.0
    fib.inputs["Roughness"].default_value = 0.65
    L.new(comb.outputs[0], fib.inputs["Vector"])
    # stretch: the fibres run radially, so sample noise with x (angle) at high frequency
    sc = _node(nt, "ShaderNodeMapping", (-300, -260))
    sc.inputs["Scale"].default_value = (9.0, 0.9, 1.0)
    L.new(comb.outputs[0], sc.inputs["Vector"])
    fib2 = _node(nt, "ShaderNodeTexNoise", (-100, -260))
    fib2.inputs["Scale"].default_value = 1.0
    fib2.inputs["Detail"].default_value = 6.0
    L.new(sc.outputs[0], fib2.inputs["Vector"])
    crypt = _node(nt, "ShaderNodeTexVoronoi", (-100, -440))
    crypt.inputs["Scale"].default_value = 1.6
    L.new(comb.outputs[0], crypt.inputs["Vector"])
    # iris base colour: mix two pigments by fibre noise, darken in crypts
    mix1 = _node(nt, "ShaderNodeMix", (150, -100)); mix1.data_type = "RGBA"
    mix1.inputs[6].default_value = (*iris_a, 1); mix1.inputs[7].default_value = (*iris_b, 1)
    L.new(fib2.outputs["Fac"], mix1.inputs[0])
    cr = _node(nt, "ShaderNodeMapRange", (150, -400))
    cr.inputs["From Min"].default_value = 0.0; cr.inputs["From Max"].default_value = 0.35
    cr.inputs["To Min"].default_value = 0.55; cr.inputs["To Max"].default_value = 1.0
    L.new(crypt.outputs["Distance"], cr.inputs["Value"])
    dark = _node(nt, "ShaderNodeMix", (350, -150)); dark.data_type = "RGBA"; dark.blend_type = "MULTIPLY"
    dark.inputs[0].default_value = 1.0
    L.new(mix1.outputs[2], dark.inputs[6])
    gray = _node(nt, "ShaderNodeCombineColor", (250, -400))
    L.new(cr.outputs[0], gray.inputs[0]); L.new(cr.outputs[0], gray.inputs[1]); L.new(cr.outputs[0], gray.inputs[2])
    L.new(gray.outputs[0], dark.inputs[7])
    # radial profile: pupil black, collarette lighter at ~3.2 mm, limbal ring dark at 5.4-5.8 mm
    ramp = _node(nt, "ShaderNodeValToRGB", (350, 200))
    cra = ramp.color_ramp
    cra.interpolation = "EASE"
    stops = [(0.0, (0.0, 0.0, 0.0)), (PUPIL * 1000 / 6.5, (0.0, 0.0, 0.0)), (2.15 / 6.5, (0.55, 0.5, 0.45)),
             (3.2 / 6.5, (1.25, 1.15, 0.95)), (4.6 / 6.5, (0.95, 0.95, 0.95)), (5.35 / 6.5, (0.45, 0.45, 0.45)),
             (5.8 / 6.5, (0.10, 0.10, 0.10)), (6.2 / 6.5, (1.0, 1.0, 1.0))]
    cra.elements[0].position, cra.elements[0].color = stops[0][0], (*stops[0][1], 1)
    cra.elements[1].position, cra.elements[1].color = stops[-1][0], (*stops[-1][1], 1)
    for pos, col in stops[1:-1]:
        e = cra.elements.new(pos)
        e.color = (*col, 1)
    rn = _node(nt, "ShaderNodeMath", (200, 200)); rn.operation = "DIVIDE"; rn.inputs[1].default_value = 6.5
    L.new(mm.outputs[0], rn.inputs[0])
    L.new(rn.outputs[0], ramp.inputs[0])
    irisc = _node(nt, "ShaderNodeMix", (600, 0)); irisc.data_type = "RGBA"; irisc.blend_type = "MULTIPLY"
    irisc.inputs[0].default_value = 1.0
    L.new(dark.outputs[2], irisc.inputs[6]); L.new(ramp.outputs[0], irisc.inputs[7])
    # sclera: off-white, warmer and pinker towards the periphery, faint vessels
    vein = _node(nt, "ShaderNodeTexNoise", (100, 450))
    vein.inputs["Scale"].default_value = 260.0; vein.inputs["Detail"].default_value = 10.0
    vein.inputs["Distortion"].default_value = 1.4
    L.new(tc.outputs["Object"], vein.inputs["Vector"])
    vr = _node(nt, "ShaderNodeMapRange", (300, 450))
    vr.inputs["From Min"].default_value = 0.56; vr.inputs["From Max"].default_value = 0.62
    L.new(vein.outputs["Fac"], vr.inputs["Value"])
    per = _node(nt, "ShaderNodeMapRange", (300, 650))           # 0 at the limbus, 1 towards the back
    per.inputs["From Min"].default_value = 7.0; per.inputs["From Max"].default_value = 11.5
    L.new(mm.outputs[0], per.inputs["Value"])
    vmask = _node(nt, "ShaderNodeMath", (500, 520)); vmask.operation = "MULTIPLY"
    L.new(vr.outputs[0], vmask.inputs[0]); L.new(per.outputs[0], vmask.inputs[1])
    vm2 = _node(nt, "ShaderNodeMath", (650, 520)); vm2.operation = "MULTIPLY"; vm2.inputs[1].default_value = 0.6
    L.new(vmask.outputs[0], vm2.inputs[0])
    scl = _node(nt, "ShaderNodeMix", (650, 350)); scl.data_type = "RGBA"
    scl.inputs[6].default_value = (0.62, 0.57, 0.51, 1); scl.inputs[7].default_value = (0.60, 0.45, 0.40, 1)
    L.new(per.outputs[0], scl.inputs[0])
    sclv = _node(nt, "ShaderNodeMix", (820, 350)); sclv.data_type = "RGBA"
    sclv.inputs[7].default_value = (0.45, 0.06, 0.05, 1)
    L.new(vm2.outputs[0], sclv.inputs[0]); L.new(scl.outputs[2], sclv.inputs[6])
    # iris vs sclera by rho with a soft 0.25 mm limbus blend, front hemisphere only
    lim = _node(nt, "ShaderNodeMapRange", (820, 100))
    lim.inputs["From Min"].default_value = LIMBUS * 1000 - 0.15; lim.inputs["From Max"].default_value = LIMBUS * 1000 + 0.25
    L.new(mm.outputs[0], lim.inputs["Value"])
    back = _node(nt, "ShaderNodeMath", (950, -150)); back.operation = "SUBTRACT"; back.inputs[0].default_value = 1.0
    L.new(front.outputs[0], back.inputs[1])
    selm = _node(nt, "ShaderNodeMath", (1000, 100)); selm.operation = "MAXIMUM"
    L.new(lim.outputs[0], selm.inputs[0]); L.new(back.outputs[0], selm.inputs[1])
    final = _node(nt, "ShaderNodeMix", (1000, 250)); final.data_type = "RGBA"
    L.new(selm.outputs[0], final.inputs[0])
    L.new(irisc.outputs[2], final.inputs[6]); L.new(sclv.outputs[2], final.inputs[7])
    L.new(final.outputs[2], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.35
    bsdf.inputs["Subsurface Weight"].default_value = 0.25
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.5, 0.35)
    bsdf.inputs["Subsurface Scale"].default_value = 0.0015
    return mat


def cornea_material(name="Cornea"):
    """Refractive cornea + tear film in Cycles; in EEVEE a clear coat over the iris (EEVEE's
    screen-space refraction turns small closed refractive shells black)."""
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    p = nt.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (1, 1, 1, 1)
    p.inputs["Roughness"].default_value = 0.0
    p.inputs["IOR"].default_value = 1.376
    p.inputs["Transmission Weight"].default_value = 1.0
    out = nt.nodes.get("Material Output")
    out.target = "CYCLES"
    ev = nt.nodes.new("ShaderNodeOutputMaterial")
    ev.target = "EEVEE"
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    gl = nt.nodes.new("ShaderNodeBsdfGlossy")
    gl.inputs["Roughness"].default_value = 0.02
    fr = nt.nodes.new("ShaderNodeFresnel")
    fr.inputs["IOR"].default_value = 1.376
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fr.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(gl.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], ev.inputs[0])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "BLENDED"
    return mat


def build_eye(name, center, radius, look=Vector((0.0, -1.0, 0.0)), mats=None, parent=None):
    """Create the two eye objects; returns (inner, cornea). Pole (+Z local) aims along `look`."""
    mats = mats or {}
    inner_mat = mats.get("inner") or iris_material()
    cornea_mat = mats.get("cornea") or cornea_material()
    inner = bpy.data.objects.new(name, _globe_mesh(name, radius))
    cornea = bpy.data.objects.new(name + "_cornea", _cornea_mesh(name + "_cornea", radius))
    inner.data.materials.append(inner_mat)
    cornea.data.materials.append(cornea_mat)
    rot = Vector((0, 0, 1)).rotation_difference(Vector(look).normalized()).to_matrix().to_4x4()
    for ob in (inner, cornea):
        bpy.context.scene.collection.objects.link(ob)
        ob.matrix_world = Matrix.Translation(Vector(center)) @ rot
        if parent is not None:
            ob.parent = parent
            ob.matrix_parent_inverse = parent.matrix_world.inverted()
    cornea.visible_shadow = False
    return inner, cornea
