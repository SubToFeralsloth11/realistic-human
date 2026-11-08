"""Procedural compact 9 mm striker-fired pistol (generic, no markings).

Everything is generated from code: meshes are profile extrusions, lofts and lathes combined with exact
booleans, then bevelled and given weighted normals; materials are procedural node trees.

Import:      from pistol import build_pistol; parts = build_pistol()
Standalone:  blender -b --factory-startup --python blender/pistol.py -- <out.blend> <render_prefix>

Pistol space (root empty): +Y muzzle, +Z up (sights), +X gun's right side, metres.
Root origin = grip_socket = palm centre on the grip axis, 25 mm below the slide bottom.

Animation contract
  slide        location.y 0 -> -0.028 = full recoil / slide lock.
  barrel       origin on the bore axis where it passes the slide's front bushing when locked back.
               Locked back: location.y -= 0.003 and rotation_euler.x = +5 deg (chamber drops).
               Follow the slide for its first 3 mm, then tilt.
  trigger      origin at its pin; pull = rotation_euler.x -12 deg.
  magazine     origin at its top centre; local Z is the insertion axis (18 deg grip rake);
               eject by translating along its own -Z (≈0.085 m to clear the magwell).
  mag_release  press = local +X 0.0012.
  slide_stop   origin at its pivot (trigger pin); engage = rotation_euler.x -6 deg.
  casing / cartridge  hidden copies at the chamber (under the ejection port), +Y = bullet direction.
  muzzle_flash hidden, origin at the muzzle, +Y forward.
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

MM = 0.001

# ---------------------------------------------------------------------------------------------
# Layout, in "gun space" millimetres (slide rear = y 0, slide bottom = z 0, bore axis z 14.5)
# ---------------------------------------------------------------------------------------------
RAKE = math.radians(18.0)
SR, CR = math.sin(RAKE), math.cos(RAKE)
UAX = Vector((0.0, SR, CR))       # grip / magazine insertion axis (points up into the gun)
FAX = Vector((0.0, CR, -SR))      # grip forward axis, perpendicular to UAX
BORE_Z = 14.5
SLIDE_LEN = 157.0
BREECH_Y = 71.0
MAG_ORIGIN = Vector((0.0, 59.2, 7.5))       # top centre of the magazine (feed lips)
MAG_S = 3.4                                  # magazine axis offset forward of the grip centre line
FRONT_BOTTOM_Z = -75.5
T0 = (MAG_ORIGIN.z - MAG_S * FAX.z + 22.0 * FAX.z - FRONT_BOTTOM_Z) / UAX.z
GRIP_O = MAG_ORIGIN - MAG_S * FAX - T0 * UAX  # grip bottom centre
MAG_LEN = T0                                 # magazine body length below its origin
TRIGGER_PIVOT = Vector((0.0, 84.0, -7.0))
BARREL_PIVOT = Vector((0.0, 125.0, BORE_Z))
GUIDE_Z = 4.6


def gp(s, t):
    """Grip-local (s forward, t up the grip axis) -> gun-space (y, z)."""
    v = GRIP_O + s * FAX + t * UAX
    return (v.y, v.z)


def t_at(s, z):
    return (z - GRIP_O.z - s * FAX.z) / UAX.z


SOCKET = GRIP_O + t_at(0.0, -25.0) * UAX     # palm centre: on the grip axis, 25 mm below the slide


def hw_grip(t):
    """Half width of the grip at height t along the grip axis."""
    return float(np.interp(t, [-4, 0, 10, 40, 55, 62, 67, 95], [12.3, 12.4, 12.9, 13.0, 12.6, 11.6, 10.4, 10.0]))


# ---------------------------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------------------------
def _link(obj):
    bpy.context.scene.collection.objects.link(obj)


def mesh_obj(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], faces)
    me.validate()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    _link(obj)
    return obj


def dedup(pts, eps=1e-4):
    out = []
    for p in pts:
        if not out or math.dist(p, out[-1]) > eps:
            out.append(tuple(float(v) for v in p))
    if len(out) > 2 and math.dist(out[0], out[-1]) <= eps:
        out.pop()
    return out


def fillet(pts, step_deg=7.5):
    """Closed polygon of (a, b[, r]) -> polygon with each corner rounded by radius r."""
    P = [np.array(p[:2], float) for p in pts]
    R = [p[2] if len(p) > 2 else 0.0 for p in pts]
    n = len(P)
    out = []
    for i in range(n):
        p, a, b, r = P[i], P[i - 1], P[(i + 1) % n], R[i]
        if r <= 0:
            out.append(tuple(p))
            continue
        d1, d2 = a - p, b - p
        l1, l2 = np.linalg.norm(d1), np.linalg.norm(d2)
        d1, d2 = d1 / l1, d2 / l2
        ang = math.acos(float(np.clip(d1 @ d2, -1, 1)))
        if ang > math.pi - 1e-3 or ang < 1e-3:
            out.append(tuple(p))
            continue
        t = r / math.tan(ang / 2)
        tmax = 0.49 * min(l1, l2)
        if t > tmax:
            t = tmax
            r = t * math.tan(ang / 2)
        bis = d1 + d2
        bis /= np.linalg.norm(bis)
        c = p + bis * (r / math.sin(ang / 2))
        p1, p2 = p + d1 * t, p + d2 * t
        a1 = math.atan2(p1[1] - c[1], p1[0] - c[0])
        a2 = math.atan2(p2[1] - c[1], p2[0] - c[0])
        da = a2 - a1
        while da > math.pi:
            da -= 2 * math.pi
        while da < -math.pi:
            da += 2 * math.pi
        segs = max(2, int(math.ceil(abs(da) / math.radians(step_deg))))
        for k in range(segs + 1):
            ak = a1 + da * k / segs
            out.append((c[0] + r * math.cos(ak), c[1] + r * math.sin(ak)))
    return dedup(out)


def catmull(pts, n=8, closed=True):
    P = np.array(pts, float)
    N = len(P)
    out = []
    for i in range(N if closed else N - 1):
        p0 = P[i - 1] if (closed or i > 0) else P[i]
        p1, p2 = P[i], P[(i + 1) % N]
        p3 = P[(i + 2) % N] if (closed or i + 2 < N) else P[i + 1]
        for k in range(n):
            t = k / n
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(P[-1])
    return [tuple(p) for p in out]


def rrect(a0, a1, b0, b1, r):
    return fillet([(a0, b0, r), (a1, b0, r), (a1, b1, r), (a0, b1, r)])


def prism(name, pts, axis, lo, hi):
    """Extrude a closed 2D polygon. axis 'X': pts are (y, z); 'Y': (x, z); 'Z': (x, y)."""
    pts = dedup(pts)
    n = len(pts)

    def P(a, b, w):
        return (w, a, b) if axis == "X" else (a, w, b) if axis == "Y" else (a, b, w)

    verts = [P(a, b, lo) for a, b in pts] + [P(a, b, hi) for a, b in pts]
    faces = [list(range(n)), list(range(2 * n - 1, n - 1, -1))]
    faces += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return mesh_obj(name, verts, faces)


def box(name, x0, x1, y0, y1, z0, z1):
    return prism(name, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "Z", z0, z1)


def loft(name, rings):
    rings = [np.asarray(r, float) for r in rings]
    m, n = len(rings), len(rings[0])
    verts = np.concatenate(rings)
    faces = [list(range(n - 1, -1, -1)), list(range((m - 1) * n, m * n))]
    for j in range(m - 1):
        for i in range(n):
            a, b = j * n + i, j * n + (i + 1) % n
            faces.append([a, b, b + n, a + n])
    return mesh_obj(name, verts, faces)


def lathe(name, prof, seg=64, rfun=None):
    """Revolve a closed (r, y) profile about the Y axis. r == 0 points become poles."""
    verts, idx = [], []
    th = [2 * math.pi * k / seg for k in range(seg)]
    for r, y in prof:
        if r < 1e-6:
            idx.append([len(verts)])
            verts.append((0.0, y, 0.0))
        else:
            ring = []
            for t in th:
                rr = rfun(r, y, t) if rfun else r
                ring.append(len(verts))
                verts.append((rr * math.cos(t), y, rr * math.sin(t)))
            idx.append(ring)
    faces = []
    N = len(prof)
    for k in range(N):
        A, B = idx[k], idx[(k + 1) % N]
        if len(A) == 1 and len(B) == 1:
            continue
        if len(A) == 1:
            faces += [[A[0], B[(i + 1) % seg], B[i]] for i in range(seg)]
        elif len(B) == 1:
            faces += [[A[i], A[(i + 1) % seg], B[0]] for i in range(seg)]
        else:
            faces += [[A[i], A[(i + 1) % seg], B[(i + 1) % seg], B[i]] for i in range(seg)]
    return mesh_obj(name, verts, faces)


def cyl(name, r, y0, y1, seg=48):
    return lathe(name, [(0, y0), (r, y0), (r, y1), (0, y1)], seg)


def xform(obj, M):
    obj.data.transform(M)
    return obj


def at(obj, x=0.0, y=0.0, z=0.0):
    return xform(obj, Matrix.Translation((x, y, z)))


def x_axis(obj):
    """Turn something built along +Y so it runs along +X."""
    return xform(obj, Matrix.Rotation(-math.pi / 2, 4, "Z"))


def delete(obj):
    me = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


def apply_mods(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    me.name = obj.name
    if old.users == 0:
        bpy.data.meshes.remove(old)


def join(objs, name=None):
    objs = [o for o in objs if o is not None]
    base = objs[0]
    if len(objs) > 1:
        with bpy.context.temp_override(active_object=base, object=base, selected_objects=objs,
                                       selected_editable_objects=objs):
            bpy.ops.object.join()
    if name:
        base.name = name
        base.data.name = name
    return base


def boolean(obj, cutters, op="DIFFERENCE"):
    cutters = cutters if isinstance(cutters, (list, tuple)) else [cutters]
    cut = join(list(cutters))
    m = obj.modifiers.new("Bool", "BOOLEAN")
    m.operation = op
    m.object = cut
    m.solver = "EXACT"
    m.use_self = len(cutters) > 1
    apply_mods(obj)
    delete(cut)
    return obj


def clean(obj, angle=0.3):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(angle), verts=bm.verts, edges=bm.edges,
                             delimit={"NORMAL"})
    bm.to_mesh(obj.data)
    bm.free()
    return obj


_TAG = None


def _tag_mat():
    global _TAG
    if _TAG is None or _TAG.name not in bpy.data.materials:
        _TAG = bpy.data.materials.new("_bevel_tag")
    return _TAG


def finish(obj, mat, width, segs=3, angle=25.0, smooth_angle=38.0):
    """Bevel every sharp edge, tag bevel faces as 'edge' (for wear masks), weighted normals."""
    me = obj.data
    me.materials.clear()
    me.materials.append(mat)
    if width > 0:
        me.materials.append(_tag_mat())
        b = obj.modifiers.new("Bevel", "BEVEL")
        b.width = width
        b.segments = segs
        b.limit_method = "ANGLE"
        b.angle_limit = math.radians(angle)
        b.use_clamp_overlap = True
        b.miter_outer = "MITER_ARC"
        b.material = 1
        apply_mods(obj)
        me = obj.data
        n = len(me.polygons)
        mi = np.zeros(n, np.int32)
        me.polygons.foreach_get("material_index", mi)
        att = me.attributes.new("edge", "FLOAT", "FACE")
        att.data.foreach_set("value", (mi == 1).astype(np.float32))
        me.polygons.foreach_set("material_index", np.zeros(n, np.int32))
        me.materials.pop(index=1)
    me.shade_smooth()
    me.set_sharp_from_angle(angle=math.radians(smooth_angle))
    w = obj.modifiers.new("WN", "WEIGHTED_NORMAL")
    w.keep_sharp = True
    w.weight = 50
    w.mode = "FACE_AREA"
    apply_mods(obj)
    return obj


def tris(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons) if obj.type == "MESH" else 0


# ---------------------------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------------------------
def _sock(socks, ident):
    for s in socks:
        if s.identifier == ident:
            return s
    return socks[ident]


def _in(nt, sock, v):
    if isinstance(v, (int, float)):
        sock.default_value = v
    elif isinstance(v, tuple):
        sock.default_value = v if len(v) == 4 or len(sock.default_value) == 3 else (*v, 1.0)
    else:
        nt.links.new(v, sock)


def _math(nt, op, a, b=0.0, c=None, clamp=False):
    nd = nt.nodes.new("ShaderNodeMath")
    nd.operation = op
    nd.use_clamp = clamp
    _in(nt, nd.inputs[0], a)
    _in(nt, nd.inputs[1], b)
    if c is not None:
        _in(nt, nd.inputs[2], c)
    return nd.outputs[0]


def _mixc(nt, fac, a, b):
    nd = nt.nodes.new("ShaderNodeMix")
    nd.data_type = "RGBA"
    _in(nt, _sock(nd.inputs, "Factor_Float"), fac)
    _in(nt, _sock(nd.inputs, "A_Color"), a)
    _in(nt, _sock(nd.inputs, "B_Color"), b)
    return _sock(nd.outputs, "Result_Color")


def _mixf(nt, fac, a, b):
    nd = nt.nodes.new("ShaderNodeMix")
    nd.data_type = "FLOAT"
    _in(nt, _sock(nd.inputs, "Factor_Float"), fac)
    _in(nt, _sock(nd.inputs, "A_Float"), a)
    _in(nt, _sock(nd.inputs, "B_Float"), b)
    return _sock(nd.outputs, "Result_Float")


def _noise(nt, vec, scale, detail=2.0, rough=0.5, out="Fac"):
    nd = nt.nodes.new("ShaderNodeTexNoise")
    _in(nt, nd.inputs["Vector"], vec)
    nd.inputs["Scale"].default_value = scale
    nd.inputs["Detail"].default_value = detail
    nd.inputs["Roughness"].default_value = rough
    return nd.outputs[out]


def _objco(nt):
    return nt.nodes.new("ShaderNodeTexCoord").outputs["Object"]


def _smooth(nt, v, lo, hi):
    nd = nt.nodes.new("ShaderNodeMapRange")
    nd.interpolation_type = "SMOOTHSTEP"
    _in(nt, nd.inputs["Value"], v)
    nd.inputs["From Min"].default_value = lo
    nd.inputs["From Max"].default_value = hi
    return nd.outputs["Result"]


def _edge_attr(nt):
    nd = nt.nodes.new("ShaderNodeAttribute")
    nd.attribute_type = "GEOMETRY"
    nd.attribute_name = "edge"
    return nd.outputs["Fac"]


def _new_mat(name):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    return m, nt, nt.nodes.get("Principled BSDF")


def mat_metal(name, base, rough, wear_col=None, wear_amt=0.0, brushed=0.0, metallic=1.0, blotch=0.10,
              wear_rough=0.22):
    m, nt, p = _new_mat(name)
    p.inputs["Metallic"].default_value = metallic
    co = _objco(nt)
    bl = _noise(nt, co, 90.0, 3.0, 0.55)
    rough_v = _math(nt, "MULTIPLY_ADD", _math(nt, "SUBTRACT", bl, 0.5), blotch * 1.6, rough)
    if brushed > 0:
        mp = nt.nodes.new("ShaderNodeMapping")
        _in(nt, mp.inputs["Vector"], co)
        mp.inputs["Scale"].default_value = (1.0, 0.025, 1.0)
        br = _noise(nt, mp.outputs[0], 2600.0, 1.0, 0.4)
        rough_v = _math(nt, "MULTIPLY_ADD", _math(nt, "SUBTRACT", br, 0.5), brushed, rough_v)
    col = base
    if wear_col is not None and wear_amt > 0:
        wn = _smooth(nt, _noise(nt, co, 700.0, 6.0, 0.6), 0.38, 0.62)
        wear = _math(nt, "MULTIPLY", _edge_attr(nt), _math(nt, "MULTIPLY_ADD", wn, 0.75, 0.25), clamp=True)
        wear = _math(nt, "MULTIPLY", wear, wear_amt, clamp=True)
        col = _mixc(nt, wear, base, wear_col)
        rough_v = _mixf(nt, wear, rough_v, wear_rough)
    _in(nt, p.inputs["Base Color"], col)
    _in(nt, p.inputs["Roughness"], rough_v)
    return m


def mat_polymer(name, base=(0.016, 0.016, 0.018), rough=0.55, stipple=False):
    m, nt, p = _new_mat(name)
    p.inputs["Specular IOR Level"].default_value = 0.45
    co = _objco(nt)
    n1 = _noise(nt, co, 160.0, 4.0, 0.6)
    rough_v = _math(nt, "MULTIPLY_ADD", _math(nt, "SUBTRACT", n1, 0.5), 0.14, rough)
    micro = _noise(nt, co, 4200.0, 2.0, 0.5)
    height = _math(nt, "MULTIPLY", micro, 0.25)
    if stipple:
        # grip band (t along the grip axis) excluding the trigger guard (s beyond the front strap)
        def axis_coord(vec, off):
            vm = nt.nodes.new("ShaderNodeVectorMath")
            vm.operation = "DOT_PRODUCT"
            _in(nt, vm.inputs[0], co)
            vm.inputs[1].default_value = tuple(vec)
            return _math(nt, "MULTIPLY_ADD", _sock(vm.outputs, "Value"), 1000.0, off)

        t = axis_coord(UAX, (SOCKET - GRIP_O).dot(UAX))
        s = axis_coord(FAX, (SOCKET - GRIP_O).dot(FAX))
        mask = _math(nt, "MULTIPLY", _smooth(nt, t, 6.0, 6.5), _math(nt, "SUBTRACT", 1.0, _smooth(nt, t, 56.5, 57.0)))
        mask = _math(nt, "MULTIPLY", mask, _math(nt, "SUBTRACT", 1.0, _smooth(nt, s, 22.4, 22.9)))
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        _in(nt, vor.inputs["Vector"], co)
        vor.inputs["Scale"].default_value = 1500.0
        vor.inputs["Randomness"].default_value = 0.85
        dimple = _smooth(nt, vor.outputs["Distance"], 0.05, 0.55)
        height = _mixf(nt, mask, height, _math(nt, "MULTIPLY_ADD", dimple, 1.0, _math(nt, "MULTIPLY", micro, 0.2)))
        rough_v = _mixf(nt, mask, rough_v, _math(nt, "MULTIPLY_ADD", dimple, 0.12, 0.6))
        col = _mixc(nt, mask, base, tuple(c * 0.85 for c in base))
        _in(nt, p.inputs["Base Color"], col)
    else:
        _in(nt, p.inputs["Base Color"], base)
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.55 if stipple else 0.12
    bump.inputs["Distance"].default_value = 0.00025
    _in(nt, bump.inputs["Height"], height)
    nt.links.new(bump.outputs["Normal"], p.inputs["Normal"])
    _in(nt, p.inputs["Roughness"], rough_v)
    return m


def mat_plain(name, base, rough, metallic=0.0):
    m, nt, p = _new_mat(name)
    p.inputs["Base Color"].default_value = (*base, 1.0)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metallic
    return m


def mat_flash(name, length_m=0.045):
    m, nt, p = _new_mat(name)
    nt.nodes.remove(p)
    out = nt.nodes.get("Material Output")
    co = _objco(nt)
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    _in(nt, ln.inputs[0], co)
    d = _math(nt, "DIVIDE", _sock(ln.outputs, "Value"), length_m, clamp=True)
    inv = _math(nt, "SUBTRACT", 1.0, d, clamp=True)
    fall = _math(nt, "POWER", inv, 2.2)
    n = _noise(nt, co, 260.0, 6.0, 0.62)
    alpha = _math(nt, "MULTIPLY", fall, _math(nt, "MULTIPLY_ADD", n, 1.6, -0.15), clamp=True)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    _in(nt, ramp.inputs[0], d)
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (1.0, 0.97, 0.88, 1)
    els[1].position, els[1].color = 1.0, (0.45, 0.06, 0.01, 1)
    e = els.new(0.22)
    e.color = (1.0, 0.78, 0.38, 1)
    e = els.new(0.55)
    e.color = (1.0, 0.40, 0.07, 1)
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
    _in(nt, em.inputs["Strength"], _math(nt, "MULTIPLY", fall, 90.0))
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    _in(nt, mix.inputs[0], alpha)
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
        if hasattr(m, attr):
            try:
                setattr(m, attr, val)
            except (TypeError, ValueError):
                pass
    return m


def make_materials():
    return {
        "slide": mat_metal("Pistol_SlideNitride", (0.055, 0.056, 0.06), 0.34, (0.42, 0.42, 0.43), 0.9,
                           brushed=0.07),
        "barrel": mat_metal("Pistol_BarrelNitride", (0.075, 0.073, 0.072), 0.46, (0.45, 0.44, 0.42), 0.8,
                            brushed=0.05),
        "steel": mat_metal("Pistol_Steel", (0.10, 0.10, 0.105), 0.38, (0.5, 0.5, 0.5), 0.6),
        "sight": mat_metal("Pistol_SightSteel", (0.03, 0.03, 0.032), 0.48, (0.3, 0.3, 0.3), 0.4, metallic=0.7),
        "dot": mat_plain("Pistol_SightDot", (0.86, 0.86, 0.82), 0.35),
        "frame": mat_polymer("Pistol_FramePolymer", stipple=True),
        "polymer": mat_polymer("Pistol_Polymer", rough=0.5),
        "follower": mat_polymer("Pistol_Follower", base=(0.05, 0.05, 0.055), rough=0.45),
        "mag": mat_metal("Pistol_MagSteel", (0.028, 0.028, 0.03), 0.42, (0.35, 0.35, 0.36), 0.7, metallic=0.85),
        "brass": mat_metal("Pistol_Brass", (0.80, 0.56, 0.20), 0.24, blotch=0.06),
        "primer": mat_metal("Pistol_Primer", (0.72, 0.64, 0.50), 0.3, blotch=0.05),
        "copper": mat_metal("Pistol_Copper", (0.86, 0.40, 0.25), 0.27, blotch=0.06),
        "flash": mat_flash("Pistol_MuzzleFlash"),
    }


def add_mat(obj, mat):
    """Append a material slot and point every face at it (for single-material pieces)."""
    me = obj.data
    me.materials.append(mat)
    idx = len(me.materials) - 1
    me.polygons.foreach_set("material_index", np.full(len(me.polygons), idx, np.int32))
    return obj


# ---------------------------------------------------------------------------------------------
# Parts (all built in gun-space mm unless noted)
# ---------------------------------------------------------------------------------------------
def grip_loft(name, off=0.0, t_range=(-4.0, 95.0), n=112, e=2.2, b=25.0):
    rings = []
    for t in np.linspace(t_range[0], t_range[1], 44):
        a = hw_grip(t) + off
        bb = b + off
        ring = []
        for k in range(n):
            th = 2 * math.pi * k / n
            c, s = math.cos(th), math.sin(th)
            x = a * math.copysign(abs(c) ** (2 / e), c)
            sv = bb * math.copysign(abs(s) ** (2 / e), s)
            p = GRIP_O + sv * FAX + float(t) * UAX
            ring.append((x, p.y, p.z))
        rings.append(ring)
    return loft(name, rings)


def magrelease_outline(grow=0.0):
    s0, t0 = 15.0, t_at(15.0, -23.0)
    hs, ht = 3.6 + grow, 3.0 + grow
    pts = fillet([(s0 - hs, t0 - ht, 1.6 + grow), (s0 + hs, t0 - ht, 1.6 + grow),
                  (s0 + hs, t0 + ht, 1.6 + grow), (s0 - hs, t0 + ht, 1.6 + grow)])
    return [gp(s, t) for s, t in pts], gp(s0, t0)


WEB = [(-6.0, -3.0), (-3.4, -3.2), (-2.4, -6.4), (3.0, -7.8), (11.5, -9.4), (19.0, -12.6), (24.5, -17.2)]


def tang_rounding(side):
    """Cutter that rounds the lower outer corners of the beavertail tang (large radius, fades out forward)."""
    ys, zs = zip(*WEB)
    rings = []
    for y in np.linspace(-6.0, 19.0, 26):
        zb = float(np.interp(y, ys, zs))
        r = float(np.interp(y, [-6.0, 9.0, 19.0], [5.5, 5.5, 0.6]))
        xc, zc = 10.3 - r, zb + r
        ring = [(12.5, zc + 0.3), (12.5, zb - 3.0), (xc - 0.3, zb - 3.0), (xc - 0.3, zb - 0.02)]
        for k in range(13):
            a = -math.pi / 2 + (math.pi / 2) * k / 12
            ring.append((xc + r * math.cos(a), zc + r * math.sin(a)))
        rings.append([(side * x, float(y), z) for x, z in ring])
    return loft("c_tang", rings)


def build_frame(M):
    fs = lambda z: gp(22.0, t_at(22.0, z))[0]
    sil = [
        (-3.0, -0.3, 0.8), (154.6, -0.3, 0.8), (154.6, -11.5, 2.5), (103.0, -11.5, 2.5),
        (101.2, -21.0, 12.0), (97.8, -33.0, 4.0), (92.0, -35.8, 3.0), (fs(-35.8), -35.8, 3.0),
        (*gp(22.5, 28.0), 40.0), (*gp(22.0, 0.0), 2.5), (*gp(-22.0, 0.0), 3.5), (*gp(-22.0, 12.0), 20.0),
        (*gp(-23.2, 30.0), 30.0), (*gp(-22.0, 47.0), 10.0),
        (19.0, -12.6, 6.0), (11.5, -9.4, 8.0), (3.0, -7.8, 4.0), (-2.4, -6.4, 2.0), (-3.4, -3.2, 1.0),
    ]
    frame = prism("frame", fillet(sil), "X", -14.0, 14.0)
    upper = box("f_upper", -10.3, 10.3, -12.0, 162.0, -11.8, 3.0)
    guard = box("f_guard", -4.9, 4.9, 58.0, 112.0, -40.0, -10.0)
    grip = grip_loft("f_grip")
    boolean(upper, guard, "UNION")
    boolean(upper, grip, "UNION")
    boolean(frame, upper, "INTERSECT")

    hole = [(fs(-11.0), -11.0, 1.0), (97.6, -11.0, 1.5), (96.8, -20.0, 12.0), (93.5, -29.5, 5.0),
            (88.0, -32.3, 3.0), (fs(-32.3), -32.3, 3.0)]
    cutters = [prism("c_hole", fillet(hole), "X", -20, 20)]
    # magazine well with a flared, bevelled mouth
    rings = []
    for t, hwm, hdm in [(-4, 9.8, 18.6), (-0.5, 9.5, 18.3), (0.6, 8.9, 17.6), (1.6, 8.35, 17.15),
                        (2.8, 8.15, 16.95), (95.0, 8.15, 16.95)]:
        ring = []
        for x, s in rrect(-hwm, hwm, -hdm, hdm, 2.4):
            p = GRIP_O + (s + MAG_S) * FAX + t * UAX
            ring.append((x, p.y, p.z))
        rings.append(ring)
    cutters.append(loft("c_magwell", rings))
    cutters.append(box("c_lugpocket", -6.2, 6.2, 72.0, 101.5, -6.0, 3.0))
    cutters.append(box("c_trigslot", -3.7, 3.7, 80.0, 90.5, -14.0, -3.5))
    cutters.append(box("c_takedown", -11.0, 11.0, 88.0, 94.5, -5.6, -2.0))
    # slide stop recess (left side)
    cutters.append(box("c_ss_bar", -11.5, -9.75, 57.6, 84.0, -4.8, -0.8))
    cutters.append(box("c_ss_pad", -11.5, -9.75, 57.6, 64.0, -7.0, -0.8))
    cutters.append(at(x_axis(cyl("c_ss_hub", 3.1, -11.5, -9.75)), 0, TRIGGER_PIVOT.y, TRIGGER_PIVOT.z))
    # magazine release recess following the grip surface (left side)
    outline, _ = magrelease_outline(0.4)
    cutters.append(tang_rounding(1))
    cutters.append(tang_rounding(-1))
    rec = prism("c_magrel", outline, "X", -20.0, 0.0)
    boolean(rec, grip_loft("c_inner", -1.3), "DIFFERENCE")
    cutters.append(rec)
    boolean(frame, cutters, "DIFFERENCE")
    clean(frame)
    finish(frame, M["frame"], 0.6, 3)

    # steel inserts: rails, pins, takedown lever
    steel = []
    for y0, y1 in ((104.0, 150.0), (1.0, 21.0)):
        for sx in (-1, 1):
            r = box("rail", min(sx * 4.9, sx * 8.3), max(sx * 4.9, sx * 8.3), y0, y1, -0.6, 2.4)
            steel.append(finish(r, M["steel"], 0.2, 2))
    for y, z in ((TRIGGER_PIVOT.y, TRIGGER_PIVOT.z), (100.5, -6.0), (13.0, -5.5)):
        pin = lathe("pin", [(0, -10.45), (1.15, -10.45), (1.5, -10.15), (1.5, 10.15), (1.15, 10.45), (0, 10.45)], 32)
        steel.append(finish(at(x_axis(pin), 0, y, z), M["steel"], 0.0))
    td = box("takedown", -10.62, 10.62, 88.4, 94.1, -5.2, -2.4)
    rid = [box("td_r", -10.8, 10.8, yy, yy + 0.6, -5.0, -2.6) for yy in (89.4, 91.0, 92.6)]
    boolean(td, rid, "UNION")
    steel.append(finish(td, M["steel"], 0.18, 2))
    return join([frame] + steel, "Frame")


def build_guide():
    prof = [(0, 92.0), (3.9, 92.0), (3.9, 93.8), (1.9, 94.4), (1.9, 151.4), (2.9, 152.0), (2.95, 155.8),
            (2.55, 156.3), (1.1, 156.3), (0.8, 156.0), (0, 156.0)]
    return lathe("RecoilGuide", prof, 48)


def build_slide(M):
    prof = fillet([(-11, 0, 0.35), (11, 0, 0.35), (11, 20.5, 1.8), (7.6, 27.0, 1.6), (-7.6, 27.0, 1.6),
                   (-11, 20.5, 1.8)])
    slide = prism("slide", prof, "Y", 0.0, SLIDE_LEN)
    c = [box("c_chan", -8.8, 8.8, -1, 158, -1, 4.5),
         box("c_breech", -8.5, 8.5, 40.0, BREECH_Y, -1, 9.3),
         prism("c_cav", fillet([(-7.25, -1), (7.25, -1), (7.25, 21.6, 2.5), (-7.25, 21.6, 2.5)]), "Y",
               BREECH_Y, 150.5),
         at(cyl("c_bore", 6.5, 148.0, 160.0), z=BORE_Z),
         at(cyl("c_rod", 3.25, 148.0, 160.0), z=GUIDE_Z),
         prism("c_port", fillet([(70.9, 15.2, 1.5), (99.8, 15.2, 1.5), (99.8, 40.0), (70.9, 40.0)]), "X",
               -4.6, 20.0),
         prism("c_nose", [(141.0, 29.0), (160.0, 29.0), (160.0, 22.5)], "X", -12, 12),
         prism("c_nose_r", [(11.6, 146.0), (11.6, 160.0), (9.9, 160.0)], "Z", -1, 30),
         prism("c_nose_l", [(-11.6, 146.0), (-9.9, 160.0), (-11.6, 160.0)], "Z", -1, 30),
         prism("c_chin", [(153.0, -1.0), (160.0, -1.0), (160.0, 2.2)], "X", -12, 12),
         box("c_plate", -4.6, 4.6, -1.0, 0.4, 8.0, 20.8),
         box("c_stopnotch", -11.5, -8.6, 88.0, 96.0, -1.0, 2.4),
         box("c_extractor", 10.35, 12.0, 58.6, 71.6, 15.7, 20.5)]
    for yc in [4.2 + 2.5 * k for k in range(9)] + [118.0 + 2.5 * k for k in range(6)]:
        for sx in (-1, 1):
            g = prism("c_serr", rrect(yc - 0.55, yc + 0.55, 2.6, 24.0, 0.5), "X",
                      min(sx * 10.22, sx * 13.0), max(sx * 10.22, sx * 13.0))
            c.append(g)
    boolean(slide, c, "DIFFERENCE")
    clean(slide)
    finish(slide, M["slide"], 0.35, 3)

    ext = prism("extractor", rrect(59.0, 71.2, 16.1, 20.1, 0.6), "X", 10.2, 11.2)
    finish(ext, M["steel"], 0.2, 2)
    plate = prism("rearplate", rrect(-4.3, 4.3, 8.3, 20.5, 1.0), "Y", 0.05, 0.4)
    boolean(plate, at(cyl("c_ind", 0.9, -1, 1), z=17.0), "DIFFERENCE")
    finish(plate, M["steel"], 0.08, 2)

    # rear sight: U notch, sloped front, two dots
    rs = prism("rsight", fillet([(-7.2, 26.7), (7.2, 26.7), (6.3, 31.5, 0.5), (1.7, 31.5, 0.3), (1.7, 29.5, 0.4),
                                 (-1.7, 29.5, 0.4), (-1.7, 31.5, 0.3), (-6.3, 31.5, 0.5)]), "Y", 6.0, 14.5)
    boolean(rs, prism("c_rs", [(11.5, 32.0), (15.0, 32.0), (15.0, 28.6)], "X", -8, 8), "DIFFERENCE")
    finish(rs, M["sight"], 0.2, 2)
    fsight = prism("fsight", fillet([(-1.9, 26.7), (1.9, 26.7), (1.65, 31.4, 0.6), (-1.65, 31.4, 0.6)]), "Y",
                   141.0, 146.5)
    boolean(fsight, prism("c_fs", [(144.0, 32.0), (147.0, 32.0), (147.0, 28.0)], "X", -3, 3), "DIFFERENCE")
    finish(fsight, M["sight"], 0.15, 2)
    dots = [at(cyl("dot", 0.9, 5.75, 6.15, 32), x=sx * 4.1, z=29.4) for sx in (-1, 1)]
    dots.append(at(cyl("dot", 0.8, 140.75, 141.15, 32), z=29.8))
    for d in dots:
        finish(d, M["dot"], 0.12, 2)
    return join([slide, ext, plate, rs, fsight] + dots, "Slide")


def build_barrel(M):
    def rifling(r, y, th):
        return r - 0.11 * math.cos(6 * th) if r < 4.56 else r

    prof = [(4.98, BREECH_Y), (7.0, BREECH_Y), (7.0, 99.0), (6.3, 101.0), (6.3, 155.9), (5.85, 156.6),
            (5.05, 156.6), (4.52, 156.05), (4.5, 155.5), (4.5, 90.5), (4.98, 89.8)]
    tube = at(lathe("barrel", prof, 96, rifling), z=BORE_Z)
    hood = prism("hood", fillet([(-7.0, 12.0), (7.0, 12.0), (7.0, 26.4, 1.4), (-7.0, 26.4, 1.4)]), "Y",
                 BREECH_Y, 99.3)
    lug = prism("lug", fillet([(74.0, 9.5), (95.5, 9.5), (95.5, 5.5, 1.0), (90.5, 2.5, 1.5), (80.0, 2.5, 1.0),
                               (74.0, 6.5, 1.5)]), "X", -4.9, 4.9)
    boolean(tube, [hood, lug], "UNION")
    boolean(tube, [at(cyl("c_ch", 4.98, BREECH_Y - 1, 89.8, 96), z=BORE_Z),
                   prism("c_ramp", [(68.0, 4.0), (68.0, 10.6), (70.6, 10.6), (75.5, 9.3), (75.5, 4.0)], "X",
                         -3.6, 3.6)], "DIFFERENCE")
    clean(tube)
    return finish(tube, M["barrel"], 0.3, 3, angle=30.0)


def build_trigger(M):
    front = [(87.0, -4.5), (86.8, -11.0), (86.2, -15.5), (85.0, -20.0), (83.4, -24.3), (81.6, -27.4),
             (80.2, -28.4)]
    back = [(79.0, -27.8), (79.6, -24.5), (80.8, -20.5), (81.8, -16.0), (82.4, -11.0), (82.6, -4.5)]
    outline = catmull(front + back, 6)
    trig = prism("trigger", outline, "X", -3.1, 3.1)
    fc = np.array(catmull(front, 10, closed=False))
    sel = fc[(fc[:, 1] < -11.5) & (fc[:, 1] > -26.6)]
    d = np.gradient(sel, axis=0)
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)

    def band(o_out, o_in):
        return [tuple(p) for p in sel + nrm * o_out] + [tuple(p) for p in (sel - nrm * o_in)[::-1]]

    boolean(trig, prism("c_slot", band(3.0, 2.2), "X", -1.3, 1.3), "DIFFERENCE")
    clean(trig)
    finish(trig, M["polymer"], 0.7, 4, angle=30.0)
    blade = prism("safety", band(0.9, 2.0), "X", -0.95, 0.95)
    finish(blade, M["polymer"], 0.3, 3, angle=30.0)
    return join([trig, blade], "Trigger")


def build_slide_stop(M):
    poly = fillet([(58.0, -1.2, 0.8), (83.5, -1.2, 1.2), (86.4, -5.5, 2.0), (85.6, -9.0, 2.0), (82.4, -9.2, 2.0),
                   (81.4, -6.0, 1.0), (80.0, -4.4, 0.8), (64.5, -4.4, 1.0), (63.2, -6.6, 1.2), (58.0, -6.4, 1.0)])
    ss = prism("slidestop", poly, "X", -11.0, -9.8)
    boolean(ss, [box("c_g", -11.5, -10.75, yy, yy + 0.5, -7.0, -1.8) for yy in (58.9, 60.3, 61.7)], "DIFFERENCE")
    clean(ss)
    return finish(ss, M["steel"], 0.2, 2)


def build_mag_release(M):
    outline, _ = magrelease_outline(0.0)
    btn = prism("magrelease", outline, "X", -20.0, -4.0)
    boolean(btn, grip_loft("c_outer", 0.9), "INTERSECT")
    boolean(btn, grip_loft("c_inner2", -1.2), "DIFFERENCE")
    s0, t0 = 15.0, t_at(15.0, -23.0)
    grooves = []
    for ds in (-2.0, 0.0, 2.0):
        pts = [gp(s, t) for s, t in rrect(s0 + ds - 0.3, s0 + ds + 0.3, t0 - 2.3, t0 + 2.3, 0.25)]
        g = prism("c_mg", pts, "X", -20.0, 0.0)
        boolean(g, grip_loft("c_go", 0.6), "DIFFERENCE")
        grooves.append(g)
    boolean(btn, grooves, "DIFFERENCE")
    clean(btn)
    return finish(btn, M["polymer"], 0.3, 3, angle=30.0)


# ---------------------------------------------------------------------------------------------
# Ammunition (local mm, along +Y, centred on the origin)
# ---------------------------------------------------------------------------------------------
CASE_L = 19.15
OAL = 29.69


def _case(spent):
    p = [(0, 0.62), (0.35, 0.58), (0.65, 0.45), (0.85, 0.28), (1.0, 0.14)] if spent else [(0, 0.12)]
    p += [(2.05, 0.12), (2.18, 0.22), (2.3, 0.02), (4.55, 0.0), (4.98, 0.3), (4.98, 1.02), (4.8, 1.27),
          (4.42, 1.45), (4.42, 1.95), (4.965, 2.75), (4.83, CASE_L - 0.2), (4.76, CASE_L), (4.52, CASE_L),
          (4.47, CASE_L - 0.6), (4.47, 6.0), (3.0, 5.2), (0, 5.0)]
    obj = lathe("case", p, 64)
    me = obj.data
    me.materials.append(None)
    me.materials.append(None)
    for poly in me.polygons:
        c = poly.center
        poly.material_index = 1 if (c.y < 0.3 and math.hypot(c.x, c.z) < 2.12) else 0
    return obj


def _bullet():
    R = 13.1
    p = [(0, 13.4), (4.40, 13.4), (4.51, 14.0), (4.51, 19.8)]
    for k in range(1, 15):
        y = 19.8 + k / 15 * (OAL - 19.8)
        r = math.sqrt(max(R * R - (y - 19.8) ** 2, 0)) - (R - 4.51)
        if r < 0.7:
            break
        p.append((r, y))
    p += [(0.7, OAL - 0.05), (0, OAL)]
    return lathe("bullet", p, 64)


def _ammo_finish(obj, M, mats):
    obj.data.materials.clear()
    for k in mats:
        obj.data.materials.append(M[k])
    obj.data.shade_smooth()
    obj.data.set_sharp_from_angle(angle=math.radians(50))
    return obj


def build_casing(M):
    c = at(_case(True), y=-CASE_L / 2)
    return _ammo_finish(c, M, ["brass", "primer"])


def build_cartridge(M):
    c = _ammo_finish(_case(False), M, ["brass", "primer"])
    b = _ammo_finish(_bullet(), M, ["copper"])
    obj = join([c, b], "Cartridge")
    return at(obj, y=-OAL / 2)


# ---------------------------------------------------------------------------------------------
# Magazine (local mm: origin top centre, +Z insertion axis, +Y forward)
# ---------------------------------------------------------------------------------------------
MAG_HW, MAG_HD = 7.75, 16.4
ROUND_TILT = math.radians(10.0)
ROUND_PITCH = 9.95 / math.cos(ROUND_TILT)
ROUND_TOP_Z = -3.5


def to_mag_local(p):
    d = Vector(p) - MAG_ORIGIN
    return Vector((d.x, d.dot(FAX), d.dot(UAX)))


def build_magazine(M):
    zb = -MAG_LEN
    body = prism("mag", rrect(-MAG_HW, MAG_HW, -MAG_HD, MAG_HD, 2.6), "Z", zb, -1.8)
    cut = [prism("c_in", rrect(-MAG_HW + 0.65, MAG_HW - 0.65, -MAG_HD + 0.65, MAG_HD - 0.65, 2.0), "Z", zb + 1.0, 5.0),
           prism("c_front", [(5.0, 12.0), (5.0, -1.2), (9.5, -6.0), (20.0, -6.0), (20.0, 12.0)], "X", -10, 10)]
    for k in range(1, 6):
        zc = ROUND_TOP_Z - k * ROUND_PITCH - 2.6
        cut.append(at(cyl("c_wit", 1.35, -18.0, -14.9, 32), z=zc))
    _, rel = magrelease_outline()
    lz = to_mag_local((0.0, rel[0], rel[1])).z
    cut.append(box("c_notch", -2.6, 2.6, 15.0, 18.0, lz - 1.8, lz + 1.8))
    boolean(body, cut, "DIFFERENCE")
    clean(body)
    finish(body, M["mag"], 0.3, 2)

    lip_r = catmull([(7.75, -3.0), (7.6, -1.4), (6.8, -0.45), (5.6, -0.05), (4.3, -0.1), (3.85, -0.45),
                     (4.2, -0.9), (5.4, -0.7), (6.4, -1.0), (6.95, -1.7), (7.1, -3.0)], 4)
    lips = []
    for sx in (1, -1):
        lip = prism("lip", [(sx * x, z) for x, z in lip_r], "Y", -MAG_HD + 0.3, 5.5)
        lips.append(finish(lip, M["mag"], 0.0))

    plate = prism("baseplate", rrect(-8.4, 8.4, -15.8, 18.4, 3.5), "Z", zb - 2.8, zb + 0.6)
    finish(plate, M["polymer"], 0.9, 4)

    rounds = []
    for k in range(6):
        r = build_cartridge(M)
        xform(r, Matrix.Rotation(ROUND_TILT, 4, "X"))
        at(r, z=ROUND_TOP_Z - k * ROUND_PITCH)
        rounds.append(r)
    zf = ROUND_TOP_Z - 5 * ROUND_PITCH - 4.98
    tan = math.tan(ROUND_TILT)
    fol = prism("follower", fillet([(-15.4, zf - 0.3 - 15.4 * tan, 1.5), (15.4, zf - 0.3 + 15.4 * tan, 1.5),
                                    (15.4, zf - 9.0, 1.0), (-15.4, zf - 9.0, 1.0)]), "X", -7.0, 7.0)
    finish(fol, M["follower"], 0.4, 2)
    return join([body] + lips + [plate, fol] + rounds, "Magazine")


# ---------------------------------------------------------------------------------------------
# Muzzle flash (local mm, origin at the muzzle, +Y forward)
# ---------------------------------------------------------------------------------------------
def build_flash(M):
    parts = [lathe("core", catmull([(0.01, -1.0), (4.5, 0.5), (7.5, 5.0), (7.0, 12.0), (4.5, 24.0), (1.5, 38.0),
                                    (0.01, 44.0)], 5, closed=False), 48)]
    rng = np.random.default_rng(7)
    for k in range(6):
        length = 22.0 + rng.uniform(-4, 6)
        cone = lathe("petal", [(0.01, 0.0), (3.4, 2.5), (2.6, length * 0.45), (0.9, length * 0.85), (0.01, length)], 24)
        xform(cone, Matrix.Diagonal((0.33, 1.0, 1.0, 1.0)))
        xform(cone, Matrix.Rotation(math.radians(68 + rng.uniform(-8, 8)), 4, "X"))
        xform(cone, Matrix.Rotation(math.radians(k * 60 + 15 + rng.uniform(-10, 10)), 4, "Y"))
        at(cone, y=2.0)
        parts.append(cone)
    fl = join(parts, "MuzzleFlash")
    fl.data.materials.append(M["flash"])
    fl.data.shade_smooth()
    return fl


# ---------------------------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------------------------
def _empty(name, size, parent=None, display="PLAIN_AXES"):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = display
    e.empty_display_size = size
    _link(e)
    if parent is not None:
        e.parent = parent
    return e


def _place(obj, root, origin_g, rot_x=0.0, local=False):
    """Move mesh data so its origin is origin_g (gun-space mm), convert to metres, parent to root."""
    if not local:
        obj.data.transform(Matrix.Translation(-Vector(origin_g)))
    obj.data.transform(Matrix.Scale(MM, 4))
    obj.location = (Vector(origin_g) - SOCKET) * MM
    obj.rotation_euler = (rot_x, 0.0, 0.0)
    obj.parent = root
    return obj


def build_pistol(collection=None):
    M = make_materials()
    root = _empty("Pistol", 0.05, display="ARROWS")
    grip_socket = _empty("Pistol_GripSocket", 0.02, root)

    frame = build_frame(M)
    slide = build_slide(M)
    barrel = build_barrel(M)
    guide = build_guide()
    finish(guide, M["steel"], 0.0)
    trigger = build_trigger(M)
    stop = build_slide_stop(M)
    release = build_mag_release(M)
    mag = build_magazine(M)
    casing = build_casing(M)
    cart = build_cartridge(M)
    flash = build_flash(M)

    _, rel = magrelease_outline()
    rel_c = Vector((-10.3, rel[0], rel[1]))
    for o, n in ((frame, "Pistol_Frame"), (slide, "Pistol_Slide"), (barrel, "Pistol_Barrel"),
                 (guide, "Pistol_RecoilGuide"), (trigger, "Pistol_Trigger"), (stop, "Pistol_SlideStop"),
                 (release, "Pistol_MagRelease"), (mag, "Pistol_Magazine"), (casing, "Pistol_Casing"),
                 (cart, "Pistol_Cartridge"), (flash, "Pistol_MuzzleFlash")):
        o.name = n
        o.data.name = n

    _place(frame, root, SOCKET)
    _place(slide, root, (0.0, 0.0, 0.0))
    _place(barrel, root, BARREL_PIVOT)
    _place(guide, root, (0.0, 0.0, GUIDE_Z), local=True)
    _place(trigger, root, TRIGGER_PIVOT)
    _place(stop, root, (-10.4, TRIGGER_PIVOT.y, TRIGGER_PIVOT.z))
    _place(release, root, rel_c)
    _place(mag, root, MAG_ORIGIN, rot_x=-RAKE, local=True)
    _place(casing, root, (0.0, BREECH_Y + CASE_L / 2, BORE_Z), local=True)
    _place(cart, root, (0.0, BREECH_Y + OAL / 2, BORE_Z), local=True)
    _place(flash, root, (0.0, SLIDE_LEN + 0.2, BORE_Z), local=True)

    mag_socket = _empty("Pistol_MagSocket", 0.015, root)
    mag_socket.location = (MAG_ORIGIN - SOCKET) * MM
    mag_socket.rotation_euler = (-RAKE, 0.0, 0.0)

    for o in (casing, cart, flash):
        o.hide_render = True
        o.hide_viewport = True

    objs = [root, grip_socket, mag_socket, frame, slide, barrel, guide, trigger, stop, release, mag, casing,
            cart, flash]
    if collection is not None:
        for o in objs:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            collection.objects.link(o)
    global _TAG
    if _TAG is not None and _TAG.users == 0:
        bpy.data.materials.remove(_TAG)
        _TAG = None
    return {"root": root, "frame": frame, "slide": slide, "barrel": barrel, "trigger": trigger, "magazine": mag,
            "mag_release": release, "slide_stop": stop, "muzzle_flash": flash, "casing": casing,
            "cartridge": cart, "grip_socket": grip_socket, "mag_socket": mag_socket, "recoil_guide": guide}


# ---------------------------------------------------------------------------------------------
# Standalone preview
# ---------------------------------------------------------------------------------------------
def _studio(C, target):
    t = Vector(target)
    C.world_color((0.006, 0.0062, 0.007), 1.0)
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    fm, nt, p = _new_mat("Floor")
    p.inputs["Base Color"].default_value = (0.022, 0.022, 0.024, 1)
    p.inputs["Roughness"].default_value = 0.42
    floor.data.materials.append(fm)
    C.area_light("Key", t + Vector((-0.25, -0.35, 0.75)), t, 6.0, 0.7, (1.0, 0.97, 0.93))
    s1 = C.area_light("StripR", t + Vector((0.55, 0.25, 0.25)), t, 3.0, 0.1, (0.93, 0.96, 1.0))
    s1.data.shape, s1.data.size, s1.data.size_y = "RECTANGLE", 0.06, 0.9
    s2 = C.area_light("StripL", t + Vector((-0.55, 0.3, 0.2)), t, 2.5, 0.1, (0.93, 0.96, 1.0))
    s2.data.shape, s2.data.size, s2.data.size_y = "RECTANGLE", 0.06, 0.9
    C.area_light("Top", t + Vector((0.05, 0.1, 0.9)), t, 2.0, 1.2, (1.0, 1.0, 1.0))
    C.area_light("Fill", t + Vector((0.6, -0.6, 0.15)), t, 0.8, 0.8, (0.9, 0.93, 1.0))


def _standalone():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import common as C

    a = C.args()
    out_blend = os.path.abspath(a[0] if a else os.path.join(C.BUILD, "pistol.blend"))
    prefix = os.path.abspath(a[1] if len(a) > 1 else os.path.join(C.RENDERS, "pistol"))
    samples = int(os.environ.get("PISTOL_SAMPLES", "160"))
    C.reset_scene()
    parts = build_pistol()
    root = parts["root"]
    root.location = (0.0, 0.0, 0.085)
    total = sum(tris(o) for o in bpy.data.objects if o.type == "MESH")
    print("PISTOL_TRIS", total, {k: tris(v) for k, v in parts.items() if v.type == "MESH"})

    C.setup_cycles(samples=samples, res=(1600, 1000))
    sc = bpy.context.scene
    sc.cycles.blur_glossy = 0.5
    sc.cycles.max_bounces = 8
    bpy.context.view_layer.update()
    center = root.matrix_world @ ((Vector((0.0, 78.0, -23.0)) - SOCKET) * MM)
    _studio(C, center)
    cam = C.add_camera((0, -1, 0.1), center, lens=90, name="PistolCam")
    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    C.orbit_camera(cam, center, 0.62, 62, 16)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)
    print("SAVED", out_blend)

    C.render(f"{prefix}_34.png")
    C.orbit_camera(cam, center, 0.62, -90, 4)
    C.render(f"{prefix}_left.png")

    # exploded: slide locked back, barrel tilted, magazine dropped, a spent case in flight
    parts["slide"].location.y -= 0.028
    parts["barrel"].location.y -= 0.003
    parts["barrel"].rotation_euler.x = math.radians(5)
    parts["slide_stop"].rotation_euler.x = math.radians(-6)
    mag = parts["magazine"]
    mag.location -= (mag.matrix_basis.to_3x3() @ Vector((0, 0, 1))) * 0.058
    loose = parts["casing"].copy()
    bpy.context.scene.collection.objects.link(loose)
    loose.hide_render = loose.hide_viewport = False
    loose.location = Vector((0.032, 0.02, 0.05))
    loose.rotation_euler = (math.radians(35), math.radians(20), math.radians(60))
    rnd = parts["cartridge"].copy()
    bpy.context.scene.collection.objects.link(rnd)
    rnd.hide_render = rnd.hide_viewport = False
    rnd.parent = None
    rnd.location = root.matrix_world @ ((Vector((26.0, 70.0, 0.0)) - SOCKET) * MM)
    rnd.location.z = 0.00498
    rnd.rotation_euler = (0.0, 0.0, math.radians(-30))
    C.orbit_camera(cam, center + Vector((0, -0.01, -0.012)), 0.70, 70, 14)
    C.render(f"{prefix}_exploded.png")


if __name__ == "__main__":
    _standalone()
