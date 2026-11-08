"""Animation runtime: write pose-engine output as keyframes, key props, stage, camera, render to MP4."""
import math
import os
import subprocess
import sys

import bpy
import numpy as np
from bpy_extras import anim_utils
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

FPS = 30


def _channelbag(ob, name):
    ob.animation_data_create()
    action = bpy.data.actions.new(name)
    ob.animation_data.action = action
    slot = ob.animation_data.action_slot
    if slot is None:
        slot = action.slots.new(id_type="OBJECT", name=ob.name)
        ob.animation_data.action_slot = slot
    return anim_utils.action_ensure_channelbag_for_slot(action, slot)


def _fcurve(cb, path, index, frames, values, group=None, constant=False):
    fc = cb.fcurves.new(path, index=index)
    if group:
        g = cb.groups.get(group) or cb.groups.new(group)
        fc.group = g
    n = len(frames)
    fc.keyframe_points.add(n)
    co = np.empty(2 * n, np.float32)
    co[0::2] = frames
    co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)
    interp = 0 if constant else 1    # CONSTANT, LINEAR
    fc.keyframe_points.foreach_set("interpolation", np.full(n, interp, np.int32))
    fc.update()
    return fc


def continuous(q):
    """Flip quaternion signs so consecutive frames take the short path."""
    q = np.array(q, float)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def key_rig(rig_ob, frames, quats, root_locs, name="Clip"):
    """quats: list over frames of {bone: (w,x,y,z)}; root_locs: list of bone-local root locations."""
    for pb in rig_ob.pose.bones:
        pb.rotation_mode = "QUATERNION"
    cb = _channelbag(rig_ob, name)
    frames = np.asarray(frames, np.float32)
    for bone in quats[0]:
        series = continuous([q[bone] for q in quats])
        path = f'pose.bones["{bone}"].rotation_quaternion'
        for i in range(4):
            _fcurve(cb, path, i, frames, series[:, i], group=bone)
    locs = np.asarray(root_locs, float)
    for i in range(3):
        _fcurve(cb, 'pose.bones["root"].location', i, frames, locs[:, i], group="root")
    return cb


def key_object(ob, frames, mats, name=None, visible=None):
    """mats: list of 4x4 world matrices (numpy or Matrix); visible: optional list of bools."""
    ob.rotation_mode = "QUATERNION"
    cb = _channelbag(ob, name or ob.name + "_anim")
    locs, rots, scl = [], [], []
    for m in mats:
        M = Matrix(np.asarray(m).tolist()) if not isinstance(m, Matrix) else m
        loc, rot, sc = M.decompose()
        locs.append(loc[:]); rots.append(rot[:]); scl.append(sc[:])
    rots = continuous(rots)
    scl = np.asarray(scl, float)
    if visible is not None:
        scl = scl * np.asarray(visible, float)[:, None]
    frames = np.asarray(frames, np.float32)
    for i in range(3):
        _fcurve(cb, "location", i, frames, np.asarray(locs)[:, i])
        _fcurve(cb, "scale", i, frames, scl[:, i], constant=visible is not None)
    for i in range(4):
        _fcurve(cb, "rotation_quaternion", i, frames, rots[:, i])
    return cb


# ------------------------------------------------------------------ stage
def outdoor_stage(sun_elev=24.0, sun_az=215.0, strength=1.0):
    scene = bpy.context.scene
    world = bpy.data.worlds.new("Sky")
    scene.world = world
    nt = world.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(sun_elev)
    sky.sun_rotation = math.radians(sun_az)
    sky.altitude = 50
    sky.air_density = 1.0
    sky.aerosol_density = 1.2
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = 0.22 * strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(sky.outputs[0], bg.inputs[0])
    nt.links.new(bg.outputs[0], out.inputs[0])
    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.2 * strength
    sun_data.angle = math.radians(0.6)
    sun_data.color = (1.0, 0.93, 0.84)
    sun = bpy.data.objects.new("Sun", sun_data)
    scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(90 - sun_elev), 0, math.radians(sun_az))
    ground = asphalt_ground()
    return sun, ground


def asphalt_ground(size=60.0):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    g = bpy.context.active_object
    g.name = "Ground"
    mat = bpy.data.materials.new("Asphalt")
    nt = mat.node_tree
    p = nt.nodes.get("Principled BSDF")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise"); n1.inputs["Scale"].default_value = 900.0; n1.inputs["Detail"].default_value = 6
    n2 = nt.nodes.new("ShaderNodeTexNoise"); n2.inputs["Scale"].default_value = 2.5; n2.inputs["Detail"].default_value = 4
    v = nt.nodes.new("ShaderNodeTexVoronoi"); v.inputs["Scale"].default_value = 450.0
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], n2.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], v.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.035, 0.035, 0.037, 1)
    ramp.color_ramp.elements[1].color = (0.11, 0.105, 0.10, 1)
    mixv = nt.nodes.new("ShaderNodeMath"); mixv.operation = "MULTIPLY_ADD"
    mixv.inputs[1].default_value = 0.6
    nt.links.new(n1.outputs["Fac"], mixv.inputs[0])
    nt.links.new(n2.outputs["Fac"], mixv.inputs[2])
    nt.links.new(mixv.outputs[0], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.85
    bump = nt.nodes.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.6; bump.inputs["Distance"].default_value = 0.002
    nt.links.new(v.outputs["Distance"], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], p.inputs["Normal"])
    g.data.materials.append(mat)
    return g


def setup_eevee(res=(1280, 720), samples=48, blur=True):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = samples
    scene.eevee.use_raytracing = True
    scene.eevee.use_shadows = True
    scene.eevee.use_fast_gi = True
    scene.render.use_motion_blur = blur
    scene.render.motion_blur_shutter = 0.4
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.fps = FPS
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.render.image_settings.file_format = "PNG"
    for m in bpy.data.materials:
        if hasattr(m, "use_raytrace_refraction") and m.name.startswith("Cornea"):
            m.use_raytrace_refraction = True
    return scene


def follow_camera(name, frames, targets, offset, lens=40.0, smooth=8):
    """Camera that tracks a (smoothed) target path from a fixed relative offset."""
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    t = np.asarray(targets, float)
    k = np.ones(smooth) / smooth
    sm = np.stack([np.convolve(np.pad(t[:, i], (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), k, "valid")
                   for i in range(3)], 1)
    mats = []
    for p in sm:
        loc = Vector(p) + Vector(offset)
        rot = (Vector(p) - loc).to_track_quat("-Z", "Y")
        mats.append(Matrix.LocRotScale(loc, rot, Vector((1, 1, 1))))
    key_object(cam, frames, mats, name=name + "_anim")
    return cam


def render_mp4(out_mp4, frame_start, frame_end, tmp_dir=None, crf=20):
    scene = bpy.context.scene
    tmp_dir = tmp_dir or os.path.splitext(out_mp4)[0] + "_frames"
    os.makedirs(tmp_dir, exist_ok=True)
    scene.frame_start, scene.frame_end = frame_start, frame_end
    scene.render.filepath = os.path.join(tmp_dir, "f_")
    bpy.ops.render.render(animation=True)
    cmd = ["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-start_number", str(frame_start),
           "-i", os.path.join(tmp_dir, "f_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
           "-preset", "slow", "-movflags", "+faststart", out_mp4]
    subprocess.run(cmd, check=True)
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)
    print("MP4", out_mp4)
