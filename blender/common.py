"""Helpers shared by the Blender scripts (run inside Blender 5.1's Python)."""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
RENDERS = os.path.join(ROOT, "renders")


def args():
    """Arguments after `--` on the Blender command line."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    return scene


def mesh_from_faces(name, verts, faces, sizes, smooth=True):
    """Mesh from a flat vertex-index array and per-face sizes (mixed quads / triangles)."""
    verts = np.asarray(verts, np.float32)
    faces = np.asarray(faces, np.int32)
    sizes = np.asarray(sizes, np.int32)
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(len(faces))
    me.loops.foreach_set("vertex_index", faces)
    me.polygons.add(len(sizes))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int32)
    me.polygons.foreach_set("loop_start", starts)
    me.update(calc_edges=True)
    if smooth:
        me.shade_smooth()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_from_quads(name, verts, quads, smooth=True):
    q = np.asarray(quads, np.int32)
    return mesh_from_faces(name, verts, q.ravel(), np.full(len(q), 4, np.int32), smooth)


def load_npz_mesh(path, name):
    data = np.load(path)
    if "faces" in data.files:
        obj = mesh_from_faces(name, data["verts"], data["faces"], data["sizes"])
    else:
        obj = mesh_from_quads(name, data["verts"], data["quads"])
    for key in data.files:
        if key.startswith("attr_"):
            a = data[key]
            attr = obj.data.attributes.new(key[5:], "FLOAT", "POINT")
            attr.data.foreach_set("value", a.astype(np.float32))
    return obj


def clay_material(color=(0.62, 0.6, 0.58), rough=0.55, name="Clay"):
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    p = nt.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = rough
    return mat


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def add_camera(loc, target, lens=85, name="Cam"):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    cam.clip_start = 0.01
    cam.clip_end = 100
    obj = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = loc
    look_at(obj, target)
    bpy.context.scene.camera = obj
    return obj


def orbit_camera(cam, target, dist, yaw_deg, pitch_deg=0.0):
    """Place cam on a sphere around target. yaw 0 = in front of the character (-Y side)."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    t = Vector(target)
    cam.location = t + Vector((math.sin(yaw) * math.cos(pitch) * dist, -math.cos(yaw) * math.cos(pitch) * dist, math.sin(pitch) * dist))
    look_at(cam, t)


def area_light(name, loc, target, power, size, color=(1, 1, 1)):
    light = bpy.data.lights.new(name, "AREA")
    light.energy = power
    light.size = size
    light.color = color
    obj = bpy.data.objects.new(name, light)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = loc
    look_at(obj, target)
    return obj


def studio_lights(target=(0, 0, 1.0), scale=1.0):
    """Key / fill / rim area lights around the target."""
    t = Vector(target)
    area_light("Key", t + Vector((-1.6, -2.2, 1.2)) * scale, t, 900 * scale * scale, 1.6 * scale, (1.0, 0.96, 0.9))
    area_light("Fill", t + Vector((2.4, -1.6, 0.3)) * scale, t, 250 * scale * scale, 2.5 * scale, (0.9, 0.95, 1.0))
    area_light("Rim", t + Vector((1.2, 2.4, 1.4)) * scale, t, 700 * scale * scale, 1.0 * scale, (1.0, 1.0, 1.0))


def world_color(color=(0.05, 0.05, 0.055), strength=1.0):
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (*color, 1)
    bg.inputs["Strength"].default_value = strength
    return world


def ground(size=20, color=(0.18, 0.18, 0.19)):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    g = bpy.context.active_object
    g.name = "Ground"
    g.data.materials.append(clay_material(color, 0.8, "GroundMat"))
    return g


def setup_cycles(samples=128, res=(1600, 1000), denoise=True):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = denoise
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    return scene


def setup_workbench(res=(1600, 1000)):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = (0.75, 0.72, 0.68)
    sh.show_cavity = True
    sh.cavity_type = "BOTH"
    sh.show_shadows = True
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    return scene


def render(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = path
    scene.render.image_settings.file_format = "PNG" if path.endswith(".png") else "JPEG"
    if path.endswith(".jpg"):
        scene.render.image_settings.quality = 90
    bpy.ops.render.render(write_still=True)
    print("RENDERED", path)
