from __future__ import annotations

from pathlib import Path
import math

import bpy
from mathutils import Vector


def clear_scene() -> None:
    """Remove all objects from the current Blender scene."""
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_render_settings(
    resolution: int = 128,
    samples: int = 64,
    engine: str = "CYCLES",
    use_gpu: bool = True,
    background_color=(0.5, 0.5, 0.5, 1.0),
    background_strength: float = 1.0,
) -> bpy.types.Scene:
    """Configure render settings for an offline RGB render."""
    scene = bpy.context.scene

    scene.render.engine = engine

    if engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = True
        if use_gpu:
            scene.cycles.device = "GPU"

    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    if scene.world is not None:
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg is not None:
            bg.inputs["Color"].default_value = background_color
            bg.inputs["Strength"].default_value = float(background_strength)

    return scene


def import_largest_mesh_from_blend(object_path: str | Path) -> bpy.types.Object:
    """
    Import all objects from a .blend file and return the largest mesh.
    This is useful for object files containing lights, cameras, empties, etc.
    """
    object_path = Path(object_path)

    before_names = set(bpy.data.objects.keys())

    with bpy.data.libraries.load(str(object_path), link=False) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects]

    for obj in data_to.objects:
        if obj is not None:
            try:
                bpy.context.scene.collection.objects.link(obj)
            except RuntimeError:
                pass

    after_names = set(bpy.data.objects.keys())
    new_objects = [bpy.data.objects[name] for name in after_names - before_names]

    meshes = [
        obj for obj in new_objects
        if obj.type == "MESH" and obj.data is not None and len(obj.data.vertices) > 0
    ]

    if not meshes:
        raise RuntimeError(f"No mesh object found in {object_path}")

    obj = max(meshes, key=lambda x: len(x.data.vertices))
    obj.name = "target_object"

    return obj


def center_and_scale_object(
    obj: bpy.types.Object,
    target_size: float = 0.9,
    apply_transform: bool = True,
) -> bpy.types.Object:
    """
    Center object at origin and scale it to fit approximately inside a target box.
    """
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Compute world-space bounding box
    bpy.context.view_layer.update()
    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    min_corner = Vector((
        min(v.x for v in bbox_world),
        min(v.y for v in bbox_world),
        min(v.z for v in bbox_world),
    ))
    max_corner = Vector((
        max(v.x for v in bbox_world),
        max(v.y for v in bbox_world),
        max(v.z for v in bbox_world),
    ))

    center = 0.5 * (min_corner + max_corner)
    size = max(max_corner - min_corner)

    obj.location -= center

    if size > 0:
        scale = float(target_size) / float(size)
        obj.scale *= scale

    if apply_transform:
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

    return obj


def rotate_object(
    obj: bpy.types.Object,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
) -> bpy.types.Object:
    """Apply a simple object orientation."""
    obj.rotation_euler[1] += math.radians(float(azimuth_deg))
    obj.rotation_euler[0] += math.radians(float(elevation_deg))

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    return obj


def add_area_light(
    location=(0.0, 0.0, 5.0),
    rotation=(0.0, 0.0, 0.0),
    size: float = 5.0,
    strength: float = 300.0,
    name: str = "main_area_light",
) -> bpy.types.Object:
    """Add a large area light."""
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = float(strength)
    light_data.size = float(size)

    light = bpy.data.objects.new(name, light_data)
    bpy.context.scene.collection.objects.link(light)

    light.location = location
    light.rotation_euler = rotation

    return light


def create_camera(
    location=(0.0, 0.0, 2.5),
    focal_length: float = 50.0,
    sensor_width_mm: float = 32.0,
    name: str = "render_camera",
) -> bpy.types.Object:
    """Create a camera looking along its local -Z axis."""
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)

    cam.location = location
    cam.rotation_mode = "QUATERNION"

    cam.data.lens = float(focal_length)
    cam.data.sensor_width = float(sensor_width_mm)

    bpy.context.scene.camera = cam

    return cam