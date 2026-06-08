from __future__ import annotations

from pathlib import Path
import math

import bpy
from mathutils import Vector, Matrix


def clear_scene() -> None:
    """Remove all objects from the current Blender scene."""
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def setup_render_settings(
    resolution: int = 128,
    samples: int = 32,
    engine: str = "CYCLES",
    use_gpu: bool = True,
    background_color=(0.5, 0.5, 0.5, 1.0),
    background_strength: float = 1.0,
) -> bpy.types.Scene:
    """Configure Blender for offline RGB rendering."""
    scene = bpy.context.scene
    scene.render.engine = engine

    if engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = True

        if hasattr(scene.cycles, "device"):
            scene.cycles.device = "GPU" if use_gpu else "CPU"

        # Set only properties that exist in the installed Blender version.
        cycles_settings = {
            "max_bounces": 2,
            "diffuse_bounces": 2,
            "glossy_bounces": 2,
            "transmission_bounces": 2,
            "transparent_max_bounces": 2,
            "transparent_min_bounces": 2,
            "caustics_reflective": False,
            "caustics_refractive": False,
            "use_persistent_data": True,
        }

        for name, value in cycles_settings.items():
            if hasattr(scene.cycles, name):
                setattr(scene.cycles, name, value)

    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    scene.render.film_transparent = False

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

    This matches the important Toys4K preprocessing from the online repo:

        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (-90 deg, 0, 0)
        origin = geometry bounds
        location = (0, 0, 0)
        apply scale, location, rotation

    The -90 deg X canonicalization is applied before the independent
    base_azim/base_elev rotations.
    """
    object_path = Path(object_path)

    before_names = set(bpy.data.objects.keys())

    with bpy.data.libraries.load(str(object_path), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)

    for obj in data_to.objects:
        if obj is not None:
            try:
                bpy.context.scene.collection.objects.link(obj)
            except RuntimeError:
                pass

    after_names = set(bpy.data.objects.keys())
    new_objects = [bpy.data.objects[name] for name in after_names - before_names]

    meshes = [
        obj
        for obj in new_objects
        if obj.type == "MESH"
        and obj.data is not None
        and len(obj.data.vertices) > 0
    ]

    if not meshes:
        raise RuntimeError(f"No mesh object found in {object_path}")

    obj = max(meshes, key=lambda x: len(x.data.vertices))
    obj.name = "object"

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Toys4K / ActiveVisSim canonical orientation.
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = (math.radians(-90.0), 0.0, 0.0)

    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0.0, 0.0, 0.0)

    # Bake canonical transform before applying base_azim/base_elev.
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

    return obj


def center_and_scale_object(
    obj: bpy.types.Object,
    target_size: float = 0.45,
    apply_transform: bool = True,
) -> bpy.types.Object:
    """
    Scale object using the ActiveVisSim convention.

    In the online repo the scale is effectively:

        obj.scale = obj.scale * 0.45 / max(abs(vertices))

    Therefore target_size should be 0.45 if you want to match that repo.
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    obj.location = (0.0, 0.0, 0.0)

    bpy.context.view_layer.update()

    max_abs = 0.0
    for v in obj.data.vertices:
        max_abs = max(
            max_abs,
            abs(float(v.co.x)),
            abs(float(v.co.y)),
            abs(float(v.co.z)),
        )

    if max_abs > 0.0:
        obj.scale = obj.scale * (float(target_size) / float(max_abs))

    if apply_transform:
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

    return obj


def apply_global_axis_rotation(
    obj: bpy.types.Object,
    axis: str,
    angle_deg: float,
) -> None:
    """
    Apply an independent rotation around a fixed global axis.

    This is the same convention as the online repo's utils.apply_rot:

        rot_mat = Matrix.Rotation(angle, 4, axis)
        obj.matrix_world = T @ rot_mat @ R_current @ S

    This is not spherical azimuth/elevation. It is just an object transform.
    """
    if axis not in {"X", "Y", "Z"}:
        raise ValueError(f"axis must be 'X', 'Y' or 'Z', got {axis}")

    rot_mat = Matrix.Rotation(math.radians(float(angle_deg)), 4, axis)

    obj_loc, obj_rot, obj_scale = obj.matrix_world.decompose()

    loc_mat = Matrix.Translation(obj_loc)
    rot_current_mat = obj_rot.to_matrix().to_4x4()
    scale_mat = (
        Matrix.Scale(obj_scale[0], 4, (1.0, 0.0, 0.0))
        @ Matrix.Scale(obj_scale[1], 4, (0.0, 1.0, 0.0))
        @ Matrix.Scale(obj_scale[2], 4, (0.0, 0.0, 1.0))
    )

    obj.matrix_world = loc_mat @ rot_mat @ rot_current_mat @ scale_mat


def rotate_object(
    obj: bpy.types.Object,
    azimuth_deg: float = 0.0,
    elevation_deg: float = 0.0,
) -> bpy.types.Object:
    """
    Apply the online repo orientation convention.

    Important:
    These are not true camera azimuth/elevation angles.

    Online convention:
        base_azim -> independent global Y rotation
        base_elev -> independent global X rotation

    Order:
        1. rotate around global Y by azimuth_deg
        2. rotate around global X by elevation_deg
        3. bake rotation into the mesh
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Same as the online repo:
    # utils.apply_rot(obj, "Y", base_azim)
    # utils.apply_rot(obj, "X", base_elev)
    apply_global_axis_rotation(obj, "Y", azimuth_deg)
    apply_global_axis_rotation(obj, "X", elevation_deg)

    bpy.context.view_layer.update()

    # Bake rotation into mesh.
    bpy.ops.object.transform_apply(scale=False, location=False, rotation=True)

    return obj


def add_area_light(
    location=(0.0, 0.0, 5.0),
    size: float = 10.0,
    strength: float = 30.0,
    name: str = "main_area_light",
) -> bpy.types.Object:
    """Add a large area light, matching the online default more closely."""
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = float(strength)
    light_data.size = float(size)

    light = bpy.data.objects.new(name, light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = location

    return light


def create_camera(
    location=(0.0, -4.0, 1.8),
    focal_length: float = 50.0,
    sensor_width_mm: float = 32.0,
    name: str = "render_camera",
) -> bpy.types.Object:
    """Create a Blender camera."""
    cam_data = bpy.data.cameras.new(name)
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)

    cam.location = location
    cam.data.lens = float(focal_length)
    cam.data.sensor_width = float(sensor_width_mm)
    cam.rotation_mode = "QUATERNION"

    bpy.context.scene.camera = cam

    return cam


def look_at(
    obj: bpy.types.Object,
    target=(0.0, 0.0, 0.0),
) -> None:
    """Rotate an object so its local -Z axis points to target."""
    target = Vector(target)
    direction = target - obj.location

    if direction.length == 0:
        raise ValueError("Camera/object location is identical to target; cannot compute look_at.")

    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")