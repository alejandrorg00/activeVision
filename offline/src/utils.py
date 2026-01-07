# -*- coding: utf-8 -*-
"""
Util funcions


Alejandro Rodriguez-Garcia
"""
import bpy

import numpy as np
import matplotlib.pyplot as plt
from bpy_extras.object_utils import world_to_camera_view
# import random
# import math
from mathutils import Matrix, Vector, Quaternion
import os


def make_area_lamp(location, rotation, size_x=0, size_y=0, strength=10, temp=5000):
    """
    inputs:
        location  - (x,y,z) location of area light
        rotation  - (x,y,z) rotation of area light in radians
        size_x    - size in x direction of area light
        size_y    - size in y direction of area light
        strength  - strength (brightness) of area light
        temp      - color temperature in Kelvin of area light
    """

    # initialize ligth and set size
    bpy.context.view_layer.objects.active = None
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.light_add(type="AREA", location=location, rotation=rotation)

    lamp = bpy.data.lights[bpy.context.active_object.name]
    lamp.shape = "RECTANGLE"
    lamp.size = size_x
    lamp.size_y = size_y

    # create blackbody nodes for color temperature control
    lamp.use_nodes = True
    nodes = lamp.node_tree.nodes

    for node in nodes:
        nodes.remove(node)

    node_blackbody = nodes.new(type="ShaderNodeBlackbody")
    node_emission = nodes.new(type="ShaderNodeEmission")
    node_output = nodes.new(type="ShaderNodeOutputLight")

    node_output.location[1] = 400
    node_emission.location[1] = 200

    lamp.node_tree.links.new(node_blackbody.outputs[0], node_emission.inputs[0])
    lamp.node_tree.links.new(node_emission.outputs[0], node_output.inputs[0])

    node_emission.inputs[1].default_value = strength
    node_blackbody.inputs[0].default_value = temp
    lamp_obj = bpy.data.objects[lamp.name]
    lamp_obj.select_set(False)


def apply_rot(obj, axis, angle):
    """
    inputs:
        obj   - bpy.data.objects object to rotate globally
        axis  - axis along which to rotate - 'X', 'Y', or 'Z'
        angle - angle in global coordinates along axis in degrees
    """

    rot_mat = Matrix.Rotation(np.radians(angle), 4, axis)

    o_loc, o_rot, o_scl = obj.matrix_world.decompose()
    o_loc_mat = Matrix.Translation(o_loc)
    o_rot_mat = o_rot.to_matrix().to_4x4()
    o_scl_mat = (
        Matrix.Scale(o_scl[0], 4, (1, 0, 0))
        @ Matrix.Scale(o_scl[1], 4, (0, 1, 0))
        @ Matrix.Scale(o_scl[2], 4, (0, 0, 1))
    )

    # assemble the new matrix
    obj.matrix_world = o_loc_mat @ rot_mat @ o_rot_mat @ o_scl_mat


def reset_rot(obj):
    obj.rotation_euler = (0, 0, 0)

def apply_settings(scn, render_parameters):
    # ----------------------------
    # Render resolution
    # ----------------------------
    scn.render.resolution_x = int(render_parameters["resolution"])
    scn.render.resolution_y = int(render_parameters["resolution"])
    scn.render.resolution_percentage = int(render_parameters["resolution_percentage"])
    scn.render.use_persistent_data = bool(render_parameters["use_persistent_data"])

    # ----------------------------
    # ViewLayer (default name often "ViewLayer")
    # ----------------------------
    vl = scn.view_layers.get("ViewLayer")
    if vl is None:
        vl = scn.view_layers[0]

    vl.samples = int(render_parameters["render_samples"])

    vl.cycles.use_denoising = bool(render_parameters["use_denoising"])
    if hasattr(vl.cycles, "denoising_radius"):
        vl.cycles.denoising_radius = int(render_parameters["denoising_radius"])

    # ----------------------------
    # Cycles settings (drop deprecated)
    # ----------------------------
    if hasattr(scn.cycles, "debug_use_spatial_splits"):
        scn.cycles.debug_use_spatial_splits = bool(render_parameters["use_spatial_splits"])

    if hasattr(scn.cycles, "max_bounces"):
        scn.cycles.max_bounces = int(render_parameters["max_bounces"])

    # NOTE: transparent_min_bounces removed in your version -> dropped
    if hasattr(scn.cycles, "transparent_max_bounces"):
        scn.cycles.transparent_max_bounces = int(render_parameters["transparent_max_bounces"])

    if hasattr(scn.cycles, "glossy_bounces"):
        scn.cycles.glossy_bounces = int(render_parameters["glossy_bounces"])

    if hasattr(scn.cycles, "transmission_bounces"):
        scn.cycles.transmission_bounces = int(render_parameters["transmission_bounces"])

    # Caustics toggles may not exist depending on version
    if hasattr(scn.cycles, "caustics_refractive"):
        scn.cycles.caustics_refractive = bool(render_parameters["use_caustics_refractive"])
    if hasattr(scn.cycles, "caustics_reflective"):
        scn.cycles.caustics_reflective = bool(render_parameters["use_caustics_reflective"])

    # Device (GPU/CPU)
    if hasattr(scn.cycles, "device"):
        scn.cycles.device = render_parameters["rendering_device"]

    # Film transparency moved around between versions
    if hasattr(scn.cycles, "film_transparent"):
        scn.cycles.film_transparent = bool(render_parameters["use_film_transparent"])
    elif hasattr(scn.render, "film_transparent"):
        scn.render.film_transparent = bool(render_parameters["use_film_transparent"])

    # ----------------------------
    # Output color mode
    # ----------------------------
    scn.render.image_settings.color_mode = render_parameters["color_mode"]

    # ----------------------------
    # World background
    # ----------------------------
    scn.world.use_nodes = True
    bg = scn.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = tuple(render_parameters["background_color"])
        bg.inputs["Strength"].default_value = float(render_parameters["background_strength"])




def load_obj(scn, path, dataset_type):
    """
    Loads an object for either ShapeNet, ModelNet or Toys

    Inputs:
        scn - bpy.context.scene object
        path - absolute path to load from
        dataset_type - "toys", "modelnet" or "shapenet" string
                       used to determine initial transform after loading
    """

    if dataset_type == "toys":

        # Track what exists BEFORE loading so we can isolate newly loaded objects
        before_names = set(bpy.data.objects.keys())

        with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects]

        # Link all loaded objects to the scene collection
        for o in data_to.objects:
            if o is not None:
                try:
                    scn.collection.objects.link(o)
                except RuntimeError:
                    # Already linked
                    pass

        # Prefer meshes among JUST loaded objects
        loaded_objs = [o for o in data_to.objects if o is not None]
        meshes = [
            o for o in loaded_objs
            if o.type == "MESH" and o.data is not None and len(o.data.vertices) > 0
        ]

        # Fallback: if the .blend uses collection instances or odd linking, search newly added datablocks
        if not meshes:
            after_names = set(bpy.data.objects.keys())
            new_names = list(after_names - before_names)
            new_objs = [bpy.data.objects[n] for n in new_names]
            meshes = [
                o for o in new_objs
                if o.type == "MESH" and o.data is not None and len(o.data.vertices) > 0
            ]

        if not meshes:
            # Last resort: any mesh in scene (avoid lights/empties)
            meshes = [
                o for o in bpy.data.objects
                if o.type == "MESH" and o.data is not None and len(o.data.vertices) > 0
            ]

        if not meshes:
            # Helpful debug
            print("[load_obj/toys] Loaded objects were:")
            for o in loaded_objs:
                print(" -", o.name, o.type, "data:", type(o.data).__name__ if o.data else None)
            raise RuntimeError(f"[load_obj/toys] No MESH with vertices found in: {path}")

        # If multiple meshes, pick the largest one (most vertices)
        obj = sorted(meshes, key=lambda m: len(m.data.vertices), reverse=True)[0]
        obj.name = "object"

        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (np.radians(-90), 0, 0)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    if dataset_type == "modelnet":
        # Track existing meshes before import to isolate newly imported object(s)
        before_names = set(bpy.data.objects.keys())

        bpy.ops.import_scene.obj(filepath=path)

        after_names = set(bpy.data.objects.keys())
        new_names = list(after_names - before_names)
        new_objs = [bpy.data.objects[n] for n in new_names]

        meshes = [
            o for o in new_objs
            if o.type == "MESH" and o.data is not None and len(o.data.vertices) > 0
        ]
        if not meshes:
            meshes = [
                o for o in bpy.data.objects
                if o.type == "MESH" and o.data is not None and len(o.data.vertices) > 0
            ]
        if not meshes:
            raise RuntimeError(f"[load_obj/modelnet] No MESH with vertices loaded from: {path}")

        obj = sorted(meshes, key=lambda m: len(m.data.vertices), reverse=True)[0]
        obj.name = "object"

        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (np.radians(-90), np.radians(180), 0)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    if dataset_type == "shapenet":
        before_names = set(bpy.data.objects.keys())

        bpy.ops.import_scene.obj(filepath=path)

        after_names = set(bpy.data.objects.keys())
        new_names = list(after_names - before_names)
        new_objs = [bpy.data.objects[n] for n in new_names]

        meshes = [
            o for o in new_objs
            if o.type == "MESH" and o.data is not None and len(o.data.vertices) > 0
        ]
        if not meshes:
            meshes = [
                o for o in bpy.data.objects
                if o.type == "MESH" and o.data is not None and len(o.data.vertices) > 0
            ]
        if not meshes:
            raise RuntimeError(f"[load_obj/shapenet] No MESH with vertices loaded from: {path}")

        obj = sorted(meshes, key=lambda m: len(m.data.vertices), reverse=True)[0]
        obj.name = "object"

        obj.rotation_mode = "XYZ"
        obj.rotation_euler = (0, np.radians(180), 0)

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        obj.location = (0, 0, 0)
        bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

        return obj

    raise ValueError(f"Unknown dataset_type: {dataset_type}")





# ----------------------------
# Helpers
# ----------------------------
def world_to_pixel(scene, cam_obj, world_co, W, H):
    """Project world coordinate to image pixels (origin top-left)."""
    co_ndc = world_to_camera_view(scene, cam_obj, world_co)  # x,y in [0,1]
    x_px = float(co_ndc.x) * float(W)
    y_px = (1.0 - float(co_ndc.y)) * float(H)
    return x_px, y_px


def contiguous_segments(mask: np.ndarray):
    mask = np.asarray(mask).astype(bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    return [(int(g[0]), int(g[-1])) for g in groups]


def plot_scanROI(
    xy_px,                     # (T, 2) pixels
    W, H,                      # full image size
    out_png="scanROI.png",
    stride=1,
    show_points=False,
    point_size=10,
    line_width=1.5,
):
    """
    Plot scanpath restricted to a ROI of size 0.1*W x 0.1*H
    centered at the initial fixation point.

    Coordinate system:
    - Origin: top-left
    - x to the right, y down
    """
    xy = np.asarray(xy_px, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy_px must have shape (T, 2)")

    if stride > 1:
        xy = xy[::stride]

    # ROI definition (centered on initial fixation)
    cx, cy = xy[0]                 # initial fixation
    roi_w = 0.1 * W
    roi_h = 0.1 * H

    x0 = cx - roi_w / 2
    # x1 = cx + roi_w / 2
    y0 = cy - roi_h / 2
    # y1 = cy + roi_h / 2

    # Shift trajectory into ROI coordinates
    xy_roi = xy.copy()
    xy_roi[:, 0] -= x0
    xy_roi[:, 1] -= y0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor("white")

    ax.plot(xy_roi[:, 0], xy_roi[:, 1], "-", linewidth=line_width, alpha=0.9)

    if show_points:
        ax.scatter(xy_roi[:, 0], xy_roi[:, 1], s=point_size, alpha=0.9)

    ax.scatter(
        xy_roi[0, 0], xy_roi[0, 1],
        s=80, color="red", edgecolors="black",
        linewidth=1.0, zorder=5, label="start"
    )
    ax.scatter(
        xy_roi[-1, 0], xy_roi[-1, 1],
        s=80, color="green", edgecolors="black",
        linewidth=1.0, zorder=5, label="end"
    )

    ax.set_xlim([0, roi_w])
    ax.set_ylim([roi_h, 0])

    ax.set_xlabel("x (ROI pixels)")
    ax.set_ylabel("y (ROI pixels)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    return out_png


def plot_scanpath(
    xy_px,                     # (T, 2) pixels
    W, H,                      # image extent
    out_png="scanpath.png",
    stride=1,
    show_points=True,
    point_size=10,
    line_width=1.5,
):
    xy = np.asarray(xy_px, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy_px must have shape (T, 2)")

    if stride > 1:
        xy = xy[::stride]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("white")

    ax.plot(xy[:, 0], xy[:, 1], "-", linewidth=line_width, alpha=0.9)

    if show_points:
        ax.scatter(xy[:, 0], xy[:, 1], s=point_size, alpha=0.9)

    ax.scatter(
        xy[0, 0], xy[0, 1],
        s=80, color="red", edgecolors="black", linewidth=1.0, zorder=5,
        label="start"
    )
    ax.scatter(
        xy[-1, 0], xy[-1, 1],
        s=80, color="green", edgecolors="black", linewidth=1.0, zorder=5,
        label="end"
    )

    ax.set_xlim([0, W])
    ax.set_ylim([H, 0])

    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png
