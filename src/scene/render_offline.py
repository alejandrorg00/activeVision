from __future__ import annotations

from pathlib import Path
import csv
import json

import bpy

from scene.blender_scene import (
    clear_scene,
    setup_render_settings,
    import_largest_mesh_from_blend,
    center_and_scale_object,
    rotate_object,
    add_area_light,
    create_camera,
    look_at,
)

from scene.camera_motion import MicrosaccadeCameraController


def setup_object_scene(
    object_path: str | Path,
    resolution: int = 128,
    samples: int = 32,
    use_gpu: bool = True,
    object_target_size: float = 1.2,
    object_azimuth_deg: float = 0.0,
    object_elevation_deg: float = 0.0,
    camera_position=(0.0, -4.0, 1.8),
    camera_target=(0.0, 0.0, 0.0),
    focal_length: float = 50.0,
    sensor_width_mm: float = 32.0,
    light_location=(2.5, -2.5, 4.0),
    light_size: float = 5.0,
    light_strength: float = 2000.0,
):
    """
    Build a clean Blender scene with one centered object, one light and one camera.
    """
    clear_scene()

    scene = setup_render_settings(
        resolution=resolution,
        samples=samples,
        engine="CYCLES",
        use_gpu=use_gpu,
    )

    obj = import_largest_mesh_from_blend(object_path)
    obj = center_and_scale_object(obj, target_size=object_target_size)
    obj = rotate_object(
        obj,
        azimuth_deg=object_azimuth_deg,
        elevation_deg=object_elevation_deg,
    )

    light = add_area_light(
        location=light_location,
        size=light_size,
        strength=light_strength,
    )

    cam = create_camera(
        location=camera_position,
        focal_length=focal_length,
        sensor_width_mm=sensor_width_mm,
    )
    look_at(cam, target=camera_target)

    return scene, obj, cam, light


def render_still(
    object_path: str | Path,
    output_path: str | Path,
    resolution: int = 256,
    samples: int = 32,
    use_gpu: bool = True,
    object_target_size: float = 1.2,
    object_azimuth_deg: float = 0.0,
    object_elevation_deg: float = 0.0,
    camera_position=(0.0, -4.0, 1.8),
    camera_target=(0.0, 0.0, 0.0),
    focal_length: float = 50.0,
    sensor_width_mm: float = 32.0,
    light_location=(2.5, -2.5, 4.0),
    light_size: float = 5.0,
    light_strength: float = 2000.0,
) -> dict:
    """
    Render a single still image to verify object placement, lighting and camera.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scene, obj, cam, light = setup_object_scene(
        object_path=object_path,
        resolution=resolution,
        samples=samples,
        use_gpu=use_gpu,
        object_target_size=object_target_size,
        object_azimuth_deg=object_azimuth_deg,
        object_elevation_deg=object_elevation_deg,
        camera_position=camera_position,
        camera_target=camera_target,
        focal_length=focal_length,
        sensor_width_mm=sensor_width_mm,
        light_location=light_location,
        light_size=light_size,
        light_strength=light_strength,
    )

    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)

    return {
        "image_path": str(output_path),
        "object_name": obj.name,
        "camera_name": cam.name,
    }


def render_camera_motion_sequence(
    object_path: str | Path,
    output_dir: str | Path,
    num_frames: int = 120,
    fps: int = 60,
    resolution: int = 128,
    samples: int = 16,
    use_gpu: bool = True,
    object_target_size: float = 1.2,
    object_azimuth_deg: float = 0.0,
    object_elevation_deg: float = 0.0,
    camera_position=(0.0, -4.0, 1.8),
    camera_target=(0.0, 0.0, 0.0),
    focal_length: float = 50.0,
    sensor_width_mm: float = 32.0,
    light_location=(2.5, -2.5, 4.0),
    light_size: float = 5.0,
    light_strength: float = 2000.0,
    drift_sigma_deg=(0.005, 0.005),
    microsaccade_rate_hz: float = 5.0,
    microsaccade_amp_deg=(0.05, 0.30),
    microsaccade_dur_ms=(10, 30),
    jitter_sigma_deg=(0.005, 0.005),
    seed: int = 0,
    save_metadata: bool = True,
) -> dict:
    """
    Render RGB frames with camera drift, microsaccades and jitter.

    This function does not generate DVS events.
    It only saves RGB frames and camera metadata.
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "camera_motion.csv"
    config_path = output_dir / "render_config.json"

    scene, obj, cam, light = setup_object_scene(
        object_path=object_path,
        resolution=resolution,
        samples=samples,
        use_gpu=use_gpu,
        object_target_size=object_target_size,
        object_azimuth_deg=object_azimuth_deg,
        object_elevation_deg=object_elevation_deg,
        camera_position=camera_position,
        camera_target=camera_target,
        focal_length=focal_length,
        sensor_width_mm=sensor_width_mm,
        light_location=light_location,
        light_size=light_size,
        light_strength=light_strength,
    )

    base_rotation = cam.rotation_quaternion.copy()

    controller = MicrosaccadeCameraController(
        camera=cam,
        base_rotation_quaternion=base_rotation,
        drift_sigma_deg=drift_sigma_deg,
        microsaccade_rate_hz=microsaccade_rate_hz,
        microsaccade_amp_deg=microsaccade_amp_deg,
        microsaccade_dur_ms=microsaccade_dur_ms,
        jitter_sigma_deg=jitter_sigma_deg,
        seed=seed,
    )

    scene.frame_start = 0
    scene.frame_end = int(num_frames) - 1
    scene.render.fps = int(fps)

    dt = 1.0 / float(fps)
    rows = []

    config = {
        "object_path": str(object_path),
        "output_dir": str(output_dir),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "resolution": int(resolution),
        "samples": int(samples),
        "object_target_size": float(object_target_size),
        "object_azimuth_deg": float(object_azimuth_deg),
        "object_elevation_deg": float(object_elevation_deg),
        "camera_position": list(camera_position),
        "camera_target": list(camera_target),
        "focal_length": float(focal_length),
        "sensor_width_mm": float(sensor_width_mm),
        "light_location": list(light_location),
        "light_size": float(light_size),
        "light_strength": float(light_strength),
        "drift_sigma_deg": list(drift_sigma_deg),
        "microsaccade_rate_hz": float(microsaccade_rate_hz),
        "microsaccade_amp_deg": list(microsaccade_amp_deg),
        "microsaccade_dur_ms": list(microsaccade_dur_ms),
        "jitter_sigma_deg": list(jitter_sigma_deg),
        "seed": int(seed),
    }

    if save_metadata:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    for frame_idx in range(int(num_frames)):
        scene.frame_set(frame_idx)

        motion_info = controller.update(dt)

        frame_path = frames_dir / f"frame_{frame_idx:05d}.png"
        scene.render.filepath = str(frame_path)
        bpy.ops.render.render(write_still=True)

        row = {
            "frame": frame_idx,
            "time_s": frame_idx * dt,
            "frame_path": str(frame_path),
            "camera_x": float(cam.location.x),
            "camera_y": float(cam.location.y),
            "camera_z": float(cam.location.z),
            **motion_info,
        }
        rows.append(row)

    if save_metadata and rows:
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return {
        "output_dir": str(output_dir),
        "frames_dir": str(frames_dir),
        "metadata_path": str(metadata_path),
        "config_path": str(config_path),
        "num_frames": int(num_frames),
        "fps": int(fps),
    }


def frames_to_video(
    frames_dir: str | Path,
    output_video: str | Path,
    fps: int = 60,
) -> str:
    """
    Convert rendered PNG frames to an mp4 video.
    """
    import cv2

    frames_dir = Path(frames_dir)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("frame_*.png"))

    if len(frames) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")

    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )

    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")
        writer.write(img)

    writer.release()

    return str(output_video)