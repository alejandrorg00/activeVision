from __future__ import annotations

from pathlib import Path
import json
import csv

import bpy

from scene.blender_scene import (
    clear_scene,
    setup_render_settings,
    import_largest_mesh_from_blend,
    center_and_scale_object,
    rotate_object,
    add_area_light,
    create_camera,
)

from attention.camera_motion import MicrosaccadeCameraController


def render_object_sequence(
    object_path: str | Path,
    output_dir: str | Path,
    num_frames: int = 300,
    fps: int = 60,
    resolution: int = 128,
    samples: int = 64,
    camera_position=(0.0, 0.0, 2.5),
    focal_length: float = 50.0,
    sensor_width_mm: float = 32.0,
    object_target_size: float = 0.9,
    object_azimuth_deg: float = 0.0,
    object_elevation_deg: float = 0.0,
    light_location=(0.0, 0.0, 5.0),
    light_size: float = 5.0,
    light_strength: float = 300.0,
    drift_sigma_deg=(0.01, 0.01),
    microsaccade_rate_hz: float = 10.0,
    microsaccade_amp_deg=(0.2, 1.0),
    microsaccade_dur_ms=(10, 30),
    jitter_sigma_deg=(0.0, 0.0),
    seed: int = 0,
    save_metadata: bool = True,
) -> dict:
    """
    Offline RGB renderer.

    It renders frames only.
    No DVS.
    No event generation.
    No online attention loop.
    """
    object_path = Path(object_path)
    output_dir = Path(output_dir)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "camera_motion.csv"
    config_path = output_dir / "render_config.json"

    clear_scene()

    scene = setup_render_settings(
        resolution=resolution,
        samples=samples,
        engine="CYCLES",
        use_gpu=True,
    )

    obj = import_largest_mesh_from_blend(object_path)
    obj = center_and_scale_object(obj, target_size=object_target_size)
    obj = rotate_object(
        obj,
        azimuth_deg=object_azimuth_deg,
        elevation_deg=object_elevation_deg,
    )

    add_area_light(
        location=light_location,
        size=light_size,
        strength=light_strength,
    )

    cam = create_camera(
        location=camera_position,
        focal_length=focal_length,
        sensor_width_mm=sensor_width_mm,
    )

    controller = MicrosaccadeCameraController(
        camera=cam,
        drift_sigma_deg=drift_sigma_deg,
        microsaccade_rate_hz=microsaccade_rate_hz,
        microsaccade_amp_deg=microsaccade_amp_deg,
        microsaccade_dur_ms=microsaccade_dur_ms,
        jitter_sigma_deg=jitter_sigma_deg,
        seed=seed,
    )

    dt = 1.0 / float(fps)
    rows = []

    scene.frame_start = 0
    scene.frame_end = int(num_frames) - 1
    scene.render.fps = int(fps)

    config = {
        "object_path": str(object_path),
        "output_dir": str(output_dir),
        "num_frames": int(num_frames),
        "fps": int(fps),
        "resolution": int(resolution),
        "samples": int(samples),
        "camera_position": list(camera_position),
        "focal_length": float(focal_length),
        "sensor_width_mm": float(sensor_width_mm),
        "object_target_size": float(object_target_size),
        "object_azimuth_deg": float(object_azimuth_deg),
        "object_elevation_deg": float(object_elevation_deg),
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