from __future__ import annotations

from pathlib import Path
import csv
import json
import sys

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

    obj = center_and_scale_object(
        obj,
        target_size=object_target_size,
    )

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
    Render a single still image.
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
    seed: int = 0,
    save_metadata: bool = True,
) -> dict:
    """
    Render RGB frames with camera drift, microsaccades and jitter.

    This only saves RGB frames and camera metadata.
    DVS conversion is done later from the saved PNG frames.
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


def frames_to_gif(
    frames_dir: str | Path,
    output_gif: str | Path,
    fps: int = 30,
    loop: int = 0,
) -> str:
    """
    Convert rendered PNG frames to an animated GIF.

    loop=0 means loop indefinitely.
    """
    from PIL import Image

    frames_dir = Path(frames_dir)
    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {frames_dir}")

    frames = [
        Image.open(frame_path).convert("P", palette=Image.Palette.ADAPTIVE)
        for frame_path in frame_paths
    ]

    duration_ms = int(round(1000.0 / float(fps)))

    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=int(loop),
        optimize=True,
    )

    for frame in frames:
        frame.close()

    return str(output_gif)


def _add_iebcs_lux_to_path() -> tuple[Path, Path]:
    """
    Add IEBCS module and lux-noise directories to sys.path.

    Expected repo layout:

        ./src/scene/render_offline.py
        ./src/IEBCS/event_buffer.py
        ./src/IEBCS/dvs_sensor.py
        ./src/IEBCS/dat_files.py
        ./src/IEBCS/lux/noise_pos_*.npy
        ./src/IEBCS/lux/noise_neg_*.npy

    Returns
    -------
    iebcs_dir:
        Directory containing event_buffer.py, dvs_sensor.py, dat_files.py.

    lux_dir:
        Directory containing lux noise files.
    """
    this_file = Path(__file__).resolve()

    # this_file = .../src/scene/render_offline.py
    # src_dir   = .../src
    src_dir = this_file.parents[1]

    iebcs_dir = src_dir / "IEBCS"
    lux_dir = iebcs_dir / "lux"

    if not iebcs_dir.exists():
        raise FileNotFoundError(
            f"Could not find IEBCS module directory at: {iebcs_dir}\n"
            "Expected path: ./src/IEBCS"
        )

    if not lux_dir.exists():
        raise FileNotFoundError(
            f"Could not find IEBCS lux directory at: {lux_dir}\n"
            "Expected path: ./src/IEBCS/lux"
        )

    for path in (iebcs_dir, lux_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return iebcs_dir, lux_dir


def _read_frame_luminance_klux(frame_path: str | Path):
    """
    Read a rendered RGB frame and convert it to the luminance signal expected
    by the IEBCS DVS code.

    Convention from your original script:
        uint8 255 -> 1e4

    In other words:
        L = L_channel / 255 * 1e4

    The name says klux because this is the same luminance scaling used by the
    IEBCS example code.
    """
    import cv2
    import numpy as np

    frame_path = Path(frame_path)

    img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read frame: {frame_path}")

    # OpenCV reads BGR. Convert to LUV and use the L channel.
    luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    lum = luv[:, :, 0].astype(np.float32) / 255.0 * 1e4

    return lum


def _events_from_dat_to_npy(
    dat_path: str | Path,
    output_npy: str | Path,
) -> str:
    """
    Convert IEBCS .dat events to .npy.

    Output array shape:
        (N, 4)

    Columns:
        x, y, polarity, timestamp_seconds
    """
    import numpy as np

    _add_iebcs_lux_to_path()
    from dat_files import load_dat_event

    dat_path = Path(dat_path)
    output_npy = Path(output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)

    ts, x, y, p = load_dat_event(str(dat_path))

    timestamps_sec = ts.astype(np.float64) * 1e-6
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    p = p.astype(np.int32)

    data = np.stack([x, y, p, timestamps_sec], axis=1)
    np.save(output_npy, data)

    return str(output_npy)


def _dat_to_event_gif(
    dat_path: str | Path,
    output_gif: str | Path,
    res: tuple[int, int],
    window_us: int = 1000,
    gif_fps: int = 20,
    loop: int = 0,
    timestamp_text: bool = True,
) -> str:
    """
    Convert IEBCS .dat events to an animated GIF.

    loop=0 means infinite looping.

    res:
        (width, height)
    """
    import cv2
    import numpy as np
    from PIL import Image

    _add_iebcs_lux_to_path()
    from dat_files import load_dat_event

    dat_path = Path(dat_path)
    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    ts, x, y, p = load_dat_event(str(dat_path))

    width, height = int(res[0]), int(res[1])

    if ts.size == 0:
        blank = np.full((height, width, 3), 125, dtype=np.uint8)
        pil = Image.fromarray(blank)
        pil.save(
            output_gif,
            save_all=True,
            append_images=[],
            duration=int(round(1000.0 / float(gif_fps))),
            loop=int(loop),
        )
        pil.close()
        return str(output_gif)

    frames = []

    img = np.zeros((height, width), dtype=np.float32)
    tsurface = np.zeros((height, width), dtype=np.int64)
    indsurface = np.zeros((height, width), dtype=np.int8)

    t0 = int(ts[0])
    t1 = int(ts[-1])

    duration_ms = int(round(1000.0 / float(gif_fps)))
    decay_tau = max(float(window_us) / 30.0, 1.0)

    for t in range(t0, t1 + 1, int(window_us)):
        ind = (ts >= t) & (ts < t + window_us)

        tsurface[:, :] = 0
        indsurface[:, :] = 0

        if ind.any():
            tsurface[y[ind], x[ind]] = t + window_us
            indsurface[y[ind], x[ind]] = 2 * p[ind].astype(np.int8) - 1

        img[:, :] = 125.0
        active = tsurface > 0

        if active.any():
            img[active] = 125.0 + indsurface[active] * np.exp(
                -(t + window_us - tsurface[active].astype(np.float32)) / decay_tau
            ) * 125.0

        gray = np.clip(img, 0, 255).astype(np.uint8)
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if timestamp_text:
            color = cv2.putText(
                color,
                f"{t + window_us} us",
                (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        color = cv2.applyColorMap(color, cv2.COLORMAP_VIRIDIS)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        frames.append(Image.fromarray(color))

    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=int(loop),
        optimize=True,
    )

    for frame in frames:
        frame.close()

    return str(output_gif)


def frames_to_events_npy_and_gif(
    frames_dir: str | Path,
    output_dir: str | Path,
    fps: int,
    th_pos: float = 0.15,
    th_neg: float = 0.15,
    th_noise: float = 0.05,
    lat: int = 500,
    tau: int = 300,
    jit: int = 100,
    bgnp: float = 0.0001,
    bgnn: float = 0.0001,
    ref: int = 40,
    skip_frames: int = 0,
    noise_pos: str | Path | None = None,
    noise_neg: str | Path | None = None,
    gif_fps: int = 20,
    gif_window_us: int | None = None,
    loop: int = 0,
    timestamp_text: bool = True,
    keep_dat: bool = True,
) -> dict:
    """
    Convert rendered PNG frames to IEBCS DVS events.

    Outputs:
        events.dat
        events.npy
        events.gif

    This is still offline:
        rendered PNG frames -> DVS conversion -> npy/gif

    It uses the IEBCS DVS code located at:
        ./src/IEBCS/lux

    The .npy format is:
        columns = [x, y, polarity, timestamp_seconds]

    Parameters
    ----------
    frames_dir:
        Directory containing rendered frames named frame_*.png.

    output_dir:
        Directory where event outputs are saved.

    fps:
        Frame rate of the rendered sequence.

    th_pos, th_neg:
        ON/OFF thresholds.

    th_noise:
        Threshold noise.

    lat:
        Latency in microseconds.

    tau:
        Front-end time constant in microseconds.

    jit:
        Temporal jitter standard deviation in microseconds.

    bgnp, bgnn:
        ON/OFF background event noise rates.

    ref:
        Refractory period in microseconds.

    skip_frames:
        Number of initial RGB frames to skip before initializing the DVS.

    gif_fps:
        Playback fps for the event GIF.

    gif_window_us:
        Temporal event accumulation window for the GIF.
        If None, uses dt_us = 1e6 / fps.

    loop:
        GIF loop setting. 0 means infinite loop.

    keep_dat:
        If False, deletes events.dat after creating events.npy and events.gif.
    """
    _add_iebcs_lux_to_path()

    from event_buffer import EventBuffer
    from dvs_sensor import DvsSensor

    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))

    if len(frame_paths) < 2:
        raise RuntimeError(f"Need at least 2 frames in {frames_dir} to generate events.")

    if fps <= 0:
        raise ValueError("fps must be > 0")

    if skip_frames < 0:
        raise ValueError("skip_frames must be >= 0")

    if skip_frames >= len(frame_paths) - 1:
        raise ValueError(
            f"skip_frames={skip_frames} is too large for {len(frame_paths)} frames."
        )

    dt_us = int(round(1e6 / float(fps)))

    if gif_window_us is None:
        gif_window_us = dt_us

    init_frame_path = frame_paths[skip_frames]
    init_im = _read_frame_luminance_klux(init_frame_path)

    height, width = init_im.shape[:2]

    dvs = DvsSensor("OfflineDVS")

    dvs.initCamera(
        int(width),
        int(height),
        lat=int(lat),
        jit=int(jit),
        ref=int(ref),
        tau=int(tau),
        th_pos=float(th_pos),
        th_neg=float(th_neg),
        th_noise=float(th_noise),
        bgnp=float(bgnp),
        bgnn=float(bgnn),
    )

    if noise_pos is not None and noise_neg is not None:
        dvs.init_bgn_hist(str(noise_pos), str(noise_neg))

    dvs.init_image(init_im)

    ev_full = EventBuffer(1)

    for frame_path in frame_paths[skip_frames + 1:]:
        im = _read_frame_luminance_klux(frame_path)
        ev = dvs.update(im, dt_us)
        ev_full.increase_ev(ev)

    dat_path = output_dir / "events.dat"
    npy_path = output_dir / "events.npy"
    gif_path = output_dir / "events.gif"

    ev_full.write(str(dat_path))

    _events_from_dat_to_npy(
        dat_path=dat_path,
        output_npy=npy_path,
    )

    _dat_to_event_gif(
        dat_path=dat_path,
        output_gif=gif_path,
        res=(width, height),
        window_us=int(gif_window_us),
        gif_fps=int(gif_fps),
        loop=int(loop),
        timestamp_text=timestamp_text,
    )

    if not keep_dat:
        dat_path.unlink(missing_ok=True)

    return {
        "dat_path": str(dat_path) if keep_dat else None,
        "npy_path": str(npy_path),
        "gif_path": str(gif_path),
        "frames_dir": str(frames_dir),
        "fps": int(fps),
        "dt_us": int(dt_us),
        "width": int(width),
        "height": int(height),
        "th_pos": float(th_pos),
        "th_neg": float(th_neg),
        "th_noise": float(th_noise),
        "lat": int(lat),
        "tau": int(tau),
        "jit": int(jit),
        "bgnp": float(bgnp),
        "bgnn": float(bgnn),
        "ref": int(ref),
    }