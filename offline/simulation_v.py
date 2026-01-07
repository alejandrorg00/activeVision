
import bpy
import numpy as np
import os, sys
import cv2
import shutil
from mathutils import Vector
from pathlib import Path

# ----------------------------
# Local imports (project/src)
# ----------------------------
PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.append(str(SRC_DIR))

import utils
from dvs_sensor_blender import Blender_DvsSensor
from event_buffer import EventBuffer
from event_display import EventDisplay

from bpy_extras.object_utils import world_to_camera_view
import matplotlib.pyplot as plt

# ----------------------------
# Helpers
# ----------------------------
def world_to_pixel(scene, cam_obj, world_co, W, H):
    co_ndc = world_to_camera_view(scene, cam_obj, world_co)
    x_px = float(co_ndc.x) * float(W)
    y_px = (1.0 - float(co_ndc.y)) * float(H)
    return x_px, y_px


def ensure_object_mode():
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def clear_scene():
    ensure_object_mode()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def plot_scanROI(xy_px, W, H, out_png):
    xy = np.asarray(xy_px, dtype=np.float32)
    cx, cy = xy[0]
    roi_w = 0.1 * W
    roi_h = 0.1 * H
    x0 = cx - roi_w / 2
    y0 = cy - roi_h / 2

    xy_roi = xy.copy()
    xy_roi[:, 0] -= x0
    xy_roi[:, 1] -= y0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(xy_roi[:, 0], xy_roi[:, 1], "-", linewidth=1.5, alpha=0.9)
    ax.scatter(xy_roi[0, 0], xy_roi[0, 1], s=80, c="red", edgecolors="black", zorder=5)
    ax.scatter(xy_roi[-1, 0], xy_roi[-1, 1], s=80, c="green", edgecolors="black", zorder=5)
    ax.set_xlim([0, roi_w])
    ax.set_ylim([roi_h, 0])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def pngs_to_video(png_dir: Path, out_avi: Path, fps: float, pattern: str):
    frames = sorted(png_dir.glob(pattern))
    if not frames:
        print(f"[pngs_to_video] No PNGs found in {png_dir} with pattern {pattern}")
        return

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read {frames[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    out = cv2.VideoWriter(str(out_avi), fourcc, float(fps), (w, h))

    for f in frames:
        im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Could not read {f}")
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
        out.write(im)

    out.release()
    print(f"[pngs_to_video] Wrote {out_avi}")


def pngs_to_gif(png_dir: Path, out_gif: Path, fps: float, pattern: str):
    frames = sorted(png_dir.glob(pattern))
    if not frames:
        print(f"[pngs_to_gif] No PNGs found in {png_dir} with pattern {pattern}")
        return

    # imageio is the cleanest approach for GIF
    try:
        import imageio.v2 as imageio  # pip install imageio
    except Exception as e:
        raise RuntimeError(
            "GIF writing requires imageio. Install it in your Python env:\n"
            "  pip install imageio\n"
            f"Original error: {e}"
        )

    duration = 1.0 / float(fps)
    imgs = [imageio.imread(str(f)) for f in frames]
    imageio.mimsave(str(out_gif), imgs, duration=duration, loop=0)
    print(f"[pngs_to_gif] Wrote {out_gif}")


# ----------------------------
# IO
# ----------------------------
OUT_ROOT = PROJECT_DIR / "out"
OBJ_NAME = "obj3"
OUT_OBJ = OUT_ROOT / OBJ_NAME
OUT_EVENTS = OUT_OBJ / "events"
OUT_IMAGES = OUT_OBJ / "images"
OUT_TRAJ = OUT_OBJ / "traj"

# DELETE EVERYTHING for this object before overwriting
if OUT_OBJ.exists():
    shutil.rmtree(OUT_OBJ)

OUT_EVENTS.mkdir(parents=True, exist_ok=True)
OUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUT_TRAJ.mkdir(parents=True, exist_ok=True)

# Noise files
NOISE_POS = str(PROJECT_DIR / "data" / "lux" / "noise_pos_161lux.npy")
NOISE_NEG = str(PROJECT_DIR / "data" / "lux" / "noise_neg_161lux.npy")

# Object path (toys .blend)
OBJ_BLEND = PROJECT_DIR / "data" / "objects" / "airplane" / "airplane_010" / "airplane_010.blend"
# OBJ_BLEND = PROJECT_DIR / "data" / "objects" / "tv" / "tv_015" / "tv_015.blend"
# OBJ_BLEND = PROJECT_DIR / "data" / "objects" / "penguin" / "penguin_020" / "penguin_020.blend"

def main():
    scn = bpy.context.scene

    # ----------------------------
    # Clean scene + world settings
    # ----------------------------
    clear_scene()

    scn.render.engine = "CYCLES"
    scn.cycles.device = "GPU"
    scn.view_settings.view_transform = "Standard"
    scn.view_settings.look = "None"
    scn.view_settings.exposure = 0.0
    scn.view_settings.gamma = 1.0

    render_parameters = {
        "background_color": [0.5, 0.5, 0.5, 1.0],
        "background_strength": 1.0,
        "use_persistent_data": True,
        "transparent_min_bounces": 2,
        "color_mode": "RGBA",
        "max_bounces": 2,
        "render_samples": 50,
        "use_spatial_splits": True,
        "transparent_max_bounces": 2,
        "rendering_device": "GPU",
        "use_caustics_refractive": False,
        "resolution_percentage": 100,
        "denoising_radius": 5,
        "glossy_bounces": 2,
        "min_bounces": 2,
        "transmission_bounces": 2,
        "use_film_transparent": False,
        "use_denoising": True,
        "use_caustics_reflective": False,
        "resolution": 128,
    }
    utils.apply_settings(scn, render_parameters)

    # ----------------------------
    # Load object
    # ----------------------------
    print("Using object .blend:", str(OBJ_BLEND))
    obj = utils.load_obj(scn, str(OBJ_BLEND), "toys")

    bpy.ops.mesh.customdata_custom_splitnormals_clear()
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.editmode_toggle()

    vertices = np.array([v.co for v in obj.data.vertices])
    obj.scale = obj.scale * 0.45 / np.max(np.abs(vertices))
    bpy.ops.object.transform_apply(scale=True, location=True, rotation=True)

    obj.location = Vector((0.0, 0.0, 0.0))
    utils.apply_rot(obj, "Y", 315.0)
    utils.apply_rot(obj, "X", 30.0)
    bpy.ops.object.transform_apply(scale=False, location=False, rotation=True)
    omega_el = 0 # X
    omega_az = 0 # Y

    # ----------------------------
    # Light
    # ----------------------------
    lp = {
        "area_light_location": [0, 0, 5],
        "area_light_rotation": [0, 0, 0],
        "area_size_x": 10,
        "area_size_y": 10,
        "area_strength": 30,
        "light_temperature": 6000,
    }
    utils.make_area_lamp(
        lp["area_light_location"],
        lp["area_light_rotation"],
        size_x=lp["area_size_x"],
        size_y=lp["area_size_y"],
        strength=lp["area_strength"],
        temp=lp["light_temperature"],
    )

    # ----------------------------
    # Sensor / camera
    # ----------------------------
    ppsee = Blender_DvsSensor("Sensor")
    ppsee.set_sensor(nx=128, ny=128, pp=0.015)
    ppsee.set_dvs_sensor(
        th_pos=0.15, th_neg=0.15, th_n=0.05,
        lat=500, tau=300, jit=100,
        bgnp=0.0001, bgnn=0.0001,
        ref=40
    )
    ppsee.set_sensor_optics(50)

    ppsee.cam.data.sensor_width = 32
    ppsee.cam.data.sensor_height = 32 * (ppsee.def_y / ppsee.def_x)

    ppsee.set_position([0.0, 0.0, 2.5])
    ppsee.set_speeds([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    ppsee.enable_look_at_drift_and_microsaccades(
        target_obj=obj,
        drift_sigma_deg=(0.01, 0.009),
        microsaccade_rate_hz=10.0,
        microsaccade_amp_deg=(0.2, 1.0),
        microsaccade_dur_ms=(10, 30),
        seed=2,
    )

    ppsee.init_thresholds()
    if os.path.exists(NOISE_POS) and os.path.exists(NOISE_NEG):
        ppsee.init_bgn_hist(NOISE_POS, NOISE_NEG)
    else:
        print("WARNING: noise hist files not found; continuing without init_bgn_hist")

    # ----------------------------
    # Render setup
    # ----------------------------
    if ppsee.cam.name not in scn.collection.objects:
        scn.collection.objects.link(ppsee.cam)
    scn.camera = ppsee.cam

    scn.render.image_settings.file_format = "PNG"
    scn.render.resolution_x = int(ppsee.def_x)
    scn.render.resolution_y = int(ppsee.def_y)

    W = int(ppsee.def_x)
    H = int(ppsee.def_y)

    # ----------------------------
    # Event pipeline
    # ----------------------------
    ev = EventBuffer(0)

    # IMPORTANT: frametime is microseconds
    # 10000 = 10 ms window, 1000 = 1 ms window
    ed = EventDisplay("Events", ppsee.def_x, ppsee.def_y, 10000, render=0, display_time=True)

    # Trajectory buffers (time vs x,y)
    traj_t_us, traj_x, traj_y = [], [], []

    num_frames = 50
    fps = 1000
    dt_s = 1.0 / float(fps)
    dt_us = int(round(dt_s * 1e6))
    lux_scale = 1e4

    dvs_initialized = False
    ev_png_idx = 0  # consecutive index for saved event-display PNGs
    

    for p in range(num_frames):
        # Update internal drift/microsaccade timing
        ppsee.update_time(dt_s)
        
        # --- ROTATE OBJECT (global rotation) ---
        utils.apply_rot(obj, "Y", omega_az * dt_s) 
        utils.apply_rot(obj, "X", omega_el * dt_s)

        # Time (microseconds) for trajectory log
        t_us = p * dt_us

        # Scanpath proxy (projection of object origin)
        tgt_w = obj.matrix_world.translation
        x_px, y_px = world_to_pixel(scn, ppsee.cam, tgt_w, W, H)
        traj_t_us.append(t_us)
        traj_x.append(x_px)
        traj_y.append(y_px)

        # Render -> PNG
        frame_path = OUT_IMAGES / f"frame_{p:05d}.png"
        scn.render.filepath = str(frame_path)
        bpy.ops.render.render(write_still=1)

        im_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if im_bgr is None:
            raise RuntimeError(f"Could not read render: {frame_path}")

        # Convert to L (lux)
        L = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2LUV)[:, :, 0].astype(np.float32) / 255.0 * lux_scale

        if not dvs_initialized:
            ppsee.init_image(L)
            dvs_initialized = True
        else:
            # NOTE: ppsee.update expects dt in microseconds in this codebase
            pk = ppsee.update(L, dt_us)

            # EventDisplay.update also uses dt in microseconds
            did_render = ed.update(pk, dt_us)

            if did_render:
                png_path = OUT_EVENTS / f"events_{ev_png_idx:05d}.png"
                ed.save_png(str(png_path))
                ev_png_idx += 1

            ev.increase_ev(pk)

        cv2.imshow("Blender", im_bgr)
        cv2.waitKey(1)

    # Write DVS events to .dat inside events/
    ev_path = OUT_EVENTS / "events.dat"
    ev.write(str(ev_path), width=W, height=H)

    # Trajectory plot + trajectory .dat inside traj/
    xy = np.stack([np.asarray(traj_x, np.float32), np.asarray(traj_y, np.float32)], axis=1)
    scan_png = OUT_TRAJ / "scanpath_axes_only.png"
    plot_scanROI(xy_px=xy, W=W, H=H, out_png=str(scan_png))

    traj_dat = OUT_TRAJ / "traj_txy.dat"
    traj_arr = np.stack(
        [np.asarray(traj_t_us, np.int64),
         np.asarray(traj_x, np.float32),
         np.asarray(traj_y, np.float32)],
        axis=1
    )
    np.savetxt(
        str(traj_dat),
        traj_arr,
        fmt=["%d", "%.6f", "%.6f"],
        header="t_us x_px y_px",
        comments=""
    )

    # Build videos + gifs from PNGs and save INSIDE images/ and events/
    # Playback FPS is purely for visualization (20 is fine)
    pngs_to_video(OUT_IMAGES, OUT_IMAGES / "render_from_pngs.avi", fps=20.0, pattern="frame_*.png")
    pngs_to_gif(OUT_IMAGES, OUT_IMAGES / "render_from_pngs.gif", fps=20.0, pattern="frame_*.png")

    pngs_to_video(OUT_EVENTS, OUT_EVENTS / "events_from_pngs.avi", fps=20.0, pattern="events_*.png")
    pngs_to_gif(OUT_EVENTS, OUT_EVENTS / "events_from_pngs.gif", fps=20.0, pattern="events_*.png")

    print("Saved outputs:")
    print(" - images PNGs:", str(OUT_IMAGES))
    print(" - images AVI:", str(OUT_IMAGES / "render_from_pngs.avi"))
    print(" - images GIF:", str(OUT_IMAGES / "render_from_pngs.gif"))
    print(" - events PNGs:", str(OUT_EVENTS))
    print(" - events DAT:", str(ev_path))
    print(" - events AVI:", str(OUT_EVENTS / "events_from_pngs.avi"))
    print(" - events GIF:", str(OUT_EVENTS / "events_from_pngs.gif"))
    print(" - traj plot:", str(scan_png))
    print(" - traj dat:", str(traj_dat))


if __name__ == "__main__":
    main()


