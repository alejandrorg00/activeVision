from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from attention.foveation import foveate_black_gray

DEFAULT_ATTENTION_PARAMS = {
    # Giulia / NeuromorphicAttentionSim-style parameters
    "size_krn": 16,
    "r0": 14,
    "rho": 0.05,
    "theta": np.pi * 3 / 2,
    "thetas": np.arange(0, 2 * np.pi, np.pi / 4),
    "thick": 3,
    "fltr_resize_perc": [2, 2],
    "offsetpxs": 0,
    "offset": (0, 0),
    "num_pyr": 6,
    "tau_mem": 0.3,
    "stride": 1,
    "out_ch": 1,

    # Softmax beta added on top of Giulia attention.
    # Larger beta gives sharper sampling.
    # Smaller beta gives more diffuse sampling.
    "beta": 10.0,

    # RNG seed.
    "seed": 0,
}


def _get_device(device: str | torch.device | None = None) -> torch.device:
    """
    Resolve torch device.
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _load_events_npy(events_npy: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load events.npy with expected columns:
        x, y, polarity, timestamp_seconds

    Returns:
        x, y, p, t_ms
    """
    events = np.load(events_npy)

    if events.ndim != 2 or events.shape[1] < 4:
        raise ValueError(f"Expected events with shape (N, 4). Got {events.shape}.")

    if events.shape[0] == 0:
        raise ValueError("events array is empty.")

    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    p = events[:, 2].astype(np.int64)
    t_ms = events[:, 3].astype(np.float64) * 1e3

    order = np.argsort(t_ms)
    return x[order], y[order], p[order], t_ms[order]


def build_binary_event_windows(
    events_npy: str | Path,
    resolution: tuple[int, int] | None = None,
    window_period_ms: float = 100.0,
    max_windows: int | None = None,
    use_polarity: bool = False,
) -> tuple[list[np.ndarray], list[float], tuple[int, int]]:
    """
    Build Giulia-style fixed temporal windows from events.npy.

    This is offline binning over saved events.

    It follows the logic:

        if ti <= time:
            window[0][yi][xi] = 255
        else:
            process window
            time += window_period
            reset window

    Important:
        This is binary accumulation.
        If a pixel spikes at least once within the temporal window, it becomes 255.
        It does not normalize by event count.
        It does not fade.
        It does not render online.
    """
    x, y, p, t_ms = _load_events_npy(events_npy)

    if resolution is None:
        height = int(y.max()) + 1
        width = int(x.max()) + 1
    else:
        height, width = int(resolution[0]), int(resolution[1])

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)

    x = x[valid]
    y = y[valid]
    p = p[valid]
    t_ms = t_ms[valid]

    if x.size == 0:
        raise ValueError("No valid events remain after resolution filtering.")

    window_period_ms = float(window_period_ms)
    if window_period_ms <= 0:
        raise ValueError("window_period_ms must be > 0.")

    t0 = float(t_ms.min())
    t1 = float(t_ms.max())

    # Start exactly like Giulia, but shifted to the first event time.
    # First window is [t0, t0 + window_period_ms].
    time_boundary = t0 + window_period_ms

    if use_polarity:
        current = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        current = np.zeros((height, width), dtype=np.uint8)

    windows: list[np.ndarray] = []
    window_times_s: list[float] = []

    n_windows = 0

    for xi, yi, pi, ti in zip(x, y, p, t_ms):
        # Process all elapsed empty windows if needed.
        while ti > time_boundary:
            if use_polarity:
                gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
                windows.append(gray.copy())
            else:
                windows.append(current.copy())

            window_times_s.append(time_boundary * 1e-3)

            n_windows += 1
            if max_windows is not None and n_windows >= int(max_windows):
                return windows, window_times_s, (height, width)

            if use_polarity:
                current.fill(0)
            else:
                current.fill(0)

            time_boundary += window_period_ms

        # Giulia-style binary accumulation.
        if use_polarity:
            # BGR display convention:
            # ON events white, OFF events gray.
            if pi > 0:
                current[int(yi), int(xi)] = (255, 255, 255)
            else:
                current[int(yi), int(xi)] = (120, 120, 120)
        else:
            current[int(yi), int(xi)] = 255

    # Flush final partial window.
    if max_windows is None or n_windows < int(max_windows):
        if use_polarity:
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            windows.append(gray.copy())
        else:
            windows.append(current.copy())

        window_times_s.append(time_boundary * 1e-3)

    return windows, window_times_s, (height, width)


def compute_giulia_attention_softmax(
    window_frame: np.ndarray,
    net_attention,
    device: torch.device,
    resolution: tuple[int, int],
    attention_params: dict,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Giulia/att_module attention on one accumulated event window.

    Returns:
        sal_vis:
            uint8 saliency visualization map.

        peak_xy:
            sampled fixation as (x, y).

        probs_vis:
            full probability map.

        sal_raw:
            raw saliency map.

        probs_raw:
            raw probability map.
    """
    height, width = int(resolution[0]), int(resolution[1])

    window_u8 = np.asarray(window_frame, dtype=np.uint8)

    # att_module.run_attention expects [1, H, W] or compatible.
    window_torch = torch.from_numpy(window_u8).unsqueeze(0).to(device=device, dtype=torch.float32)

    sal_vis, salmax_coords, probs_vis, sal_raw, probs_raw = run_attention(
        window_torch,
        net_attention,
        device,
        (height, width),
        int(attention_params["num_pyr"]),
        beta=float(attention_params.get("beta", 10.0)),
        object_mask=None,
    )

    # salmax_coords from Giulia code is row, col = y, x
    peak_y = int(salmax_coords[0])
    peak_x = int(salmax_coords[1])

    return sal_vis, (peak_x, peak_y), probs_vis, sal_raw, probs_raw


def run_attention_black_gaussian_dataset(
    events_npy: str | Path,
    output_dir: str | Path,
    resolution: tuple[int, int] | None = None,
    attention_params: dict | None = None,
    window_period_ms: float = 100.0,
    sigma: float | None = None,
    max_windows: int | None = None,
    use_polarity: bool = False,
    clear_existing: bool = True,
    image_prefix: str = "roi",
    device: str | torch.device | None = None,
    verbose: bool = False,
) -> dict:
    """
    Offline attention dataset generation.

    Pipeline:
        events.npy
            -> Giulia-style fixed temporal windows
            -> binary accumulated event window
            -> Giulia/att_module attention
            -> softmax(beta * saliency)
            -> sampled fixation
            -> black Gaussian foveation over the accumulated event window
            -> save fov/roi_00000.png, ...
            -> save saccades.dat

    Output layout:
        output_dir/
            fov/
                roi_00000.png
                roi_00001.png
                ...
            saccades.dat
    """
    events_npy = Path(events_npy)
    output_dir = Path(output_dir)

    fov_dir = output_dir / "fov"
    saccades_path = output_dir / "saccades.dat"

    output_dir.mkdir(parents=True, exist_ok=True)
    fov_dir.mkdir(parents=True, exist_ok=True)

    if clear_existing:
        for old_path in fov_dir.glob(f"{image_prefix}_*.png"):
            old_path.unlink()

        if saccades_path.exists():
            saccades_path.unlink()

    merged_params = dict(DEFAULT_ATTENTION_PARAMS)
    if attention_params is not None:
        merged_params.update(attention_params)

    torch.manual_seed(int(merged_params.get("seed", 0)))
    np.random.seed(int(merged_params.get("seed", 0)))

    device_torch = _get_device(device)

    windows, window_times_s, (height, width) = build_binary_event_windows(
        events_npy=events_npy,
        resolution=resolution,
        window_period_ms=window_period_ms,
        max_windows=max_windows,
        use_polarity=use_polarity,
    )

    net_attention = initialise_attention(device_torch, merged_params)
    net_attention = net_attention.to(device_torch)
    net_attention.eval()

    if sigma is None:
        sigma = max(1.0, 0.10 * float(min(height, width)))

    saccades = []
    fov_paths = []

    previous_peak: tuple[int, int] | None = None

    with torch.no_grad():
        for idx, (window_frame, time_s) in enumerate(zip(windows, window_times_s)):
            if verbose:
                print(
                    idx,
                    "time_s=", float(time_s),
                    "nonzero=", int(np.count_nonzero(window_frame)),
                    "max=", int(window_frame.max()),
                    "sum=", int(window_frame.sum()),
                )

            _, (peak_x, peak_y), _, _, _ = compute_giulia_attention_softmax(
                window_frame=window_frame,
                net_attention=net_attention,
                device=device_torch,
                resolution=(height, width),
                attention_params=merged_params,
            )

            fov_black = foveate_black_gray(
                window_frame,
                center_x=peak_x,
                center_y=peak_y,
                sigma=float(sigma),
            )

            fov_path = fov_dir / f"{image_prefix}_{idx:05d}.png"
            ok = cv2.imwrite(str(fov_path), fov_black)

            if not ok:
                raise RuntimeError(f"Could not write foveation image: {fov_path}")

            fov_paths.append(str(fov_path))

            if previous_peak is None:
                pre_x, pre_y = peak_x, peak_y
                dx, dy = 0.0, 0.0
            else:
                pre_x, pre_y = previous_peak
                dx = float(peak_x - pre_x)
                dy = float(peak_y - pre_y)

            saccades.append(
                [
                    int(idx),
                    float(time_s),
                    int(pre_x),
                    int(pre_y),
                    int(peak_x),
                    int(peak_y),
                    float(dx),
                    float(dy),
                ]
            )

            previous_peak = (peak_x, peak_y)

    saccades_arr = np.asarray(saccades, dtype=np.float64)

    header = "idx time_s pre_x pre_y post_x post_y dx dy"
    np.savetxt(
        saccades_path,
        saccades_arr,
        fmt=[
            "%d",
            "%.9f",
            "%d",
            "%d",
            "%d",
            "%d",
            "%.6f",
            "%.6f",
        ],
        header=header,
        comments="",
    )

    return {
        "output_dir": str(output_dir),
        "fov_dir": str(fov_dir),
        "saccades_path": str(saccades_path),
        "num_foveations": int(len(fov_paths)),
        "resolution": (int(height), int(width)),
        "sigma": float(sigma),
        "beta": float(merged_params.get("beta", 10.0)),
        "window_period_ms": float(window_period_ms),
        "device": str(device_torch),
        "fov_paths": fov_paths,
    }


def foveations_to_gif(
    fov_dir: str | Path,
    output_gif: str | Path,
    fps: int = 20,
    loop: int = 0,
    image_prefix: str = "roi",
) -> str:
    """
    Create a GIF from saved foveation images.
    """
    from PIL import Image

    fov_dir = Path(fov_dir)
    output_gif = Path(output_gif)

    output_gif.parent.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(fov_dir.glob(f"{image_prefix}_*.png"))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No foveation images found in {fov_dir}")

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


def _normalise_to_uint8(x: np.ndarray) -> np.ndarray:
    """
    Normalize numeric array to uint8 [0, 255].
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    xmin = float(x.min())
    xmax = float(x.max())

    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.uint8)

    y = 255.0 * (x - xmin) / (xmax - xmin)
    return np.clip(y, 0, 255).astype(np.uint8)


def _make_saliency_display_panel(
    saliency_map: np.ndarray,
    peak_x: int,
    peak_y: int,
) -> np.ndarray:
    """
    Saliency map panel.
    """
    sal_u8 = _normalise_to_uint8(saliency_map)

    cmap = getattr(cv2, "COLORMAP_PARULA", cv2.COLORMAP_VIRIDIS)
    panel = cv2.applyColorMap(sal_u8, cmap)

    cv2.circle(
        panel,
        (int(peak_x), int(peak_y)),
        5,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        panel,
        "saliency",
        (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return panel


def _make_window_action_display_panel(
    window_frame: np.ndarray,
    pre_x: int,
    pre_y: int,
    post_x: int,
    post_y: int,
) -> np.ndarray:
    """
    Accumulated binary event-window panel with green action vector.
    """
    window_u8 = np.asarray(window_frame, dtype=np.uint8)

    # Important:
    # keep Giulia-style binary accumulated event map.
    # Do not normalize by event counts.
    panel = cv2.cvtColor(window_u8, cv2.COLOR_GRAY2BGR)

    green = (20, 255, 57)

    cv2.arrowedLine(
        panel,
        (int(pre_x), int(pre_y)),
        (int(post_x), int(post_y)),
        green,
        2,
        cv2.LINE_AA,
        tipLength=0.25,
    )

    cv2.circle(
        panel,
        (int(pre_x), int(pre_y)),
        3,
        green,
        -1,
    )

    cv2.circle(
        panel,
        (int(post_x), int(post_y)),
        5,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        panel,
        "accumulated window + action",
        (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        green,
        1,
        cv2.LINE_AA,
    )

    return panel


def make_saliency_window_action_gif(
    events_npy: str | Path,
    saccades_path: str | Path,
    output_gif: str | Path,
    resolution: tuple[int, int] | None = None,
    attention_params: dict | None = None,
    window_period_ms: float = 100.0,
    max_windows: int | None = None,
    use_polarity: bool = False,
    fps: int = 20,
    loop: int = 0,
    device: str | torch.device | None = None,
) -> str:
    """
    Display GIF.

    Creates:
        saliency map | accumulated binary event window + action vector

    This reads events.npy and saccades.dat.
    It does not use foveation images.
    """
    from PIL import Image

    events_npy = Path(events_npy)
    saccades_path = Path(saccades_path)
    output_gif = Path(output_gif)

    output_gif.parent.mkdir(parents=True, exist_ok=True)

    merged_params = dict(DEFAULT_ATTENTION_PARAMS)
    if attention_params is not None:
        merged_params.update(attention_params)

    torch.manual_seed(int(merged_params.get("seed", 0)))
    np.random.seed(int(merged_params.get("seed", 0)))

    device_torch = _get_device(device)

    windows, _, (height, width) = build_binary_event_windows(
        events_npy=events_npy,
        resolution=resolution,
        window_period_ms=window_period_ms,
        max_windows=max_windows,
        use_polarity=use_polarity,
    )

    net_attention = initialise_attention(device_torch, merged_params)
    net_attention = net_attention.to(device_torch)
    net_attention.eval()

    if not saccades_path.exists():
        raise FileNotFoundError(f"saccades.dat not found: {saccades_path}")

    saccades = np.loadtxt(
        saccades_path,
        skiprows=1,
        dtype=np.float64,
    )

    if saccades.ndim == 1:
        saccades = saccades[None, :]

    n = min(len(windows), saccades.shape[0])

    frames = []

    with torch.no_grad():
        for idx in range(n):
            window_frame = windows[idx]

            sal_vis, (peak_x, peak_y), _, sal_raw, _ = compute_giulia_attention_softmax(
                window_frame=window_frame,
                net_attention=net_attention,
                device=device_torch,
                resolution=(height, width),
                attention_params=merged_params,
            )

            # saccades.dat columns:
            # idx time_s pre_x pre_y post_x post_y dx dy
            pre_x = int(round(saccades[idx, 2]))
            pre_y = int(round(saccades[idx, 3]))
            post_x = int(round(saccades[idx, 4]))
            post_y = int(round(saccades[idx, 5]))

            saliency_panel = _make_saliency_display_panel(
                saliency_map=sal_raw,
                peak_x=peak_x,
                peak_y=peak_y,
            )

            window_action_panel = _make_window_action_display_panel(
                window_frame=window_frame,
                pre_x=pre_x,
                pre_y=pre_y,
                post_x=post_x,
                post_y=post_y,
            )

            composite_bgr = np.concatenate(
                [
                    saliency_panel,
                    window_action_panel,
                ],
                axis=1,
            )

            composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(composite_rgb))

    if len(frames) == 0:
        blank = np.zeros((height, width * 2, 3), dtype=np.uint8)
        frames = [Image.fromarray(blank)]

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