from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .foveation import foveate_black_gray
except ImportError:
    from attention.foveation import foveate_black_gray


DEFAULT_ATTENTION_PARAMS = {
    # Multiscale center-surround saliency.
    "num_pyr": 6,
    "center_sigma": 2.0,
    "surround_sigma": 8.0,

    # Softmax beta.
    # Larger beta means sharper fixation.
    # Smaller beta means more diffuse fixation.
    "beta": 10.0,

    # RNG seed for sampling from the softmax map.
    "seed": 0,
}


def _build_event_windows(
    events: np.ndarray,
    resolution: tuple[int, int] | None = None,
    window_period_ms: float = 10.0,
    max_windows: int | None = None,
    use_polarity: bool = False,
) -> tuple[list[np.ndarray], list[float], tuple[int, int]]:
    """
    Convert event array to offline temporal event frames.

    Expected event format:
        events[:, 0] = x
        events[:, 1] = y
        events[:, 2] = polarity
        events[:, 3] = timestamp_seconds

    Returns:
        windows: list of accumulated event windows, each (H, W) uint8
        window_times_s: end time of each window, in seconds
        resolution: (H, W)
    """
    if events.ndim != 2 or events.shape[1] < 4:
        raise ValueError(f"Expected events with shape (N, 4). Got {events.shape}.")

    if events.shape[0] == 0:
        raise ValueError("events array is empty.")

    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    p = events[:, 2].astype(np.int64)
    t_ms = events[:, 3].astype(np.float64) * 1e3

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

    t_start = float(t_ms.min())
    t_stop = float(t_ms.max())

    windows: list[np.ndarray] = []
    window_times_s: list[float] = []

    left = t_start
    right = left + window_period_ms
    n_windows = 0

    while left < t_stop:
        ind = (t_ms >= left) & (t_ms < right)

        frame = np.zeros((height, width), dtype=np.uint8)

        if ind.any():
            xi = x[ind]
            yi = y[ind]
            pi = p[ind]

            if use_polarity:
                on = pi > 0
                off = ~on

                if on.any():
                    frame[yi[on], xi[on]] = 255

                if off.any():
                    frame[yi[off], xi[off]] = 120
            else:
                frame[yi, xi] = 255

        windows.append(frame)
        window_times_s.append(right * 1e-3)

        n_windows += 1
        if max_windows is not None and n_windows >= int(max_windows):
            break

        left = right
        right += window_period_ms

    return windows, window_times_s, (height, width)


def compute_saliency_map(
    event_frame: np.ndarray,
    center_sigma: float = 2.0,
    surround_sigma: float = 8.0,
    num_pyr: int = 6,
    beta: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """
    Compute saliency map and sample fixation from softmax.

    The selected fixation is always sampled from:

        prob_map = softmax(beta * saliency_norm)

    There is no selection argument.

    Larger beta gives sharper fixation.
    Smaller beta gives more diffuse fixation.
    """
    frame = np.asarray(event_frame, dtype=np.float32)

    if frame.ndim != 2:
        raise ValueError(f"Expected HxW event frame. Got {frame.shape}.")

    height, width = frame.shape
    saliency = np.zeros((height, width), dtype=np.float32)

    for scale_idx in range(int(num_pyr)):
        scale = 1.0 + float(scale_idx)

        center_sigma_s = max(float(center_sigma) * scale, 0.5)
        surround_sigma_s = max(
            float(surround_sigma) * scale,
            center_sigma_s + 0.5,
        )

        center = cv2.GaussianBlur(frame, (0, 0), center_sigma_s)
        surround = cv2.GaussianBlur(frame, (0, 0), surround_sigma_s)

        diff = center - surround
        diff[diff < 0] = 0.0

        saliency += diff

    if rng is None:
        rng = np.random.default_rng()

    smax = float(saliency.max())

    if smax <= 0:
        saliency_norm = np.zeros_like(saliency, dtype=np.float32)
        prob_map = np.full(
            (height, width),
            1.0 / float(height * width),
            dtype=np.float32,
        )

        flat_idx = int(rng.choice(prob_map.size, p=prob_map.reshape(-1)))
        peak_y, peak_x = np.unravel_index(flat_idx, prob_map.shape)

        return saliency_norm, (int(peak_x), int(peak_y)), prob_map

    saliency_norm = saliency / (smax + 1e-12)

    logits = float(beta) * saliency_norm
    logits = logits - float(logits.max())

    exp_logits = np.exp(logits).astype(np.float32)
    prob_map = exp_logits / (float(exp_logits.sum()) + 1e-12)

    flat_idx = int(rng.choice(prob_map.size, p=prob_map.reshape(-1)))
    peak_y, peak_x = np.unravel_index(flat_idx, prob_map.shape)

    return saliency_norm.astype(np.float32), (int(peak_x), int(peak_y)), prob_map


def run_attention_black_gaussian_dataset(
    events_npy: str | Path,
    output_dir: str | Path,
    resolution: tuple[int, int] | None = None,
    attention_params: dict | None = None,
    window_period_ms: float = 10.0,
    sigma: float | None = None,
    max_windows: int | None = None,
    use_polarity: bool = False,
    clear_existing: bool = True,
    image_prefix: str = "roi",
) -> dict:
    """
    Fully offline attention + black Gaussian foveation dataset generation.

    Pipeline:
        events.npy
            -> temporal event windows
            -> accumulated event window
            -> multiscale center-surround saliency
            -> softmax(beta * saliency)
            -> sample saliency/fixation point
            -> black Gaussian foveation centered on sampled saliency point
            -> save fov/roi_00000.png, fov/roi_00001.png, ...
            -> save saccades.dat

    Output layout:
        output_dir/
            fov/
                roi_00000.png
                roi_00001.png
                ...
            saccades.dat

    saccades.dat columns:
        idx time_s pre_x pre_y post_x post_y dx dy
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

    rng = np.random.default_rng(int(merged_params.get("seed", 0)))

    events = np.load(events_npy)

    windows, window_times_s, (height, width) = _build_event_windows(
        events=events,
        resolution=resolution,
        window_period_ms=window_period_ms,
        max_windows=max_windows,
        use_polarity=use_polarity,
    )

    if sigma is None:
        sigma = max(1.0, 0.10 * float(min(height, width)))

    saccades = []
    fov_paths = []

    previous_peak: tuple[int, int] | None = None

    for idx, (window_frame, time_s) in enumerate(zip(windows, window_times_s)):
        _, (peak_x, peak_y), _ = compute_saliency_map(
            window_frame,
            center_sigma=merged_params.get("center_sigma", 2.0),
            surround_sigma=merged_params.get("surround_sigma", 8.0),
            num_pyr=merged_params.get("num_pyr", 6),
            beta=merged_params.get("beta", 10.0),
            rng=rng,
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

    This is only for display or inspection.
    It is not part of the dataset output.

    loop=0 means loop indefinitely.
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
    Saliency map panel for display only.
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
    Accumulated event-window panel with green action vector.

    The vector goes from previous saliency point to current saliency point:
        (pre_x, pre_y) -> (post_x, post_y)
    """
    window_u8 = np.asarray(window_frame, dtype=np.uint8)
    panel = cv2.cvtColor(window_u8, cv2.COLOR_GRAY2BGR)

    green = (20, 255, 57)  # BGR phosphor green

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
    window_period_ms: float = 10.0,
    max_windows: int | None = None,
    use_polarity: bool = False,
    fps: int = 20,
    loop: int = 0,
) -> str:
    """
    Display-only GIF.

    Creates:
        saliency map | accumulated event window + action vector

    This does not write dataset outputs.
    It only creates a GIF for notebook inspection.

    The action vector is read from saccades.dat:
        pre_x, pre_y -> post_x, post_y
    """
    from PIL import Image

    events_npy = Path(events_npy)
    saccades_path = Path(saccades_path)
    output_gif = Path(output_gif)

    output_gif.parent.mkdir(parents=True, exist_ok=True)

    merged_params = dict(DEFAULT_ATTENTION_PARAMS)
    if attention_params is not None:
        merged_params.update(attention_params)

    rng = np.random.default_rng(int(merged_params.get("seed", 0)))

    events = np.load(events_npy)

    windows, _, (height, width) = _build_event_windows(
        events=events,
        resolution=resolution,
        window_period_ms=window_period_ms,
        max_windows=max_windows,
        use_polarity=use_polarity,
    )

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

    for idx in range(n):
        window_frame = windows[idx]

        saliency_map, (peak_x, peak_y), _ = compute_saliency_map(
            window_frame,
            center_sigma=merged_params.get("center_sigma", 2.0),
            surround_sigma=merged_params.get("surround_sigma", 8.0),
            num_pyr=merged_params.get("num_pyr", 6),
            beta=merged_params.get("beta", 10.0),
            rng=rng,
        )

        # saccades.dat columns:
        # idx time_s pre_x pre_y post_x post_y dx dy
        pre_x = int(round(saccades[idx, 2]))
        pre_y = int(round(saccades[idx, 3]))
        post_x = int(round(saccades[idx, 4]))
        post_y = int(round(saccades[idx, 5]))

        saliency_panel = _make_saliency_display_panel(
            saliency_map=saliency_map,
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