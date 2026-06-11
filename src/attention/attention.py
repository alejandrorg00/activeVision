# Alejandro Rodriguez-Garcia
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .foveation import foveate_black_gray
except ImportError:
    from attention.foveation import foveate_black_gray


DEFAULT_ATTENTION_PARAMS = {
    # Saliency backend. Options:
    #   "center_surround" : multiscale centre-surround filtering.
    #   "lif_vm"          : Von Mises orientation filters + Sinabs LIF neurons.
    "saliency_backend": "center_surround",

    # Multiscale processing.
    "num_pyr": 6,

    # Centre-surround saliency parameters.
    "center_sigma": 2.0,
    "surround_sigma": 8.0,

    # LIF/Von-Mises saliency parameters.
    "lif_tau_mem": 10.0,
    "lif_thetas": None,              # None -> 8 orientations in [0, 2*pi).
    "lif_size_krn": 31,
    "lif_rho": 0.1,
    "lif_r0": 8,
    "lif_thick": 0.5,
    "lif_offset": (0, 0),
    "lif_filter_resize_perc": 1.0,
    "lif_stride": 1,
    "lif_device": "auto",          # "auto", "cpu", or "cuda".
    "lif_stateful": True,            # Keep LIF membrane state across event windows.

    # Softmax beta.
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

    Returns
    -------
    windows:
        List of grayscale accumulated event windows, each shape (H, W), uint8.

    window_times_s:
        End time of each event window, in seconds.

    resolution:
        Tuple (H, W).
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


def _softmax_sample_from_saliency(
    saliency: np.ndarray,
    beta: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """
    Normalize a saliency/activity map and sample a fixation from
    softmax(beta * saliency_norm).

    Returns
    -------
    saliency_norm:
        HxW normalized activity map in [0, 1].
    fixation:
        Tuple (x, y) sampled from the softmax probability map.
    prob_map:
        HxW probability map.
    """
    saliency = np.asarray(saliency, dtype=np.float32)

    if saliency.ndim != 2:
        raise ValueError(f"Expected HxW saliency map. Got {saliency.shape}.")

    if rng is None:
        rng = np.random.default_rng()

    height, width = saliency.shape
    saliency = np.nan_to_num(saliency, nan=0.0, posinf=0.0, neginf=0.0)
    saliency[saliency < 0] = 0.0

    smax = float(saliency.max())

    if smax <= 0:
        saliency_norm = np.zeros_like(saliency, dtype=np.float32)
        prob_map = np.full(
            (height, width),
            1.0 / float(height * width),
            dtype=np.float32,
        )
    else:
        saliency_norm = saliency / (smax + 1e-12)
        logits = float(beta) * saliency_norm
        logits = logits - float(logits.max())
        exp_logits = np.exp(logits).astype(np.float32)
        prob_map = exp_logits / (float(exp_logits.sum()) + 1e-12)

    flat_idx = int(rng.choice(prob_map.size, p=prob_map.reshape(-1)))
    peak_y, peak_x = np.unravel_index(flat_idx, prob_map.shape)

    return saliency_norm.astype(np.float32), (int(peak_x), int(peak_y)), prob_map


def _zero_2pi_tan(x: float, y: float) -> float:
    """
    Angle in [0, 2*pi].
    """
    return float(np.arctan2(y, x) % (2.0 * np.pi))


def _vm_filter(
    theta: float,
    scale: int,
    rho: float = 0.1,
    r0: int = 0,
    thick: float = 0.5,
    offset: tuple[float, float] = (0, 0),
) -> np.ndarray:
    """
    Generate one Von Mises-like orientation filter.
    Adapted from the original LIF attention code.
    """
    from scipy.special import iv

    height, width = int(scale), int(scale)
    vm = np.empty((height, width), dtype=np.float32)
    offset_x, offset_y = offset

    for x in range(width):
        for y in range(height):
            X = (x - width / 2.0) + r0 * np.cos(theta) - offset_x * np.cos(theta)
            Y = (height / 2.0 - y) + r0 * np.sin(theta) - offset_y * np.sin(theta)
            r = np.sqrt(X**2 + Y**2)
            angle = _zero_2pi_tan(X, Y)
            denom = float(iv(0, r - r0)) + 1e-12
            vm[y, x] = np.exp(float(thick) * float(rho) * float(r0) * np.cos(angle - theta)) / denom

    vm = np.nan_to_num(vm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Stabilize filter magnitude for Conv2d/LIF.
    vm = vm - float(vm.mean())
    vmax = float(np.max(np.abs(vm)))
    if vmax > 0:
        vm = vm / vmax

    return vm.astype(np.float32)


def _vm_kernels(
    thetas: np.ndarray,
    size: int,
    rho: float,
    r0: int,
    thick: float,
    offset: tuple[float, float],
    fltr_resize_perc: float,
) -> np.ndarray:
    """
    Create a bank of Von Mises filters with different orientations.
    Returns array with shape (n_orientations, kernel_h, kernel_w).
    """
    filters = []

    for theta in thetas:
        filt = _vm_filter(
            theta=float(theta),
            scale=int(size),
            rho=float(rho),
            r0=int(r0),
            thick=float(thick),
            offset=offset,
        )

        if float(fltr_resize_perc) != 1.0:
            from skimage.transform import rescale

            filt = rescale(
                filt,
                float(fltr_resize_perc),
                anti_aliasing=False,
                preserve_range=True,
            ).astype(np.float32)

        filters.append(filt.astype(np.float32))

    # Ensure all kernels have same shape after optional rescaling.
    min_h = min(f.shape[0] for f in filters)
    min_w = min(f.shape[1] for f in filters)
    filters = [f[:min_h, :min_w] for f in filters]

    return np.stack(filters).astype(np.float32)


def _get_lif_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def _build_lif_vm_attention_net(params: dict):
    """
    Build a simple feedforward spiking attention module:

        event window -> Von Mises Conv2d filters -> Sinabs LIF neurons

    The network returns orientation-selective spiking/activity maps.
    The spatial saliency map is obtained by summing across orientations
    and pyramid scales.
    """
    import torch
    import torch.nn as nn
    import sinabs.layers as sl

    device = _get_lif_device(str(params.get("lif_device", "auto")))

    thetas = params.get("lif_thetas", None)
    if thetas is None:
        thetas = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    else:
        thetas = np.asarray(thetas, dtype=np.float32)

    kernels = _vm_kernels(
        thetas=thetas,
        size=int(params.get("lif_size_krn", 31)),
        rho=float(params.get("lif_rho", 0.1)),
        r0=int(params.get("lif_r0", 8)),
        thick=float(params.get("lif_thick", 0.5)),
        offset=tuple(params.get("lif_offset", (0, 0))),
        fltr_resize_perc=float(params.get("lif_filter_resize_perc", 1.0)),
    )

    kernel_h, kernel_w = int(kernels.shape[-2]), int(kernels.shape[-1])
    stride = int(params.get("lif_stride", 1))

    conv = nn.Conv2d(
        in_channels=1,
        out_channels=int(kernels.shape[0]),
        kernel_size=(kernel_h, kernel_w),
        stride=stride,
        padding=(kernel_h // 2, kernel_w // 2),
        bias=False,
    )

    conv.weight.data = torch.tensor(kernels, dtype=torch.float32).unsqueeze(1)
    conv.weight.requires_grad_(False)

    lif = sl.LIF(tau_mem=float(params.get("lif_tau_mem", 10.0)))

    net = nn.Sequential(conv, lif).to(device)
    net.eval()

    return net, device


def _reset_lif_state(net) -> None:
    """
    Reset Sinabs LIF state when supported by the installed version.
    """
    for module in net.modules():
        if hasattr(module, "reset_states"):
            module.reset_states()
        elif hasattr(module, "reset_state"):
            module.reset_state()


def compute_lif_vm_saliency_map(
    event_frame: np.ndarray,
    net,
    device,
    num_pyr: int = 6,
    beta: float = 10.0,
    rng: np.random.Generator | None = None,
    reset_state: bool = False,
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """
    Compute a saliency map using Von Mises orientation filters followed by
    Sinabs LIF neurons, then sample a fixation from softmax(beta * saliency).

    This backend makes the saliency map an explicit spiking population
    activity map rather than a purely image-filtered centre-surround map.
    """
    import torch
    import torch.nn.functional as F

    frame = np.asarray(event_frame, dtype=np.float32)

    if frame.ndim != 2:
        raise ValueError(f"Expected HxW event frame. Got {frame.shape}.")

    height, width = frame.shape

    # Normalize event window to [0, 1].
    if float(frame.max()) > 0:
        frame = frame / (float(frame.max()) + 1e-12)

    x0 = torch.tensor(frame, dtype=torch.float32, device=device)[None, None, :, :]
    saliency = np.zeros((height, width), dtype=np.float32)

    if reset_state:
        _reset_lif_state(net)

    with torch.no_grad():
        for scale_idx in range(int(num_pyr)):
            scale = 1.0 + float(scale_idx)
            h_s = max(1, int(round(height / scale)))
            w_s = max(1, int(round(width / scale)))

            if h_s != height or w_s != width:
                xs = F.interpolate(x0, size=(h_s, w_s), mode="bilinear", align_corners=False)
                xs = F.interpolate(xs, size=(height, width), mode="bilinear", align_corners=False)
            else:
                xs = x0

            y = net(xs)

            if isinstance(y, (tuple, list)):
                y = y[0]

            # y shape: [1, n_orientations, H', W'].
            y_sum = y.sum(dim=1, keepdim=True)

            if y_sum.shape[-2:] != (height, width):
                y_sum = F.interpolate(y_sum, size=(height, width), mode="bilinear", align_corners=False)

            saliency += y_sum[0, 0].detach().cpu().numpy().astype(np.float32)

    return _softmax_sample_from_saliency(saliency, beta=beta, rng=rng)


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
    Accumulated event-window panel with green action vector for display.

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


def _save_saliency_window_action_gif(
    frames: list,
    output_gif: str | Path,
    fps: int = 20,
    loop: int = 0,
) -> str:
    """
    Save precomputed saliency/window/action frames as GIF.
    """
    from PIL import Image

    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    if len(frames) == 0:
        raise RuntimeError("No frames were provided for saliency_window_action GIF.")

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


def run_attention(
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

    # Optional saliency/window/action GIF.
    plot: bool = False,
    plot_gif_path: str | Path | None = None,
    plot_fps: int = 20,
    plot_loop: int = 0,
) -> dict:
    """
    Fully offline attention + black Gaussian foveation dataset generation.
    Adapted from Giulia D'angelo (https://github.com/GiuliaDAngelo/CTU-EDNeuromorphic/tree/main)

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

    If plot=True, it also saves:
        saliency map | accumulated event window + action vector

    Output layout
    -------------
    output_dir/
        fov/
            roi_00000.png
            roi_00001.png
            ...
        saccades.dat
        saliency_window_action.gif  # only if plot=True

    saccades.dat columns
    --------------------
    idx time_s pre_x pre_y post_x post_y dx dy
    """
    from PIL import Image

    events_npy = Path(events_npy)
    output_dir = Path(output_dir)

    fov_dir = output_dir / "fov"
    saccades_path = output_dir / "saccades.dat"

    if plot_gif_path is None:
        plot_gif_path = output_dir / "saliency_window_action.gif"
    else:
        plot_gif_path = Path(plot_gif_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    fov_dir.mkdir(parents=True, exist_ok=True)

    if clear_existing:
        for old_path in fov_dir.glob(f"{image_prefix}_*.png"):
            old_path.unlink()

        if saccades_path.exists():
            saccades_path.unlink()

        if plot and plot_gif_path.exists():
            plot_gif_path.unlink()

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

    saliency_backend = str(merged_params.get("saliency_backend", "center_surround")).lower()

    lif_net = None
    lif_device = None

    if saliency_backend in {"lif_vm", "lif", "spiking"}:
        lif_net, lif_device = _build_lif_vm_attention_net(merged_params)
        _reset_lif_state(lif_net)
        print(f"[attention] Using LIF/Von-Mises saliency backend on {lif_device}.")
    elif saliency_backend in {"center_surround", "cs", "classic"}:
        print("[attention] Using centre-surround saliency backend.")
    else:
        raise ValueError(
            "Unknown saliency_backend: "
            f"{saliency_backend}. Use 'center_surround' or 'lif_vm'."
        )

    saccades = []
    fov_paths = []
    plot_frames = []

    previous_peak: tuple[int, int] | None = None

    for idx, (window_frame, time_s) in enumerate(zip(windows, window_times_s)):
        if saliency_backend in {"lif_vm", "lif", "spiking"}:
            saliency_map, (peak_x, peak_y), _ = compute_lif_vm_saliency_map(
                window_frame,
                net=lif_net,
                device=lif_device,
                num_pyr=merged_params.get("num_pyr", 6),
                beta=merged_params.get("beta", 10.0),
                rng=rng,
                reset_state=not bool(merged_params.get("lif_stateful", True)),
            )
        else:
            saliency_map, (peak_x, peak_y), _ = compute_saliency_map(
                window_frame,
                center_sigma=merged_params.get("center_sigma", 2.0),
                surround_sigma=merged_params.get("surround_sigma", 8.0),
                num_pyr=merged_params.get("num_pyr", 6),
                beta=merged_params.get("beta", 10.0),
                rng=rng,
            )

        if previous_peak is None:
            pre_x, pre_y = peak_x, peak_y
            dx, dy = 0.0, 0.0
        else:
            pre_x, pre_y = previous_peak
            dx = float(peak_x - pre_x)
            dy = float(peak_y - pre_y)

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

        if plot:
            saliency_panel = _make_saliency_display_panel(
                saliency_map=saliency_map,
                peak_x=peak_x,
                peak_y=peak_y,
            )

            window_action_panel = _make_window_action_display_panel(
                window_frame=window_frame,
                pre_x=pre_x,
                pre_y=pre_y,
                post_x=peak_x,
                post_y=peak_y,
            )

            composite_bgr = np.concatenate(
                [
                    saliency_panel,
                    window_action_panel,
                ],
                axis=1,
            )

            composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
            plot_frames.append(Image.fromarray(composite_rgb))

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

    saliency_window_action_gif_path = None

    if plot:
        saliency_window_action_gif_path = _save_saliency_window_action_gif(
            frames=plot_frames,
            output_gif=plot_gif_path,
            fps=plot_fps,
            loop=plot_loop,
        )

    return {
        "output_dir": str(output_dir),
        "fov_dir": str(fov_dir),
        "saccades_path": str(saccades_path),
        "saliency_window_action_gif_path": saliency_window_action_gif_path,
        "num_foveations": int(len(fov_paths)),
        "resolution": (int(height), int(width)),
        "sigma": float(sigma),
        "beta": float(merged_params.get("beta", 10.0)),
        "window_period_ms": float(window_period_ms),
        "fov_paths": fov_paths,
    }


def plot_attention_exploration(
    events_npy: str | Path,
    saccades_path: str | Path,
    output_dir: str | Path,
    resolution: tuple[int, int] | None = None,
    beta: float | None = None,
    object_label: str | None = None,

    # Grid/object mask parameters.
    per: float = 0.1,
    noise_thresh: int = 100,

    # Fixation handling.
    exclude_initial_fixation: bool = True,
    plot_trajectory: bool = True,

    # Figure output.
    save_name: str = "attention_exploration",
    save_png: bool = True,
    save_pdf: bool = True,
    show: bool = True,
) -> dict:
    """
    Plot explored object area from sampled attention fixations.

    - creates:
        left: global event map + object grid + saccades
        right: fixation heatmap over object cells

    - computes:
        area_explored_coeff = visited object cells / total object cells

    """
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    events_npy = Path(events_npy)
    saccades_path = Path(saccades_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not events_npy.exists():
        raise FileNotFoundError(f"Events npy not found: {events_npy}")

    if not saccades_path.exists():
        raise FileNotFoundError(f"saccades.dat not found: {saccades_path}")

    if object_label is None:
        object_label = events_npy.stem

    # ============================================================
    # Load events
    # ============================================================

    data = np.load(events_npy)

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Expected events with shape (N, 4), got {data.shape}")

    x = data[:, 0].astype(np.int64)
    y = data[:, 1].astype(np.int64)

    if resolution is None:
        W = int(x.max()) + 1
        H = int(y.max()) + 1
    else:
        H, W = int(resolution[0]), int(resolution[1])

    valid_events = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    x = x[valid_events]
    y = y[valid_events]

    if x.size == 0:
        raise RuntimeError("No valid events after resolution filtering.")

    # ============================================================
    # Global event-density image
    # ============================================================

    counts = np.zeros((H, W), dtype=np.int32)
    np.add.at(counts, (y, x), 1)

    event_binary = counts > 0

    # ============================================================
    # Load sampled fixations from saccades.dat
    # ============================================================

    saccades = np.loadtxt(
        saccades_path,
        skiprows=1,
        dtype=np.float64,
    )

    if saccades.ndim == 1:
        saccades = saccades[None, :]

    # saccades.dat columns:
    attention_xy = saccades[:, [4, 5]].astype(np.float32)

    if attention_xy.ndim != 2 or attention_xy.shape[1] != 2:
        raise ValueError(
            f"Expected attention_xy with shape (N, 2), got {attention_xy.shape}"
        )

    if exclude_initial_fixation and attention_xy.shape[0] > 1:
        attention_xy_eff = attention_xy[1:]
    else:
        attention_xy_eff = attention_xy

    if attention_xy_eff.shape[0] == 0:
        raise RuntimeError("No effective fixations available after filtering.")

    # ============================================================
    # Define object grid
    # ============================================================

    crop_x = max(1, int(W * float(per)))
    crop_y = max(1, int(H * float(per)))

    n_cols = int(np.ceil(W / crop_x))
    n_rows = int(np.ceil(H / crop_y))

    object_mask = np.zeros((n_rows, n_cols), dtype=bool)

    for i in range(n_rows):
        for j in range(n_cols):
            y0 = i * crop_y
            x0 = j * crop_x

            y1 = min(y0 + crop_y, H)
            x1 = min(x0 + crop_x, W)

            cell_count = counts[y0:y1, x0:x1].sum()

            if cell_count >= int(noise_thresh):
                object_mask[i, j] = True

    # ============================================================
    # Count fixations per object cell
    # ============================================================

    fix_counts = np.zeros((n_rows, n_cols), dtype=np.int32)

    xs = np.clip(attention_xy_eff[:, 0].astype(np.int64), 0, W - 1)
    ys = np.clip(attention_xy_eff[:, 1].astype(np.int64), 0, H - 1)

    js = np.minimum(xs // crop_x, n_cols - 1)
    is_ = np.minimum(ys // crop_y, n_rows - 1)

    for i_cell, j_cell in zip(is_, js):
        fix_counts[i_cell, j_cell] += 1

    heatmap = fix_counts.astype(np.float32)
    heatmap[~object_mask] = np.nan

    total_obj_cells = int(object_mask.sum())
    visited_obj_cells = int(np.nansum((heatmap > 0).astype(np.int32)))

    if total_obj_cells > 0:
        area_explored_coeff = visited_obj_cells / total_obj_cells
    else:
        area_explored_coeff = np.nan

    print(
        f"[Exploration | beta={beta}] "
        f"Object cells: {total_obj_cells} | "
        f"Visited: {visited_obj_cells} | "
        f"Area explored coeff: {area_explored_coeff:.3f}"
    )

    # ============================================================
    # Plot
    # ============================================================

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        constrained_layout=True,
    )

    # ------------------------------------------------------------
    # Left: event map + object grid + saccades
    # ------------------------------------------------------------

    ax1.imshow(
        event_binary,
        cmap="gray",
        origin="upper",
        vmin=0,
        vmax=1,
    )

    ax1.set_aspect("equal")
    ax1.set_xlabel(f"x ({W} px)", fontsize=14)
    ax1.set_ylabel(f"y ({H} px)", fontsize=14)
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)

    ax1.tick_params(
        length=4,
        width=1.2,
        labelsize=10,
        direction="in",
    )

    for i in range(n_rows):
        for j in range(n_cols):
            y0 = i * crop_y
            x0 = j * crop_x

            y1 = min(y0 + crop_y, H)
            x1 = min(x0 + crop_x, W)

            if object_mask[i, j]:
                rect = plt.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="limegreen",
                    alpha=0.25,
                    edgecolor="black",
                    linewidth=0.6,
                )
            else:
                rect = plt.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="none",
                    edgecolor="black",
                    linewidth=0.3,
                )

            ax1.add_patch(rect)

    if plot_trajectory and attention_xy.shape[0] >= 2:
        ax1.plot(
            attention_xy[:, 0],
            attention_xy[:, 1],
            "-",
            color="royalblue",
            linewidth=2.0,
            alpha=0.5,
            zorder=5,
        )

    ax1.scatter(
        attention_xy_eff[:, 0],
        attention_xy_eff[:, 1],
        s=40,
        color="royalblue",
        edgecolors="white",
        linewidth=0.6,
        zorder=6,
    )

    if attention_xy.shape[0] >= 1:
        ax1.scatter(
            [attention_xy[0, 0]],
            [attention_xy[0, 1]],
            s=30,
            facecolors="white",
            edgecolors="royalblue",
            linewidth=1.5,
            zorder=7,
        )

    fix_patch = mlines.Line2D(
        [],
        [],
        color="royalblue",
        marker="o",
        linestyle="None",
        markersize=6,
        label="Fixations",
    )

    sac_line = mlines.Line2D(
        [],
        [],
        color="royalblue",
        linestyle="-",
        linewidth=2,
        alpha=0.5,
        label="Saccades",
    )

    obj_patch = mpatches.Patch(
        facecolor="limegreen",
        alpha=0.25,
        edgecolor="limegreen",
        label="Object cells",
    )

    ax1.legend(
        handles=[fix_patch, sac_line, obj_patch],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        facecolor="none",
        edgecolor="none",
        fontsize=14,
        columnspacing=1.0,
        handletextpad=0.6,
    )

    # ------------------------------------------------------------
    # Right: fixation heatmap over object cells
    # ------------------------------------------------------------

    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(alpha=0.0)

    extent = [0, W, H, 0]

    im2 = ax2.imshow(
        heatmap,
        cmap=cmap,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )

    cb = plt.colorbar(
        im2,
        ax=ax2,
        fraction=0.033,
        pad=0.01,
    )

    cb.set_label(
        "Number of fixations per object cell",
        fontsize=14,
    )

    for i in range(n_rows):
        for j in range(n_cols):
            if object_mask[i, j]:
                x0 = j * crop_x
                y0 = i * crop_y

                x1 = min(x0 + crop_x, W)
                y1 = min(y0 + crop_y, H)

                ax2.add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        fill=False,
                        edgecolor="white",
                        linewidth=0.4,
                        alpha=0.6,
                    )
                )

    ax2.set_xlabel(f"x ({W} px)", fontsize=14)
    ax2.set_ylabel(f"y ({H} px)", fontsize=14)
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)

    ax2.set_title(
        f"Explored object area = {area_explored_coeff:.3f}",
        fontsize=16,
        pad=22,
    )

    fig.suptitle(
        f"{object_label} | beta={beta}",
        fontsize=18,
        y=1.04,
    )

    png_path = None
    pdf_path = None

    if save_png:
        png_path = output_dir / f"{save_name}.png"
        plt.savefig(png_path, dpi=300, bbox_inches="tight")

    if save_pdf:
        pdf_path = output_dir / f"{save_name}.pdf"
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "attention_exploration_png_path": str(png_path) if png_path else None,
        "attention_exploration_pdf_path": str(pdf_path) if pdf_path else None,
        "area_explored_coeff": float(area_explored_coeff),
        "total_obj_cells": int(total_obj_cells),
        "visited_obj_cells": int(visited_obj_cells),
        "crop_x": int(crop_x),
        "crop_y": int(crop_y),
        "noise_thresh": int(noise_thresh),
    }