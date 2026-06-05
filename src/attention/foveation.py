# -*- coding: utf-8 -*-
"""
Alejandro Rodriguez-Garcia 01/02/2026

    This module implements foveation mechanisms for both image frames and event
    streams, including centered cropping, Gaussian masking, and 
    eccentricity-dependent multi-scale blurring. For DVS events, it provides 
    deterministic ROI cropping around a fixation point and probabilistic 
    foveation using a 2D Gaussian keep-probability map. The selected events are
    repackaged into an EventBuffer-compatible structure for downstream 
    neuromorphic processing.

"""
import numpy as np
import cv2


import numpy as np
import cv2


# =============================================================================
# FOVEATION: CROP
# =============================================================================
def foveate_crop(frame, cx, cy, crop_w, crop_h, pad_value=0):
    """ Center crop with padding """
    frame = np.asarray(frame)
    if frame.ndim not in (2, 3):
        raise ValueError(f"foveate_crop expects (H,W) or (H,W,C). Got {frame.shape}")

    H, W = frame.shape[:2]
    half_w = crop_w // 2
    half_h = crop_h // 2

    x0, x1 = cx - half_w, cx - half_w + crop_w
    y0, y1 = cy - half_h, cy - half_h + crop_h

    # Allocate output
    if frame.ndim == 2:
        out = np.full((crop_h, crop_w), pad_value, dtype=np.uint8)
    else:
        C = frame.shape[2]
        out = np.zeros((crop_h, crop_w, C), dtype=np.uint8)
        out[...] = np.array(pad_value, dtype=np.uint8)

    # Source bounds
    src_x0 = max(0, x0)
    src_x1 = min(W, x1)
    src_y0 = max(0, y0)
    src_y1 = min(H, y1)

    # Destination bounds
    dst_x0 = src_x0 - x0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y0 = src_y0 - y0
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if (src_x1 > src_x0) and (src_y1 > src_y0):
        out[dst_y0:dst_y1, dst_x0:dst_x1, ...] = frame[src_y0:src_y1, src_x0:src_x1, ...]

    return out


# =============================================================================
# FOVEATION: BLACK (GAUSSIAN MASK)
# =============================================================================
def foveate_black(frame, peak_x, peak_y, sigma, eps=1e-8):
    """ Gaussian multiplicative mask centered at (peak_x, peak_y) """
    frame = np.asarray(frame)
    if frame.ndim not in (2, 3):
        raise ValueError(f"foveate_black expects (H,W) or (H,W,C). Got {frame.shape}")

    H, W = frame.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    g = np.exp(
        -0.5 * (
            ((xx - peak_x) / (sigma + eps)) ** 2 +
            ((yy - peak_y) / (sigma + eps)) ** 2
        )
    ).astype(np.float32)

    g = (g - g.min()) / (g.max() - g.min() + eps)  # normalize to [0,1]

    if frame.ndim == 2:
        out = frame.astype(np.float32) * g
    else:
        out = frame.astype(np.float32) * g[:, :, None]

    return np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================
# FOVEATION: BLUR (MULTI-SCALE)
# =============================================================================
def foveate_blur(frame, cx, cy, sigma0=1e-8, sigma_max=50.0, n_scales=6):
    """
    Multi-scale foveated blur.
    Sharp near (cx, cy), increasingly blurred in the periphery.

    """
    frame = np.asarray(frame)
    if frame.ndim not in (2, 3):
        raise ValueError(f"foveate_blur expects (H,W) or (H,W,C). Got {frame.shape}")

    H, W = frame.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = np.sqrt(W ** 2 + H ** 2)

    sigma_map = sigma0 + (sigma_max - sigma0) * (r / (r_max + 1e-8))

    sigmas = np.linspace(sigma0, sigma_max, n_scales).astype(np.float32)
    step = np.mean(np.diff(sigmas)) if n_scales > 1 else (sigma_max - sigma0)
    delta = max(1e-3, 0.5 * step)

    # Blur pyramid
    blurred = [cv2.GaussianBlur(frame, (0, 0), float(s)) for s in sigmas]
    blurred_stack = np.stack(blurred, axis=-1).astype(np.float32)
    # gray:  H,W,S
    # color: H,W,C,S

    diff = (sigma_map[..., None] - sigmas[None, None, :]) ** 2
    w = np.exp(-diff / (2.0 * delta ** 2 + 1e-8)).astype(np.float32)
    w /= np.sum(w, axis=-1, keepdims=True) + 1e-8

    if frame.ndim == 2:
        out = np.sum(w * blurred_stack, axis=-1)
    else:
        out = np.sum(blurred_stack * w[:, :, None, :], axis=-1)

    return np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================
# EVENT FOVEATION: Crop
# =============================================================================
def event_crop_buffer(ev_fix, pk, cx, cy, w, h, ts_default_us=None, W=None, H=None):
    """ Crops events centered on the fixational point """

    if pk is None or getattr(pk, "i", 0) <= 0:
        return

    i = int(pk.i)

    xs = pk.x[:i].astype(np.int32, copy=False)
    ys = pk.y[:i].astype(np.int32, copy=False)

    # bounds (optional clamp if W/H provided)
    x0 = cx - w // 2
    x1 = cx + (w - w // 2)
    y0 = cy - h // 2
    y1 = cy + (h - h // 2)

    if W is not None:
        x0 = max(0, x0)
        x1 = min(int(W), x1)
    if H is not None:
        y0 = max(0, y0)
        y1 = min(int(H), y1)

    m = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    if not np.any(m):
        return

    # Grab timestamps from pk under common names
    if hasattr(pk, "ts"):
        ts_arr = pk.ts[:i]
    elif hasattr(pk, "t"):
        ts_arr = pk.t[:i]
    elif hasattr(pk, "t_us"):
        ts_arr = pk.t_us[:i]
    else:
        if ts_default_us is None:
            raise AttributeError(
                "pk has no timestamp field (ts/t/t_us) and ts_default_us was not provided."
            )
        # fallback: all events in this packet share the same timestamp
        ts_arr = np.full((i,), int(ts_default_us), dtype=np.uint64)

    # Build a minimal ev-like object compatible with EventBuffer.increase_ev
    class _Ev:
        pass

    ev = _Ev()
    ev.x = pk.x[:i][m].astype(np.uint16, copy=False)
    ev.y = pk.y[:i][m].astype(np.uint16, copy=False)
    ev.p = pk.p[:i][m].astype(np.uint8, copy=False)
    ev.ts = ts_arr[m].astype(np.uint64, copy=False)
    ev.i = int(ev.x.shape[0])

    ev_fix.increase_ev(ev)


def gaussian_2d(cx, cy, percentage, H, W, theta=0.0, eps=1e-12):
    """
    Returns a (H,W) gaussian mask in [0,1], centered at (cx,cy).
    Percentage defines sigma as percentage of resolution.
    """
    sigma_y = max(1.0, float(percentage) * float(H))
    sigma_x = max(1.0, float(percentage) * float(W))

    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    ct = np.cos(theta)
    st = np.sin(theta)
    a = (ct * ct) / (2.0 * sigma_x * sigma_x + eps) + (st * st) / (2.0 * sigma_y * sigma_y + eps)
    b = (-np.sin(2.0 * theta)) / (4.0 * sigma_x * sigma_x + eps) + (np.sin(2.0 * theta)) / (4.0 * sigma_y * sigma_y + eps)
    c = (st * st) / (2.0 * sigma_x * sigma_x + eps) + (ct * ct) / (2.0 * sigma_y * sigma_y + eps)

    G = np.exp(-(a * (X - cx) ** 2 + 2.0 * b * (X - cx) * (Y - cy) + c * (Y - cy) ** 2))
    return G


def event_fov_buffer(ev_foveated,pk,cx,cy,percentage,W,H,ts_default_us=None,rng=None,gaussian_cache=None,mode="resample_unique",):
    """
    Foveate events probabilistically and append into ev_foveated.

    - Builds a gaussian keep-probability map centered at (cx,cy).
    - For events in pk, computes keep_prob = gaussian[y,x].
    - mode:
        * "resample_unique" : probability-weighted resampling with replacement
        * "bernoulli"       : independent per-event acceptance with probability
    """

    if pk is None or getattr(pk, "i", 0) <= 0:
        return

    if rng is None:
        rng = np.random.default_rng()

    i = int(pk.i)
    xs = pk.x[:i].astype(np.int32, copy=False)
    ys = pk.y[:i].astype(np.int32, copy=False)

    valid = (0 <= xs) & (xs < int(W)) & (0 <= ys) & (ys < int(H))
    if not np.any(valid):
        return
    xs = xs[valid]
    ys = ys[valid]

    # timestamps
    if hasattr(pk, "ts"):
        ts_all = pk.ts[:i][valid].astype(np.uint64, copy=False)
    elif hasattr(pk, "t"):
        ts_all = pk.t[:i][valid].astype(np.uint64, copy=False)
    elif hasattr(pk, "t_us"):
        ts_all = pk.t_us[:i][valid].astype(np.uint64, copy=False)
    else:
        if ts_default_us is None:
            raise AttributeError("pk has no ts/t/t_us and ts_default_us was not provided.")
        ts_all = np.full(xs.shape[0], int(ts_default_us), dtype=np.uint64)

    ps_all = pk.p[:i][valid].astype(np.uint8, copy=False)

    # gaussian map (cacheable)
    if gaussian_cache is not None:
        G = gaussian_cache
    else:
        G = gaussian_2d(cx, cy, float(percentage), int(H), int(W))

    keep_prob = G[ys, xs].astype(np.float64, copy=False)
    n = int(keep_prob.shape[0])
    if n == 0:
        return

    if mode == "resample_unique":
        s = float(keep_prob.sum())
        if s <= 0.0:
            return
        pnorm = keep_prob / s
        idx = rng.choice(n, size=n, replace=True, p=pnorm)
        idx = np.unique(idx)

    elif mode == "bernoulli":
        u = rng.random(n)
        idx = np.where(u < keep_prob)[0]
        if idx.size == 0:
            return

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Build an ev-like object compatible with EventBuffer.increase_ev
    class _Ev:
        pass

    ev = _Ev()
    ev.x = xs[idx].astype(np.uint16, copy=False)
    ev.y = ys[idx].astype(np.uint16, copy=False)
    ev.p = ps_all[idx].astype(np.uint8, copy=False)
    ev.ts = ts_all[idx].astype(np.uint64, copy=False)
    ev.i = int(ev.x.shape[0])

    ev_foveated.increase_ev(ev)