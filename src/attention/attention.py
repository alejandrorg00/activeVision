# -*- coding: utf-8 -*-
"""
Alejandro Rodriguez-Garcia 01/02/2026

    Adapted from CTU-EDNeuromorphic (Giulia D'Angelo). 

    It implements an event-driven attention mechanism that transforms
    DVS event windows into a multi-scale pyramid and processes them through a 
    Conv2D + LIF (spiking) network with Von Mises kernels to generate a saliency
    map.

"""

import numpy as np
import cv2
from collections import deque
from scipy.special import iv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sinabs.layers as sl
from skimage.transform import rescale


# -----------------------------
# Event-window visual utilities
# -----------------------------
def time_window(events, camera_events, height, width, window_period):
    e_x = events["data"][camera_events]["dvs"]["x"]
    e_y = events["data"][camera_events]["dvs"]["y"]
    e_ts = np.multiply(events["data"][camera_events]["dvs"]["ts"], 1e3)
    e_pol = events["data"][camera_events]["dvs"]["pol"]

    time = window_period
    window_pos = np.zeros((height, width), dtype=np.uint8)
    window_neg = np.zeros((height, width), dtype=np.uint8)

    for x, y, ts, pol in zip(e_x, e_y, e_ts, e_pol):
        if ts <= time:
            if pol == 1:
                window_pos[y, x] = 255
            else:
                window_neg[y, x] = 255
        else:
            cv2.imshow("Event Pos and Neg", np.hstack((window_pos, window_neg)))
            cv2.waitKey(1)
            time += window_period
            window_pos.fill(0)
            window_neg.fill(0)


def sliding_window(events, camera_events, height, width, initial_window_period, sliding_wdw, time_buff):
    e_x = events["data"][camera_events]["dvs"]["x"]
    e_y = events["data"][camera_events]["dvs"]["y"]
    e_ts = np.multiply(events["data"][camera_events]["dvs"]["ts"], 1e3)
    e_pol = events["data"][camera_events]["dvs"]["pol"]

    sliding_window_pos = np.zeros((height, width), dtype=np.uint8)
    sliding_window_neg = np.zeros((height, width), dtype=np.uint8)
    event_queue = deque()

    for x, y, ts, pol in zip(e_x, e_y, e_ts, e_pol):
        if ts <= initial_window_period:
            if pol == 1:
                sliding_window_pos[y, x] = 255
            else:
                sliding_window_neg[y, x] = 255
            event_queue.append((x, y, ts, pol))
        else:
            if ts <= initial_window_period + time_buff:
                while event_queue and event_queue[0][2] < ts - initial_window_period:
                    x_old, y_old, _, pol_old = event_queue.popleft()
                    if pol_old == 1:
                        sliding_window_pos[y_old, x_old] = 0
                    else:
                        sliding_window_neg[y_old, x_old] = 0

                if pol == 1:
                    sliding_window_pos[y, x] = 255
                else:
                    sliding_window_neg[y, x] = 255
                event_queue.append((x, y, ts, pol))
            else:
                cv2.imshow("Event Pos and Neg", np.hstack((sliding_window_pos, sliding_window_neg)))
                cv2.waitKey(1)
                time_buff += sliding_wdw


def number_events(events, camera_events, height, width, num_events):
    e_x = events["data"][camera_events]["dvs"]["x"]
    e_y = events["data"][camera_events]["dvs"]["y"]
    e_pol = events["data"][camera_events]["dvs"]["pol"]

    window_pos = np.zeros((height, width), dtype=np.uint8)
    window_neg = np.zeros((height, width), dtype=np.uint8)

    for i in range(0, len(e_x), num_events):
        window_pos.fill(0)
        window_neg.fill(0)

        for j in range(i, min(i + num_events, len(e_x))):
            x, y, pol = int(e_x[j]), int(e_y[j]), int(e_pol[j])
            if pol == 1:
                window_pos[y, x] = 255
            else:
                window_neg[y, x] = 255

        cv2.imshow("Event Pos and Neg", np.hstack((window_pos, window_neg)))
        cv2.waitKey(1)


# -----------------------------
# Von Mises filter bank
# -----------------------------
def zero_2pi_tan(x, y):
    return np.arctan2(y, x) % (2 * np.pi)


def vm_filter(theta, scale, rho=0.1, r0=0, thick=0.5, offset=(0, 0)):
    height, width = scale, scale
    vm = np.empty((height, width), dtype=np.float32)
    offset_x, offset_y = offset

    for x in range(width):
        for y in range(height):
            X = (x - width / 2) + r0 * np.cos(theta) - offset_x * np.cos(theta)
            Y = (height / 2 - y) + r0 * np.sin(theta) - offset_y * np.sin(theta)
            r = np.sqrt(X ** 2 + Y ** 2)
            angle = zero_2pi_tan(X, Y)
            vm[y, x] = np.exp(thick * rho * r0 * np.cos(angle - theta)) / iv(0, r - r0)

    return vm


def VMkernels(thetas, size, rho, r0, thick, offset, fltr_resize_perc):
    filters = []
    for theta in thetas:
        f = vm_filter(theta, size, rho=rho, r0=r0, thick=thick, offset=offset)
        f = rescale(f, fltr_resize_perc, anti_aliasing=False)
        filters.append(f)
    filters = torch.tensor(np.stack(filters).astype(np.float32))  # [n_filters, Kh, Kw]
    return filters


# -----------------------------
# Attention network definition
# -----------------------------
def net_def(filters, tau_mem, in_ch, out_ch, size_krn, device, stride):
    """
    Conv(in_ch=num_pyr) + LIF.
    IMPORTANT: in_ch must match pyramid channels built in run_attention (== num_pyr).
    """
    net = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, (size_krn, size_krn), stride=stride, bias=False),
        sl.LIF(tau_mem),
    )
    # filters expected shape [out_ch, Kh, Kw]; we load into conv as [out_ch, 1, Kh, Kw] only if in_ch==1
    # Here in_ch == num_pyr, so we replicate the same spatial filter across channels (simple and stable).
    # Shape needed: [out_ch, in_ch, Kh, Kw]
    w = filters.unsqueeze(1).repeat(1, in_ch, 1, 1)  # replicate across pyramid channels
    net[0].weight.data = w.to(device)

    # keep your original state-device handling
    net[1].v_mem = net[1].tau_mem * net[1].v_mem.to(device)
    return net


def initialise_attention(device, ATTENTION_PARAMS):
    vm_kernels = VMkernels(
        ATTENTION_PARAMS["thetas"],
        ATTENTION_PARAMS["size_krn"],
        ATTENTION_PARAMS["rho"],
        ATTENTION_PARAMS["r0"],
        ATTENTION_PARAMS["thick"],
        ATTENTION_PARAMS["offset"],
        ATTENTION_PARAMS["fltr_resize_perc"],
    )

    net_attention = net_def(
        vm_kernels,
        ATTENTION_PARAMS["tau_mem"],
        in_ch=int(ATTENTION_PARAMS["num_pyr"]),   # <-- MUST match pyramid channels
        out_ch=int(ATTENTION_PARAMS["out_ch"]),
        size_krn=int(ATTENTION_PARAMS["size_krn"]),
        device=device,
        stride=int(ATTENTION_PARAMS["stride"]),
    )
    return net_attention


# -----------------------------
# Attention computation
# -----------------------------
def _standardize_window_to_4d(window, device):
    """
    Accepts:
      - [1,H,W]   (your current att_window)
      - [H,W]
      - [T,H,W]   (future time bins)
      - [T,1,H,W]
      - [B,1,H,W]
    Returns:
      X: [N,1,H,W] where N is batch-like dimension (time or batch)
      mode: "single" or "time"
    """
    if isinstance(window, np.ndarray):
        window = torch.from_numpy(window)

    window = window.to(device=device, dtype=torch.float32)

    if window.ndim == 2:
        # [H,W] -> [1,1,H,W]
        X = window.unsqueeze(0).unsqueeze(0)
        return X, "single"

    if window.ndim == 3:
        # could be [1,H,W] or [T,H,W]
        # treat first dim as N
        X = window.unsqueeze(1)  # [N,1,H,W]
        mode = "time" if window.shape[0] > 1 else "single"
        return X, mode

    if window.ndim == 4:
        # assume already [N,1,H,W] or [B,1,H,W]
        if window.shape[1] != 1:
            raise ValueError(f"Expected channel=1 in 4D window, got {tuple(window.shape)}")
        mode = "time" if window.shape[0] > 1 else "single"
        return window, mode

    raise ValueError(f"Unexpected window shape: {tuple(window.shape)}")


def run_attention(window, net, device, resolution, num_pyr, beta: float = 0.0, object_mask=None):
    """
    Builds a pyramid with num_pyr channels:
      channel k corresponds to downscale by factor (k+1), then upsample back to (H,W).
    Then runs Conv+LIF on [N, num_pyr, H, W].

    Compatible with the current att_window: [1,H,W] with 0/255 events.

    Behavior
    --------
    - Probabilities are computed normally over the full image.
    - If object_mask is provided, sampling is restricted to pixels inside the mask,
      but using the original full-image probabilities as weights.
    - For visualization, probs_vis is always the full-image probability map.
      The mask is only used later as an overlay contour in the debug panel.

    Returns
    -------
    sal_vis : np.ndarray, uint8, shape [H, W]
        Saliency map normalized for visualization.
    salmax_coords : tuple
        (row, col) coordinates of selected peak.
    probs_vis : np.ndarray, float32, shape [H, W]
        Full-image probability map for plotting.
    sal_raw : np.ndarray, float32, shape [H, W]
        Raw saliency/grouping map before visualization normalization.
    probs_raw : np.ndarray, float32, shape [H, W]
        Raw full-image fixation probability map.
    """

    H, W = int(resolution[0]), int(resolution[1])
    num_pyr = int(num_pyr)

    # 1) standardize
    X1, mode = _standardize_window_to_4d(window, device=device)  # [N,1,H,W]

    # normalize if 0..255
    if X1.max() > 1.5:
        X1 = X1 / 255.0
    X1 = X1.clamp(min=0.0)

    # ensure spatial size matches expected resolution
    if X1.shape[-2:] != (H, W):
        X1 = F.interpolate(X1, size=(H, W), mode="nearest")

    # 2) build pyramid channels
    pyr = []
    for k in range(num_pyr):
        factor = k + 1
        h = max(1, H // factor)
        w = max(1, W // factor)

        small = F.interpolate(X1, size=(h, w), mode="bilinear", align_corners=False)
        back = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
        pyr.append(back)

    # [N, num_pyr, H, W]
    X = torch.cat(pyr, dim=1)

    # 3) forward Conv+LIF
    out = net(X)
    out = out.type(torch.float32)

    if out.shape[-2:] != (H, W):
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

    # 4) aggregate over N and channels
    out_sum = out.sum(dim=0, keepdim=True)
    out_sum = out_sum.sum(dim=1, keepdim=True)
    salmap_torch = out_sum.squeeze(0).squeeze(0)  # [H,W]

    # 5) fixation distribution
    logits_full = float(beta) * (salmap_torch - salmap_torch.max())
    logits_full = logits_full.flatten().clone()

    # Full-image probability distribution
    probs_full = torch.softmax(logits_full, dim=0)

    mask_2d = None
    mask_flat = None

    if object_mask is not None:
        mask_2d = torch.as_tensor(object_mask, device=device) > 0

        if mask_2d.ndim != 2:
            raise ValueError(f"object_mask must have shape [H,W], got {tuple(mask_2d.shape)}")

        if mask_2d.shape != (H, W):
            mask_2d = mask_2d.float().unsqueeze(0).unsqueeze(0)
            mask_2d = F.interpolate(mask_2d, size=(H, W), mode="nearest")
            mask_2d = mask_2d.squeeze(0).squeeze(0) > 0.5

        mask_flat = mask_2d.flatten()

        if mask_flat.sum() == 0:
            raise ValueError("object_mask contains no valid pixels")

        # Restrict sampling to mask, but keep original full-image probabilities
        valid_idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
        valid_weights = probs_full[valid_idx]

        if torch.sum(valid_weights) <= 0:
            raise ValueError("Masked region has zero probability mass")

        sampled_local = torch.multinomial(valid_weights, 1).item()
        idx = valid_idx[sampled_local].item()
    else:
        idx = torch.multinomial(probs_full, 1).item()

    salmax_coords = np.unravel_index(idx, (H, W))

    # 6) raw maps
    sal_raw = salmap_torch.detach().cpu().numpy().astype(np.float32)
    probs_raw = probs_full.view(H, W).detach().cpu().numpy().astype(np.float32)

    # 7) visualization maps
    den = sal_raw.max() - sal_raw.min()
    if den < 1e-8:
        sal_vis = np.zeros((H, W), dtype=np.uint8)
    else:
        sal_vis = ((sal_raw - sal_raw.min()) / (den + 1e-8) * 255.0).astype(np.uint8)

    # Always plot the full probability map
    probs_vis = probs_raw.copy()

    return sal_vis, salmax_coords, probs_vis, sal_raw, probs_raw


###############################################################################
###############################################################################
###############################################################################

# Debug helpers
def draw_peak_arrow(im_bgr, peak_x, peak_y, color=(255, 255, 255),
                    circle_r=6, circle_thick=2, arrow_thick=2, tip_length=0.2):
    """
    Dibuja círculo y flecha desde el centro hacia el punto atencional.
    """
    H, W = im_bgr.shape[:2]
    cx, cy = W // 2, H // 2
    px, py = int(peak_x), int(peak_y)

    cv2.circle(im_bgr, (px, py), circle_r, color, circle_thick)
    cv2.arrowedLine(im_bgr, (cx, cy), (px, py), color, arrow_thick, tipLength=tip_length)
    return im_bgr


def make_attention_debug_panel(raw_frame_u8, saliency_map_f32, probs_map, object_mask, peak_x, peak_y):
    """
    Construye un panel horizontal con:
      1) events window
      2) saliency map
      3) probability map
      4) object mask

    Mantiene los colores originales y las flechas.
    Añade la silueta de la máscara sobre events, saliency y probs.
    Devuelve imagen BGR lista para guardar o mostrar.
    """
    if not isinstance(raw_frame_u8, np.ndarray):
        raw_frame_u8 = np.asarray(raw_frame_u8)
    if raw_frame_u8.dtype != np.uint8:
        raw_frame_u8 = np.clip(raw_frame_u8, 0, 255).astype(np.uint8)

    H, W = raw_frame_u8.shape[:2]

    # ----------------------------
    # Prepare mask
    # ----------------------------
    if object_mask is None:
        mask_bin = np.zeros((H, W), dtype=np.uint8)
    else:
        if not isinstance(object_mask, np.ndarray):
            object_mask = np.asarray(object_mask)

        if object_mask.shape[:2] != (H, W):
            mask_bin = cv2.resize(
                (object_mask > 0).astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_bin = (object_mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ----------------------------
    # Events window
    # ----------------------------
    events_vis = cv2.applyColorMap(raw_frame_u8, cv2.COLORMAP_JET)
    if len(contours) > 0:
        cv2.drawContours(events_vis, contours, -1, (255, 255, 255), 1)
    events_vis = draw_peak_arrow(events_vis, peak_x, peak_y, color=(255, 255, 255))
    cv2.putText(events_vis, "Events", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # ----------------------------
    # Saliency map
    # ----------------------------
    sal_u8 = cv2.normalize(saliency_map_f32, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_PARULA", cv2.COLORMAP_VIRIDIS)
    sal_vis = cv2.applyColorMap(sal_u8, cmap)
    if len(contours) > 0:
        cv2.drawContours(sal_vis, contours, -1, (255, 255, 255), 1)
    sal_vis = draw_peak_arrow(sal_vis, peak_x, peak_y, color=(255, 255, 255))
    cv2.putText(sal_vis, "Saliency", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # ----------------------------
    # Probability map
    # ----------------------------
    if not isinstance(probs_map, np.ndarray):
        probs_map = np.asarray(probs_map)
    if probs_map.dtype != np.uint8:
        probs_u8 = cv2.normalize(probs_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        probs_u8 = probs_map

    prob_vis = cv2.applyColorMap(probs_u8, cv2.COLORMAP_HOT)
    if len(contours) > 0:
        cv2.drawContours(prob_vis, contours, -1, (255, 255, 255), 1)
    prob_vis = draw_peak_arrow(prob_vis, peak_x, peak_y, color=(255, 255, 255))
    cv2.putText(prob_vis, "Probs", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # ----------------------------
    # Mask panel
    # ----------------------------
    mask_u8 = (mask_bin * 255).astype(np.uint8)
    mask_vis = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    if len(contours) > 0:
        cv2.drawContours(mask_vis, contours, -1, (255, 255, 255), 1)
    mask_vis = draw_peak_arrow(mask_vis, peak_x, peak_y, color=(255, 255, 255))
    cv2.putText(mask_vis, "Mask", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    panel = np.hstack([events_vis, sal_vis, prob_vis, mask_vis])
    return panel


def save_attention_debug(panel_bgr, out_path, show=False, win_name="Attention debug"):
    cv2.imwrite(str(out_path), panel_bgr)
    if show:
        cv2.imshow(win_name, panel_bgr)
        cv2.waitKey(1)