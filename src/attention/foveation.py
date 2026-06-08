from __future__ import annotations

import numpy as np


def foveate_black_gray(
    frame_gray: np.ndarray,
    center_x: float,
    center_y: float,
    sigma: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Black Gaussian foveation.

    Keeps the image around the Gaussian center and suppresses the periphery.

    Parameters
    ----------
    frame_gray:
        Grayscale frame, shape (H, W).

    center_x, center_y:
        Gaussian center in pixel coordinates.

    sigma:
        Gaussian standard deviation in pixels.

    Returns
    -------
    out:
        Foveated grayscale frame, shape (H, W), uint8.
    """
    frame_gray = np.asarray(frame_gray)

    if frame_gray.ndim != 2:
        raise ValueError(
            f"foveate_black_gray expects a grayscale frame with shape (H, W). "
            f"Got {frame_gray.shape}."
        )

    height, width = frame_gray.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)

    sigma = max(float(sigma), 1.0)

    gaussian = np.exp(
        -0.5
        * (
            ((xx - float(center_x)) / sigma) ** 2
            + ((yy - float(center_y)) / sigma) ** 2
        )
    ).astype(np.float32)

    gaussian = (gaussian - gaussian.min()) / (
        gaussian.max() - gaussian.min() + eps
    )

    out = frame_gray.astype(np.float32) * gaussian
    return np.clip(out, 0, 255).astype(np.uint8)