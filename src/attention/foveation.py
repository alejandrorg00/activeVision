from __future__ import annotations

from pathlib import Path

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


def foveations_to_gif(
    fov_dir: str | Path,
    output_gif: str | Path,
    fps: int = 20,
    loop: int = 0,
    image_prefix: str = "roi",
) -> str:
    """
    Create a GIF from saved foveation images.

    This is only for notebook display or inspection.

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