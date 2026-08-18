from __future__ import annotations

from pathlib import Path
import re

import numpy as np


def _sanitize_token(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-\.]", "", text)
    return text


def _format_beta(beta: float | None) -> str:
    if beta is None:
        return "unknown"
    beta_str = f"{float(beta):.3f}".rstrip("0").rstrip(".")
    return beta_str.replace(".", "p")


def _default_figures_dir() -> Path:
    """
    Assumes analisys_helpers.py is inside the repo (e.g. root or src/).
    If it is in src/, this returns repo_root/figures.
    If it is in repo root, this also returns repo_root/figures.
    """
    this_file = Path(__file__).resolve()
    if this_file.parent.name == "src":
        repo_root = this_file.parent.parent
    else:
        repo_root = this_file.parent
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir

def plot_saccade_three_panel(
    rgb_path,
    attention_gif_path,
    attention_frame_idx=1,
    object_label="object",
    beta=10,
    figures_dir="figures",
    figsize=(8.4, 2.8),
    dpi=600,
    save_png=True,
    save_pdf=True,
    save_svg=True,
    show=True,
):
    """
    Plot 1x3 figure for the first real saccade using:
        1) last RGB frame of the relevant time window
        2) saliency panel
        3) accumulated window + action panel

    The saliency and accumulated/action panels are extracted from
    saliency_window_action.gif, whose frames are horizontally concatenated:
        left half  -> saliency
        right half -> accumulated window + action
    """
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    from PIL import Image

    rgb_path = Path(rgb_path)
    attention_gif_path = Path(attention_gif_path)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not attention_gif_path.exists():
        raise FileNotFoundError(f"Attention GIF not found: {attention_gif_path}")

    # ------------------------------------------------------------
    # Load RGB image
    # ------------------------------------------------------------
    rgb_img = np.array(Image.open(rgb_path))

    # ------------------------------------------------------------
    # Load selected frame from attention GIF
    # ------------------------------------------------------------
    gif_frames = imageio.mimread(attention_gif_path)
    if len(gif_frames) == 0:
        raise RuntimeError(f"No frames found in GIF: {attention_gif_path}")

    if attention_frame_idx < 0 or attention_frame_idx >= len(gif_frames):
        raise IndexError(
            f"attention_frame_idx={attention_frame_idx} is out of range "
            f"for GIF with {len(gif_frames)} frames."
        )

    composite = np.array(gif_frames[attention_frame_idx])

    # If GIF frame has alpha, drop it
    if composite.ndim == 3 and composite.shape[2] == 4:
        composite = composite[:, :, :3]

    H, W = composite.shape[:2]
    mid = W // 2

    saliency_img = composite[:, :mid]
    accumulated_img = composite[:, mid:]

    # ------------------------------------------------------------
    # Save name
    # ------------------------------------------------------------
    beta_str = str(beta).replace(".", "p")
    save_name = f"first_saccade_3panel_{object_label}_beta{beta_str}"

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.86, wspace=0.03)

    panels = [
        (rgb_img, "RGB frame"),
        (saliency_img, "Saliency"),
        (accumulated_img, "DVS frame"),
    ]

    for ax, (img, title) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=8, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    png_path = None
    pdf_path = None
    svg_path = None

    if save_png:
        png_path = figures_dir / f"{save_name}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)

    if save_pdf:
        pdf_path = figures_dir / f"{save_name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)

    if save_svg:
        svg_path = figures_dir / f"{save_name}.svg"
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "png_path": str(png_path) if png_path else None,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "svg_path": str(svg_path) if svg_path else None,
    }

def _compute_attention_exploration_data(
    events_npy: str | Path,
    saccades_path: str | Path,
    resolution: tuple[int, int] | None = None,
    per: float = 0.1,
    noise_thresh: int = 100,
    exclude_initial_fixation: bool = True,
) -> dict:
    """
    Shared computation for attention exploration analysis.

    Computes:
        - event-defined object grid
        - fixation counts per grid cell
        - explored object area
        - fixation entropy
        - normalized fixation entropy

    Also returns the intermediate arrays required by
    plot_attention_exploration().
    """
    events_npy = Path(events_npy)
    saccades_path = Path(saccades_path)

    if not events_npy.exists():
        raise FileNotFoundError(f"Events npy not found: {events_npy}")

    if not saccades_path.exists():
        raise FileNotFoundError(f"saccades.dat not found: {saccades_path}")

    # ============================================================
    # Load events
    # ============================================================
    data = np.load(events_npy)

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(
            f"Expected events with shape (N, 4), got {data.shape}"
        )

    if data.shape[0] == 0:
        raise RuntimeError("events.npy contains no events.")

    x = data[:, 0].astype(np.int64)
    y = data[:, 1].astype(np.int64)

    if resolution is None:
        W = int(x.max()) + 1
        H = int(y.max()) + 1
    else:
        H, W = int(resolution[0]), int(resolution[1])

    valid_events = (
        (x >= 0)
        & (x < W)
        & (y >= 0)
        & (y < H)
    )

    x = x[valid_events]
    y = y[valid_events]

    if x.size == 0:
        raise RuntimeError(
            "No valid events after resolution filtering."
        )

    # ============================================================
    # Global event-density image
    # ============================================================
    counts = np.zeros(
        (H, W),
        dtype=np.int32,
    )

    np.add.at(
        counts,
        (y, x),
        1,
    )

    event_binary = counts > 0

    # ============================================================
    # Load fixation sequence
    # ============================================================
    saccades = np.loadtxt(
        saccades_path,
        skiprows=1,
        dtype=np.float64,
    )

    if saccades.ndim == 1:
        saccades = saccades[None, :]

    # saccades.dat:
    # idx time_s pre_x pre_y post_x post_y dx dy
    attention_xy = saccades[:, [4, 5]].astype(
        np.float32
    )

    if attention_xy.ndim != 2 or attention_xy.shape[1] != 2:
        raise ValueError(
            f"Expected attention_xy with shape (N, 2), "
            f"got {attention_xy.shape}"
        )

    if (
        exclude_initial_fixation
        and attention_xy.shape[0] > 1
    ):
        attention_xy_eff = attention_xy[1:]
    else:
        attention_xy_eff = attention_xy

    if attention_xy_eff.shape[0] == 0:
        raise RuntimeError(
            "No effective fixations available after filtering."
        )

    # ============================================================
    # Event-defined object grid
    # ============================================================
    crop_x = max(
        1,
        int(W * float(per)),
    )

    crop_y = max(
        1,
        int(H * float(per)),
    )

    n_cols = int(np.ceil(W / crop_x))
    n_rows = int(np.ceil(H / crop_y))

    object_mask = np.zeros(
        (n_rows, n_cols),
        dtype=bool,
    )

    for i in range(n_rows):
        for j in range(n_cols):

            y0 = i * crop_y
            x0 = j * crop_x

            y1 = min(
                y0 + crop_y,
                H,
            )

            x1 = min(
                x0 + crop_x,
                W,
            )

            cell_count = counts[
                y0:y1,
                x0:x1,
            ].sum()

            if cell_count >= int(noise_thresh):
                object_mask[i, j] = True

    # ============================================================
    # Fixation counts per grid cell
    # ============================================================
    fix_counts = np.zeros(
        (n_rows, n_cols),
        dtype=np.int32,
    )

    xs = np.clip(
        attention_xy_eff[:, 0].astype(np.int64),
        0,
        W - 1,
    )

    ys = np.clip(
        attention_xy_eff[:, 1].astype(np.int64),
        0,
        H - 1,
    )

    js = np.minimum(
        xs // crop_x,
        n_cols - 1,
    )

    is_ = np.minimum(
        ys // crop_y,
        n_rows - 1,
    )

    for i_cell, j_cell in zip(is_, js):
        fix_counts[i_cell, j_cell] += 1

    # Heatmap only displays object cells
    heatmap = fix_counts.astype(np.float32)
    heatmap[~object_mask] = np.nan

    # ============================================================
    # Explored object area
    # ============================================================
    total_obj_cells = int(
        object_mask.sum()
    )

    visited_obj_cells = int(
        np.sum(
            (fix_counts > 0)
            & object_mask
        )
    )

    area_explored_coeff = (
        visited_obj_cells / total_obj_cells
        if total_obj_cells > 0
        else np.nan
    )

    # ============================================================
    # Fixation entropy
    # ============================================================
    object_fix_counts = fix_counts[
        object_mask
    ].astype(np.float64)

    num_fixations_on_object = int(
        object_fix_counts.sum()
    )

    if (
        total_obj_cells <= 1
        or num_fixations_on_object == 0
    ):
        fixation_entropy = 0.0
        fixation_entropy_norm = 0.0

    else:
        p = (
            object_fix_counts
            / object_fix_counts.sum()
        )

        # zero-probability cells do not contribute to H
        p_nonzero = p[p > 0]

        fixation_entropy = float(
            -np.sum(
                p_nonzero
                * np.log(p_nonzero)
            )
        )

        fixation_entropy_norm = float(
            fixation_entropy
            / np.log(total_obj_cells)
        )

    return {
        # Metrics
        "area_explored_coeff": float(
            area_explored_coeff
        ),
        "fixation_entropy": fixation_entropy,
        "fixation_entropy_norm": fixation_entropy_norm,

        # Counts
        "total_obj_cells": total_obj_cells,
        "visited_obj_cells": visited_obj_cells,
        "num_fixations": int(
            attention_xy_eff.shape[0]
        ),
        "num_fixations_on_object": (
            num_fixations_on_object
        ),

        # Data required for plotting
        "attention_xy": attention_xy,
        "attention_xy_eff": attention_xy_eff,
        "object_mask": object_mask,
        "fix_counts": fix_counts,
        "heatmap": heatmap,
        "event_binary": event_binary,

        # Geometry
        "crop_x": int(crop_x),
        "crop_y": int(crop_y),
        "H": int(H),
        "W": int(W),
    }


def compute_attention_exploration_metrics(
    events_npy: str | Path,
    saccades_path: str | Path,
    resolution: tuple[int, int] | None = None,
    per: float = 0.1,
    noise_thresh: int = 100,
    exclude_initial_fixation: bool = True,
) -> dict:
    """
    Compute scalar exploration metrics without generating a figure.
    """

    data = _compute_attention_exploration_data(
        events_npy=events_npy,
        saccades_path=saccades_path,
        resolution=resolution,
        per=per,
        noise_thresh=noise_thresh,
        exclude_initial_fixation=exclude_initial_fixation,
    )

    return {
        "area_explored_coeff": data[
            "area_explored_coeff"
        ],
        "fixation_entropy": data[
            "fixation_entropy"
        ],
        "fixation_entropy_norm": data[
            "fixation_entropy_norm"
        ],
        "total_obj_cells": data[
            "total_obj_cells"
        ],
        "visited_obj_cells": data[
            "visited_obj_cells"
        ],
        "num_fixations": data[
            "num_fixations"
        ],
        "num_fixations_on_object": data[
            "num_fixations_on_object"
        ],
    }

def plot_attention_exploration(
    events_npy: str | Path,
    saccades_path: str | Path,
    resolution: tuple[int, int] | None = None,
    beta: float | None = None,
    object_label: str | None = None,

    # Grid/object mask parameters
    per: float = 0.1,
    noise_thresh: int = 100,

    # Fixation handling
    exclude_initial_fixation: bool = True,
    plot_trajectory: bool = True,

    # Figure style/output
    heatmap_cmap: str = "magma",
    figsize: tuple[float, float] = (6.6, 3.0),
    dpi: int = 600,
    figures_dir: str | Path | None = None,
    save_png: bool = True,
    save_pdf: bool = True,
    save_svg: bool = True,
    show: bool = True,
) -> dict:
    """
    Plot explored object area from sampled attention fixations.

    The figure contains:
        left: global event map, object grid, fixations and saccades
        right: fixation heatmap over object cells

    The underlying computation is shared with
    compute_attention_exploration_metrics().
    """
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    events_npy = Path(events_npy)

    if figures_dir is None:
        figures_dir = _default_figures_dir()
    else:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if object_label is None:
        object_label = events_npy.stem

    object_label_clean = _sanitize_token(
        object_label
    )

    beta_token = _format_beta(beta)

    save_name = (
        f"attention_exploration_"
        f"{object_label_clean}_"
        f"beta{beta_token}"
    )

    # ============================================================
    # Shared computation
    # ============================================================
    data = _compute_attention_exploration_data(
        events_npy=events_npy,
        saccades_path=saccades_path,
        resolution=resolution,
        per=per,
        noise_thresh=noise_thresh,
        exclude_initial_fixation=exclude_initial_fixation,
    )

    attention_xy = data["attention_xy"]
    attention_xy_eff = data["attention_xy_eff"]

    object_mask = data["object_mask"]
    heatmap = data["heatmap"]
    event_binary = data["event_binary"]

    crop_x = data["crop_x"]
    crop_y = data["crop_y"]

    H = data["H"]
    W = data["W"]

    total_obj_cells = data[
        "total_obj_cells"
    ]

    visited_obj_cells = data[
        "visited_obj_cells"
    ]

    area_explored_coeff = data[
        "area_explored_coeff"
    ]

    print(
        f"[Exploration | beta={beta}] "
        f"Object cells: {total_obj_cells} | "
        f"Visited: {visited_obj_cells} | "
        f"Area explored coeff: "
        f"{area_explored_coeff:.3f}"
    )

    # ============================================================
    # Figure
    # ============================================================
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=figsize,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.94,
        bottom=0.18,
        top=0.84,
        wspace=0.34,
    )

    label_fs = 8
    tick_fs = 7
    legend_fs = 7
    title_fs = 8

    def _style_image_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_position(
            ("outward", 5)
        )
        ax.spines["bottom"].set_position(
            ("outward", 5)
        )

        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        ax.tick_params(
            axis="both",
            direction="out",
            length=3,
            width=0.8,
            labelsize=tick_fs,
            pad=2,
        )

        ax.set_xlabel(
            "x (px)",
            fontsize=label_fs,
            labelpad=3,
        )

        ax.set_ylabel(
            "y (px)",
            fontsize=label_fs,
            labelpad=3,
        )

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")

    # ============================================================
    # Left panel
    # ============================================================
    ax1.imshow(
        event_binary,
        cmap="gray",
        origin="upper",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    n_rows, n_cols = object_mask.shape

    for i in range(n_rows):
        for j in range(n_cols):

            y0 = i * crop_y
            x0 = j * crop_x

            y1 = min(
                y0 + crop_y,
                H,
            )

            x1 = min(
                x0 + crop_x,
                W,
            )

            if object_mask[i, j]:
                rect = plt.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="mediumseagreen",
                    alpha=0.20,
                    edgecolor="0.25",
                    linewidth=0.35,
                )
            else:
                rect = plt.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor="none",
                    edgecolor="0.35",
                    linewidth=0.25,
                    alpha=0.75,
                )

            ax1.add_patch(rect)

    if (
        plot_trajectory
        and attention_xy.shape[0] >= 2
    ):
        ax1.plot(
            attention_xy[:, 0],
            attention_xy[:, 1],
            "-",
            color="tab:blue",
            linewidth=1.15,
            alpha=0.65,
            zorder=5,
        )

    ax1.scatter(
        attention_xy_eff[:, 0],
        attention_xy_eff[:, 1],
        s=15,
        color="tab:blue",
        edgecolors="white",
        linewidth=0.45,
        zorder=6,
    )

    if attention_xy.shape[0] >= 1:
        ax1.scatter(
            [attention_xy[0, 0]],
            [attention_xy[0, 1]],
            s=17,
            facecolors="white",
            edgecolors="tab:blue",
            linewidth=1.0,
            zorder=7,
        )

    fix_patch = mlines.Line2D(
        [],
        [],
        color="tab:blue",
        marker="o",
        linestyle="None",
        markeredgecolor="white",
        markeredgewidth=0.45,
        markersize=4.2,
        label="Fixations",
    )

    sac_line = mlines.Line2D(
        [],
        [],
        color="tab:blue",
        linestyle="-",
        linewidth=1.15,
        alpha=0.65,
        label="Saccades",
    )

    obj_patch = mpatches.Patch(
        facecolor="mediumseagreen",
        alpha=0.20,
        edgecolor="0.25",
        linewidth=0.35,
        label="Object cells",
    )

    ax1.legend(
        handles=[
            fix_patch,
            sac_line,
            obj_patch,
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        fontsize=legend_fs,
        handlelength=1.3,
        columnspacing=0.8,
        handletextpad=0.4,
        borderaxespad=0.0,
    )

    _style_image_axis(ax1)

    # ============================================================
    # Right panel
    # ============================================================
    cmap = plt.get_cmap(
        heatmap_cmap
    ).copy()

    cmap.set_bad(alpha=0.0)

    extent = [
        0,
        W,
        H,
        0,
    ]

    if np.any(object_mask):
        heatmap_max = max(
            1.0,
            float(np.nanmax(heatmap)),
        )
    else:
        heatmap_max = 1.0

    im2 = ax2.imshow(
        heatmap,
        cmap=cmap,
        vmin=0.0,
        vmax=heatmap_max,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )

    for i in range(n_rows):
        for j in range(n_cols):

            if not object_mask[i, j]:
                continue

            x0 = j * crop_x
            y0 = i * crop_y

            x1 = min(
                x0 + crop_x,
                W,
            )

            y1 = min(
                y0 + crop_y,
                H,
            )

            ax2.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.55,
                )
            )

    ax2.set_title(
        f"Explored object area: "
        f"{area_explored_coeff:.3f}",
        fontsize=title_fs,
        pad=6,
    )

    _style_image_axis(ax2)

    cb = fig.colorbar(
        im2,
        ax=ax2,
        fraction=0.045,
        pad=0.06,
    )

    cb.locator = MaxNLocator(
        nbins=5,
        integer=True,
    )

    cb.update_ticks()

    cb.set_label(
        "Fixations per cell",
        fontsize=label_fs,
        labelpad=4,
    )

    cb.ax.tick_params(
        direction="out",
        length=2.5,
        width=0.7,
        labelsize=tick_fs,
    )

    cb.outline.set_linewidth(0.7)

    # ============================================================
    # Save
    # ============================================================
    png_path = None
    pdf_path = None
    svg_path = None

    if save_png:
        png_path = (
            figures_dir
            / f"{save_name}.png"
        )

        fig.savefig(
            png_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03,
        )

    if save_pdf:
        pdf_path = (
            figures_dir
            / f"{save_name}.pdf"
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.03,
        )

    if save_svg:
        svg_path = (
            figures_dir
            / f"{save_name}.svg"
        )

        fig.savefig(
            svg_path,
            bbox_inches="tight",
            pad_inches=0.03,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "attention_exploration_png_path": (
            str(png_path)
            if png_path
            else None
        ),
        "attention_exploration_pdf_path": (
            str(pdf_path)
            if pdf_path
            else None
        ),
        "attention_exploration_svg_path": (
            str(svg_path)
            if svg_path
            else None
        ),

        "save_name": save_name,
        "figures_dir": str(figures_dir),

        "area_explored_coeff": data[
            "area_explored_coeff"
        ],
        "fixation_entropy": data[
            "fixation_entropy"
        ],
        "fixation_entropy_norm": data[
            "fixation_entropy_norm"
        ],

        "total_obj_cells": data[
            "total_obj_cells"
        ],
        "visited_obj_cells": data[
            "visited_obj_cells"
        ],
        "num_fixations": data[
            "num_fixations"
        ],
        "num_fixations_on_object": data[
            "num_fixations_on_object"
        ],

        "crop_x": data["crop_x"],
        "crop_y": data["crop_y"],
        "noise_thresh": int(noise_thresh),
    }

def plot_first_n_foveations(
    fov_dir,
    n=5,
    start_idx=0,
    object_label="object",
    beta=10,
    figures_dir="figures",
    figsize_per_panel=2.2,
    dpi=600,
    save_png=True,
    save_pdf=True,
    save_svg=True,
    show=True,
):
    """
    Plot the first n foveation images in a 1 x n layout.

    Parameters
    ----------
    fov_dir : str or Path
        Directory containing foveation PNGs saved by run_attention.
    n : int
        Number of foveations to display.
    start_idx : int
        Starting foveation index. Use:
            - 0 to include the very first fixation/foveation
            - 1 to start from the first real saccade
    object_label : str
        Object name for output filename.
    beta : float or int
        Beta value for output filename.
    figures_dir : str or Path
        Directory where the figure will be saved.
    figsize_per_panel : float
        Width/height scale per panel.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    fov_dir = Path(fov_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not fov_dir.exists():
        raise FileNotFoundError(f"Foveation directory not found: {fov_dir}")

    fov_paths = sorted(fov_dir.glob("*.png"))

    if len(fov_paths) == 0:
        raise RuntimeError(f"No PNG foveation images found in: {fov_dir}")

    selected_paths = fov_paths[start_idx:start_idx + n]

    if len(selected_paths) == 0:
        raise RuntimeError(
            f"No foveation images available for start_idx={start_idx}, n={n}."
        )

    actual_n = len(selected_paths)

    imgs = [np.array(Image.open(p)) for p in selected_paths]

    beta_str = str(beta).replace(".", "p")
    save_name = f"first_{actual_n}_foveations_{object_label}_beta{beta_str}"

    fig, axes = plt.subplots(
        1,
        actual_n,
        figsize=(figsize_per_panel * actual_n, figsize_per_panel),
    )

    if actual_n == 1:
        axes = [axes]

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995, wspace=0.02)

    for ax, img in zip(axes, imgs):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    png_path = None
    pdf_path = None
    svg_path = None

    if save_png:
        png_path = figures_dir / f"{save_name}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)

    if save_pdf:
        pdf_path = figures_dir / f"{save_name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)

    if save_svg:
        svg_path = figures_dir / f"{save_name}.svg"
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.01)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "png_path": str(png_path) if png_path else None,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "svg_path": str(svg_path) if svg_path else None,
        "foveation_paths": [str(p) for p in selected_paths],
    }


import base64
import mimetypes
from pathlib import Path

import numpy as np
from IPython.display import HTML, display


def display_gif_grid(
    paths,
    titles=None,
    ncols=3,
    cell_width=240,
    title_prefix="",
    gap_px=4,
    title_fontsize_px=13,
):
    """
    Display GIFs in a compact HTML grid inside a notebook.
    Titles are overlaid at the top-right corner of each GIF.
    """
    if titles is None:
        titles = [f"item_{i}" for i in range(len(paths))]

    if len(paths) != len(titles):
        raise ValueError("paths and titles must have the same length.")

    def file_to_data_uri(path):
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"GIF not found: {path}")

        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type is None:
            mime_type = "image/gif"

        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    html_parts = [
        f"""
        <div style="
            display: grid;
            grid-template-columns: repeat({ncols}, minmax(0, 1fr));
            gap: {gap_px}px;
            width: 100%;
            padding: 0;
            margin: 0;
            background: transparent;
        ">
        """
    ]

    for path, title in zip(paths, titles):
        data_uri = file_to_data_uri(path)

        html_parts.append(
            f"""
            <div style="
                padding: 0;
                margin: 0;
                background: transparent;
                text-align: center;
            ">
                <div style="
                    position: relative;
                    display: inline-block;
                    margin: 0;
                    padding: 0;
                ">
                    <img
                        src="{data_uri}"
                        style="
                            width: 100%;
                            max-width: {cell_width}px;
                            height: auto;
                            display: block;
                            margin: 0 auto;
                            padding: 0;
                        "
                    />

                    <div style="
                        position: absolute;
                        top: 6px;
                        right: 6px;
                        color: black;
                        font-weight: 600;
                        font-size: {title_fontsize_px}px;
                        line-height: 1.1;
                        background: rgba(255, 255, 255, 0.65);
                        padding: 2px 6px;
                        border-radius: 6px;
                    ">
                        {title_prefix}{title}
                    </div>
                </div>
            </div>
            """
        )

    html_parts.append("</div>")
    display(HTML("".join(html_parts)))

def _render_dvs_snapshot_from_npy(
    events_npy,
    resolution=(256, 256),
    t_end_us=20000,
    window_us=1000,
    timestamp_text=True,
    cmap_name="viridis",
):
    """
    Render a single DVS accumulated snapshot from events.npy, in a style
    similar to the event GIF.

    Parameters
    ----------
    events_npy : str or Path
        Path to events.npy with columns [x, y, polarity, timestamp_seconds].
    resolution : tuple[int, int]
        (H, W)
    t_end_us : int
        End time of the accumulation window, in microseconds.
    window_us : int
        Accumulation window size in microseconds.
    """
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.pyplot as plt

    events_npy = Path(events_npy)
    if not events_npy.exists():
        raise FileNotFoundError(f"events.npy not found: {events_npy}")

    H, W = resolution
    data = np.load(events_npy)

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Expected events with shape (N, 4), got {data.shape}")

    x = data[:, 0].astype(np.int64)
    y = data[:, 1].astype(np.int64)
    p = data[:, 2]
    ts_us = data[:, 3].astype(np.float64) * 1e6

    t_start_us = t_end_us - window_us
    ind = np.where((ts_us >= t_start_us) & (ts_us < t_end_us))[0]

    gray = np.full((H, W), 125, dtype=np.uint8)

    if ind.size > 0:
        xx = x[ind]
        yy = y[ind]
        pp = p[ind]

        valid = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        xx = xx[valid]
        yy = yy[valid]
        pp = pp[valid]

        # Accept either polarity in {0,1} or {-1,1}
        if np.min(pp) < 0:
            signs = np.sign(pp).astype(np.int8)
        else:
            signs = (2 * pp.astype(np.int8) - 1).astype(np.int8)

        # Last event at each pixel wins
        for xi, yi, si in zip(xx, yy, signs):
            gray[yi, xi] = 250 if si > 0 else 0

    cmap = plt.get_cmap(cmap_name)
    rgb = (cmap(gray.astype(np.float32) / 255.0)[..., :3] * 255).astype(np.uint8)

    if timestamp_text:
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        draw.text((6, 6), f"{int(t_end_us)} us", fill=(255, 255, 0))
        rgb = np.array(img)

    return rgb


def plot_rgb_dvs_snapshot_grid(
    object_runs,
    rgb_frame_idx=20,
    resolution=(256, 256),
    fps=1000,
    dvs_window_us=1000,
    figures_dir=None,
    title_fs=8,
    figsize_per_col=2.2,
    dpi=600,
    save_png=True,
    save_pdf=True,
    save_svg=True,
    show=True,
):
    """
    Plot a 2 x N grid:
        top row    -> RGB snapshot
        bottom row -> DVS accumulated snapshot reconstructed from events.npy

    This is useful for showing all objects at the same temporal snapshot.

    Parameters
    ----------
    object_runs : list[dict]
        Each dict should contain at least:
            - "name"
            - "frames_dir" or "sequence_dir"
            - "events_npy"
    rgb_frame_idx : int
        RGB frame index to display.
    resolution : tuple[int, int]
        (H, W)
    fps : int
        Frame rate used for the rendered RGB sequence.
    dvs_window_us : int
        Event accumulation window, matching the GIF style.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    if figures_dir is None:
        figures_dir = _default_figures_dir()
    else:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

    H, W = resolution
    dt_us = int(round(1e6 / float(fps)))
    t_end_us = int(rgb_frame_idx * dt_us)

    n = len(object_runs)
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(figsize_per_col * n, figsize_per_col * 2.0),
    )

    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, run in enumerate(object_runs):
        object_name = run["name"]

        if "frames_dir" in run:
            frames_dir = Path(run["frames_dir"])
        else:
            frames_dir = Path(run["sequence_dir"]) / "frames"

        rgb_path = frames_dir / f"frame_{rgb_frame_idx:05d}.png"
        events_npy = Path(run["events_npy"])

        if not rgb_path.exists():
            raise FileNotFoundError(f"RGB frame not found: {rgb_path}")
        if not events_npy.exists():
            raise FileNotFoundError(f"events.npy not found: {events_npy}")

        rgb_img = np.array(Image.open(rgb_path))
        dvs_img = _render_dvs_snapshot_from_npy(
            events_npy=events_npy,
            resolution=(H, W),
            t_end_us=t_end_us,
            window_us=dvs_window_us,
            timestamp_text=True,
            cmap_name="viridis",
        )

        ax_rgb = axes[0, col]
        ax_dvs = axes[1, col]

        ax_rgb.imshow(rgb_img)
        ax_dvs.imshow(dvs_img)

        ax_rgb.set_title(object_name, fontsize=title_fs, pad=3)

        for ax in (ax_rgb, ax_dvs):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.92,
        wspace=0.03,
        hspace=0.06,
    )

    save_name = f"rgb_dvs_snapshot_grid_frame{rgb_frame_idx:05d}"

    png_path = None
    pdf_path = None
    svg_path = None

    if save_png:
        png_path = figures_dir / f"{save_name}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)

    if save_pdf:
        pdf_path = figures_dir / f"{save_name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)

    if save_svg:
        svg_path = figures_dir / f"{save_name}.svg"
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "png_path": str(png_path) if png_path else None,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "svg_path": str(svg_path) if svg_path else None,
        "rgb_frame_idx": int(rgb_frame_idx),
        "t_end_us": int(t_end_us),
        "dvs_window_us": int(dvs_window_us),
    }

def plot_attentional_gain_concept(
    figures_dir=None,
    figsize=(10.8, 3.4),
    dpi=600,
    save_png=True,
    save_pdf=True,
    save_svg=True,
    show=True,
):
    """
    Conceptual illustration of increasing attentional gain beta.

    (a) Left:
        Selection probability as a function of relative saliency
        for increasing beta.

    (b) Right:
        Conceptual attention maps transitioning from exploratory
        sampling to winner-take-all selection.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    # ------------------------------------------------------------
    # Local helpers
    # ------------------------------------------------------------
    def gauss2d(X, Y, x0, y0, sx, sy, amp=1.0):
        return amp * np.exp(
            -(
                ((X - x0) ** 2) / (2 * sx ** 2)
                + ((Y - y0) ** 2) / (2 * sy ** 2)
            )
        )

    def normalize01(A):
        A = A - A.min()
        if A.max() > 0:
            A = A / A.max()
        return A

    # ------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------
    if figures_dir is None:
        figures_dir = _default_figures_dir()
    else:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Proto-object saliency layout
    # ------------------------------------------------------------
    x = np.linspace(-1, 1, 220)
    y = np.linspace(-1, 1, 220)
    X, Y = np.meshgrid(x, y)

    S = (
        gauss2d(X, Y, x0=0.28,  y0=-0.02, sx=0.13, sy=0.13, amp=1.00)
        + gauss2d(X, Y, x0=-0.32, y0=0.28, sx=0.16, sy=0.16, amp=0.92)
        + gauss2d(X, Y, x0=0.02,  y0=-0.42, sx=0.18, sy=0.18, amp=0.22)
    )
    S = normalize01(S)

    # ------------------------------------------------------------
    # Conceptual attention maps
    # ------------------------------------------------------------
    A_explore = (
        0.62
        + gauss2d(X, Y, 0.28, -0.02, 0.34, 0.34, amp=0.16)
        + gauss2d(X, Y, -0.32, 0.28, 0.38, 0.38, amp=0.14)
        + gauss2d(X, Y, 0.02, -0.42, 0.42, 0.42, amp=0.05)
    )
    A_explore = normalize01(A_explore)

    A_intermediate = (
        0.08
        + gauss2d(X, Y, 0.28, -0.02, 0.16, 0.16, amp=1.00)
        + gauss2d(X, Y, -0.32, 0.28, 0.18, 0.18, amp=0.88)
        + gauss2d(X, Y, 0.02, -0.42, 0.24, 0.24, amp=0.12)
    )
    A_intermediate = normalize01(A_intermediate)

    A_wta = (
        0.02
        + gauss2d(X, Y, 0.28, -0.02, 0.10, 0.10, amp=1.00)
        + gauss2d(X, Y, -0.32, 0.28, 0.14, 0.14, amp=0.10)
    )
    A_wta = normalize01(A_wta)

    attention_maps = [A_explore, A_intermediate, A_wta]

    # ------------------------------------------------------------
    # Gain curves
    # ------------------------------------------------------------
    betas = [0.5, 5, 25]
    labels = ["Exploratory", "Intermediate", "Winner-take-all"]
    beta_colors = plt.cm.Oranges([0.45, 0.68, 0.90])

    relative_saliency = np.linspace(0, 1, 400)
    reference_saliency = 0.5

    # ------------------------------------------------------------
    # Figure style
    # ------------------------------------------------------------
    label_fs = 10
    tick_fs = 9
    title_fs = 12
    legend_fs = 7
    panel_fs = 13
    arrow_fs = 10
    lowhigh_fs = 9

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        4,
        figsize=figsize,
        gridspec_kw={
            "width_ratios": [1, 1, 1, 1],
            "wspace": 0.22,
        },
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.86,
        bottom=0.24,
    )

    # ------------------------------------------------------------
    # Left panel: gain curves
    # ------------------------------------------------------------
    ax = axes[0]

    for beta, label, color in zip(betas, labels, beta_colors):
        selection_probability = 1 / (
            1 + np.exp(-beta * (relative_saliency - reference_saliency))
        )

        ax.plot(
            relative_saliency,
            selection_probability,
            linewidth=2.2,
            color=color,
            label=fr"{label}",
        )

    ax.set_xlabel("Relative saliency", fontsize=label_fs, labelpad=4)
    ax.set_ylabel("Selection probability", fontsize=label_fs, labelpad=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.01, 1.01)

    # keep physically square
    ax.set_box_aspect(1)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.8,
        pad=3,
        labelsize=tick_fs,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 4))
    ax.spines["left"].set_position(("outward", 4))
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)

    ax.legend(
        frameon=False,
        fontsize=legend_fs,
        loc="upper left",
        handlelength=2.0,
        borderaxespad=0.2,
    )

    # ------------------------------------------------------------
    # Right panels: conceptual maps
    # ------------------------------------------------------------
    for ax, A, label, color in zip(axes[1:], attention_maps, labels, beta_colors):
        ax.imshow(
            A,
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=1,
            interpolation="bilinear",
        )

        ax.contour(
            S,
            levels=[0.30, 0.60],
            colors="white",
            linewidths=0.7,
            alpha=0.15,
        )

        ax.set_title(
            label,
            color=color,
            pad=6,
            fontsize=title_fs,
        )

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

    # ------------------------------------------------------------
    # Panel labels
    # ------------------------------------------------------------
    p0 = axes[0].get_position()
    p1 = axes[1].get_position()

    fig.text(
        p0.x0 - 0.06,
        p0.y1 + 0.01,
        "a",
        fontsize=panel_fs,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    fig.text(
        p1.x0 - 0.03,
        p1.y1 + 0.01,
        "b",
        fontsize=panel_fs,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # ------------------------------------------------------------
    # Global arrow indicating increasing beta
    # ------------------------------------------------------------
    p_first = axes[1].get_position()
    p_last = axes[3].get_position()

    x_start = p_first.x0 + 0.01
    x_end = p_last.x1 - 0.01
    y_arrow = min(p_first.y0, p_last.y0) - 0.075

    arrow = FancyArrowPatch(
        (x_start, y_arrow),
        (x_end, y_arrow),
        transform=fig.transFigure,
        arrowstyle="->",
        mutation_scale=13,
        linewidth=1.3,
        color="0.45",
    )
    fig.add_artist(arrow)

    fig.text(
        (x_start + x_end) / 2,
        y_arrow - 0.045,
        r"Increasing attentional gain, $\beta$",
        ha="center",
        va="center",
        fontsize=arrow_fs,
        color="0.30",
    )

    fig.text(
        x_start,
        y_arrow - 0.045,
        "low",
        ha="left",
        va="center",
        fontsize=lowhigh_fs,
        color="0.45",
    )

    fig.text(
        x_end,
        y_arrow - 0.045,
        "high",
        ha="right",
        va="center",
        fontsize=lowhigh_fs,
        color="0.45",
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    save_name = "attentional_gain_concept"

    png_path = None
    pdf_path = None
    svg_path = None

    if save_png:
        png_path = figures_dir / f"{save_name}.png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)

    if save_pdf:
        pdf_path = figures_dir / f"{save_name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)

    if save_svg:
        svg_path = figures_dir / f"{save_name}.svg"
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "png_path": str(png_path) if png_path else None,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "svg_path": str(svg_path) if svg_path else None,
    }

def exact_sign_flip_test(x, y):
    """
    Exact two-sided paired sign-flip permutation test.

    The test statistic is the absolute mean paired difference.
    Intended here for the object-level paired comparisons.
    """
    from itertools import product

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got "
            f"{x.shape} and {y.shape}"
        )

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.nan

    d = x - y

    observed = abs(np.mean(d))

    permutation_stats = []

    for signs in product(
        [-1.0, 1.0],
        repeat=len(d),
    ):
        signs = np.asarray(
            signs,
            dtype=float,
        )

        stat = abs(
            np.mean(d * signs)
        )

        permutation_stats.append(stat)

    permutation_stats = np.asarray(
        permutation_stats
    )

    p = np.mean(
        permutation_stats
        >= observed - 1e-12
    )

    return float(p)


def _p_to_stars(p):
    """
    Convert p-value to significance annotation.
    """
    if not np.isfinite(p):
        return "n/a"

    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"

    return "ns"


def _add_star_bracket(
    ax,
    x1,
    x2,
    y,
    h,
    p,
    fontsize=13,
):
    ax.plot(
        [
            x1,
            x1,
            x2,
            x2,
        ],
        [
            y,
            y + h,
            y + h,
            y,
        ],
        color="black",
        linewidth=1.1,
        clip_on=False,
    )

    ax.text(
        (x1 + x2) / 2,
        y + 1.15 * h,
        _p_to_stars(p),
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color="black",
    )


def _paired_comparison_panel(
    ax,
    data,
    labels,
    colors,
    ylabel,
    comparisons,
    pvalues,
    panel_letter,
):
    """
    Shared paired violin/boxplot style for either
    two or three conditions.
    """
    data = [
        np.asarray(
            values,
            dtype=float,
        )
        for values in data
    ]

    n_conditions = len(data)

    if n_conditions not in (2, 3):
        raise ValueError(
            "This plot is designed for "
            "2 or 3 conditions."
        )

    n_objects = len(data[0])

    if any(
        len(values) != n_objects
        for values in data
    ):
        raise ValueError(
            "All conditions must contain "
            "the same number of paired objects."
        )

    positions = np.arange(
        n_conditions,
        dtype=float,
    )

    # --------------------------------------------------------
    # 1. Violin distributions
    # --------------------------------------------------------
    vp = ax.violinplot(
        data,
        positions=positions,
        widths=0.68,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in vp["bodies"]:
        body.set_facecolor(
            "gray"
        )
        body.set_edgecolor(
            "none"
        )
        body.set_alpha(0.16)
        body.set_zorder(0)

    # --------------------------------------------------------
    # 2. Paired object trajectories
    # --------------------------------------------------------
    for values in zip(*data):
        ax.plot(
            positions,
            values,
            color="slategray",
            linewidth=1.0,
            alpha=0.45,
            zorder=1,
        )

    # --------------------------------------------------------
    # 3. Thin transparent boxplots
    # --------------------------------------------------------
    ax.boxplot(
        data,
        positions=positions,
        widths=0.08,
        patch_artist=True,
        showfliers=False,
        zorder=2,

        medianprops=dict(
            color="black",
            linewidth=1.3,
        ),

        boxprops=dict(
            facecolor="none",
            edgecolor="black",
            linewidth=1.15,
        ),

        whiskerprops=dict(
            color="black",
            linewidth=1.05,
        ),

        capprops=dict(
            color="black",
            linewidth=1.05,
        ),
    )

    # --------------------------------------------------------
    # 4. Individual object points
    # --------------------------------------------------------
    if n_objects == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(
            -0.035,
            0.035,
            n_objects,
        )

    for (
        x,
        values,
        color,
    ) in zip(
        positions,
        data,
        colors,
    ):
        ax.scatter(
            x + offsets,
            values,
            s=38,
            facecolor="none",
            edgecolor=color,
            linewidth=1.5,
            alpha=0.85,
            zorder=4,
        )

    # --------------------------------------------------------
    # 5. Labels
    # --------------------------------------------------------
    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        fontsize=10.5,
        color="black",
    )

    ax.set_ylabel(
        ylabel,
        fontsize=12,
        color="black",
    )

    # --------------------------------------------------------
    # 6. Significance annotations
    # --------------------------------------------------------
    data_min = min(
        np.min(values)
        for values in data
    )

    data_max = max(
        np.max(values)
        for values in data
    )

    yrange = (
        data_max
        - data_min
    )

    if yrange <= 0:
        yrange = max(
            abs(data_max),
            1.0,
        )

    h = 0.025 * yrange

    # Different height for each bracket
    bracket_step = (
        0.14 * yrange
    )

    first_y = (
        data_max
        + 0.08 * yrange
    )

    for k, (
        comparison,
        p,
    ) in enumerate(
        zip(
            comparisons,
            pvalues,
        )
    ):
        x1, x2 = comparison

        y = (
            first_y
            + k * bracket_step
        )

        _add_star_bracket(
            ax=ax,
            x1=x1,
            x2=x2,
            y=y,
            h=h,
            p=p,
            fontsize=13,
        )

    top_extra = (
        0.27
        if len(comparisons) == 1
        else 0.42
    )

    ax.set_ylim(
        data_min
        - 0.08 * yrange,
        data_max
        + top_extra * yrange,
    )

    # --------------------------------------------------------
    # 7. Aesthetics
    # --------------------------------------------------------
    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.spines[
        "bottom"
    ].set_visible(False)

    ax.spines[
        "left"
    ].set_visible(True)

    ax.spines[
        "left"
    ].set_color("black")

    ax.spines[
        "left"
    ].set_linewidth(1.0)

    ax.tick_params(
        axis="y",
        colors="black",
        width=1.0,
        length=4,
        direction="out",
        labelsize=10,
    )

    ax.tick_params(
        axis="x",
        colors="black",
        length=0,
        width=0,
        pad=8,
    )

    ax.grid(False)

    ax.text(
        -0.25,
        1.02,
        panel_letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color="black",
        va="top",
    )




def _save_comparison_figure(
    fig,
    save_name,
    figures_dir=None,
    dpi=600,
    save_png=True,
    save_pdf=True,
    save_svg=True,
):
    if figures_dir is None:
        figures_dir = _default_figures_dir()
    else:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    png_path = None
    pdf_path = None
    svg_path = None

    if save_png:
        png_path = (
            figures_dir
            / f"{save_name}.png"
        )

        fig.savefig(
            png_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.03,
        )

    if save_pdf:
        pdf_path = (
            figures_dir
            / f"{save_name}.pdf"
        )

        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.03,
        )

    if save_svg:
        svg_path = (
            figures_dir
            / f"{save_name}.svg"
        )

        fig.savefig(
            svg_path,
            bbox_inches="tight",
            pad_inches=0.03,
        )

    return {
        "png_path": (
            str(png_path)
            if png_path
            else None
        ),
        "pdf_path": (
            str(pdf_path)
            if pdf_path
            else None
        ),
        "svg_path": (
            str(svg_path)
            if svg_path
            else None
        ),
    }

def plot_softmax_argmax_overview_1x4(
    softmax_argmax_df,
    softmax_argmax_object_df=None,
    objects_order=None,
    softmax_beta=0.5,
    figures_dir=None,
    figsize=(15.0, 4.6),
    dpi=600,
    save_png=True,
    save_pdf=True,
    save_svg=True,
    show=True,
):
    """
    Combined Softmax-vs-Argmax overview.

    Layout:
        1) by-object explored object area
        2) paired explored object area
        3) by-object normalised fixation entropy
        4) paired normalised fixation entropy

    The paired panels use the same aesthetics as
    plot_exploration_balance_exploitation_paired():
        - light gray violin
        - slategray paired trajectories
        - thin transparent boxplot
        - hollow condition-coloured object points
        - significance bracket with stars

    The by-object panels show:
        - outlined condition bars
        - black seed-level observations
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ============================================================
    # Validation
    # ============================================================
    required_seed_columns = {
        "object",
        "mode",
        "area_explored_coeff",
        "fixation_entropy_norm",
    }

    missing = (
        required_seed_columns
        - set(softmax_argmax_df.columns)
    )

    if missing:
        raise ValueError(
            "softmax_argmax_df is missing columns: "
            + ", ".join(sorted(missing))
        )

    # ============================================================
    # Object order
    # ============================================================
    if objects_order is None:
        objects_order = list(
            dict.fromkeys(
                softmax_argmax_df[
                    "object"
                ].tolist()
            )
        )

    # ============================================================
    # Seed-level dataframe
    # ============================================================
    seed_df = (
        softmax_argmax_df
        .copy()
    )

    seed_df["mode"] = (
        seed_df["mode"]
        .astype(str)
        .str.lower()
    )

    # ------------------------------------------------------------
    # Softmax seeds
    # ------------------------------------------------------------
    if "beta" in seed_df.columns:

        softmax_seed_df = seed_df[
            (
                seed_df["mode"]
                == "softmax"
            )
            & np.isclose(
                seed_df["beta"].astype(float),
                float(softmax_beta),
            )
        ].copy()

    else:

        softmax_seed_df = seed_df[
            seed_df["mode"]
            == "softmax"
        ].copy()

    # ------------------------------------------------------------
    # Argmax
    # ------------------------------------------------------------
    argmax_seed_df = seed_df[
        seed_df["mode"]
        == "argmax"
    ].copy()

    # ============================================================
    # Object-level means
    # ============================================================
    if softmax_argmax_object_df is None:

        object_df = (
            seed_df
            .groupby(
                [
                    "object",
                    "mode",
                ],
                as_index=False,
            )
            .agg(
                area_explored_coeff=(
                    "area_explored_coeff",
                    "mean",
                ),
                fixation_entropy_norm=(
                    "fixation_entropy_norm",
                    "mean",
                ),
            )
        )

    else:

        object_df = (
            softmax_argmax_object_df
            .copy()
        )

    object_df["mode"] = (
        object_df["mode"]
        .astype(str)
        .str.lower()
    )

    softmax_object_df = object_df[
        object_df["mode"]
        == "softmax"
    ].copy()

    argmax_object_df = object_df[
        object_df["mode"]
        == "argmax"
    ].copy()

    # ============================================================
    # Extract paired object-level values
    # ============================================================
    def get_paired_values(metric):

        softmax_map = (
            softmax_object_df
            .set_index("object")[
                metric
            ]
        )

        argmax_map = (
            argmax_object_df
            .set_index("object")[
                metric
            ]
        )

        valid_objects = [
            obj
            for obj in objects_order
            if (
                obj in softmax_map.index
                and obj in argmax_map.index
                and np.isfinite(
                    softmax_map.loc[obj]
                )
                and np.isfinite(
                    argmax_map.loc[obj]
                )
            )
        ]

        if len(valid_objects) == 0:
            raise RuntimeError(
                f"No paired objects available "
                f"for {metric}."
            )

        softmax = np.asarray(
            [
                softmax_map.loc[obj]
                for obj in valid_objects
            ],
            dtype=float,
        )

        argmax = np.asarray(
            [
                argmax_map.loc[obj]
                for obj in valid_objects
            ],
            dtype=float,
        )

        return (
            valid_objects,
            softmax,
            argmax,
        )

    # ------------------------------------------------------------
    # Area
    # ------------------------------------------------------------
    (
        objects_area,
        area_softmax,
        area_argmax,
    ) = get_paired_values(
        "area_explored_coeff"
    )

    # ------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------
    (
        objects_entropy,
        entropy_softmax,
        entropy_argmax,
    ) = get_paired_values(
        "fixation_entropy_norm"
    )

    if objects_area != objects_entropy:
        raise RuntimeError(
            "Area and entropy do not contain "
            "the same paired objects."
        )

    # ============================================================
    # Statistics
    # ============================================================
    p_area = exact_sign_flip_test(
        area_softmax,
        area_argmax,
    )

    p_entropy = exact_sign_flip_test(
        entropy_softmax,
        entropy_argmax,
    )

    print(
        "Softmax vs Argmax"
    )

    print(
        f"  Explored object area: "
        f"p = {p_area:.5f}"
    )

    print(
        f"  Normalised fixation entropy: "
        f"p = {p_entropy:.5f}"
    )

    # ============================================================
    # Colours
    # ============================================================
    COLOR_SOFTMAX = "navy"
    COLOR_ARGMAX = "crimson"

    colors = [
        COLOR_SOFTMAX,
        COLOR_ARGMAX,
    ]

    # ============================================================
    # Figure
    # ============================================================
    fig, axes = plt.subplots(
        1,
        4,
        figsize=figsize,
        gridspec_kw={
            "width_ratios": [
                1.25,
                0.72,
                1.25,
                0.72,
            ],
        },
    )

    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.25,
        top=0.84,
        wspace=0.38,
    )

    # ============================================================
    # Shared style
    # ============================================================
    label_fs = 12
    tick_fs = 10
    panel_fs = 14
    legend_fs = 12

    bar_width = 0.34

    # ============================================================
    # By-object aesthetics
    # ============================================================
    def style_by_object_axis(ax):

        ax.set_facecolor(
            "white"
        )

        ax.grid(
            False
        )

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

        # Same visual language as paired plots
        ax.spines[
            "bottom"
        ].set_visible(
            False
        )

        ax.spines[
            "left"
        ].set_visible(
            True
        )

        ax.spines[
            "left"
        ].set_color(
            "black"
        )

        ax.spines[
            "left"
        ].set_linewidth(
            1.0
        )

        ax.tick_params(
            axis="y",
            colors="black",
            width=1.0,
            length=4,
            direction="out",
            labelsize=tick_fs,
        )

        ax.tick_params(
            axis="x",
            colors="black",
            length=0,
            width=0,
            pad=8,
            labelsize=tick_fs,
        )

    # ============================================================
    # By-object plot helper
    # ============================================================
    def plot_by_object_panel(
        ax,
        metric,
        ylabel,
        panel_letter,
    ):

        x = np.arange(
            len(objects_order),
            dtype=float,
        )

        # --------------------------------------------------------
        # Object-level means
        # --------------------------------------------------------
        softmax_map = (
            softmax_object_df
            .set_index("object")[
                metric
            ]
        )

        argmax_map = (
            argmax_object_df
            .set_index("object")[
                metric
            ]
        )

        softmax_means = np.asarray(
            [
                (
                    softmax_map.loc[obj]
                    if obj in softmax_map.index
                    else np.nan
                )
                for obj in objects_order
            ],
            dtype=float,
        )

        argmax_means = np.asarray(
            [
                (
                    argmax_map.loc[obj]
                    if obj in argmax_map.index
                    else np.nan
                )
                for obj in objects_order
            ],
            dtype=float,
        )

        # --------------------------------------------------------
        # Outline bars
        # --------------------------------------------------------
        ax.bar(
            x - bar_width / 2,
            softmax_means,
            width=bar_width,
            facecolor="none",
            edgecolor=COLOR_SOFTMAX,
            linewidth=2.5,
            zorder=1,
        )

        ax.bar(
            x + bar_width / 2,
            argmax_means,
            width=bar_width,
            facecolor="none",
            edgecolor=COLOR_ARGMAX,
            linewidth=2.5,
            zorder=1,
        )

        # --------------------------------------------------------
        # Seed-level observations
        # --------------------------------------------------------
        for i, obj in enumerate(
            objects_order
        ):

            softmax_values = (
                softmax_seed_df[
                    softmax_seed_df[
                        "object"
                    ] == obj
                ][metric]
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            argmax_values = (
                argmax_seed_df[
                    argmax_seed_df[
                        "object"
                    ] == obj
                ][metric]
                .dropna()
                .to_numpy(
                    dtype=float
                )
            )

            # Softmax seeds
            if len(
                softmax_values
            ) > 0:

                if len(
                    softmax_values
                ) == 1:

                    jitter = np.array(
                        [0.0]
                    )

                else:

                    jitter = np.linspace(
                        -0.04,
                        0.04,
                        len(
                            softmax_values
                        ),
                    )

                ax.scatter(
                    (
                        x[i]
                        - bar_width / 2
                        + jitter
                    ),
                    softmax_values,
                    s=28,
                    facecolor="black",
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.90,
                    zorder=4,
                )

            # Argmax observation(s)
            if len(
                argmax_values
            ) > 0:

                if len(
                    argmax_values
                ) == 1:

                    jitter = np.array(
                        [0.0]
                    )

                else:

                    jitter = np.linspace(
                        -0.04,
                        0.04,
                        len(
                            argmax_values
                        ),
                    )

                ax.scatter(
                    (
                        x[i]
                        + bar_width / 2
                        + jitter
                    ),
                    argmax_values,
                    s=28,
                    facecolor="black",
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.90,
                    zorder=4,
                )

        # --------------------------------------------------------
        # Dynamic y-limits
        # --------------------------------------------------------
        all_values = []

        for values in (
            softmax_means,
            argmax_means,
        ):

            valid = values[
                np.isfinite(values)
            ]

            if len(valid) > 0:
                all_values.extend(
                    valid.tolist()
                )

        all_values = np.asarray(
            all_values,
            dtype=float,
        )

        y_min = float(
            np.min(all_values)
        )

        y_max = float(
            np.max(all_values)
        )

        yrange = max(
            y_max - y_min,
            1e-6,
        )

        ax.set_ylim(
            max(
                0.0,
                y_min
                - 0.08 * yrange,
            ),
            y_max
            + 0.12 * yrange,
        )

        # --------------------------------------------------------
        # Labels
        # --------------------------------------------------------
        ax.set_xlim(
            -0.65,
            len(objects_order)
            - 0.35,
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            objects_order,
            rotation=40,
            ha="right",
            fontsize=tick_fs,
            color="black",
        )

        ax.set_ylabel(
            ylabel,
            fontsize=label_fs,
            color="black",
        )

        style_by_object_axis(
            ax
        )

        # --------------------------------------------------------
        # Panel label
        # --------------------------------------------------------
        ax.text(
            -0.22,
            1.12,
            panel_letter,
            transform=ax.transAxes,
            fontsize=panel_fs,
            fontweight="bold",
            color="black",
            va="top",
        )

    # ============================================================
    # Labels for paired plots
    # ============================================================
    paired_labels = [
        (
            "Softmax\n"
            "Exploration\n"
            + rf"($\beta={softmax_beta:g}$)"
        ),
        (
            "Argmax\n"
            "Exploitation"
        ),
    ]

    # ============================================================
    # a — explored object area
    # ============================================================
    plot_by_object_panel(
        ax=axes[0],
        metric=(
            "area_explored_coeff"
        ),
        ylabel=(
            "Explored object area"
        ),
        panel_letter="a",
    )

    _paired_comparison_panel(
        ax=axes[1],
        data=[
            area_softmax,
            area_argmax,
        ],
        labels=paired_labels,
        colors=colors,
        ylabel=(
            "Explored object area"
        ),
        comparisons=[
            (0, 1),
        ],
        pvalues=[
            p_area,
        ],
        # no additional panel letter:
        # axes[0] already represents panel a
        panel_letter="",
    )

    # ============================================================
    # b — normalised fixation entropy
    # ============================================================
    plot_by_object_panel(
        ax=axes[2],
        metric=(
            "fixation_entropy_norm"
        ),
        ylabel=(
            "Normalised fixation entropy"
        ),
        panel_letter="b",
    )

    _paired_comparison_panel(
        ax=axes[3],
        data=[
            entropy_softmax,
            entropy_argmax,
        ],
        labels=paired_labels,
        colors=colors,
        ylabel=(
            "Normalised fixation entropy"
        ),
        comparisons=[
            (0, 1),
        ],
        pvalues=[
            p_entropy,
        ],
        panel_letter="",
    )

    # ============================================================
    # One-row legend
    # ============================================================
    legend_handles = [
        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="None",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=(
                COLOR_SOFTMAX
            ),
            markeredgewidth=2.0,
            label=(
                "Softmax Exploration "
                + rf"($\beta={softmax_beta:g}$)"
            ),
        ),

        plt.Line2D(
            [],
            [],
            marker="s",
            linestyle="None",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor=(
                COLOR_ARGMAX
            ),
            markeredgewidth=2.0,
            label=(
                "Argmax Exploitation"
            ),
        ),

        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=5.5,
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=(
                "Individual run"
            ),
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.995,
        ),
        ncol=3,
        frameon=False,
        fontsize=legend_fs,
        handlelength=1.0,
        handletextpad=1.0,
        columnspacing=1.5,
    )

    # ============================================================
    # Save
    # ============================================================
    paths = _save_comparison_figure(
        fig=fig,
        save_name=(
            "softmax_argmax_overview_1x4"
        ),
        figures_dir=figures_dir,
        dpi=dpi,
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        **paths,
        "p_area": p_area,
        "p_entropy": p_entropy,
        "objects": objects_area,
    }

def plot_beta_sweep_by_object(
    beta_object_df,
    objects_order=None,
    figures_dir=None,
    figsize=(8.6, 3.8),
    dpi=600,
    save_png=True,
    save_pdf=True,
    save_svg=True,
    show=True,
):
    """
    Plot object-level trajectories across the Softmax beta sweep.

    Panels:
        a) explored object area
        b) normalised fixation entropy

    Each coloured trajectory corresponds to one object.

    A dashed black curve shows the overall second-order trend
    across object-level observations, with a 95% confidence interval.

    Beta is represented internally as log10(beta), while the
    displayed x-axis ticks retain the original beta values.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------
    required_columns = {
        "object",
        "beta",
        "area_explored_coeff",
        "fixation_entropy_norm",
    }

    missing = (
        required_columns
        - set(beta_object_df.columns)
    )

    if missing:
        raise ValueError(
            "beta_object_df is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # ------------------------------------------------------------
    # Prepare dataframe
    # ------------------------------------------------------------
    df = beta_object_df.copy()

    df["beta"] = (
        df["beta"]
        .astype(float)
    )

    if (df["beta"] <= 0).any():
        raise ValueError(
            "All beta values must be > 0."
        )

    # Work in log10(beta), but display original beta values
    df["log_beta"] = np.log10(
        df["beta"]
    )

    # ------------------------------------------------------------
    # Object order
    # ------------------------------------------------------------
    if objects_order is None:
        objects_order = list(
            dict.fromkeys(
                df["object"].tolist()
            )
        )

    # ------------------------------------------------------------
    # Beta values
    # ------------------------------------------------------------
    betas = np.sort(
        df["beta"].unique()
    )

    log_betas = np.log10(
        betas
    )

    # ------------------------------------------------------------
    # Colour palette
    # ------------------------------------------------------------
    palette_colors = sns.color_palette(
        "flare",
        n_colors=len(objects_order),
    )

    object_palette = {
        obj: palette_colors[i]
        for i, obj in enumerate(
            objects_order
        )
    }

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
    )

    # Space for one-row legend
    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.20,
        top=0.80,
        wspace=0.30,
    )

    panel_specs = [
        (
            axes[0],
            "area_explored_coeff",
            "Explored object area",
            "a",
        ),
        (
            axes[1],
            "fixation_entropy_norm",
            "Normalised fixation entropy",
            "b",
        ),
    ]

    # ------------------------------------------------------------
    # Style parameters
    # ------------------------------------------------------------
    label_fs = 10
    tick_fs = 9
    legend_fs = 10
    panel_fs = 13

    # ============================================================
    # Panels
    # ============================================================
    for (
        ax,
        metric,
        ylabel,
        panel_letter,
    ) in panel_specs:

        # --------------------------------------------------------
        # Individual object trajectories
        # --------------------------------------------------------
        for obj in objects_order:

            sub = (
                df[
                    df["object"] == obj
                ]
                .copy()
                .sort_values("beta")
            )

            if len(sub) == 0:
                continue

            ax.plot(
                sub["log_beta"],
                sub[metric],
                color=object_palette[obj],
                linewidth=1.5,
                alpha=0.50,
                zorder=2,
            )

        # --------------------------------------------------------
        # Object-level points
        # --------------------------------------------------------
        for obj in objects_order:

            sub = (
                df[
                    df["object"] == obj
                ]
                .copy()
                .sort_values("beta")
            )

            if len(sub) == 0:
                continue

            ax.scatter(
                sub["log_beta"],
                sub[metric],
                s=28,
                color=object_palette[obj],
                edgecolor="white",
                linewidth=0.55,
                alpha=0.90,
                zorder=3,
            )

        # --------------------------------------------------------
        # Overall trend + 95% CI
        # --------------------------------------------------------
        sns.regplot(
            data=df,
            x="log_beta",
            y=metric,
            scatter=False,
            order=2,
            truncate=False,
            ci=95,
            n_boot=1000,
            seed=0,
            color="0.20",
            line_kws={
                "linewidth": 1.8,
                "alpha": 0.90,
                "linestyle": "--",
                "zorder": 4,
            },
            ax=ax,
        )

        # --------------------------------------------------------
        # X-axis
        # --------------------------------------------------------
        ax.set_xticks(
            log_betas
        )

        ax.set_xticklabels(
            [
                f"{beta:g}"
                for beta in betas
            ]
        )

        ax.set_xlabel(
            r"Attentional gain $\beta$",
            fontsize=label_fs,
            labelpad=5,
        )

        ax.set_ylabel(
            ylabel,
            fontsize=label_fs,
            labelpad=5,
        )

        # --------------------------------------------------------
        # Dynamic Y limits
        # --------------------------------------------------------
        y_values = (
            df[metric]
            .astype(float)
            .to_numpy()
        )

        y_min = float(
            np.nanmin(y_values)
        )

        y_max = float(
            np.nanmax(y_values)
        )

        y_range = max(
            y_max - y_min,
            1e-6,
        )

        ax.set_ylim(
            max(
                0.0,
                y_min - 0.05 * y_range,
            ),
            y_max + 0.08 * y_range,
        )

        # --------------------------------------------------------
        # Clean paper-style axes
        # --------------------------------------------------------
        ax.set_facecolor(
            "white"
        )

        ax.grid(
            False
        )

        ax.spines[
            "top"
        ].set_visible(
            False
        )

        ax.spines[
            "right"
        ].set_visible(
            False
        )

        ax.spines[
            "left"
        ].set_position(
            ("outward", 5)
        )

        ax.spines[
            "bottom"
        ].set_position(
            ("outward", 5)
        )

        ax.spines[
            "left"
        ].set_linewidth(
            0.8
        )

        ax.spines[
            "bottom"
        ].set_linewidth(
            0.8
        )

        ax.tick_params(
            axis="both",
            direction="out",
            length=3,
            width=0.8,
            pad=3,
            labelsize=tick_fs,
        )

        # --------------------------------------------------------
        # Panel letter
        # --------------------------------------------------------
        ax.text(
            -0.25,
            1.04,
            panel_letter,
            transform=ax.transAxes,
            fontsize=panel_fs,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="black",
        )

    # ============================================================
    # Shared legend — objects only
    # ============================================================
    legend_handles = []

    for obj in objects_order:

        legend_handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="-",
                linewidth=1.4,
                markersize=5,
                color=object_palette[obj],
                alpha=0.75,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=obj,
            )
        )

    # One single row
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(objects_order),
        frameon=False,
        fontsize=legend_fs,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=1.2,
    )

    # ============================================================
    # Save
    # ============================================================
    paths = _save_comparison_figure(
        fig=fig,
        save_name="beta_sweep_by_object",
        figures_dir=figures_dir,
        dpi=dpi,
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        **paths,
        "objects": objects_order,
        "betas": betas.tolist(),
    }


def plot_exploration_balance_exploitation_paired(
    beta_object_df,
    softmax_argmax_object_df,
    exploration_beta=0.5,
    balance_beta=10.0,
    objects_order=None,
    figures_dir=None,
    figsize=(8.6, 4.6),
    dpi=600,
    save_png=True,
    save_pdf=True,
    save_svg=True,
    show=True,
):
    """
    Object-level paired comparison of three attentional regimes:

        Softmax Exploration  -> exploration_beta
        Softmax Balance      -> balance_beta
        Argmax Exploitation

    Panels:
        a) explored object area
        b) normalised fixation entropy

    Statistical comparisons:
        Exploration vs Balance
        Balance vs Exploitation

    Exact two-sided paired sign-flip permutation tests are used.
    Exact p-values are printed; significance stars are shown
    on the figure.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # ============================================================
    # Validation
    # ============================================================
    beta_required = {
        "object",
        "beta",
        "area_explored_coeff",
        "fixation_entropy_norm",
    }

    argmax_required = {
        "object",
        "mode",
        "area_explored_coeff",
        "fixation_entropy_norm",
    }

    missing_beta = (
        beta_required
        - set(beta_object_df.columns)
    )

    missing_argmax = (
        argmax_required
        - set(softmax_argmax_object_df.columns)
    )

    if missing_beta:
        raise ValueError(
            "beta_object_df is missing columns: "
            + ", ".join(
                sorted(missing_beta)
            )
        )

    if missing_argmax:
        raise ValueError(
            "softmax_argmax_object_df is missing columns: "
            + ", ".join(
                sorted(missing_argmax)
            )
        )

    # ============================================================
    # Object order
    # ============================================================
    if objects_order is None:
        objects_order = list(
            dict.fromkeys(
                beta_object_df[
                    "object"
                ].tolist()
            )
        )

    # ============================================================
    # Extract the three paired conditions
    # ============================================================
    def get_three_values(metric):

        # --------------------------------------------------------
        # Softmax exploration
        # --------------------------------------------------------
        exploration_map = (
            beta_object_df[
                np.isclose(
                    beta_object_df[
                        "beta"
                    ].astype(float),
                    float(exploration_beta),
                )
            ]
            .set_index("object")[
                metric
            ]
        )

        # --------------------------------------------------------
        # Softmax balance
        # --------------------------------------------------------
        balance_map = (
            beta_object_df[
                np.isclose(
                    beta_object_df[
                        "beta"
                    ].astype(float),
                    float(balance_beta),
                )
            ]
            .set_index("object")[
                metric
            ]
        )

        # --------------------------------------------------------
        # Argmax exploitation
        # --------------------------------------------------------
        exploitation_map = (
            softmax_argmax_object_df[
                softmax_argmax_object_df[
                    "mode"
                ]
                .astype(str)
                .str.lower()
                == "argmax"
            ]
            .set_index("object")[
                metric
            ]
        )

        # --------------------------------------------------------
        # Only objects available in all three conditions
        # --------------------------------------------------------
        valid_objects = [
            obj
            for obj in objects_order
            if (
                obj in exploration_map.index
                and obj in balance_map.index
                and obj in exploitation_map.index
                and np.isfinite(
                    exploration_map.loc[obj]
                )
                and np.isfinite(
                    balance_map.loc[obj]
                )
                and np.isfinite(
                    exploitation_map.loc[obj]
                )
            )
        ]

        if len(valid_objects) == 0:
            raise RuntimeError(
                f"No paired objects available for {metric}."
            )

        exploration = np.asarray(
            [
                exploration_map.loc[obj]
                for obj in valid_objects
            ],
            dtype=float,
        )

        balance = np.asarray(
            [
                balance_map.loc[obj]
                for obj in valid_objects
            ],
            dtype=float,
        )

        exploitation = np.asarray(
            [
                exploitation_map.loc[obj]
                for obj in valid_objects
            ],
            dtype=float,
        )

        return (
            valid_objects,
            exploration,
            balance,
            exploitation,
        )

    # ============================================================
    # Area
    # ============================================================
    (
        objects_area,
        area_exploration,
        area_balance,
        area_exploitation,
    ) = get_three_values(
        "area_explored_coeff"
    )

    # ============================================================
    # Entropy
    # ============================================================
    (
        objects_entropy,
        entropy_exploration,
        entropy_balance,
        entropy_exploitation,
    ) = get_three_values(
        "fixation_entropy_norm"
    )

    if objects_area != objects_entropy:
        raise RuntimeError(
            "Area and entropy do not contain "
            "the same paired objects."
        )

    # ============================================================
    # Statistics
    # ============================================================

    # ------------------------------------------------------------
    # Area
    # ------------------------------------------------------------
    p_area_exploration_balance = (
        exact_sign_flip_test(
            area_exploration,
            area_balance,
        )
    )

    p_area_balance_exploitation = (
        exact_sign_flip_test(
            area_balance,
            area_exploitation,
        )
    )

    # ------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------
    p_entropy_exploration_balance = (
        exact_sign_flip_test(
            entropy_exploration,
            entropy_balance,
        )
    )

    p_entropy_balance_exploitation = (
        exact_sign_flip_test(
            entropy_balance,
            entropy_exploitation,
        )
    )

    # ============================================================
    # Print exact p-values
    # ============================================================
    print(
        "Exploration / Balance / Exploitation"
    )

    print()

    print(
        "Explored object area"
    )

    print(
        "  Exploration vs Balance: "
        f"p = {p_area_exploration_balance:.5f}"
    )

    print(
        "  Balance vs Exploitation: "
        f"p = {p_area_balance_exploitation:.5f}"
    )

    print()

    print(
        "Normalised fixation entropy"
    )

    print(
        "  Exploration vs Balance: "
        f"p = {p_entropy_exploration_balance:.5f}"
    )

    print(
        "  Balance vs Exploitation: "
        f"p = {p_entropy_balance_exploitation:.5f}"
    )

    # ============================================================
    # Figure
    # ============================================================
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
    )

    # ============================================================
    # Labels
    # ============================================================
    labels = [
        (
            "Softmax\n"
            "Exploration\n"
            + rf"($\beta={exploration_beta:g}$)"
        ),
        (
            "Softmax\n"
            "Balance\n"
            + rf"($\beta={balance_beta:g}$)"
        ),
        (
            "Argmax\n"
            "Exploitation"
        ),
    ]

    # ============================================================
    # Colours
    # ============================================================
    colors = [
        "navy",
        "blueviolet",
        "crimson",
    ]

    # ============================================================
    # Panel a — explored object area
    # ============================================================
    _paired_comparison_panel(
        ax=axes[0],

        data=[
            area_exploration,
            area_balance,
            area_exploitation,
        ],

        labels=labels,

        colors=colors,

        ylabel=(
            "Explored object area"
        ),

        # Only compare Balance against the two extremes
        comparisons=[
            (0, 1),
            (1, 2),
        ],

        pvalues=[
            p_area_exploration_balance,
            p_area_balance_exploitation,
        ],

        panel_letter="a",
    )

    # ============================================================
    # Panel b — fixation entropy
    # ============================================================
    _paired_comparison_panel(
        ax=axes[1],

        data=[
            entropy_exploration,
            entropy_balance,
            entropy_exploitation,
        ],

        labels=labels,

        colors=colors,

        ylabel=(
            "Normalised fixation entropy"
        ),

        comparisons=[
            (0, 1),
            (1, 2),
        ],

        pvalues=[
            p_entropy_exploration_balance,
            p_entropy_balance_exploitation,
        ],

        panel_letter="b",
    )

    # ============================================================
    # Layout
    # ============================================================
    fig.tight_layout(
        w_pad=2.4,
    )

    # ============================================================
    # Save
    # ============================================================
    paths = _save_comparison_figure(
        fig=fig,

        save_name=(
            "exploration_balance_"
            "exploitation_paired"
        ),

        figures_dir=figures_dir,
        dpi=dpi,
        save_png=save_png,
        save_pdf=save_pdf,
        save_svg=save_svg,
    )

    # ============================================================
    # Show / close
    # ============================================================
    if show:
        plt.show()
    else:
        plt.close(fig)

    # ============================================================
    # Return
    # ============================================================
    return {
        **paths,

        "p_area_exploration_balance": (
            p_area_exploration_balance
        ),

        "p_area_balance_exploitation": (
            p_area_balance_exploitation
        ),

        "p_entropy_exploration_balance": (
            p_entropy_exploration_balance
        ),

        "p_entropy_balance_exploitation": (
            p_entropy_balance_exploitation
        ),

        "objects": objects_area,
    }
