"""
Run a beta sweep for the activeVision repo as a single script.

This script can:
1. Render a drift-only active vision sequence from a .blend object.
2. Convert rendered RGB frames into DVS-style events.
3. Run proto-object attention for multiple beta values and seeds.
4. Quantify coverage, policy entropy, fixation recurrence, and saccade/action statistics.
5. Save CSV summaries and summary plots.

Run from the root of the activeVision repo, for example:

    python run_activevision_beta_sweep.py \
        --object-path data/airplane_010.blend \
        --output-root data/renders/airplane_beta_sweep \
        --render \
        --events \
        --num-frames 500 \
        --fps 60 \
        --resolution 256 \
        --betas 5 10 25 50 100 200 \
        --seeds 0 1 2 3 4 5 6 7 8 9

If you already rendered frames and generated events.npy, skip rendering/events:

    python run_activevision_beta_sweep.py \
        --output-root data/renders/airplane_beta_sweep \
        --events-npy data/renders/airplane_beta_sweep/events/events.npy \
        --betas 5 10 25 50 100 200 \
        --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def add_src_to_path(repo_root: Path) -> None:
    src_dir = repo_root / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"Could not find src directory at {src_dir}. Run from repo root or pass --repo-root.")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def load_saccades(saccades_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(saccades_path, sep=r"\s+", engine="python")


def fixation_recurrence_metrics(
    saccades_path: str | Path,
    resolution: tuple[int, int],
    cell_fraction: float = 0.10,
    exclude_initial: bool = True,
) -> dict:
    """Quantify how often sampled fixations revisit the same coarse spatial cell."""
    saccades = load_saccades(saccades_path)
    xy = saccades[["post_x", "post_y"]].to_numpy(dtype=float)

    if exclude_initial and len(xy) > 1:
        xy = xy[1:]

    height, width = resolution
    cell_w = max(1, int(width * cell_fraction))
    cell_h = max(1, int(height * cell_fraction))

    if xy.shape[0] == 0:
        return {
            "n_fixations": 0,
            "unique_cells": 0,
            "revisit_fraction": np.nan,
            "consecutive_same_cell_fraction": np.nan,
        }

    xs = np.clip(xy[:, 0].astype(int), 0, width - 1)
    ys = np.clip(xy[:, 1].astype(int), 0, height - 1)

    cells = list(zip(xs // cell_w, ys // cell_h))

    n_fix = len(cells)
    n_unique = len(set(cells))
    revisit_fraction = 1.0 - (n_unique / n_fix)

    if n_fix > 1:
        consecutive_same = float(np.mean([cells[i] == cells[i - 1] for i in range(1, n_fix)]))
    else:
        consecutive_same = np.nan

    return {
        "n_fixations": int(n_fix),
        "unique_cells": int(n_unique),
        "revisit_fraction": float(revisit_fraction),
        "consecutive_same_cell_fraction": float(consecutive_same),
    }


def saccade_action_metrics(saccades_path: str | Path, exclude_initial: bool = True) -> dict:
    """Quantify displacement statistics between consecutive fixations."""
    saccades = load_saccades(saccades_path)
    dx = saccades["dx"].to_numpy(dtype=float)
    dy = saccades["dy"].to_numpy(dtype=float)

    if exclude_initial and len(dx) > 1:
        dx = dx[1:]
        dy = dy[1:]

    if len(dx) == 0:
        return {
            "mean_saccade_length_px": np.nan,
            "median_saccade_length_px": np.nan,
            "std_saccade_length_px": np.nan,
        }

    length = np.sqrt(dx ** 2 + dy ** 2)
    return {
        "mean_saccade_length_px": float(np.mean(length)),
        "median_saccade_length_px": float(np.median(length)),
        "std_saccade_length_px": float(np.std(length, ddof=1)) if len(length) > 1 else 0.0,
    }


def attention_entropy_metrics(
    events_npy: str | Path,
    beta: float,
    resolution: tuple[int, int],
    window_period_ms: float,
    max_windows: int | None,
    seed: int,
    center_sigma: float,
    surround_sigma: float,
    num_pyr: int,
    use_polarity: bool,
) -> dict:
    """Recompute the attention policy per temporal window and measure softmax entropy."""
    from attention.attention import _build_event_windows, compute_saliency_map

    events = np.load(events_npy)
    windows, _, _ = _build_event_windows(
        events=events,
        resolution=resolution,
        window_period_ms=window_period_ms,
        max_windows=max_windows,
        use_polarity=use_polarity,
    )

    rng = np.random.default_rng(seed)
    entropies = []
    norm_entropies = []
    max_probs = []
    effective_n_pixels = []

    for window_frame in windows:
        _, _, prob_map = compute_saliency_map(
            window_frame,
            center_sigma=center_sigma,
            surround_sigma=surround_sigma,
            num_pyr=num_pyr,
            beta=beta,
            rng=rng,
        )

        p = prob_map.reshape(-1).astype(np.float64)
        p = p[p > 0]

        entropy = -np.sum(p * np.log(p))
        norm_entropy = entropy / np.log(prob_map.size)
        eff_n = np.exp(entropy)

        entropies.append(entropy)
        norm_entropies.append(norm_entropy)
        max_probs.append(float(prob_map.max()))
        effective_n_pixels.append(float(eff_n))

    return {
        "mean_entropy": float(np.mean(entropies)),
        "mean_norm_entropy": float(np.mean(norm_entropies)),
        "mean_max_prob": float(np.mean(max_probs)),
        "mean_effective_n_pixels": float(np.mean(effective_n_pixels)),
        "n_windows_entropy": int(len(windows)),
    }


def mean_sem_summary(df: pd.DataFrame, group_col: str, metric_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for group_value, group_df in df.groupby(group_col):
        row = {group_col: group_value, "n": int(len(group_df))}
        for col in metric_cols:
            values = group_df[col].astype(float).to_numpy()
            values = values[~np.isnan(values)]
            if len(values) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_sem"] = np.nan
            elif len(values) == 1:
                row[f"{col}_mean"] = float(values[0])
                row[f"{col}_sem"] = 0.0
            else:
                row[f"{col}_mean"] = float(np.mean(values))
                row[f"{col}_sem"] = float(np.std(values, ddof=1) / math.sqrt(len(values)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


def save_summary_plot(summary: pd.DataFrame, x: str, y_mean: str, y_sem: str, ylabel: str, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.errorbar(summary[x], summary[y_mean], yerr=summary[y_sem], marker="o", capsize=3)
    plt.xscale("log")
    plt.xlabel("Attentional gain beta")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run activeVision beta sweep and quantify attentional sampling statistics.")

    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Path to activeVision repo root.")
    parser.add_argument("--object-path", type=Path, default=Path("data/airplane_010.blend"), help="Path to .blend object, relative to repo root unless absolute.")
    parser.add_argument("--output-root", type=Path, default=Path("data/renders/airplane_beta_sweep"), help="Output directory, relative to repo root unless absolute.")

    parser.add_argument("--render", action="store_true", help="Render RGB frames before generating events.")
    parser.add_argument("--events", action="store_true", help="Convert rendered RGB frames to DVS events before attention.")
    parser.add_argument("--events-npy", type=Path, default=None, help="Use an existing events.npy instead of generating one.")

    parser.add_argument("--num-frames", type=int, default=500)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU rendering in Blender if available.")

    parser.add_argument("--object-target-size", type=float, default=0.45)
    parser.add_argument("--object-azimuth-deg", type=float, default=315.0)
    parser.add_argument("--object-elevation-deg", type=float, default=30.0)
    parser.add_argument("--drift-sigma-deg", type=float, nargs=2, default=(0.005, 0.005))

    parser.add_argument("--betas", type=float, nargs="+", default=[5, 10, 25, 50, 100, 200])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--window-period-ms", type=float, default=10.0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--use-polarity", action="store_true")

    parser.add_argument("--num-pyr", type=int, default=6)
    parser.add_argument("--center-sigma", type=float, default=2.0)
    parser.add_argument("--surround-sigma", type=float, default=8.0)
    parser.add_argument("--foveation-sigma-fraction", type=float, default=0.10)

    parser.add_argument("--grid-cell-fraction", type=float, default=0.10)
    parser.add_argument("--object-noise-thresh", type=int, default=100)

    parser.add_argument("--make-gifs-for-first-seed", action="store_true", help="Save saliency/action GIFs only for the first seed per beta.")
    parser.add_argument("--gif-fps", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    add_src_to_path(repo_root)

    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    object_path = args.object_path if args.object_path.is_absolute() else repo_root / args.object_path
    sequence_dir = output_root / "motion_sequence"
    frames_dir = sequence_dir / "frames"
    events_dir = output_root / "events"
    sweep_dir = output_root / "beta_sweep"
    plots_dir = output_root / "summary_plots"

    output_root.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["repo_root"] = str(repo_root)
    config["object_path"] = str(object_path)
    config["output_root"] = str(output_root)
    with open(output_root / "beta_sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    if args.render:
        from scene.render_offline import render_camera_motion_sequence, frames_to_gif

        print(f"[render] Rendering {args.num_frames} frames at {args.resolution}x{args.resolution}...")
        render_result = render_camera_motion_sequence(
            object_path=object_path,
            output_dir=sequence_dir,
            num_frames=args.num_frames,
            fps=args.fps,
            resolution=args.resolution,
            samples=args.samples,
            use_gpu=args.use_gpu,
            object_target_size=args.object_target_size,
            object_azimuth_deg=args.object_azimuth_deg,
            object_elevation_deg=args.object_elevation_deg,
            drift_sigma_deg=tuple(args.drift_sigma_deg),
            seed=0,
        )
        print(f"[render] Frames saved to {render_result['frames_dir']}")
        frames_to_gif(frames_dir=frames_dir, output_gif=sequence_dir / "rgb_motion.gif", fps=min(args.fps, 30))

    if args.events:
        from scene.render_offline import frames_to_events_npy_and_gif

        print(f"[events] Converting frames to DVS events from {frames_dir}...")
        event_result = frames_to_events_npy_and_gif(
            frames_dir=frames_dir,
            output_dir=events_dir,
            fps=args.fps,
            gif_fps=args.gif_fps,
            keep_dat=True,
        )
        events_npy = Path(event_result["npy_path"])
        print(f"[events] Events saved to {events_npy}")
    elif args.events_npy is not None:
        events_npy = args.events_npy if args.events_npy.is_absolute() else repo_root / args.events_npy
    else:
        events_npy = events_dir / "events.npy"

    if not events_npy.exists():
        raise FileNotFoundError(
            f"Could not find events.npy at {events_npy}. Pass --events-npy or run with --events."
        )

    from attention.attention import run_attention, plot_attention_exploration
    from attention.foveation import foveations_to_gif

    resolution_hw = (args.resolution, args.resolution)
    foveation_sigma = max(1.0, args.foveation_sigma_fraction * args.resolution)

    rows = []
    first_seed = args.seeds[0] if len(args.seeds) > 0 else 0

    for beta in args.betas:
        for seed in args.seeds:
            run_name = f"beta_{beta:g}_seed_{seed}"
            out_dir = sweep_dir / run_name
            make_gif = bool(args.make_gifs_for_first_seed and seed == first_seed)

            print(f"[attention] beta={beta:g}, seed={seed}")
            att = run_attention(
                events_npy=events_npy,
                output_dir=out_dir,
                resolution=resolution_hw,
                window_period_ms=args.window_period_ms,
                max_windows=args.max_windows,
                sigma=foveation_sigma,
                use_polarity=args.use_polarity,
                clear_existing=True,
                attention_params={
                    "num_pyr": args.num_pyr,
                    "center_sigma": args.center_sigma,
                    "surround_sigma": args.surround_sigma,
                    "beta": beta,
                    "seed": seed,
                },
                plot=make_gif,
                plot_gif_path=out_dir / "saliency_window_action.gif",
                plot_fps=args.gif_fps,
            )

            exploration = plot_attention_exploration(
                events_npy=events_npy,
                saccades_path=att["saccades_path"],
                output_dir=out_dir,
                resolution=resolution_hw,
                beta=beta,
                object_label=f"activeVision | beta={beta:g} | seed={seed}",
                per=args.grid_cell_fraction,
                noise_thresh=args.object_noise_thresh,
                exclude_initial_fixation=True,
                plot_trajectory=True,
                save_name="attention_exploration",
                save_png=make_gif,
                save_pdf=False,
                show=False,
            )

            if make_gif:
                foveations_to_gif(att["fov_dir"], out_dir / "foveations.gif", fps=args.gif_fps)

            recurrence = fixation_recurrence_metrics(
                saccades_path=att["saccades_path"],
                resolution=resolution_hw,
                cell_fraction=args.grid_cell_fraction,
                exclude_initial=True,
            )

            action = saccade_action_metrics(att["saccades_path"], exclude_initial=True)

            entropy = attention_entropy_metrics(
                events_npy=events_npy,
                beta=beta,
                resolution=resolution_hw,
                window_period_ms=args.window_period_ms,
                max_windows=args.max_windows,
                seed=seed,
                center_sigma=args.center_sigma,
                surround_sigma=args.surround_sigma,
                num_pyr=args.num_pyr,
                use_polarity=args.use_polarity,
            )

            row = {
                "beta": float(beta),
                "seed": int(seed),
                "num_foveations": int(att["num_foveations"]),
                "coverage": float(exploration["area_explored_coeff"]),
                "visited_obj_cells": int(exploration["visited_obj_cells"]),
                "total_obj_cells": int(exploration["total_obj_cells"]),
                "saccades_path": att["saccades_path"],
                "output_dir": str(out_dir),
                **recurrence,
                **action,
                **entropy,
            }
            rows.append(row)

            pd.DataFrame(rows).to_csv(output_root / "beta_sweep_metrics_partial.csv", index=False)

    metrics = pd.DataFrame(rows)
    metrics_path = output_root / "beta_sweep_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"[done] Metrics saved to {metrics_path}")

    metric_cols = [
        "coverage",
        "mean_norm_entropy",
        "mean_max_prob",
        "mean_effective_n_pixels",
        "revisit_fraction",
        "consecutive_same_cell_fraction",
        "mean_saccade_length_px",
    ]

    summary = mean_sem_summary(metrics, "beta", metric_cols)
    summary_path = output_root / "beta_sweep_summary_mean_sem.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[done] Summary saved to {summary_path}")

    save_summary_plot(
        summary,
        x="beta",
        y_mean="coverage_mean",
        y_sem="coverage_sem",
        ylabel="Object coverage",
        title="Effect of gain on object coverage",
        output_path=plots_dir / "coverage_vs_beta.png",
    )
    save_summary_plot(
        summary,
        x="beta",
        y_mean="mean_norm_entropy_mean",
        y_sem="mean_norm_entropy_sem",
        ylabel="Normalised policy entropy",
        title="Effect of gain on attentional entropy",
        output_path=plots_dir / "entropy_vs_beta.png",
    )
    save_summary_plot(
        summary,
        x="beta",
        y_mean="revisit_fraction_mean",
        y_sem="revisit_fraction_sem",
        ylabel="Fixation revisit fraction",
        title="Effect of gain on fixation recurrence",
        output_path=plots_dir / "recurrence_vs_beta.png",
    )
    save_summary_plot(
        summary,
        x="beta",
        y_mean="mean_saccade_length_px_mean",
        y_sem="mean_saccade_length_px_sem",
        ylabel="Mean saccade length (px)",
        title="Effect of gain on displacement length",
        output_path=plots_dir / "saccade_length_vs_beta.png",
    )

    print(f"[done] Plots saved to {plots_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
