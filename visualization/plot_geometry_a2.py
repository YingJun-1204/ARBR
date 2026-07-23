"""Plot Experiment 5.4 A1 from extracted GeoJet-TS geometry.

Default output is a support-strip visualization because it separates center,
width, saliency, and active count cleanly. Envelope mode remains available for
diagnostics, with the y-axis scaled by the largest individual response rather
than by an unplotted response sum.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


plt.rcParams.update(
    {
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.titleweight": "bold",
        "grid.color": "#d1d5db",
        "grid.linestyle": "--",
        "grid.alpha": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS_PRIMITIVE = (
    "#2563eb",
    "#d97706",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#0891b2",
    "#db2777",
    "#4b5563",
)


# -----------------------------------------------------------------------------
# Loading and common helpers
# -----------------------------------------------------------------------------

def _unwrap_object(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value


def load_selected_samples(
    raw_dir: str,
    dataset_name: str,
    pred_len: int,
) -> Optional[Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]]:
    path = Path(raw_dir) / f"{dataset_name}_H{pred_len}_selected_samples.npz"
    if not path.exists():
        print(f"[Warning] Selected-sample file not found: {path}")
        return None
    data = np.load(path, allow_pickle=True)
    selected = {
        "low": _unwrap_object(data["low"]),
        "mid": _unwrap_object(data["mid"]),
        "high": _unwrap_object(data["high"]),
    }
    metadata = _unwrap_object(data["metadata"]) if "metadata" in data.files else {}
    return selected, metadata


def _model_input(sample: Mapping[str, Any]) -> np.ndarray:
    if "input_model" in sample:
        return np.asarray(sample["input_model"], dtype=float)
    if "input_revin" in sample:
        return np.asarray(sample["input_revin"], dtype=float)
    raise KeyError("Sample has neither input_model nor input_revin.")


def _active_geometry(sample: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    x = _model_input(sample)
    length = len(x)
    mu = np.asarray(sample["mu"], dtype=float)
    sigma = np.asarray(sample["sigma"], dtype=float)
    effective_alpha = np.asarray(sample["effective_alpha"], dtype=float)
    hard_gate = np.asarray(sample["hard_gate"], dtype=float)
    active_indices = np.flatnonzero(hard_gate > 0.5)
    return {
        "x": x,
        "length": np.asarray(length),
        "active_indices": active_indices,
        "mu_idx": mu * (length - 1),
        "sigma_idx": sigma * (length - 1),
        "effective_alpha": effective_alpha,
    }


def _saliency_style(saliency: float) -> Tuple[float, float]:
    clipped = float(np.clip(saliency, 0.0, 1.0))
    marker_size = 38.0 + 145.0 * clipped
    alpha = 0.32 + 0.68 * clipped
    return marker_size, alpha


def _title(sample: Mapping[str, Any], dataset: str, complexity: str) -> str:
    channel = sample.get("channel_index", "?")
    window = sample.get("window_index", "?")
    return (
        f"{dataset} — {complexity} complexity "
        f"($\\kappa={float(sample['complexity_kappa']):.2f}$, "
        f"$K(x)={int(sample['active_count'])}$; window={window}, channel={channel})"
    )


# -----------------------------------------------------------------------------
# Panel drawing
# -----------------------------------------------------------------------------

def _plot_input(ax: plt.Axes, sample: Mapping[str, Any], dataset: str, complexity: str) -> None:
    x = _model_input(sample)
    time = np.arange(len(x))
    ax.plot(time, x, color="#111827", linewidth=1.15)
    ax.set_ylabel("RevIN-normalized input", fontsize=9)
    ax.set_title(_title(sample, dataset, complexity), fontsize=10.5, pad=7)
    ax.set_xlim(0, len(x) - 1)
    ax.grid(True, axis="both")
    ax.tick_params(labelsize=8.5)


def _plot_support_strip(
    ax: plt.Axes,
    sample: Mapping[str, Any],
    annotate_parameters: bool,
) -> None:
    geometry = _active_geometry(sample)
    length = int(geometry["length"])
    active_indices = geometry["active_indices"]
    mu_idx = geometry["mu_idx"]
    sigma_idx = geometry["sigma_idx"]
    effective_alpha = geometry["effective_alpha"]

    for row, primitive in enumerate(active_indices, start=1):
        center = float(mu_idx[primitive])
        sigma_samples = float(sigma_idx[primitive])
        saliency = float(effective_alpha[primitive])
        left_raw = center - 2.0 * sigma_samples
        right_raw = center + 2.0 * sigma_samples
        left = max(0.0, left_raw)
        right = min(length - 1.0, right_raw)
        color = COLORS_PRIMITIVE[int(primitive) % len(COLORS_PRIMITIVE)]
        marker_size, visual_alpha = _saliency_style(saliency)

        ax.hlines(
            row,
            left,
            right,
            color=color,
            linewidth=2.5,
            alpha=visual_alpha,
            zorder=2,
        )
        ax.scatter(
            [center],
            [row],
            color=color,
            s=marker_size,
            alpha=visual_alpha,
            edgecolors="white",
            linewidths=0.55,
            zorder=4,
        )
        # Arrowheads reveal that the true ±2σ interval extends outside the visible input.
        if left_raw < 0:
            ax.plot([0], [row], marker="<", markersize=5, color=color, alpha=visual_alpha)
        if right_raw > length - 1:
            ax.plot([length - 1], [row], marker=">", markersize=5, color=color, alpha=visual_alpha)

        if annotate_parameters:
            ax.text(
                length - 3,
                row,
                f"$k={int(primitive)}$   $\\mu={center:.0f}$   "
                f"$\\sigma={sigma_samples:.0f}$   "
                f"$\\tilde{{\\alpha}}={saliency:.2f}$",
                ha="right",
                va="center",
                fontsize=7.6,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.2},
                zorder=5,
            )

    ax.set_ylabel("Active primitives", fontsize=9)
    ax.set_xlabel("Input timestep $t$", fontsize=9)
    ax.set_yticks(np.arange(1, len(active_indices) + 1))
    ax.set_yticklabels([f"$k={int(index)}$" for index in active_indices], fontsize=8)
    ax.set_ylim(0.45, max(1.55, len(active_indices) + 0.55))
    ax.set_xlim(0, length - 1)
    ax.grid(True, axis="x")
    ax.tick_params(axis="x", labelsize=8.5)


def _plot_envelopes(
    ax: plt.Axes,
    sample: Mapping[str, Any],
    show_total: bool,
) -> None:
    geometry = _active_geometry(sample)
    length = int(geometry["length"])
    time = np.arange(length)
    active_indices = geometry["active_indices"]
    mu_idx = geometry["mu_idx"]
    sigma_idx = geometry["sigma_idx"]
    effective_alpha = geometry["effective_alpha"]

    total_response = np.zeros(length, dtype=float)
    maximum_individual = 0.0
    for primitive in active_indices:
        center = float(mu_idx[primitive])
        sigma_samples = max(float(sigma_idx[primitive]), 1e-8)
        saliency = float(effective_alpha[primitive])
        color = COLORS_PRIMITIVE[int(primitive) % len(COLORS_PRIMITIVE)]
        response = saliency * np.exp(-((time - center) ** 2) / (2.0 * sigma_samples**2))
        total_response += response
        maximum_individual = max(maximum_individual, float(np.max(response)))
        _, visual_alpha = _saliency_style(saliency)

        ax.plot(
            time,
            response,
            color=color,
            linewidth=1.45,
            alpha=visual_alpha,
            label=f"$k={int(primitive)}$, $\\tilde{{\\alpha}}={saliency:.2f}$",
        )
        ax.scatter([center], [saliency], color=color, s=28 + 65 * saliency, zorder=4)
        ax.hlines(
            saliency,
            max(0.0, center - 2.0 * sigma_samples),
            min(length - 1.0, center + 2.0 * sigma_samples),
            color=color,
            linewidth=1.7,
            alpha=visual_alpha,
        )

    if show_total and len(active_indices) > 0:
        ax.plot(time, total_response, color="#111827", linewidth=1.8, linestyle="--", label="Summed support mass")
        y_max = max(maximum_individual, float(np.max(total_response)))
    else:
        # Critical fix: do not scale individual curves by an unplotted sum.
        y_max = maximum_individual

    ax.set_ylim(-0.02, max(0.15, 1.15 * y_max))
    ax.set_xlim(0, length - 1)
    ax.set_ylabel("Scalar support $r_k(t)$", fontsize=9)
    ax.set_xlabel("Input timestep $t$", fontsize=9)
    ax.grid(True)
    ax.tick_params(labelsize=8.5)
    if len(active_indices) > 0:
        ax.legend(loc="upper right", fontsize=7.2, ncol=1, framealpha=0.9)


def plot_single_sample_panel(
    ax_top: plt.Axes,
    ax_bottom: plt.Axes,
    sample: Mapping[str, Any],
    dataset: str,
    complexity: str,
    mode: str,
    annotate_parameters: bool,
    show_total: bool,
) -> None:
    _plot_input(ax_top, sample, dataset, complexity)
    if mode == "strip":
        _plot_support_strip(ax_bottom, sample, annotate_parameters)
    elif mode == "envelope":
        _plot_envelopes(ax_bottom, sample, show_total)
    else:
        raise ValueError(f"Unsupported plotting mode: {mode}")


# -----------------------------------------------------------------------------
# Figure composition
# -----------------------------------------------------------------------------

def _add_strip_legend(fig: plt.Figure) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2563eb", markeredgecolor="white", markersize=8, label="Circle position: center $\\mu_k$"),
        Line2D([0], [0], color="#2563eb", linewidth=2.5, label="Horizontal span: $\\mu_k \\pm 2\\sigma_k$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2563eb", alpha=0.45, markersize=5, label="Marker size/opacity: effective saliency"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.003),
        ncol=3,
        fontsize=8.3,
        frameon=False,
    )


def plot_dataset_figure(
    dataset_name: str,
    pred_len: int,
    selected: Mapping[str, Mapping[str, Any]],
    save_dir: Path,
    mode: str,
    annotate_parameters: bool,
    show_total: bool,
) -> None:
    max_active = max(int(sample["active_count"]) for sample in selected.values())
    lower_ratio = max(0.65, 0.18 * max_active)
    fig = plt.figure(figsize=(13.2, 10.2 if mode == "strip" else 9.6))
    outer = gridspec.GridSpec(3, 1, hspace=0.38)

    levels = (("low", "Low"), ("mid", "Medium"), ("high", "High"))
    for row, (key, label) in enumerate(levels):
        sample = selected[key]
        sub = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer[row],
            height_ratios=[1.0, lower_ratio],
            hspace=0.08,
        )
        ax_top = fig.add_subplot(sub[0])
        ax_bottom = fig.add_subplot(sub[1], sharex=ax_top)
        plot_single_sample_panel(
            ax_top,
            ax_bottom,
            sample,
            dataset_name,
            label,
            mode,
            annotate_parameters,
            show_total,
        )
        plt.setp(ax_top.get_xticklabels(), visible=False)

    fig.suptitle(
        f"Adaptive Primitive Geometry (GeoJet-TS) — {dataset_name}, $H={pred_len}$",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    if mode == "strip":
        _add_strip_legend(fig)
        bottom_margin = 0.055
    else:
        bottom_margin = 0.02
    fig.subplots_adjust(top=0.945, bottom=bottom_margin)

    basename = f"{dataset_name}_H{pred_len}_geometry_a1_{mode}"
    fig.savefig(save_dir / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(save_dir / f"{basename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {save_dir / f'{basename}.png'}")


def _parse_overview_levels(text: str, number: int) -> Sequence[str]:
    levels = [item.strip().lower() for item in text.split(",") if item.strip()]
    valid = {"low", "mid", "high"}
    if not levels:
        levels = ["high"]
    if any(level not in valid for level in levels):
        raise ValueError(f"overview levels must be low/mid/high, got {levels}")
    if len(levels) < number:
        levels = [levels[index % len(levels)] for index in range(number)]
    return levels[:number]


def plot_overview(
    datasets_samples: Mapping[str, Mapping[str, Mapping[str, Any]]],
    pred_len: int,
    save_dir: Path,
    mode: str,
    overview_levels: str,
    annotate_parameters: bool,
    show_total: bool,
) -> None:
    items = list(datasets_samples.items())[:4]
    levels = _parse_overview_levels(overview_levels, len(items))
    fig = plt.figure(figsize=(15.5, 10.8))
    main = gridspec.GridSpec(2, 2, wspace=0.20, hspace=0.31)

    for index, ((dataset, selected), level) in enumerate(zip(items, levels)):
        sample = selected[level]
        sub = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=main[index],
            height_ratios=[1.0, max(0.8, 0.20 * int(sample["active_count"]))],
            hspace=0.08,
        )
        ax_top = fig.add_subplot(sub[0])
        ax_bottom = fig.add_subplot(sub[1], sharex=ax_top)
        plot_single_sample_panel(
            ax_top,
            ax_bottom,
            sample,
            dataset,
            level.capitalize(),
            mode,
            annotate_parameters=False if mode == "strip" else annotate_parameters,
            show_total=show_total,
        )
        plt.setp(ax_top.get_xticklabels(), visible=False)

    fig.suptitle(
        f"GeoJet-TS Adaptive Primitive Geometry — Experiment 5.4 A1 ($H={pred_len}$)",
        fontsize=15.5,
        fontweight="bold",
        y=0.995,
    )
    if mode == "strip":
        _add_strip_legend(fig)
        bottom_margin = 0.052
    else:
        bottom_margin = 0.02
    fig.subplots_adjust(top=0.945, bottom=bottom_margin)

    basename = f"geometry_a1_overview_H{pred_len}_{mode}"
    fig.savefig(save_dir / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(save_dir / f"{basename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {save_dir / f'{basename}.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GeoJet-TS Experiment 5.4 A1 geometry")
    parser.add_argument("--raw_dir", default="./visualization/outputs/geometry_a1/raw")
    parser.add_argument("--output_dir", default="./visualization/outputs/geometry_a1/figures")
    parser.add_argument("--datasets", default="ETTh1,ETTm1,weather,electricity")
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument(
        "--mode",
        choices=("strip", "envelope"),
        default="strip",
        help="strip is recommended for the paper; envelope is a diagnostic view.",
    )
    parser.add_argument(
        "--show_total",
        action="store_true",
        help="Envelope mode only: draw summed support mass and include it in y scaling.",
    )
    parser.add_argument(
        "--no_parameter_annotations",
        action="store_true",
        help="Hide per-primitive mu/sigma/saliency annotations in dataset figures.",
    )
    parser.add_argument(
        "--overview_levels",
        default="low,mid,high,high",
        help="Comma-separated level chosen for each dataset in the overview.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]

    datasets_samples: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for dataset in datasets:
        loaded = load_selected_samples(args.raw_dir, dataset, args.pred_len)
        if loaded is None:
            continue
        selected, metadata = loaded
        datasets_samples[dataset] = selected
        geometry_sources = metadata.get("geometry_sources") if isinstance(metadata, Mapping) else None
        if geometry_sources:
            print(f"[{dataset}] geometry source: {geometry_sources}")
        plot_dataset_figure(
            dataset,
            args.pred_len,
            selected,
            output_dir,
            args.mode,
            annotate_parameters=not args.no_parameter_annotations,
            show_total=args.show_total,
        )

    if datasets_samples:
        plot_overview(
            datasets_samples,
            args.pred_len,
            output_dir,
            args.mode,
            args.overview_levels,
            annotate_parameters=not args.no_parameter_annotations,
            show_total=args.show_total,
        )
        print(f"\n[Success] Figures saved under: {output_dir}")
    else:
        raise SystemExit("No selected-sample files were found. Run the extraction script first.")


if __name__ == "__main__":
    main()
