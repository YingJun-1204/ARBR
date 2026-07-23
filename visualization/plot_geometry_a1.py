import argparse
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Configure publication style fonts and parameters
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.color"] = "#e5e7eb"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.6


COLORS_PRIMITIVE = [
    "#2563eb",  # Blue
    "#d97706",  # Amber
    "#16a34a",  # Green
    "#dc2626",  # Red
    "#9333ea",  # Purple
    "#0891b2",  # Cyan
    "#db2777",  # Pink
    "#4b5563",  # Slate
]


def load_selected_samples(raw_dir, dataset_name, pred_len):
    sel_path = os.path.join(raw_dir, f"{dataset_name}_H{pred_len}_selected_samples.npz")
    if not os.path.exists(sel_path):
        print(f"[Warning] Selected samples file not found: {sel_path}")
        return None
    data = np.load(sel_path, allow_pickle=True)
    return {
        "low": data["low"].item(),
        "mid": data["mid"].item(),
        "high": data["high"].item(),
    }


def plot_single_sample_panel(ax_top, ax_bot, sample_dict, dataset_label, complexity_label, show_envelope=True):
    x_revin = sample_dict["input_revin"]
    L = len(x_revin)
    t_grid = np.arange(L)
    
    mu = sample_dict["mu"]
    sigma = sample_dict["sigma"]
    effective_alpha = sample_dict["effective_alpha"]
    hard_gate = sample_dict["hard_gate"]
    active_count = sample_dict["active_count"]
    kappa = sample_dict["complexity_kappa"]

    mu_idx = mu * (L - 1)
    sigma_idx = sigma * (L - 1)

    # 1. Top Panel: RevIN Input Sequence
    ax_top.plot(t_grid, x_revin, color="#111827", lw=1.3, label="RevIN-normalized Input")
    ax_top.set_ylabel("RevIN Value", fontsize=10, fontweight="medium")
    ax_top.set_title(
        f"{dataset_label} ({complexity_label} Complexity, $\kappa={kappa:.2f}$) | Active Primitives $K(x)={active_count}$",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_top.grid(True)
    ax_top.tick_params(labelsize=9)
    ax_top.set_xlim(0, L - 1)

    # 2. Bottom Panel: Active Gaussian Primitive Supports & Envelopes
    active_indices = [k for k in range(len(mu)) if hard_gate[k] > 0.5]
    
    if show_envelope:
        # Plot scalar support envelopes: r_k(t) = alpha_tilde * exp(-(t - mu)^2 / (2 * sigma^2))
        total_response = np.zeros(L)
        for idx_k, k in enumerate(active_indices):
            c_color = COLORS_PRIMITIVE[k % len(COLORS_PRIMITIVE)]
            center = mu_idx[k]
            width = sigma_idx[k]
            eff_a = effective_alpha[k]

            r_k = eff_a * np.exp(-((t_grid - center) ** 2) / (2 * (width ** 2) + 1e-5))
            total_response += r_k

            ax_bot.plot(t_grid, r_k, color=c_color, lw=1.5, alpha=0.8, label=f"Primitive k={k} ($\widetilde{{\\alpha}}={eff_a:.2f}$)")
            ax_bot.fill_between(t_grid, 0, r_k, color=c_color, alpha=0.12)
            
            # Mark center and support bar at top of response peak
            peak_val = max(0.05, eff_a)
            ax_bot.scatter([center], [peak_val], color=c_color, s=30 + 70 * eff_a, zorder=5)
            ax_bot.hlines(y=peak_val, xmin=max(0, center - 2 * width), xmax=min(L - 1, center + 2 * width), color=c_color, lw=2.0, alpha=0.7)

        ax_bot.set_ylabel("Support Response $r_k(t)$", fontsize=10, fontweight="medium")
        ax_bot.set_ylim(-0.05, max(1.05, float(np.max(total_response) + 0.1) if len(active_indices) > 0 else 1.05))
    else:
        # Plot horizontal support bars
        for idx_k, k in enumerate(active_indices):
            c_color = COLORS_PRIMITIVE[k % len(COLORS_PRIMITIVE)]
            center = mu_idx[k]
            width = 2 * sigma_idx[k]
            eff_a = effective_alpha[k]
            y_val = idx_k + 1

            ax_bot.hlines(y=y_val, xmin=max(0, center - width), xmax=min(L - 1, center + width), color=c_color, lw=3.0, alpha=max(0.3, eff_a))
            ax_bot.scatter([center], [y_val], color=c_color, s=40 + 80 * eff_a, zorder=5)
            ax_bot.text(center, y_val + 0.18, f"k={k} ($\widetilde{{\\alpha}}={eff_a:.2f}$)", fontsize=8, ha="center", color=c_color, fontweight="bold")

        ax_bot.set_ylabel("Active Primitive $k$", fontsize=10, fontweight="medium")
        ax_bot.set_yticks(np.arange(1, len(active_indices) + 1))
        ax_bot.set_yticklabels([f"k={k}" for k in active_indices])
        ax_bot.set_ylim(0.4, len(active_indices) + 0.8 if len(active_indices) > 0 else 1.5)

    ax_bot.set_xlabel("Input Timestep ($t$)", fontsize=10, fontweight="medium")
    ax_bot.grid(True)
    ax_bot.tick_params(labelsize=9)
    ax_bot.set_xlim(0, L - 1)
    
    if len(active_indices) > 0:
        ax_bot.legend(loc="upper right", fontsize=8, framealpha=0.9)


def plot_dataset_figure(dataset_name, pred_len, selected_samples, save_dir, show_envelope=True):
    """
    Generate publication-quality 3-panel figure for a dataset (Low, Mid, High complexity samples).
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.35)

    complexity_levels = [("low", "Low"), ("mid", "Medium"), ("high", "High")]

    for i, (key, comp_title) in enumerate(complexity_levels):
        sample = selected_samples[key]
        sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i], height_ratios=[1, 1], hspace=0.1)
        ax_top = fig.add_subplot(sub_gs[0])
        ax_bot = fig.add_subplot(sub_gs[1], sharex=ax_top)

        plot_single_sample_panel(ax_top, ax_bot, sample, dataset_label=dataset_name, complexity_label=comp_title, show_envelope=show_envelope)
        plt.setp(ax_top.get_xticklabels(), visible=False)

    plt.suptitle(f"Adaptive Primitive Geometry (GeoJet-TS) - Dataset: {dataset_name} (H={pred_len})", fontsize=14, fontweight="bold", y=0.99)

    pdf_path = os.path.join(save_dir, f"{dataset_name}_H{pred_len}_geometry_a1.pdf")
    png_path = os.path.join(save_dir, f"{dataset_name}_H{pred_len}_geometry_a1.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved dataset figure: {png_path}")


def plot_overview_combined(datasets_samples, pred_len, save_dir, show_envelope=True):
    """
    Generate paper-ready main overview figure combining representative samples across 4 datasets.
    """
    fig = plt.figure(figsize=(15, 11))
    
    # 2x2 grid of 2-panel subplots
    gs_main = gridspec.GridSpec(2, 2, wspace=0.22, hspace=0.32)

    ds_items = list(datasets_samples.items())[:4]
    
    for idx, (ds_name, sel_dict) in enumerate(ds_items):
        # Pick low complexity for ds 0 & 2, high complexity for ds 1 & 3 for clear contrast
        comp_key, comp_title = ("high" if idx % 2 == 1 else "low", "High" if idx % 2 == 1 else "Low")
        sample = sel_dict[comp_key]

        sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[idx], height_ratios=[1, 1.1], hspace=0.1)
        ax_top = fig.add_subplot(sub_gs[0])
        ax_bot = fig.add_subplot(sub_gs[1], sharex=ax_top)

        plot_single_sample_panel(ax_top, ax_bot, sample, dataset_label=ds_name, complexity_label=comp_title, show_envelope=show_envelope)
        plt.setp(ax_top.get_xticklabels(), visible=False)

    plt.suptitle(f"GeoJet-TS Adaptive Primitive Geometry (Experiment 5.4 A1, Horizon H={pred_len})", fontsize=15, fontweight="bold", y=0.99)

    pdf_path = os.path.join(save_dir, f"geometry_a1_overview_H{pred_len}.pdf")
    png_path = os.path.join(save_dir, f"geometry_a1_overview_H{pred_len}.png")

    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved multi-dataset overview figure: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot GeoJet-TS Experiment 5.4 A1 Geometry Figures")
    parser.add_argument("--raw_dir", type=str, default="./visualization/outputs/geometry_a1/raw", help="Path to raw extracted npz directory")
    parser.add_argument("--output_dir", type=str, default="./visualization/outputs/geometry_a1/figures", help="Path to save output figures")
    parser.add_argument("--datasets", type=str, default="ETTh1,ETTm1,weather,electricity", help="Comma-separated dataset names")
    parser.add_argument("--pred_len", type=int, default=96, help="Prediction horizon length")
    parser.add_argument("--no_envelope", action="store_true", default=False, help="Disable scalar support envelope responses and plot horizontal bars instead")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]

    datasets_samples = {}

    for ds in dataset_list:
        sel = load_selected_samples(args.raw_dir, ds, args.pred_len)
        if sel is not None:
            datasets_samples[ds] = sel
            plot_dataset_figure(ds, args.pred_len, sel, args.output_dir, show_envelope=(not args.no_envelope))

    if len(datasets_samples) > 0:
        plot_overview_combined(datasets_samples, args.pred_len, args.output_dir, show_envelope=(not args.no_envelope))
        print(f"\n[Success] All publication figures saved under: {args.output_dir}")
    else:
        print("[Error] No valid selected sample npz files found to plot. Run extract_geometry_a1.py first!")


if __name__ == "__main__":
    main()
