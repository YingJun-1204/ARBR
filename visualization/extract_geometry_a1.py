import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from data_provider.data_factory import data_provider
from models.gs_linear import Model


DATASET_CONFIGS = {
    "ETTh1": {"data": "ETTh1", "data_path": "ETTh1.csv", "enc_in": 7, "freq": "h"},
    "ETTh2": {"data": "ETTh2", "data_path": "ETTh2.csv", "enc_in": 7, "freq": "h"},
    "ETTm1": {"data": "ETTm1", "data_path": "ETTm1.csv", "enc_in": 7, "freq": "t"},
    "ETTm2": {"data": "ETTm2", "data_path": "ETTm2.csv", "enc_in": 7, "freq": "t"},
    "weather": {"data": "custom", "data_path": "weather.csv", "enc_in": 21, "freq": "h"},
    "electricity": {"data": "custom", "data_path": "electricity.csv", "enc_in": 321, "freq": "h"},
    "traffic": {"data": "custom", "data_path": "traffic.csv", "enc_in": 862, "freq": "h"},
}


def build_args(dataset_name, pred_len=96, seq_len=512, root_path="./data", checkpoints_dir="./checkpoints"):
    ds_cfg = DATASET_CONFIGS.get(dataset_name, {"data": "custom", "data_path": f"{dataset_name}.csv", "enc_in": 7, "freq": "h"})
    
    args = argparse.Namespace()
    args.task_name = "long_term_forecast"
    args.is_training = 0
    args.model_id = f"{dataset_name.lower()}_jet_{pred_len}"
    args.model = "SplatTS"
    args.data = ds_cfg.get("data", "custom")
    args.root_path = root_path
    args.data_path = ds_cfg["data_path"]
    args.features = "M"
    args.target = "OT"
    args.freq = ds_cfg["freq"]
    args.checkpoints = checkpoints_dir
    args.seq_len = seq_len
    args.pred_len = pred_len
    args.enc_in = ds_cfg["enc_in"]
    args.d_model = 128
    args.representation = "gs"
    args.patch_len = 24
    args.stride = 12
    args.head_dropout = 0.1
    args.head_dropout_position = "post"
    args.head_mode = "linear"
    args.gs_dropout = 0.3
    args.gs_weight_decay = 0.0001
    args.num_gaussians = 8
    args.density_mode = "cas"
    args.use_occlusion = False
    args.use_residual = True
    args.gs_residual_weight = 0.1
    args.k_base = -1
    args.output_dir = "loss_cas_simplify"
    args.ablation_mode = "none"
    args.num_implicit_gaussians = 4
    args.jet_max_shift_samples = 1.0
    args.jet_score_temperature = 0.01
    args.jet_density_tau = 1.0
    args.jet_detach_geometry = 1
    args.jet_scale_init = 0.1
    args.jet_sigma_init = 0.2
    args.jet_derivative_mode = "centered_legacy"
    args.fusion_mode = "geometry"
    args.fusion_hidden_dim = 16
    args.fusion_init = 0.5
    args.fusion_beta_max = 0.75
    args.fusion_detach_geometry = 1
    args.num_workers = 0
    args.batch_size = 64
    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.use_multi_gpu = False
    return args


def find_checkpoint_path(checkpoints_dir, dataset_name, pred_len):
    possible_names = [
        f"long_term_forecast_{dataset_name.lower()}_jet_{pred_len}_SplatTS",
        f"long_term_forecast_{dataset_name}_jet_{pred_len}_SplatTS",
    ]
    for name in possible_names:
        p = os.path.join(checkpoints_dir, name, "checkpoint.pth")
        if os.path.exists(p):
            return p
    
    # Fallback search by dataset substring and pred_len
    if os.path.exists(checkpoints_dir):
        for folder in os.listdir(checkpoints_dir):
            if dataset_name.lower() in folder.lower() and f"_{pred_len}_" in folder:
                p = os.path.join(checkpoints_dir, folder, "checkpoint.pth")
                if os.path.exists(p):
                    return p
    return None


@torch.no_grad()
def extract_dataset_geometry(dataset_name, pred_len, seq_len, root_path, checkpoints_dir, device, max_windows=200):
    print(f"\n[Extract] Processing dataset={dataset_name}, pred_len={pred_len}...")
    args = build_args(dataset_name, pred_len=pred_len, seq_len=seq_len, root_path=root_path, checkpoints_dir=checkpoints_dir)
    
    ckpt_path = find_checkpoint_path(checkpoints_dir, dataset_name, pred_len)
    if not ckpt_path:
        print(f"[Warning] Checkpoint for dataset {dataset_name} H={pred_len} not found in {checkpoints_dir}! Skipping.")
        return None

    print(f"  -> Found checkpoint: {ckpt_path}")
    model = Model(args).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_set, test_loader = data_provider(args, flag="test")
    
    splat_encoder = model.splatting_residual
    cas_gating = splat_encoder.cas_gating
    generator = splat_encoder.gaussian_splatting.generator
    num_gaussians = splat_encoder.gaussian_splatting.num_gaussians

    extracted_records = []
    window_count = 0

    for batch_idx, (batch_x, _, _, _) in enumerate(test_loader):
        if window_count >= max_windows:
            break
        
        batch_x = batch_x.float().to(device) # (B, T, C)
        B, T, C = batch_x.shape

        # RevIN normalization
        x_norm = model.norm(batch_x, "norm") # (B, T, C)
        x_repr_in = x_norm.permute(0, 2, 1) # (B, C, T)
        x_flat = x_repr_in.flatten(0, 1) # (B*C, T)

        # 1. CAS Gating & Complexity (curvature)
        gate_effective, pi_i = cas_gating(x_flat)
        gate_hard = (pi_i > cas_gating.tau).float() # (B*C, K)
        active_count = gate_hard.sum(dim=-1) # (B*C,)

        patch_len = cas_gating.patch_len if cas_gating.patch_len is not None else 16
        x_seq_temp = x_flat.unsqueeze(1)
        x_low = F.avg_pool1d(x_seq_temp, kernel_size=patch_len, stride=1, padding=patch_len // 2)
        if x_low.shape[-1] != x_seq_temp.shape[-1]:
            x_low = x_low[..., :x_seq_temp.shape[-1]]
        diff2 = x_low[..., 2:] - 2 * x_low[..., 1:-1] + x_low[..., :-2]
        curvature = torch.sum(torch.abs(diff2), dim=-1).squeeze(1) # (B*C,)

        # 2. Gaussian Generator parameters
        raw_params = generator(x_flat).view(B * C, num_gaussians, -1)
        mu = torch.sigmoid(raw_params[:, :, 0]) # (B*C, K)
        if hasattr(splat_encoder.gaussian_splatting, "raw_sigma"):
            sigma_shared = F.softplus(splat_encoder.gaussian_splatting.raw_sigma) + 1e-5
            sigma = sigma_shared.unsqueeze(0).expand(B * C, -1) # (B*C, K)
        else:
            sigma = F.softplus(raw_params[:, :, 1]) + 1e-5 # (B*C, K)
        alpha = torch.ones_like(mu)
        effective_alpha = gate_hard # (B*C, K)

        # Move to CPU numpy
        x_norm_np = x_norm.detach().cpu().numpy() # (B, T, C)
        x_raw_np = batch_x.detach().cpu().numpy() # (B, T, C)
        curvature_np = curvature.detach().cpu().numpy().reshape(B, C)
        mu_np = mu.detach().cpu().numpy().reshape(B, C, num_gaussians)
        sigma_np = sigma.detach().cpu().numpy().reshape(B, C, num_gaussians)
        alpha_np = alpha.detach().cpu().numpy().reshape(B, C, num_gaussians)
        gate_hard_np = gate_hard.detach().cpu().numpy().reshape(B, C, num_gaussians)
        pi_i_np = pi_i.detach().cpu().numpy().reshape(B, C, num_gaussians)
        effective_alpha_np = effective_alpha.detach().cpu().numpy().reshape(B, C, num_gaussians)
        active_count_np = active_count.detach().cpu().numpy().reshape(B, C)

        for b in range(B):
            win_idx = window_count + b
            for c in range(C):
                extracted_records.append({
                    "dataset": dataset_name,
                    "horizon": pred_len,
                    "window_index": win_idx,
                    "channel_index": c,
                    "input_revin": x_norm_np[b, :, c],
                    "input_raw": x_raw_np[b, :, c],
                    "complexity_kappa": float(curvature_np[b, c]),
                    "mu": mu_np[b, c],
                    "sigma": sigma_np[b, c],
                    "alpha": alpha_np[b, c],
                    "hard_gate": gate_hard_np[b, c],
                    "straight_through_gate": pi_i_np[b, c],
                    "effective_alpha": effective_alpha_np[b, c],
                    "active_count": int(active_count_np[b, c]),
                })
        
        window_count += B

    print(f"  -> Extracted total {len(extracted_records)} sample-channel sequences.")
    return extracted_records


def select_representative_samples(records):
    """
    Sort records by complexity_kappa and select representative samples:
    Low complexity (~10th percentile), Mid complexity (~50th percentile), High complexity (~90th percentile).
    """
    sorted_records = sorted(records, key=lambda r: r["complexity_kappa"])
    N = len(sorted_records)
    
    idx_low = int(N * 0.10)
    idx_mid = int(N * 0.50)
    idx_high = min(N - 1, int(N * 0.90))

    selected = {
        "low": sorted_records[idx_low],
        "mid": sorted_records[idx_mid],
        "high": sorted_records[idx_high],
    }
    return selected


def save_diagnostic_plot(sample_dict, save_path, title_prefix="Diagnostic"):
    """
    Generate a quick diagnostic plot for debugging internal primitive parameters.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    x_revin = sample_dict["input_revin"]
    L = len(x_revin)
    t_grid = np.arange(L)
    
    # Top Panel: Input sequence
    axes[0].plot(t_grid, x_revin, color="black", lw=1.2, label="RevIN Input")
    axes[0].set_ylabel("RevIN Value")
    axes[0].set_title(f"{title_prefix} - {sample_dict['dataset']} (H={sample_dict['horizon']}, Win={sample_dict['window_index']}, Ch={sample_dict['channel_index']}, Kappa={sample_dict['complexity_kappa']:.2f})")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right")

    # Bottom Panel: Primitive active supports
    mu = sample_dict["mu"]
    sigma = sample_dict["sigma"]
    effective_alpha = sample_dict["effective_alpha"]
    hard_gate = sample_dict["hard_gate"]
    active_count = sample_dict["active_count"]

    mu_idx = mu * (L - 1)
    sigma_idx = sigma * (L - 1)

    colors = plt.cm.tab10(np.linspace(0, 1, len(mu)))
    
    for k in range(len(mu)):
        if hard_gate[k] > 0.5:
            center = mu_idx[k]
            width = 2 * sigma_idx[k]
            eff_a = effective_alpha[k]
            
            axes[1].hlines(y=k+1, xmin=max(0, center - width), xmax=min(L-1, center + width), color=colors[k], lw=2.5, alpha=max(0.3, eff_a))
            axes[1].scatter([center], [k+1], color=colors[k], s=40 + 100 * eff_a, zorder=5)
            axes[1].text(center, k+1 + 0.15, f"k={k}\nα={eff_a:.2f}", fontsize=8, ha="center", color=colors[k])

    axes[1].set_xlabel("Input Timestep (t)")
    axes[1].set_ylabel("Active Primitive k")
    axes[1].set_yticks(np.arange(1, len(mu) + 1))
    axes[1].set_ylim(0.5, len(mu) + 0.8)
    axes[1].set_title(f"Active Gaussian Supports (Count={active_count})")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Extract GeoJet-TS Primitive Geometry (A1)")
    parser.add_argument("--datasets", type=str, default="ETTh1,ETTm1,weather,electricity", help="Comma-separated dataset names")
    parser.add_argument("--pred_len", type=int, default=96, help="Prediction horizon length")
    parser.add_argument("--seq_len", type=int, default=512, help="Input sequence length")
    parser.add_argument("--root_path", type=str, default="./data", help="Data root path")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints", help="Checkpoints directory")
    parser.add_argument("--output_dir", type=str, default="./visualization/outputs/geometry_a1", help="Output directory")
    parser.add_argument("--max_windows", type=int, default=200, help="Max test windows to evaluate per dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_output_dir = os.path.join(args.output_dir, "raw")
    diag_output_dir = os.path.join(args.output_dir, "diagnostics")
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(diag_output_dir, exist_ok=True)

    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    manifest = {
        "pred_len": args.pred_len,
        "seq_len": args.seq_len,
        "datasets": {},
    }

    for ds in dataset_list:
        records = extract_dataset_geometry(
            dataset_name=ds,
            pred_len=args.pred_len,
            seq_len=args.seq_len,
            root_path=args.root_path,
            checkpoints_dir=args.checkpoints,
            device=device,
            max_windows=args.max_windows
        )

        if not records:
            continue

        all_save_path = os.path.join(raw_output_dir, f"{ds}_H{args.pred_len}_all_samples.npz")
        np.savez_compressed(all_save_path, records=records)

        selected = select_representative_samples(records)
        sel_save_path = os.path.join(raw_output_dir, f"{ds}_H{args.pred_len}_selected_samples.npz")
        np.savez_compressed(sel_save_path, low=selected["low"], mid=selected["mid"], high=selected["high"])

        for comp_level, sample in selected.items():
            diag_path = os.path.join(diag_output_dir, f"{ds}_H{args.pred_len}_{comp_level}.png")
            save_diagnostic_plot(sample, diag_path, title_prefix=f"{ds} ({comp_level.upper()})")

        manifest["datasets"][ds] = {
            "total_sequences": len(records),
            "all_samples_npz": all_save_path,
            "selected_samples_npz": sel_save_path,
            "selected_kappa": {
                "low": selected["low"]["complexity_kappa"],
                "mid": selected["mid"]["complexity_kappa"],
                "high": selected["high"]["complexity_kappa"],
            }
        }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    print(f"\n[Success] Extraction complete! Raw data and diagnostics saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
