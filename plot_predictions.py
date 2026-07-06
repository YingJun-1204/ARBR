import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure workspace is in Python path
sys.path.append(os.path.abspath("."))

from data_provider.data_factory import data_provider
from models import gs_linear

# Hyperparameters mapped from the shell scripts run_etth1_cas.sh, run_etth2_cas.sh, run_ettm1_cas.sh, run_ettm2_cas.sh
ALL_CONFIGS = {
    "ETTh1": {
        96: {
            "gs_dropout": 0.85, "head_dropout": 0.85, "gs_residual_weight": 0.4,
            "gate_beta": 0.5, "gate_lambda": 0.1, "gate_window_half": 2
        },
        192: {
            "gs_dropout": 0.9, "head_dropout": 0.9, "gs_residual_weight": 0.6,
            "gate_beta": 0.3, "gate_lambda": 0.08, "gate_window_half": 4
        },
        336: {
            "gs_dropout": 0.9, "head_dropout": 0.9, "gs_residual_weight": 0.55,
            "gate_beta": 0.4, "gate_lambda": 0.1, "gate_window_half": 4
        },
        720: {
            "gs_dropout": 0.9, "head_dropout": 0.9, "gs_residual_weight": 0.55,
            "gate_beta": 0.4, "gate_lambda": 0.1, "gate_window_half": 4
        }
    },
    "ETTh2": {
        96: {
            "gs_dropout": 0.9, "head_dropout": 0.9, "gs_residual_weight": 0.6,
            "gate_beta": 0.5, "gate_lambda": 0.01, "gate_window_half": 1
        },
        192: {
            "gs_dropout": 0.9, "head_dropout": 0.9, "gs_residual_weight": 0.35,
            "gate_beta": 0.25, "gate_lambda": 0.1, "gate_window_half": 4
        },
        336: {
            "gs_dropout": 0.9, "head_dropout": 0.9, "gs_residual_weight": 0.6,
            "gate_beta": 0.5, "gate_lambda": 0.1, "gate_window_half": 3
        },
        720: {
            "gs_dropout": 0.9, "head_dropout": 0.9, "gs_residual_weight": 0.6,
            "gate_beta": 0.2, "gate_lambda": 0.08, "gate_window_half": 2
        }
    },
    "ETTm1": {
        96: {
            "gs_dropout": 0.35, "head_dropout": 0.35, "gs_residual_weight": 0.25,
            "gate_beta": 0.15, "gate_lambda": 0.01, "gate_window_half": 3
        },
        192: {
            "gs_dropout": 0.4, "head_dropout": 0.4, "gs_residual_weight": 0.2,
            "gate_beta": 0.5, "gate_lambda": 0.03, "gate_window_half": 2
        },
        336: {
            "gs_dropout": 0.6, "head_dropout": 0.6, "gs_residual_weight": 0.4,
            "gate_beta": 0.1, "gate_lambda": 0.1, "gate_window_half": 2
        },
        720: {
            "gs_dropout": 0.75, "head_dropout": 0.75, "gs_residual_weight": 0.35,
            "gate_beta": 0.45, "gate_lambda": 0.05, "gate_window_half": 3
        }
    },
    "ETTm2": {
        96: {
            "gs_dropout": 0.9, "head_dropout": 0.9, "gs_residual_weight": 0.35,
            "gate_beta": 0.1, "gate_lambda": 0.02, "gate_window_half": 4
        },
        192: {
            "gs_dropout": 0.65, "head_dropout": 0.65, "gs_residual_weight": 0.2,
            "gate_beta": 0.2, "gate_lambda": 0.08, "gate_window_half": 2
        },
        336: {
            "gs_dropout": 0.3, "head_dropout": 0.3, "gs_residual_weight": 0.6,
            "gate_beta": 0.5, "gate_lambda": 0.05, "gate_window_half": 3
        },
        720: {
            "gs_dropout": 0.4, "head_dropout": 0.4, "gs_residual_weight": 0.45,
            "gate_beta": 0.5, "gate_lambda": 0.1, "gate_window_half": 4
        }
    }
}

def main():
    parser = argparse.ArgumentParser(description="Plot predictions using pre-trained checkpoint.")
    parser.add_argument("--dataset", type=str, default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
                        help="Dataset name (choices: ETTh1, ETTh2, ETTm1, ETTm2).")
    parser.add_argument("--pred_len", type=int, default=192, choices=[96, 192, 336, 720], 
                        help="Prediction horizon length to load and plot.")
    
    # Support running both inside scripts and interactive IDEs
    if len(sys.argv) > 1 and sys.argv[1] != "plot_predictions.py":
        args_parsed = parser.parse_args()
    else:
        # Default fallback for interactive runs
        class ArgsParsed:
            dataset = "ETTh1"
            pred_len = 192
        args_parsed = ArgsParsed()

    dataset = args_parsed.dataset
    pred_len = args_parsed.pred_len
    cfg = ALL_CONFIGS[dataset][pred_len]
    
    dataset_lower = dataset.lower()
    setting = f"long_term_forecast_{dataset_lower}_router_{pred_len}_SplatTS_{dataset}_ftM_sl512_pl{pred_len}_dm128_df256_Ablation_{dataset}_router_0"
    checkpoint_path = os.path.join("./checkpoints", setting, "checkpoint.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}\n"
                                f"Please make sure you have run the corresponding training script first (e.g. sh scripts/run_{dataset_lower}_cas.sh).")
        
    print(f"Loading pre-trained checkpoint from: {checkpoint_path}")
    
    # Recreate the args for loading the model and dataset
    class Args:
        pass

    args = Args()
    args.task_name = "long_term_forecast"
    args.model_id = f"{dataset_lower}_router_{pred_len}"
    args.model = "SplatTS"
    args.data = dataset
    args.root_path = "./data/"
    args.data_path = f"{dataset}.csv"
    args.features = "M"
    args.target = "OT"
    args.freq = "h" if "h" in dataset_lower else "t"
    args.checkpoints = "./checkpoints/"
    args.seq_len = 512
    args.pred_len = pred_len
    args.inverse = False
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.d_model = 128
    args.d_ff = 256
    args.dropout = 0.0
    args.activation = "gelu"
    args.output_attention = False
    args.representation = "gs"
    args.patch_len = 24
    args.stride = 12
    args.head_dropout = cfg["head_dropout"]
    args.head_dropout_position = "pre"
    args.head_mode = "linear"
    args.debug_mode = False
    args.gs_dropout = cfg["gs_dropout"]
    args.gs_weight_decay = 1e-4
    args.num_gaussians = 10
    args.density_mode = "cas"
    args.gs_lambda = 0.0
    args.gate_lambda = cfg["gate_lambda"]
    args.use_occlusion = False
    args.use_residual = True
    args.gs_residual_weight = cfg["gs_residual_weight"]
    args.gate_type = "adaptive_direction"
    args.gate_beta = cfg["gate_beta"]
    args.gate_window_half = cfg["gate_window_half"]
    args.no_save_checkpoint = False
    args.num_workers = 0
    args.train_epochs = 30
    args.batch_size = 256  # Use smaller batch size for evaluation
    args.patience = 6
    args.learning_rate = 0.002
    args.des = f"Ablation_{dataset}_router"
    args.loss = "MSE"
    args.lradj = "cosine"
    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = "0"
    args.p_hidden_dims = [128, 128]
    args.p_hidden_layers = 2
    args.augmentation_ratio = 0
    args.seed = 2
    args.jitter = False
    args.scaling = False
    args.permutation = False
    args.randompermutation = False
    args.magwarp = False
    args.timewarp = False
    args.windowslice = False
    args.windowwarp = False
    args.rotation = False
    args.spawner = False
    args.dtwwarp = False
    args.shapedtwwarp = False
    args.wdba = False
    args.discdtw = False
    args.discsdtw = False
    args.extra_tag = ""

    # 1. Load test data
    test_data, test_loader = data_provider(args, flag="test")
    print(f"Test dataset size: {len(test_data)}")
    
    # 2. Recreate model and load weights
    model = gs_linear.Model(args)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 3. Find the best sample with minimum MSE on target variable
    best_batch_x = None
    best_batch_y = None
    best_sample_in_batch = None
    min_mse = float('inf')
    
    print(f">Search running... Searching for the test sample with the minimum prediction error (MSE) for {dataset}...")
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x_dev = batch_x.float().to(device)
            outputs = model(batch_x_dev, is_training=False)
            outputs = outputs.detach().cpu()
            
            # Target future is the last pred_len steps
            batch_y_target = batch_y[:, -args.pred_len :, :]
            
            # Compute MSE on the target feature (-1) for each sample in the batch
            sample_mses = torch.mean((outputs[:, :, -1] - batch_y_target[:, :, -1]) ** 2, dim=1).numpy()
            
            batch_min_idx = np.argmin(sample_mses)
            batch_min_mse = sample_mses[batch_min_idx]
            
            if batch_min_mse < min_mse:
                min_mse = batch_min_mse
                best_batch_x = batch_x.clone()
                best_batch_y = batch_y.clone()
                best_sample_in_batch = batch_min_idx

    print(f"Found best sample with minimum prediction MSE: {min_mse:.6f}")
    
    # Slice only the best sample (batch size = 1)
    batch_x = best_batch_x[best_sample_in_batch : best_sample_in_batch + 1].float().to(device)
    batch_y = best_batch_y[best_sample_in_batch : best_sample_in_batch + 1].float().to(device)
    
    # 4. Perform forward and decompose the prediction on the best sample
    with torch.no_grad():
        B, T, C = batch_x.shape
        x_norm = model.norm(batch_x, "norm")
        x_repr_in = x_norm.permute(0, 2, 1) # (B, C, T)
        
        # Get SplattingResidualEncoder
        encoder = model.splatting_residual
        
        # Calculate CAS Gating
        gate_effective = None
        if encoder.density_mode == "cas" and encoder.cas_gating is not None:
            x_flat = x_repr_in.reshape(B * C, T)
            gate_effective, pi_i, gate_hard, p_i = encoder.cas_gating(x_flat, x_repr_in)
            
        # Get Gaussian splatting rendered event (GS trend)
        x_flat = x_repr_in.reshape(B * C, T)
        rendered_event, weights, mu, sigma, alpha_effective, last_density_score = encoder.gaussian_splatting(
            x_flat, encoder.patch_num, x_repr_in.device, gate_effective
        )
        
        # Get Residual projection component
        res_proj = encoder.patch_residual(x_repr_in, B * C)
        
        # Get Gating scale
        gate_scale, gate_direction, gate_strength, gate_route_probs = encoder.residual_router(
            weights, rendered_event, res_proj, B * C, encoder.patch_num, x_repr_in.device
        )
        
        # Active scaling for CAS mode
        if encoder.gate_type not in ["forward", "reverse", "adaptive_direction"] and encoder.density_mode == "cas":
            active_ratio = gate_effective.sum(dim=-1, keepdim=True) / encoder.num_gaussians
            scale = 1.5 - encoder.gamma_complement * active_ratio
            last_gate_scale = scale.unsqueeze(1).expand(-1, encoder.patch_num, 1)
        else:
            last_gate_scale = gate_scale
            
        # Decompose combined representation
        res_component = encoder.gs_residual_weight * last_gate_scale * res_proj
        
        # Map both components independently to output space using head_srs
        latent_gs = rendered_event.reshape(B, C, encoder.patch_num, encoder.d_model)
        latent_gs = latent_gs.permute(0, 1, 3, 2).reshape(-1, encoder.d_model, encoder.patch_num)
        
        latent_res = res_component.reshape(B, C, encoder.patch_num, encoder.d_model)
        latent_res = latent_res.permute(0, 1, 3, 2).reshape(-1, encoder.d_model, encoder.patch_num)
        
        # Pass to head
        pred_gs_norm = model.head_srs(latent_gs) # (B*C, pred_len)
        pred_res_norm = model.head_srs(latent_res) # (B*C, pred_len)
        
        # Reshape to (B, pred_len, C)
        pred_gs_norm = pred_gs_norm.reshape(B, C, args.pred_len).permute(0, 2, 1)
        pred_res_norm = pred_res_norm.reshape(B, C, args.pred_len).permute(0, 2, 1)
        
        # Denormalize using the RevIN module directly to avoid dimension mismatches
        pred_gs = model.norm(pred_gs_norm, "denorm")
        pred_combined = model.norm(pred_gs_norm + pred_res_norm, "denorm")
        pred_res = pred_combined - pred_gs
        
        # Ground truth
        batch_y_target = batch_y[:, -args.pred_len :, :]
        
    # Convert all tensors to numpy arrays
    inputs_np = batch_x.cpu().numpy()
    trues_np = batch_y_target.cpu().numpy()
    pred_gs_np = pred_gs.cpu().numpy()
    pred_res_np = pred_res.cpu().numpy()
    pred_combined_np = pred_combined.cpu().numpy()
    
    # 5. Generate the plot
    feature_idx = -1
    
    history = inputs_np[0, :, feature_idx]
    gt = trues_np[0, :, feature_idx]
    p_gs = pred_gs_np[0, :, feature_idx]
    p_res = pred_res_np[0, :, feature_idx]
    p_comb = pred_combined_np[0, :, feature_idx]
    
    # Setup professional plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=False)
    
    # ----------------------------------------------------
    # Plot 1: Prediction vs Ground Truth
    # ----------------------------------------------------
    history_len = args.seq_len
    
    x_hist = np.arange(history_len)
    x_pred = np.arange(history_len, history_len + pred_len)
    
    # Plot lookback window (last 128 points for visual clarity)
    plot_hist_len = 128
    ax1.plot(x_hist[-plot_hist_len:], history[-plot_hist_len:], color='#7f8c8d', label='Historical Lookback', linewidth=1.5, linestyle='--')
    ax1.plot(x_pred, gt, color='#2c3e50', label='Ground Truth (Real)', linewidth=2.0)
    ax1.plot(x_pred, p_comb, color='#e74c3c', label='SplatTS Prediction (Combined)', linewidth=2.0)
    
    ax1.axvline(x=history_len, color='#bdc3c7', linestyle=':', linewidth=1.5)
    ax1.set_title(f'A: Long-Term Forecasting Prediction vs. Ground Truth ({dataset}, Horizon={pred_len}, OT Target)', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    ax1.set_ylabel('Scaled Value', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # ----------------------------------------------------
    # Plot 2: SplatTS Decoupled Components (Trend vs. Correction)
    # ----------------------------------------------------
    ax2.plot(x_pred, gt, color='#2c3e50', label='Ground Truth (Real)', linewidth=1.5, alpha=0.7)
    ax2.plot(x_pred, p_gs, color='#3498db', label='1D Gaussian Splatting (Low-frequency Trend)', linewidth=2.0)
    ax2.bar(x_pred, p_res, color='#2ecc71', alpha=0.6, label='Adaptive Residual (High-frequency Correction)', width=1.0)
    ax2.plot(x_pred, p_comb, color='#e74c3c', label='SplatTS Reconstructed Prediction', linewidth=2.0, linestyle=':')
    
    ax2.axvline(x=history_len, color='#bdc3c7', linestyle=':', linewidth=1.5)
    ax2.set_title(f'B: Decoupled Multi-Scale Representation (Gaussian Trend + Local Patch Residual)', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none')
    ax2.set_xlabel('Time Steps', fontsize=11)
    ax2.set_ylabel('Scaled Value', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = f"forecasting_decomposition_{dataset_lower}_{pred_len}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated and saved plot to: {output_path}")

if __name__ == "__main__":
    main()
