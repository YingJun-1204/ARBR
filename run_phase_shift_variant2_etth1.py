import argparse
import os
import random
import numpy as np
import torch
import types

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.variant_configs import apply_variant_configs

def get_parser():
    parser = argparse.ArgumentParser(description="Phase Shift Equivariance & Accuracy Evaluation (Variant 2) for ETTh1")

    # basic config
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="etth1_router_96")
    parser.add_argument("--model", type=str, default="SplatTS")

    # data loader
    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./data/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--inverse", action="store_true", default=False)
    
    # model define
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.85)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true")

    # representation model
    parser.add_argument("--representation", type=str, default="gs", choices=["gs", "patch_linear"])
    parser.add_argument("--patch_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--head_dropout_position", type=str, default="post", choices=["none", "pre", "post"])
    parser.add_argument("--head_mode", type=str, default="linear")
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--gs_dropout", type=float, default=0.3)
    parser.add_argument("--gs_weight_decay", type=float, default=0.0001)

    # model constraints
    parser.add_argument("--num_gaussians", type=int, default=10)
    parser.add_argument("--density_mode", type=str, default="cas", choices=["none", "cas"])
    parser.add_argument("--gs_lambda", type=float, default=0.1)
    parser.add_argument("--gate_lambda", type=float, default=0.1)
    parser.add_argument("--use_occlusion", action="store_true", default=False)
    parser.add_argument("--use_residual", action="store_true", default=True)
    parser.add_argument("--gs_residual_weight", type=float, default=0.4)
    parser.add_argument("--gate_type", type=str, default="adaptive_direction")
    parser.add_argument("--gate_beta", type=float, default=0.5)
    parser.add_argument("--gate_window_half", type=int, default=2)
    parser.add_argument("--k_base", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="loss_cas_simplify")

    # optimization / execution
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.0008)
    parser.add_argument("--des", type=str, default="Ablation_ETTh1_router")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="cosine")

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    # de-stationary projector params
    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)

    # augmentation
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--jitter", default=False, action="store_true")
    parser.add_argument("--scaling", default=False, action="store_true")
    parser.add_argument("--permutation", default=False, action="store_true")
    parser.add_argument("--randompermutation", default=False, action="store_true")
    parser.add_argument("--magwarp", default=False, action="store_true")
    parser.add_argument("--timewarp", default=False, action="store_true")
    parser.add_argument("--windowslice", default=False, action="store_true")
    parser.add_argument("--windowwarp", default=False, action="store_true")
    parser.add_argument("--rotation", default=False, action="store_true")
    parser.add_argument("--spawner", default=False, action="store_true")
    parser.add_argument("--dtwwarp", default=False, action="store_true")
    parser.add_argument("--shapedtwwarp", default=False, action="store_true")
    parser.add_argument("--wdba", default=False, action="store_true")
    parser.add_argument("--discdtw", default=False, action="store_true")
    parser.add_argument("--discsdtw", default=False, action="store_true")
    parser.add_argument("--extra_tag", type=str, default="")

    # evaluation specific
    parser.add_argument("--shifts", type=str, default="0,1,2,4,8,16", help="Comma-separated shift steps to test")
    
    return parser

def evaluate_shift_metrics(model, test_data, seq_len, pred_len, delta_t, batch_size, device, features):
    model.eval()
    
    total_len = len(test_data)
    max_index = total_len - delta_t
    
    equiv_errors = []
    forecast_mses = []
    forecast_maes = []
    f_dim = -1 if features == "MS" else 0
    
    with torch.no_grad():
        for i in range(0, max_index, batch_size):
            batch_x_0_list = []
            batch_x_dt_list = []
            batch_y_dt_list = []
            
            for j in range(i, min(i + batch_size, max_index)):
                # Get unshifted input
                seq_x_0, _, _, _ = test_data[j]
                # Get shifted input & target
                seq_x_dt, seq_y_dt, _, _ = test_data[j + delta_t]
                
                batch_x_0_list.append(seq_x_0)
                batch_x_dt_list.append(seq_x_dt)
                batch_y_dt_list.append(seq_y_dt)
            
            if not batch_x_0_list:
                break
                
            batch_x_0 = torch.tensor(np.array(batch_x_0_list), dtype=torch.float32).to(device)
            batch_x_dt = torch.tensor(np.array(batch_x_dt_list), dtype=torch.float32).to(device)
            batch_y_dt = torch.tensor(np.array(batch_y_dt_list), dtype=torch.float32).to(device)
            
            # Predict original (0) and shifted (delta_t)
            outputs_0 = model(batch_x_0, is_training=False)[:, -pred_len:, f_dim:]
            outputs_dt = model(batch_x_dt, is_training=False)[:, -pred_len:, f_dim:]
            batch_y_dt = batch_y_dt[:, -pred_len:, f_dim:]
            
            overlap_len = pred_len - delta_t
            if overlap_len > 0:
                # Align overlap slices in physical time
                slice_0 = outputs_0[:, delta_t :, :]
                slice_dt = outputs_dt[:, :overlap_len, :]
                
                # Calculate MSE of predictions over the overlap region
                batch_equiv = torch.mean((slice_0 - slice_dt) ** 2, dim=[1, 2])
                equiv_errors.extend(batch_equiv.cpu().numpy())
                
            # Calculate forecasting MSE and MAE on the shifted window
            batch_forecast_mse = torch.mean((outputs_dt - batch_y_dt) ** 2, dim=[1, 2])
            batch_forecast_mae = torch.mean(torch.abs(outputs_dt - batch_y_dt), dim=[1, 2])
            forecast_mses.extend(batch_forecast_mse.cpu().numpy())
            forecast_maes.extend(batch_forecast_mae.cpu().numpy())
               
    return (
        np.mean(equiv_errors) if equiv_errors else 0.0,
        np.mean(forecast_mses) if forecast_mses else 0.0,
        np.mean(forecast_maes) if forecast_maes else 0.0,
    )

def format_accuracy_cell(idx, delta_t, mode, results, baseline_idx):
    current_mse = results[mode]["mse"][idx]
    current_mae = results[mode]["mae"][idx]
    
    if np.isnan(current_mse) or np.isnan(current_mae):
        return "N/A"
        
    if delta_t == 0 or baseline_idx is None:
        return f"{current_mse:.6f} / {current_mae:.6f}"
        
    baseline_mse = results[mode]["mse"][baseline_idx]
    baseline_mae = results[mode]["mae"][baseline_idx]
    
    diff_mse = current_mse - baseline_mse
    diff_mae = current_mae - baseline_mae
    
    deg_mse_pct = (diff_mse / (baseline_mse + 1e-9)) * 100
    deg_mae_pct = (diff_mae / (baseline_mae + 1e-9)) * 100
    
    sign_mse = "+" if diff_mse >= 0 else ""
    sign_mae = "+" if diff_mae >= 0 else ""
    
    return f"{current_mse:.6f} / {current_mae:.6f} ({sign_mse}{diff_mse:.6f} / {sign_mae}{diff_mae:.6f} | {sign_mse}{deg_mse_pct:.2f}% / {sign_mae}{deg_mae_pct:.2f}%)"

def main():
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    if args.gate_type == "none":
        args.gate_type = None
    args = apply_variant_configs(args)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    print(f"Using device: {device}")

    # Reconstruct the settings path
    ii = 0
    setting = "{}_{}_{}_{}_ft{}_sl{}_pl{}_dm{}_df{}_{}_{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.d_ff,
        args.des,
        ii,
    )

    # Initialize experiment
    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    else:
        Exp = Exp_Long_Term_Forecast

    exp = Exp(args)
    
    # Load model state dict
    checkpoint_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Adapt old keys to GMLP naming schema if needed
    adapted_state_dict = {}
    for k, v in state_dict.items():
        if "patch_residual.projection.weight" in k:
            adapted_key = k.replace("patch_residual.projection.weight", "patch_residual.base_projection.weight")
            adapted_state_dict[adapted_key] = v
        else:
            adapted_state_dict[k] = v
            
    # Load state dict with strict=False to allow default initialization of GMLP specific parameters
    exp.model.load_state_dict(adapted_state_dict, strict=False)
    exp.model.to(device)
    
    # Load test dataset
    test_data, _ = exp._get_data(flag="test")
    print(f"Loaded test dataset: {args.data} (Size: {len(test_data)})")

    # Shift steps
    shift_steps = [int(x.strip()) for x in args.shifts.split(",")]
    
    print("\n" + "=" * 60)
    print(f" Evaluating Shift Equivariance & Accuracy for {args.model} (ETTh1) ")
    print("=" * 60)
    
    # Check if the representation has the required modules for monkeypatching
    rep = exp.model.representation
    has_residual = hasattr(rep, "use_residual")
    
    results = {}
    modes = ["Full Model", "Gaussian Only", "Residual Only"]
    for mode in modes:
        results[mode] = {"equiv": [], "mse": [], "mae": []}
    
    # Define custom patch methods for testing pathways
    original_forward = rep.forward
    
    # Define monkeypatched forwards
    def forward_gaussian_only(self, x_seq, patch_num, is_flat=False):
        old_use_residual = self.use_residual
        self.use_residual = False
        out = original_forward(x_seq, patch_num, is_flat)
        self.use_residual = old_use_residual
        return out
        
    def forward_residual_only(self, x_seq, patch_num, is_flat=False):
        # Run original forward to populate variables (like self.last_gate_scale, self.last_gate_direction)
        _ = original_forward(x_seq, patch_num, is_flat)
        if not is_flat:
            B, C, L = x_seq.shape
            batch_channel = B * C
        else:
            batch_channel = x_seq.shape[0]
        res_proj = self.patch_residual(x_seq, batch_channel, gate_direction=self.last_gate_direction)
        return self.gs_residual_weight * self.last_gate_scale * res_proj

    # 1. Evaluate Full Model
    print("Evaluating Mode: Full Model...")
    for delta_t in shift_steps:
        equiv, mse, mae = evaluate_shift_metrics(
            exp.model, test_data, args.seq_len, args.pred_len,
            delta_t, args.batch_size, device, args.features
        )
        results["Full Model"]["equiv"].append(equiv)
        results["Full Model"]["mse"].append(mse)
        results["Full Model"]["mae"].append(mae)
        
    # 2. Evaluate Gaussian Only
    if has_residual:
        print("Evaluating Mode: Gaussian Only...")
        rep.forward = types.MethodType(forward_gaussian_only, rep)
        for delta_t in shift_steps:
            equiv, mse, mae = evaluate_shift_metrics(
                exp.model, test_data, args.seq_len, args.pred_len,
                delta_t, args.batch_size, device, args.features
            )
            results["Gaussian Only"]["equiv"].append(equiv)
            results["Gaussian Only"]["mse"].append(mse)
            results["Gaussian Only"]["mae"].append(mae)
        # Restore
        rep.forward = original_forward
    else:
        print("Gaussian Only evaluation skipped")
        for k in ["equiv", "mse", "mae"]:
            results["Gaussian Only"][k] = [float('nan')] * len(shift_steps)

    # 3. Evaluate Residual Only
    if has_residual and rep.use_residual:
        print("Evaluating Mode: Residual Only...")
        rep.forward = types.MethodType(forward_residual_only, rep)
        for delta_t in shift_steps:
            equiv, mse, mae = evaluate_shift_metrics(
                exp.model, test_data, args.seq_len, args.pred_len,
                delta_t, args.batch_size, device, args.features
            )
            results["Residual Only"]["equiv"].append(equiv)
            results["Residual Only"]["mse"].append(mse)
            results["Residual Only"]["mae"].append(mae)
        # Restore
        rep.forward = original_forward
    else:
        print("Residual Only evaluation skipped")
        for k in ["equiv", "mse", "mae"]:
            results["Residual Only"][k] = [float('nan')] * len(shift_steps)

    # Find index of baseline (0)
    try:
        baseline_idx = shift_steps.index(0)
    except ValueError:
        baseline_idx = None

    # Print markdown table 1: Equivariance Error
    print("\n### 1. Shift Equivariance Error (Variant 2) Results (ETTh1)\n")
    print("| Shift Step ($\\Delta t$) | Full Model Equiv Error | Gaussian-Only Equiv Error | Residual-Only Equiv Error |")
    print("|---|---|---|---|")
    for idx, delta_t in enumerate(shift_steps):
        f_err = results["Full Model"]["equiv"][idx]
        g_err = results["Gaussian Only"]["equiv"][idx]
        r_err = results["Residual Only"]["equiv"][idx]
        
        f_str = f"{f_err:.6f}" if not np.isnan(f_err) else "N/A"
        g_str = f"{g_err:.6f}" if not np.isnan(g_err) else "N/A"
        r_str = f"{r_err:.6f}" if not np.isnan(r_err) else "N/A"
        
        print(f"| {delta_t} | {f_str} | {g_str} | {r_str} |")
    print()

    # Print markdown table 2: Forecasting MSE/MAE
    print("\n### 2. Forecasting Accuracy (MSE / MAE) under Shift (ETTh1)\n")
    print("| Shift Step ($\\Delta t$) | Full Model MSE/MAE | Gaussian-Only MSE/MAE | Residual-Only MSE/MAE |")
    print("|---|---|---|---|")
    for idx, delta_t in enumerate(shift_steps):
        f_str = format_accuracy_cell(idx, delta_t, "Full Model", results, baseline_idx)
        g_str = format_accuracy_cell(idx, delta_t, "Gaussian Only", results, baseline_idx)
        r_str = format_accuracy_cell(idx, delta_t, "Residual Only", results, baseline_idx)
        
        print(f"| {delta_t} | {f_str} | {g_str} | {r_str} |")
    print()

if __name__ == "__main__":
    main()
