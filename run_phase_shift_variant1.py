import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.variant_configs import apply_variant_configs

def get_parser():
    parser = argparse.ArgumentParser(description="Phase Shift Robustness Evaluation (Variant 1)")

    # basic config
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="test")
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
    parser.add_argument("--dropout", type=float, default=0.1)
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

    parser.add_argument("--num_gaussians", type=int, default=5)
    parser.add_argument("--density_mode", type=str, default="none", choices=["none", "soft", "sparse", "cas"])
    parser.add_argument("--gs_lambda", type=float, default=0.0)
    parser.add_argument("--gate_lambda", type=float, default=0.0)
    parser.add_argument("--use_occlusion", action="store_true", default=False)
    parser.add_argument("--use_residual", action="store_true", default=True)
    parser.add_argument("--gs_residual_weight", type=float, default=0.1)
    parser.add_argument("--gate_type", type=str, default=None)
    parser.add_argument("--gate_beta", type=float, default=0.25)
    parser.add_argument("--gate_window_half", type=int, default=2)
    parser.add_argument("--k_base", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="loss_cas_simplify")

    # optimization / execution
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--des", type=str, default="test")
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
    parser.add_argument("--shifts", type=str, default="0,1,2,3,4,5,6,8,10,12", help="Comma-separated shift steps to test")
    
    return parser

def evaluate_shift(model, test_data, seq_len, pred_len, delta_t, batch_size, device, features):
    model.eval()
    
    total_len = len(test_data)
    max_index = total_len - delta_t
    
    preds = []
    trues = []
    
    f_dim = -1 if features == "MS" else 0
    
    with torch.no_grad():
        for i in range(0, max_index, batch_size):
            batch_x_list = []
            batch_y_list = []
            
            for j in range(i, min(i + batch_size, max_index)):
                # Get unshifted markers from index j (dummy for SplatTS, but matches data flow)
                _, _, _, _ = test_data[j]
                # Get shifted values from index j + delta_t
                seq_x, seq_y, _, _ = test_data[j + delta_t]
                
                batch_x_list.append(seq_x)
                batch_y_list.append(seq_y)
            
            if not batch_x_list:
                break
                
            batch_x = torch.tensor(np.array(batch_x_list), dtype=torch.float32).to(device)
            batch_y = torch.tensor(np.array(batch_y_list), dtype=torch.float32).to(device)
            
            # Forward pass
            outputs = model(batch_x, is_training=False)
            
            # Slice output to prediction window
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:]
            
            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())
            
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # Calculate MSE and MAE
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    return mse, mae

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
    exp.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    exp.model.to(device)
    
    # Load test dataset
    test_data, _ = exp._get_data(flag="test")
    print(f"Loaded test dataset: {args.data} (Size: {len(test_data)})")

    # Shift steps
    shift_steps = [int(x.strip()) for x in args.shifts.split(",")]
    
    print("\n" + "=" * 60)
    print(f" Evaluating Phase Shift Robustness (Variant 1) for {args.model} ")
    print("=" * 60)
    
    results = []
    base_mse = None
    
    for delta_t in shift_steps:
        mse, mae = evaluate_shift(
            exp.model, test_data, args.seq_len, args.pred_len, 
            delta_t, args.batch_size, device, args.features
        )
        if delta_t == 0:
            base_mse = mse
        
        rel_increase = (mse - base_mse) / base_mse if base_mse is not None else 0.0
        
        print(f"Shift Delta t = {delta_t:2d} | MSE: {mse:.6f} | MAE: {mae:.6f} | Relative MSE Increase: {rel_increase:.2%}")
        results.append((delta_t, mse, mae, rel_increase))

    # Print markdown table
    print("\n### Phase Shift Robustness (Variant 1) Results (Markdown Table)\n")
    print("| Shift Step ($\\Delta t$) | Test MSE | Test MAE | Relative MSE Increase |")
    print("|---|---|---|---|")
    for delta_t, mse, mae, rel_increase in results:
        print(f"| {delta_t} | {mse:.6f} | {mae:.6f} | {rel_increase:.2%} |")
    print()

if __name__ == "__main__":
    main()
