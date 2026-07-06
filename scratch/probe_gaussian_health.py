import argparse
import os
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import dotdict
from utils.variant_configs import apply_variant_configs


def build_args(opts):
    args = dotdict(
        {
            "task_name": "long_term_forecast",
            "is_training": 0,
            "model_id": opts.model_id,
            "model": "yzy",
            "data": opts.data,
            "root_path": "./data",
            "data_path": opts.data_path,
            "features": "M",
            "target": "OT",
            "freq": "h",
            "checkpoints": "./checkpoints",
            "seq_len": opts.seq_len,
            "pred_len": opts.pred_len,
            "inverse": False,
            "mask_rate": 0.25,
            "anomaly_ratio": 0.25,
            "enc_in": 7,
            "dec_in": 7,
            "c_out": 7,
            "d_model": 128,
            "d_ff": 256,
            "dropout": opts.dropout,
            "activation": "gelu",
            "output_attention": False,
            "representation": "pure_gs",
            "patch_len": opts.patch_len,
            "stride": opts.stride,
            "head_dropout": opts.head_dropout,
            "head_dropout_position": opts.head_dropout_position,
            "head_mode": "linear",
            "debug_mode": False,
            "gs_dropout": opts.gs_dropout,
            "gs_weight_decay": opts.gs_weight_decay,
            "num_gaussians": opts.num_gaussians,
            "density_mode": "sparse",
            "gs_lambda": opts.gs_lambda,
            "gate_lambda": opts.gate_lambda,
            "use_occlusion": False,
            "use_residual": True,
            "gs_residual_weight": opts.gs_residual_weight,
            "gate_type": "adaptive_direction",
            "gate_beta": opts.gate_beta,
            "gate_router_hidden": 16,
            "gate_init": "none",
            "gate_init_strength": 0.0,
            "output_dir": "loss日志",
            "num_workers": 0,
            "itr": 1,
            "train_epochs": opts.train_epochs,
            "batch_size": opts.batch_size,
            "patience": 15,
            "learning_rate": opts.learning_rate,
            "des": opts.des,
            "loss": "MSE",
            "lradj": "cosine",
            "use_gpu": torch.cuda.is_available(),
            "gpu": 0,
            "use_multi_gpu": False,
            "devices": "0",
            "p_hidden_dims": [128, 128],
            "p_hidden_layers": 2,
            "augmentation_ratio": 0,
            "seed": 2,
            "jitter": False,
            "scaling": False,
            "permutation": False,
            "randompermutation": False,
            "magwarp": False,
            "timewarp": False,
            "windowslice": False,
            "windowwarp": False,
            "rotation": False,
            "spawner": False,
            "dtwwarp": False,
            "shapedtwwarp": False,
            "wdba": False,
            "discdtw": False,
            "discsdtw": False,
            "extra_tag": "",
        }
    )
    return apply_variant_configs(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_id", default="probe")
    parser.add_argument("--data", default="ETTh2")
    parser.add_argument("--data_path", default="ETTh2.csv")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--patch_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--num_gaussians", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--gs_residual_weight", type=float, default=0.5)
    parser.add_argument("--gate_beta", type=float, default=0.3)
    parser.add_argument("--gs_dropout", type=float, default=0.6)
    parser.add_argument("--gs_weight_decay", type=float, default=2.6e-4)
    parser.add_argument("--learning_rate", type=float, default=0.0088)
    parser.add_argument("--dropout", type=float, default=0.07)
    parser.add_argument("--head_dropout", type=float, default=0.54)
    parser.add_argument("--head_dropout_position", default="pre")
    parser.add_argument("--gate_lambda", type=float, default=0.0)
    parser.add_argument("--gs_lambda", type=float, default=0.0)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--des", default="probe")
    opts = parser.parse_args()

    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)

    args = build_args(opts)
    exp = Exp_Long_Term_Forecast(args)
    state = torch.load(opts.checkpoint, map_location=exp.device)
    exp.model.load_state_dict(state)
    exp.model.eval()

    _, test_loader = exp._get_data(flag="test")
    sigmoids = []
    egcs = []
    route_max = []
    eff_dirs = []

    with torch.no_grad():
        for batch_x, _, _, _ in test_loader:
            batch_x = batch_x.float().to(exp.device)
            exp.model(batch_x, is_training=False)
            splat = exp._get_splatting_module()
            scores = splat.last_density_score.detach()
            probs = torch.sigmoid(scores)
            sigmoids.append(probs.cpu().reshape(-1).numpy())
            egcs.append((probs > 0.5).float().sum(dim=-1).mean().item())
            if splat.last_gate_route_probs is not None:
                route = splat.last_gate_route_probs.detach()
                route_max.append(route.max(dim=-1).values.cpu().reshape(-1).numpy())
            if splat.last_gate_direction is not None and splat.last_gate_strength is not None:
                eff = (splat.last_gate_direction.detach() * splat.last_gate_strength.detach())
                eff_dirs.append(eff.cpu().reshape(-1).numpy())

    all_probs = np.concatenate(sigmoids)
    print(f"checkpoint={opts.checkpoint}")
    print(f"num_gaussians={opts.num_gaussians}")
    print(f"egc_mean={np.mean(egcs):.6f}")
    print(f"density_sigmoid_mean={all_probs.mean():.8f}")
    print(f"density_sigmoid_min={all_probs.min():.8f}")
    print(f"density_sigmoid_max={all_probs.max():.8f}")
    print(f"density_sigmoid_p50={np.percentile(all_probs, 50):.8f}")
    print(f"density_sigmoid_p95={np.percentile(all_probs, 95):.8f}")
    print(f"density_active_ratio_gt_0_5={(all_probs > 0.5).mean():.8f}")
    print(f"density_alive_ratio_gt_0_01={(all_probs > 0.01).mean():.8f}")
    if route_max:
        route_max_arr = np.concatenate(route_max)
        print(f"route_max_mean={route_max_arr.mean():.8f}")
        print(f"route_uncertain_ratio_lt_0_45={(route_max_arr < 0.45).mean():.8f}")
    if eff_dirs:
        eff_arr = np.concatenate(eff_dirs)
        print(f"eff_dir_abs_mean={np.abs(eff_arr).mean():.8f}")
        print(f"eff_dir_near_zero_ratio_le_0_05={(np.abs(eff_arr) <= 0.05).mean():.8f}")


if __name__ == "__main__":
    main()
