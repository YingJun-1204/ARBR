import argparse
import os
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from utils.variant_configs import apply_variant_configs


if __name__ == "__main__":
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    torch.cuda.manual_seed_all(fix_seed)
    np.random.seed(fix_seed)
    
    # 强制启用确定性算法以确保复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass



    parser = argparse.ArgumentParser(description="Representation baseline forecasting")

    # basic config
    parser.add_argument("--task_name", type=str, required=True, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, required=True, default=1)
    parser.add_argument("--model_id", type=str, required=True, default="test")
    parser.add_argument("--model", type=str, required=True, default="SplatTS")

    # data loader
    parser.add_argument("--data", type=str, required=True, default="ETTh1")
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

    # imputation and anomaly arguments kept for experiment compatibility
    parser.add_argument("--mask_rate", type=float, default=0.25)
    parser.add_argument("--anomaly_ratio", type=float, default=0.25)

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

    parser.add_argument("--num_gaussians", type=int, default=8)
    parser.add_argument("--density_mode", type=str, default="cas", choices=["none", "cas"])
    parser.add_argument("--gs_lambda", type=float, default=0.0)
    parser.add_argument("--use_occlusion", action="store_true", default=False)
    parser.add_argument("--use_residual", action="store_true", default=True, help="Enable Patch linear residual shortcut")
    parser.add_argument("--gs_residual_weight", type=float, default=0.1, help="Gating weight multiplier for residual shortcut")
    parser.add_argument("--k_base", type=int, default=-1, help="Manual k_base value for CAS gating (-1 means dynamic)")
    parser.add_argument("--output_dir", type=str, default="loss_cas_simplify", help="Output directory for configurations and diagnostics")

    # Gaussian Jet parameters
    parser.add_argument("--num_implicit_gaussians", type=int, default=4)
    parser.add_argument("--jet_max_shift_samples", type=float, default=1.0)
    parser.add_argument("--jet_score_temperature", type=float, default=1.0)
    parser.add_argument("--jet_density_tau", type=float, default=1.0)
    parser.add_argument("--jet_detach_geometry", type=int, default=1)
    parser.add_argument("--jet_scale_init", type=float, default=0.1)
    parser.add_argument("--jet_sigma_init", type=float, default=0.2)

    # optimization
    parser.add_argument("--no_save_checkpoint", action="store_true", default=False, help="Cache weights in RAM and skip writing checkpoints to disk.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--des", type=str, default="test")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="cosine", help="adjust learning rate strategy: [type1, type2, type3, type4, cosine, cosine_after_unfreeze, sigmoid]")

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

    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error("Unrecognized arguments: " + " ".join(unknown))
    args = apply_variant_configs(args)

    print(f"Current representation: {args.representation}")
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = "{}_{}_{}_{}_ft{}_sl{}_pl{}_dm{}_{}_{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.des,
                ii,
            )

            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            exp.train(setting)

            print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = "{}_{}_{}_{}_ft{}_sl{}_pl{}_dm{}_{}_{}".format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.des,
            ii,
        )

        exp = Exp(args)
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
