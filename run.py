import argparse
import os
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


EXPERIMENT_SEEDS = (42, 2021, 2026)


def set_global_seed(seed):
    if seed not in EXPERIMENT_SEEDS:
        raise ValueError(
            f"seed must be one of {EXPERIMENT_SEEDS}, got {seed}"
        )
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_parser():
    parser = argparse.ArgumentParser(description="ARBD forecasting")

    # basic config
    parser.add_argument("--task_name", type=str, required=True, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, required=True, default=1)
    parser.add_argument("--model_id", type=str, required=True, default="test")
    parser.add_argument("--model", type=str, required=True, default="ARBD")

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

    # model define
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=128)

    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true")

    # ARBD model
    parser.add_argument("--patch_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--head_dropout_position", type=str, default="pre", choices=["none", "pre", "post"])
    parser.add_argument("--head_mode", type=str, default="linear")
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--gs_dropout", type=float, default=0.3)
    parser.add_argument("--gs_weight_decay", type=float, default=0.0)

    parser.add_argument(
        "--num_gaussians",
        type=int,
        default=8,
        help="Fixed number K of temporal Gaussian components",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.5,
        help="Temporal center adaptation bound scalar rho (default: 0.5)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Temporal scale adaptation bound scalar beta (default: 0.5)",
    )
    parser.add_argument(
        "--local_weight_max",
        type=float,
        default=0.75,
        help="Upper bound of the local Linear correction",
    )
    parser.add_argument(
        "--static_local_weight_init",
        type=float,
        default=0.5,
        help="Initial dataset-shared Linear correction weight",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="1A",
        choices=["0", "1A"],
        help="Architecture Phase: '0' (unanchored baseline) or '1A' (non-symmetric anchors, bounded deformation)",
    )

    # optimization
    parser.add_argument("--no_save_checkpoint", action="store_true", default=False, help="Cache weights in RAM and skip writing checkpoints to disk.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--des", type=str, default="test")
    parser.add_argument("--lradj", type=str, default="cosine", help="adjust learning rate strategy: [type1, type2, type3, type4, cosine, cosine_after_unfreeze, sigmoid]")

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    # augmentation
    parser.add_argument("--augmentation_ratio", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=2021,
        choices=EXPERIMENT_SEEDS,
        help="Global experiment seed (default: 2021)",
    )
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
    parser.add_argument(
        "--print_mae",
        action="store_true",
        default=False,
        help="Print MAE alongside MSE during epoch evaluation",
    )
    parser.add_argument(
        "--eval_efficiency",
        type=int,
        default=0,
        help="Evaluate model efficiency (Params, FLOPs, Latency, Memory) during test",
    )
    return parser


parser = get_parser()

if __name__ == "__main__":
    args, unknown = parser.parse_known_args()
    if unknown:
        parser.error("Unrecognized arguments: " + " ".join(unknown))
    if args.num_gaussians <= 0:
        parser.error("--num_gaussians must be positive")
    set_global_seed(args.seed)
    print(f"[Experiment Seed] {args.seed}")
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = "{}_{}_{}".format(
                args.task_name,
                args.model_id,
                args.model,
            )

            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            exp.train(setting)

            print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = "{}_{}_{}".format(
            args.task_name,
            args.model_id,
            args.model,
        )

        exp = Exp(args)
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
