import argparse
import os
import random
import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.variant_configs import apply_variant_configs


def get_parser():
    parser = argparse.ArgumentParser(
        description="Localized Phase Shift Robustness Evaluation for Full SplatTS on ETTm2"
    )

    # basic config
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="ettm2_router_96")
    parser.add_argument("--model", type=str, default="SplatTS")

    # data loader
    parser.add_argument("--data", type=str, default="ETTm2")
    parser.add_argument("--root_path", type=str, default="./data/")
    parser.add_argument("--data_path", type=str, default="ETTm2.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="t")
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
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true")

    # representation model
    parser.add_argument("--representation", type=str, default="gs", choices=["gs", "patch_linear"])
    parser.add_argument("--patch_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--head_dropout", type=float, default=0.8)
    parser.add_argument("--head_dropout_position", type=str, default="pre", choices=["none", "pre", "post"])
    parser.add_argument("--head_mode", type=str, default="linear")
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--gs_dropout", type=float, default=0.8)
    parser.add_argument("--gs_weight_decay", type=float, default=0.0001)

    # model constraints
    parser.add_argument("--num_gaussians", type=int, default=8)
    parser.add_argument("--density_mode", type=str, default="cas", choices=["none", "cas"])
    parser.add_argument("--gs_lambda", type=float, default=0.0)
    parser.add_argument("--gate_lambda", type=float, default=0.1)
    parser.add_argument("--use_occlusion", action="store_true", default=False)
    parser.add_argument("--use_residual", action="store_true", default=True)
    parser.add_argument("--gs_residual_weight", type=float, default=0.55)
    parser.add_argument("--gate_type", type=str, default="adaptive_direction")
    parser.add_argument("--gate_beta", type=float, default=0.3)
    parser.add_argument("--gate_window_half", type=int, default=2)
    parser.add_argument("--k_base", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="loss_mdagV12")

    # optimization / execution
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.0015)
    parser.add_argument("--des", type=str, default="Ablation_ETTm2_router")
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
    parser.add_argument(
        "--shifts",
        type=str,
        default="0,1,2,4,8,16",
        help="Comma-separated local shift steps to test, e.g. 0,1,2,4,8,16",
    )

    parser.add_argument(
        "--segment_len",
        type=int,
        default=96,
        help="Length of the local segment to be shifted. Recommended: 2x or 4x patch_len.",
    )

    parser.add_argument(
        "--segment_mode",
        type=str,
        default="patch_boundary",
        choices=["middle", "random", "patch_boundary"],
        help=(
            "Where to apply the local shift. "
            "middle: fixed center segment; "
            "random: deterministic random segment per sample; "
            "patch_boundary: segment centered near a patch boundary."
        ),
    )

    parser.add_argument(
        "--num_segments",
        type=int,
        default=1,
        help="Number of local segments to perturb in each input window.",
    )

    parser.add_argument(
        "--shift_direction",
        type=str,
        default="alternate",
        choices=["right", "left", "alternate"],
        help=(
            "Direction of local temporal shift. "
            "right: delay local pattern; left: advance local pattern; "
            "alternate: deterministic left/right by sample index."
        ),
    )

    parser.add_argument(
        "--fill_mode",
        type=str,
        default="edge",
        choices=["edge", "reflect", "zero"],
        help="How to fill newly exposed positions inside the shifted local segment.",
    )

    parser.add_argument(
        "--blend_width",
        type=int,
        default=8,
        help=(
            "Smoothly blend the shifted segment back to the original signal at both boundaries. "
            "This avoids turning the test into an artificial discontinuity attack."
        ),
    )

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_setting_name(args, ii=0):
    return "{}_{}_{}_{}_ft{}_sl{}_pl{}_dm{}_df{}_{}_{}".format(
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


def adapt_state_dict_keys(state_dict):
    adapted_state_dict = {}

    for k, v in state_dict.items():
        if "patch_residual.projection.weight" in k:
            adapted_key = k.replace(
                "patch_residual.projection.weight",
                "patch_residual.base_projection.weight",
            )
            adapted_state_dict[adapted_key] = v
        else:
            adapted_state_dict[k] = v

    return adapted_state_dict


def make_rng(seed, sample_idx, delta_t, segment_id):
    value = (
        int(seed)
        + 1000003 * int(sample_idx)
        + 9176 * int(delta_t)
        + 97 * int(segment_id)
    ) % (2 ** 32 - 1)
    return np.random.RandomState(value)


def choose_segment_start(seq_len, segment_len, args, sample_idx, delta_t, segment_id):
    segment_len = min(segment_len, seq_len)

    if segment_len >= seq_len:
        return 0

    if args.segment_mode == "middle":
        return (seq_len - segment_len) // 2

    rng = make_rng(args.seed, sample_idx, delta_t, segment_id)

    if args.segment_mode == "random":
        return int(rng.randint(0, seq_len - segment_len + 1))

    if args.segment_mode == "patch_boundary":
        # Patch starts are 0, stride, 2*stride, ...
        # A rigid patch model is often sensitive near patch boundaries.
        # We center the perturbation around one such boundary.
        candidate_centers = np.arange(args.patch_len, seq_len, args.stride)

        min_center = segment_len // 2
        max_center = seq_len - (segment_len - segment_len // 2)

        candidate_centers = candidate_centers[
            (candidate_centers >= min_center) & (candidate_centers <= max_center)
        ]

        if len(candidate_centers) == 0:
            return (seq_len - segment_len) // 2

        center = int(candidate_centers[rng.randint(0, len(candidate_centers))])
        start = center - segment_len // 2
        start = max(0, min(start, seq_len - segment_len))
        return int(start)

    raise ValueError(f"Unknown segment_mode: {args.segment_mode}")


def shift_local_segment(segment, delta_t, direction, fill_mode):
    """
    Shift only the selected local segment.

    segment: numpy array with shape [segment_len, channels]
    delta_t: local displacement
    direction: right or left
    """
    seg_len = segment.shape[0]

    if delta_t <= 0:
        return np.copy(segment)

    d = min(delta_t, seg_len - 1)
    shifted = np.empty_like(segment)

    if direction == "right":
        shifted[d:] = segment[:-d]

        if fill_mode == "edge":
            shifted[:d] = segment[0:1]
        elif fill_mode == "reflect":
            if seg_len > d:
                fill = segment[1:d + 1][::-1]
                if fill.shape[0] < d:
                    pad = np.repeat(segment[0:1], d - fill.shape[0], axis=0)
                    fill = np.concatenate([fill, pad], axis=0)
                shifted[:d] = fill
            else:
                shifted[:d] = segment[0:1]
        elif fill_mode == "zero":
            shifted[:d] = 0.0
        else:
            raise ValueError(f"Unknown fill_mode: {fill_mode}")

    elif direction == "left":
        shifted[:-d] = segment[d:]

        if fill_mode == "edge":
            shifted[-d:] = segment[-1:]
        elif fill_mode == "reflect":
            if seg_len > d:
                fill = segment[-d - 1:-1][::-1]
                if fill.shape[0] < d:
                    pad = np.repeat(segment[-1:], d - fill.shape[0], axis=0)
                    fill = np.concatenate([pad, fill], axis=0)
                shifted[-d:] = fill
            else:
                shifted[-d:] = segment[-1:]
        elif fill_mode == "zero":
            shifted[-d:] = 0.0
        else:
            raise ValueError(f"Unknown fill_mode: {fill_mode}")

    else:
        raise ValueError(f"Unknown direction: {direction}")

    return shifted


def blend_shifted_segment(original_segment, shifted_segment, blend_width):
    """
    Smooth boundary blending:
    center of the segment uses shifted values;
    both boundaries gradually return to the original signal.
    """
    seg_len = original_segment.shape[0]

    if blend_width <= 0:
        return shifted_segment

    bw = min(blend_width, seg_len // 2)

    if bw <= 0:
        return shifted_segment

    weight = np.ones((seg_len, 1), dtype=original_segment.dtype)

    # At the very boundary, keep mostly original values.
    # Moving inward, use more shifted values.
    weight[:bw, 0] = np.linspace(0.0, 1.0, bw)
    weight[-bw:, 0] = np.linspace(1.0, 0.0, bw)

    return original_segment * (1.0 - weight) + shifted_segment * weight


def apply_local_phase_shift(seq_x, delta_t, args, sample_idx):
    """
    Apply local phase shift to one or more local segments of the input sequence.

    Important:
    - This does NOT shift the whole input window.
    - The forecasting target should remain unchanged outside this function.
    - The same perturbation protocol should be used for all compared models.
    """
    if delta_t == 0:
        return np.copy(seq_x)

    seq_shifted = np.copy(seq_x)
    seq_len = seq_shifted.shape[0]
    segment_len = min(args.segment_len, seq_len)

    for segment_id in range(args.num_segments):
        start = choose_segment_start(
            seq_len=seq_len,
            segment_len=segment_len,
            args=args,
            sample_idx=sample_idx,
            delta_t=delta_t,
            segment_id=segment_id,
        )
        end = start + segment_len

        if args.shift_direction == "alternate":
            direction = "right" if ((sample_idx + segment_id) % 2 == 0) else "left"
        else:
            direction = args.shift_direction

        original_segment = np.copy(seq_shifted[start:end])
        shifted_segment = shift_local_segment(
            segment=original_segment,
            delta_t=delta_t,
            direction=direction,
            fill_mode=args.fill_mode,
        )
        shifted_segment = blend_shifted_segment(
            original_segment=original_segment,
            shifted_segment=shifted_segment,
            blend_width=args.blend_width,
        )

        seq_shifted[start:end] = shifted_segment

    return seq_shifted


def evaluate_shift_metrics(model, test_data, pred_len, delta_t, batch_size, device, features, args):
    """
    Localized phase-shift robustness evaluation.

    Clean sample:
        seq_x_0 -> seq_y_0

    Perturbed sample:
        local_phase_shift(seq_x_0) -> seq_y_0

    Metrics:
    1. Prediction Stability Error:
       MSE between predictions from clean and perturbed inputs.
    2. Forecasting MSE / MAE:
       Forecasting error under locally shifted input against the original clean target.
    3. Input Perturbation MSE:
       Sanity-check magnitude of the input perturbation.
    """
    model.eval()

    total_len = len(test_data)

    stability_errors = []
    forecast_mses = []
    forecast_maes = []
    input_perturb_mses = []

    f_dim = -1 if features == "MS" else 0

    with torch.no_grad():
        for i in range(0, total_len, batch_size):
            batch_x_0_list = []
            batch_x_dt_list = []
            batch_y_0_list = []

            for j in range(i, min(i + batch_size, total_len)):
                seq_x_0, seq_y_0, _, _ = test_data[j]

                seq_x_dt = apply_local_phase_shift(
                    seq_x=seq_x_0,
                    delta_t=delta_t,
                    args=args,
                    sample_idx=j,
                )

                batch_x_0_list.append(seq_x_0)
                batch_x_dt_list.append(seq_x_dt)
                batch_y_0_list.append(seq_y_0)

            if not batch_x_0_list:
                break

            batch_x_0 = torch.tensor(np.array(batch_x_0_list), dtype=torch.float32).to(device)
            batch_x_dt = torch.tensor(np.array(batch_x_dt_list), dtype=torch.float32).to(device)
            batch_y_0 = torch.tensor(np.array(batch_y_0_list), dtype=torch.float32).to(device)

            outputs_0 = model(batch_x_0, is_training=False)[:, -pred_len:, f_dim:]
            outputs_dt = model(batch_x_dt, is_training=False)[:, -pred_len:, f_dim:]
            batch_y_0 = batch_y_0[:, -pred_len:, f_dim:]

            batch_stability = torch.mean((outputs_0 - outputs_dt) ** 2, dim=[1, 2])
            stability_errors.extend(batch_stability.cpu().numpy())

            batch_forecast_mse = torch.mean((outputs_dt - batch_y_0) ** 2, dim=[1, 2])
            batch_forecast_mae = torch.mean(torch.abs(outputs_dt - batch_y_0), dim=[1, 2])

            forecast_mses.extend(batch_forecast_mse.cpu().numpy())
            forecast_maes.extend(batch_forecast_mae.cpu().numpy())

            batch_input_perturb_mse = torch.mean((batch_x_dt - batch_x_0) ** 2, dim=[1, 2])
            input_perturb_mses.extend(batch_input_perturb_mse.cpu().numpy())

    return (
        float(np.mean(stability_errors)) if stability_errors else 0.0,
        float(np.mean(forecast_mses)) if forecast_mses else 0.0,
        float(np.mean(forecast_maes)) if forecast_maes else 0.0,
        float(np.mean(input_perturb_mses)) if input_perturb_mses else 0.0,
    )


def format_accuracy_cell(idx, delta_t, results, baseline_idx):
    current_mse = results["mse"][idx]
    current_mae = results["mae"][idx]

    if np.isnan(current_mse) or np.isnan(current_mae):
        return "N/A"

    if delta_t == 0 or baseline_idx is None:
        return f"{current_mse:.6f} / {current_mae:.6f}"

    baseline_mse = results["mse"][baseline_idx]
    baseline_mae = results["mae"][baseline_idx]

    diff_mse = current_mse - baseline_mse
    diff_mae = current_mae - baseline_mae

    deg_mse_pct = (diff_mse / (baseline_mse + 1e-9)) * 100
    deg_mae_pct = (diff_mae / (baseline_mae + 1e-9)) * 100

    sign_mse = "+" if diff_mse >= 0 else ""
    sign_mae = "+" if diff_mae >= 0 else ""

    # Use semicolon instead of "|" to avoid breaking markdown tables.
    return (
        f"{current_mse:.6f} / {current_mae:.6f} "
        f"({sign_mse}{diff_mse:.6f} / {sign_mae}{diff_mae:.6f}; "
        f"{sign_mse}{deg_mse_pct:.2f}% / {sign_mae}{deg_mae_pct:.2f}%)"
    )


def format_single_metric(idx, delta_t, values, baseline_idx):
    current = values[idx]

    if np.isnan(current):
        return "N/A"

    if delta_t == 0 or baseline_idx is None:
        return f"{current:.6f}"

    baseline = values[baseline_idx]
    diff = current - baseline
    deg_pct = (diff / (baseline + 1e-9)) * 100 if abs(baseline) > 1e-12 else 0.0
    sign = "+" if diff >= 0 else ""

    return f"{current:.6f} ({sign}{diff:.6f}; {sign}{deg_pct:.2f}%)"


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()

    if args.gate_type == "none":
        args.gate_type = None

    args = apply_variant_configs(args)
    set_seed(args.seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    setting = build_setting_name(args, ii=0)

    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    else:
        raise ValueError(f"Unsupported task_name for this script: {args.task_name}")

    exp = Exp(args)

    checkpoint_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    adapted_state_dict = adapt_state_dict_keys(state_dict)

    load_info = exp.model.load_state_dict(adapted_state_dict, strict=False)

    print(f"Missing keys: {len(load_info.missing_keys)}")
    print(f"Unexpected keys: {len(load_info.unexpected_keys)}")

    if load_info.missing_keys:
        print("First missing keys:", load_info.missing_keys[:10])

    if load_info.unexpected_keys:
        print("First unexpected keys:", load_info.unexpected_keys[:10])

    exp.model.to(device)

    test_data, _ = exp._get_data(flag="test")
    print(f"Loaded test dataset: {args.data} (Size: {len(test_data)})")

    shift_steps = [int(x.strip()) for x in args.shifts.split(",") if x.strip()]

    results = {
        "stability": [],
        "mse": [],
        "mae": [],
        "input_perturb_mse": [],
    }

    print("\n" + "=" * 88)
    print(" Evaluating Full SplatTS Localized Phase-Shift Robustness")
    print(f" Dataset: {args.data} | pred_len: {args.pred_len} | shifts: {shift_steps}")
    print(
        " Perturbation: "
        f"mode={args.segment_mode}, segment_len={args.segment_len}, "
        f"num_segments={args.num_segments}, direction={args.shift_direction}, "
        f"fill={args.fill_mode}, blend_width={args.blend_width}"
    )
    print("=" * 88)

    for delta_t in shift_steps:
        print(f"Evaluating local shift step Delta t = {delta_t} ...")

        stability, mse, mae, input_perturb_mse = evaluate_shift_metrics(
            model=exp.model,
            test_data=test_data,
            pred_len=args.pred_len,
            delta_t=delta_t,
            batch_size=args.batch_size,
            device=device,
            features=args.features,
            args=args,
        )

        results["stability"].append(stability)
        results["mse"].append(mse)
        results["mae"].append(mae)
        results["input_perturb_mse"].append(input_perturb_mse)

    try:
        baseline_idx = shift_steps.index(0)
    except ValueError:
        baseline_idx = None

    print("\n### 1. Prediction Stability Error under Localized Phase Shift\n")
    print("| Shift Step ($\\Delta t$) | Full SplatTS Stability Error | Input Perturbation MSE |")
    print("|---|---|---|")

    for idx, delta_t in enumerate(shift_steps):
        stability_str = format_single_metric(
            idx=idx,
            delta_t=delta_t,
            values=results["stability"],
            baseline_idx=baseline_idx,
        )
        input_str = format_single_metric(
            idx=idx,
            delta_t=delta_t,
            values=results["input_perturb_mse"],
            baseline_idx=baseline_idx,
        )
        print(f"| {delta_t} | {stability_str} | {input_str} |")

    print("\n### 2. Forecasting Accuracy (MSE / MAE) under Localized Phase Shift\n")
    print("| Shift Step ($\\Delta t$) | Full SplatTS MSE/MAE |")
    print("|---|---|")

    for idx, delta_t in enumerate(shift_steps):
        acc_str = format_accuracy_cell(idx, delta_t, results, baseline_idx)
        print(f"| {delta_t} | {acc_str} |")

    print("\nDone.")


if __name__ == "__main__":
    main()