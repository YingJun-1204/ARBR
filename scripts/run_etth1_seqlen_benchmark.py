import argparse
import os
import re
import subprocess
import sys
import time
import numpy as np

# Optimal V18 hyperparameters for ETTh1 by prediction horizon
ETTH1_V18_CONFIGS = {
    96: {
        "learning_rate": 0.0018,
        "gs_dropout": 0.8,
        "head_dropout": 0.9,
        "fusion_init": 0.2,
        "fusion_beta_max": 1.0,
    },
    192: {
        "learning_rate": 0.002,
        "gs_dropout": 0.85,
        "head_dropout": 0.9,
        "fusion_init": 0.35,
        "fusion_beta_max": 0.75,
    },
    336: {
        "learning_rate": 0.001,
        "gs_dropout": 0.8,
        "head_dropout": 0.9,
        "fusion_init": 0.35,
        "fusion_beta_max": 0.5,
    },
    720: {
        "learning_rate": 0.0008,
        "gs_dropout": 0.55,
        "head_dropout": 0.45,
        "fusion_init": 0.5,
        "fusion_beta_max": 1.0,
    },
}

DEFAULT_SEQLENS = [96, 192, 336, 512, 720]
DEFAULT_HORIZONS = [96, 192, 336, 720]


def run_single_experiment(seq_len, pred_len, train_epochs, extra_args):
    cfg = ETTH1_V18_CONFIGS[pred_len]
    model_id = f"etth1_seq{seq_len}_pred{pred_len}"

    cmd = [
        sys.executable,
        "-u",
        "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", "./data",
        "--data_path", "ETTh1.csv",
        "--model_id", model_id,
        "--model", "SplatTS",
        "--data", "ETTh1",
        "--features", "M",
        "--seq_len", str(seq_len),
        "--pred_len", str(pred_len),
        "--enc_in", "7",
        "--d_model", "128",
        "--itr", "1",
        "--batch_size", "256",
        "--representation", "gs",
        "--patch_len", "24",
        "--stride", "12",
        "--num_gaussians", "8",
        "--gs_dropout", str(cfg["gs_dropout"]),
        "--gs_weight_decay", "1e-4",
        "--learning_rate", str(cfg["learning_rate"]),
        "--lradj", "cosine",
        "--train_epochs", str(train_epochs),
        "--patience", "6",
        "--head_dropout", str(cfg["head_dropout"]),
        "--head_dropout_position", "pre",
        "--density_mode", "cas",
        "--num_workers", "0",
        "--use_residual",
        "--gs_residual_weight", "0.1",
        "--fusion_mode", "geometry",
        "--fusion_init", str(cfg["fusion_init"]),
        "--fusion_beta_max", str(cfg["fusion_beta_max"]),
        "--fusion_hidden_dim", "16",
        "--fusion_detach_geometry", "1",
        "--jet_score_temperature", "0.01",
        "--jet_max_shift_samples", "1",
        "--jet_derivative_mode", "exact",
        "--use_scale_jet",
        "--des", "ETTh1_GaussianJet",
    ] + extra_args

    print(f"\n[{time.strftime('%H:%M:%S')}] Executing: seq_len={seq_len}, pred_len={pred_len}")
    print(f"Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    last_mse, last_mae = None, None
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
        if line.strip().startswith("mse:"):
            m = re.search(r"mse:\s*([0-9\.]+),\s*mae:\s*([0-9\.]+)", line.strip())
            if m:
                last_mse = float(m.group(1))
                last_mae = float(m.group(2))

    process.wait()
    if process.returncode != 0:
        print(f"ERROR: Run failed with exit code {process.returncode}")
        return None, None

    return last_mse, last_mae


def print_and_save_summary(results, seq_lens, horizons, output_file="result_etth1_seqlen_benchmark.txt"):
    divider = "+" + "-"*14 + "+" + "".join(["-"*19 + "+"] * (len(horizons) + 1))
    
    header = f"| {'Seq Length':<12} |"
    for pl in horizons:
        header += f" {f'{pl} (MSE/MAE)':<17} |"
    header += f" {'Avg (MSE/MAE)':<17} |"

    summary_lines = []
    summary_lines.append("\n" + "="*115)
    summary_lines.append("           ETTh1 MULTI-SEQLEN BENCHMARK RESULTS SUMMARY TABLE (V18)")
    summary_lines.append("="*115)
    summary_lines.append(divider)
    summary_lines.append(header)
    summary_lines.append(divider)

    for sl in seq_lens:
        row_str = f"| {sl:<12} |"
        sl_mses = []
        sl_maes = []
        for pl in horizons:
            res = results.get((sl, pl))
            if res and res[0] is not None:
                mse, mae = res
                cell = f"{mse:.4f} / {mae:.4f}"
                sl_mses.append(mse)
                sl_maes.append(mae)
            else:
                cell = "N/A"
            row_str += f" {cell:<17} |"

        if sl_mses:
            avg_mse = float(np.mean(sl_mses))
            avg_mae = float(np.mean(sl_maes))
            avg_cell = f"{avg_mse:.4f} / {avg_mae:.4f}"
        else:
            avg_cell = "N/A"
        row_str += f" {avg_cell:<17} |"
        summary_lines.append(row_str)

    summary_lines.append(divider)

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary table saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Automate ETTh1 benchmark across multiple seq_lens and calculate horizon averages.")
    parser.add_argument("--seq_lens", type=int, nargs="+", default=DEFAULT_SEQLENS, help="List of seq_lens to benchmark (default: 96 192 336 512 720)")
    parser.add_argument("--pred_lens", type=int, nargs="+", default=DEFAULT_HORIZONS, help="List of pred_lens to benchmark (default: 96 192 336 720)")
    parser.add_argument("--train_epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--output_file", type=str, default="result_etth1_seqlen_benchmark.txt", help="Summary text output file")

    args, unknown_args = parser.parse_known_args()

    results = {}
    total_runs = len(args.seq_lens) * len(args.pred_lens)
    run_idx = 1

    print(f"Starting ETTh1 Multi-SeqLen Benchmark ({len(args.seq_lens)} seq_lens x {len(args.pred_lens)} horizons = {total_runs} total runs)...")

    for sl in args.seq_lens:
        for pl in args.pred_lens:
            print(f"\n==================== Run {run_idx}/{total_runs}: ETTh1 seq_len={sl}, pred_len={pl} ====================")
            mse, mae = run_single_experiment(sl, pl, args.train_epochs, unknown_args)
            results[(sl, pl)] = (mse, mae)
            run_idx += 1

    print_and_save_summary(results, args.seq_lens, args.pred_lens, args.output_file)


if __name__ == "__main__":
    main()
