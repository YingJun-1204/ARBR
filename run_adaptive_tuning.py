import argparse
import os
import subprocess
import sys
import time


def log_message(msg):
    print(f"\n==================================================")
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    print(f"==================================================")


def run_command(cmd):
    print(f"Running subprocess: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env
    )
    process.wait()
    return process.returncode


def main():
    # Note: The Gaussian freezing mechanism (--gs_freeze_epochs) has been permanently removed from the training pipeline.
    # Note: The train_epochs has been fixed to 100 and learning_rate has been fixed to 1e-4 in all HPO tuning scripts.
    parser = argparse.ArgumentParser(
        description="Automate sequential HPO studies on ETTh1, ETTh2, ETTm1 & ETTm2 for Adaptive Gating."
    )
    parser.add_argument("--gate_beta", type=float, default=0.25, help="Scale gate beta bounds (default: 0.25)")
    parser.add_argument("--pred_len", type=int, nargs="+", default=[96], help="Prediction horizons (default: [96])")
    parser.add_argument("--batches", type=int, default=7, help="Number of optuna batches (default: 10)")
    parser.add_argument("--trials_per_batch", type=int, default=30, help="Number of trials per optuna batch (default: 30)")
    parser.add_argument("--position", type=str, default="pre", choices=["none", "pre", "post"], help="Head dropout position")
    parser.add_argument("--density_mode", type=str, default="sparse", choices=["none", "soft", "sparse", "cas"], help="Density mode selection")
    parser.add_argument("--num_gaussians", type=int, default=8, help="Fixed number of gaussians (default: 8)")
    parser.add_argument("--output_dir", type=str, default="loss_lga_new", help="Output directory for best json configs")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity"],
        help="List of datasets to tune (choices: etth1, etth2, ettm1, ettm2, weather, electricity)"
    )
    parser.add_argument("--use_seed", action="store_true", help="Use previous best parameters as HPO seeds")
    parser.add_argument("--gate_type", type=str, default="adaptive_direction", choices=["none", "forward", "reverse", "adaptive_direction"], help="Gating type selection (default: adaptive_direction)")
    parser.add_argument("--batch_size", type=int, default=512, choices=[16, 256, 512, 1024], help="Fixed batch size for HPO (default: 512)")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length / lookback window (default: 512)")
    parser.add_argument("--k_base", type=int, default=-1, help="Manual k_base value for CAS gating (-1 means dynamic)")
    
    args = parser.parse_args()

    python_bin = sys.executable

    tasks_all = [
        ("ETTh1", "tune_gate_etth1.py", args.gate_type),
        ("ETTh2", "tune_gate_etth2.py", args.gate_type),
        ("ETTm1", "tune_gate_ettm1.py", args.gate_type),
        ("ETTm2", "tune_gate_ettm2.py", args.gate_type),
        ("weather", "tune_gate_weather.py", args.gate_type),
        ("electricity", "tune_gate_electricity.py", args.gate_type),
    ]

    selected = [d.lower() for d in args.datasets]
    tasks = [t for t in tasks_all if t[0].lower() in selected]

    if not tasks:
        print(f"Error: No valid datasets selected from {args.datasets}. Choices: etth1, etth2, ettm1, ettm2, weather, electricity")
        sys.exit(1)

    total_tasks = len(tasks) * len(args.pred_len)
    step_num = 1
    for p_len in args.pred_len:
        for dataset, script, gate_type in tasks:
            log_message(f"STEP {step_num}/{total_tasks}: Starting {gate_type.upper()} HPO Gating for {dataset} with pred_len={p_len}")
            
            task_batch_size = args.batch_size
            if dataset.lower() == "electricity" and args.batch_size == 512:
                task_batch_size = 16

            cmd = [
                python_bin,
                script,
                "--gate_type", gate_type,
                "--gate_beta", str(args.gate_beta),
                "--pred_len", str(p_len),
                "--batches", str(args.batches),
                "--trials_per_batch", str(args.trials_per_batch),
                "--position", args.position,
                "--density_mode", args.density_mode,
                "--output_dir", args.output_dir,
                "--num_gaussians", str(args.num_gaussians),
                "--batch_size", str(task_batch_size),
                "--seq_len", str(args.seq_len),
                "--k_base", str(args.k_base)
            ]
            
            if args.use_seed:
                cmd.append("--use_seed")
            
            ret_code = run_command(cmd)
            if ret_code != 0:
                log_message(f"ERROR: Step {step_num} ({dataset} {gate_type} with pred_len={p_len}) failed with exit code {ret_code}. Aborting downstream steps.")
                sys.exit(ret_code)
                
            log_message(f"{dataset} {gate_type.upper()} Gating HPO (pred_len={p_len}) Completed Successfully!")
            
            step_num += 1
            if step_num <= total_tasks:
                time.sleep(3)

    log_message("ALL ADAPTIVE GATING EXPERIMENTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
