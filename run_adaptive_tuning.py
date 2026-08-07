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
    parser = argparse.ArgumentParser(
        description="Automate sequential HPO studies on ETTh1, ETTh2, ETTm1 & ETTm2 for Adaptive Gating."
    )
    parser.add_argument("--pred_len", type=int, nargs="+", default=[96], help="Prediction horizons (default: [96])")
    parser.add_argument("--batches", type=int, default=7, help="Number of optuna batches (default: 10)")
    parser.add_argument("--trials_per_batch", type=int, default=30, help="Number of trials per optuna batch (default: 30)")
    parser.add_argument("--position", type=str, default="pre", choices=["pre"], help="Head dropout position")
    parser.add_argument("--density_mode", type=str, default="cas", choices=["none", "cas"], help="Density mode selection")
    parser.add_argument("--num_gaussians", type=int, default=8, help="Fixed number of gaussians (default: 8)")
    parser.add_argument("--output_dir", type=str, default="loss_jetV14", help="Output directory for best json configs")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["etth1", "etth2", "ettm1", "ettm2", "weather", "electricity", "traffic"],
        help="List of datasets to tune (choices: etth1, etth2, ettm1, ettm2, weather, electricity, traffic)"
    )
    parser.add_argument("--use_seed", action="store_true", help="Use previous best parameters as HPO seeds")
    parser.add_argument("--batch_size", type=int, default=256, choices=[8, 16, 32, 128, 256, 512, 1024], help="Fixed batch size for HPO (default: 256)")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length / lookback window (default: 512)")
    parser.add_argument("--k_base", type=int, default=-1, help="Manual k_base value for CAS gating (-1 means dynamic)")
    parser.add_argument("--jet_derivative_mode", type=str, default="exact", choices=["exact", "centered_legacy"], help="Gaussian Jet derivative mode (exact or centered_legacy)")
    parser.add_argument("--jet_max_shift_samples", type=float, default=1.0, help="Max shift samples for Gaussian Jet (default: 1.0)")
    parser.add_argument("--use_scale_jet", action="store_true", default=False, help="Enable Content-Geometry Translation-Scale Affine Gaussian Jet in HPO")
    parser.add_argument("--ablation_mode", type=str, default="none", choices=["none", "gaussian_only", "observation_only", "wo_ajc", "wo_jet"], help="Ablation mode for model components: gaussian_only, observation_only, wo_ajc, or wo_jet")
    
    args = parser.parse_args()

    python_bin = sys.executable

    tasks_all = [
        ("ETTh1", "tune_gate_etth1.py"),
        ("ETTh2", "tune_gate_etth2.py"),
        ("ETTm1", "tune_gate_ettm1.py"),
        ("ETTm2", "tune_gate_ettm2.py"),
        ("weather", "tune_gate_weather.py"),
        ("electricity", "tune_gate_electricity.py"),
        ("traffic", "tune_gate_traffic.py"),
    ]

    selected = [d.lower() for d in args.datasets]
    tasks = [t for t in tasks_all if t[0].lower() in selected]

    if not tasks:
        print(f"Error: No valid datasets selected from {args.datasets}. Choices: etth1, etth2, ettm1, ettm2, weather, electricity, traffic")
        sys.exit(1)

    total_tasks = len(tasks) * len(args.pred_len)
    step_num = 1
    for p_len in args.pred_len:
        for dataset, script in tasks:
            log_message(f"STEP {step_num}/{total_tasks}: Starting JET HPO tuning for {dataset} with pred_len={p_len}")
            
            task_batch_size = args.batch_size
            if dataset.lower() == "electricity" and args.batch_size == 512:
                task_batch_size = 16
            elif dataset.lower() == "traffic" and args.batch_size in [256, 512, 1024]:
                task_batch_size = 8

            cmd = [
                python_bin,
                script,
                "--pred_len", str(p_len),
                "--batches", str(args.batches),
                "--trials_per_batch", str(args.trials_per_batch),
                "--position", args.position,
                "--density_mode", args.density_mode,
                "--output_dir", args.output_dir,
                "--num_gaussians", str(args.num_gaussians),
                "--batch_size", str(task_batch_size),
                "--seq_len", str(args.seq_len),
                "--k_base", str(args.k_base),
                "--jet_derivative_mode", args.jet_derivative_mode,
                "--jet_max_shift_samples", str(args.jet_max_shift_samples),
                "--ablation_mode", args.ablation_mode,
            ]
            if args.use_scale_jet:
                cmd.append("--use_scale_jet")
            
            if args.use_seed:
                cmd.append("--use_seed")
            
            ret_code = run_command(cmd)
            if ret_code != 0:
                log_message(f"ERROR: Step {step_num} ({dataset} with pred_len={p_len}) failed with exit code {ret_code}. Aborting downstream steps.")
                sys.exit(ret_code)
                
            log_message(f"{dataset} JET HPO (pred_len={p_len}) Completed Successfully!")
            
            step_num += 1
            if step_num <= total_tasks:
                time.sleep(3)

    log_message("ALL JET TUNING EXPERIMENTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
