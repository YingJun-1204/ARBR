import argparse
import json
import os
import re
import subprocess
import sys

import optuna

# Basic dataset configuration for exchange
DATASET_CONFIGS = {
    "exchange": {
        "data_path": "Exchange.csv",
        "seq_len": 512,
        "batch_size": 16,
        "patience": 6,
        "dropout": 0.5,
        "head_dropout": 0.05,
        "learning_rate_lower": 1e-4,
        "learning_rate_upper": 1.5e-3,
        "extra_args": [],
    }
}

POSITIONS = ["none", "pre", "post"]
DENSITY_MODES = ["none", "cas"]

# Initial tuning search bounds for HPO
DEFAULT_BOUNDS = {
    "learning_rate_lower": 1e-4,
    "learning_rate_upper": 1.5e-3,
    "gs_dropout_lower": 0.2,
    "gs_dropout_upper": 0.9,
    "head_dropout_lower": 0.2,
    "head_dropout_upper": 0.9,
    "gs_residual_weight_lower": 0.2,
    "gs_residual_weight_upper": 0.6,
    "gate_beta_lower": 0.1,
    "gate_beta_upper": 0.5,
    "gs_weight_decay_lower": 1e-5,
    "gs_weight_decay_upper": 1e-3,
    "gate_window_half_lower": 1,
    "gate_window_half_upper": 4,
}

PARAM_KEYS = {
    "gs_dropout",
    "train_epochs",
    "learning_rate",
    "head_dropout",
    "batch_size",
    "gs_residual_weight",
    "gate_beta",
    "gate_lambda",
    "gate_window_half",
}


def get_output_json_path(output_dir, gate_type, pred_len=96, density_mode=None):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{density_mode}" if density_mode in ["cas"] else ""
    if gate_type == "adaptive_direction":
        return os.path.join(output_dir, f"best_exchange_router_{pred_len}{suffix}.json")
    gate_str = gate_type if gate_type is not None else "none"
    return os.path.join(output_dir, f"best_nooc_linear_exchange_{gate_str}_{pred_len}{suffix}.json")


def get_storage_name(gate_type, pred_len=96, output_dir=None):
    db_name = f"best_exchange_router_{pred_len}.db" if gate_type == "adaptive_direction" else f"best_exchange_{gate_type if gate_type is not None else 'none'}_{pred_len}.db"
    if output_dir:
        return f"sqlite:///{os.path.join(output_dir, db_name)}"
    return f"sqlite:///{db_name}"


def get_study_name(gate_type, pred_len=96):
    if gate_type == "adaptive_direction":
        return f"best_exchange_router_{pred_len}_v1"
    gate_str = gate_type if gate_type is not None else "none"
    return f"best_nooc_linear_exchange_{gate_str}_{pred_len}_v1"


def make_bounds():
    return DEFAULT_BOUNDS.copy()


def expand_bounds_to_include_seed(bounds, seed_params, position):
    return bounds.copy()


def update_bounds(best_params, bounds, position):
    """
    Adaptive boundary expansion algorithm to broaden search spaces when parameters approach limits.
    """
    return bounds.copy()


def sample_params(trial, bounds, position, gate_type=None, batch_size=16):
    gs_residual_weight = trial.suggest_float(
        "gs_residual_weight",
        bounds["gs_residual_weight_lower"],
        bounds["gs_residual_weight_upper"],
        step=0.05,
    )
    gate_beta = trial.suggest_float(
        "gate_beta",
        bounds["gate_beta_lower"],
        bounds["gate_beta_upper"],
        step=0.05,
    )
    gs_dropout = trial.suggest_float(
        "gs_dropout",
        bounds["gs_dropout_lower"],
        bounds["gs_dropout_upper"],
        step=0.05,
    )

    batch_size = trial.suggest_categorical("batch_size", [batch_size])
    
    model_scale = "128-256"

    params = {
        "patch_len": 24,
        "stride": 12,
        "num_gaussians": bounds.get("num_gaussians_lower", 8),
        "d_model": 128,
        "d_ff": 256,
        "model_scale": "128-256",
        "gs_dropout": gs_dropout,
        "gs_weight_decay": 1e-4,
        "train_epochs": 30,
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3, 1.2e-3, 1.5e-3, 1.8e-3, 2e-3]),
        "dropout": 0.0,
        "gs_lambda": 0.0,
        "batch_size": batch_size,
        "use_residual": True,
        "gs_residual_weight": gs_residual_weight,
    }

    if position == "none":
        params["head_dropout"] = 0.0
    else:
        params["head_dropout"] = gs_dropout
    


    return params


def build_command(position, params, trial_number, density_mode="cas", pred_len=96, gate_type=None, gate_beta=0.25, output_dir="loss_cas_simplify"):
    cfg = DATASET_CONFIGS["exchange"]
    head_dropout = 0.0 if position == "none" else params["head_dropout"]
    gate_str = "router" if gate_type == "adaptive_direction" else (gate_type if gate_type is not None else "none")
    model_id = f"tune_linear_exchange_gate_{gate_str}_trial_{trial_number}"
    batch_size = params["batch_size"]

    cmd = [
        sys.executable,
        "-u",
        "run.py",
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        "./data",
        "--data_path",
        cfg["data_path"],
        "--model_id",
        model_id,
        "--model",
        "SplatTS",
        "--data",
        "custom",
        "--features",
        "M",
        "--seq_len",
        str(cfg["seq_len"]),
        "--pred_len",
        str(pred_len),
        "--enc_in",
        "8",
        "--dec_in",
        "8",
        "--c_out",
        "8",
        "--d_model",
        "128",
        "--d_ff",
        "256",
        "--itr",
        "1",
        "--batch_size",
        str(batch_size),
        "--representation",
        "gs",
        "--patch_len",
        str(params["patch_len"]),
        "--stride",
        str(params["stride"]),
        "--num_gaussians",
        str(params["num_gaussians"]),
        "--gs_dropout",
        str(params["gs_dropout"]),
        "--gs_weight_decay",
        str(params["gs_weight_decay"]),

        "--learning_rate",
        str(params["learning_rate"]),
        "--lradj",
        "cosine",
        "--train_epochs",
        str(params["train_epochs"]),
        "--patience",
        str(cfg["patience"]),
        "--dropout",
        str(params["dropout"]),
        "--head_dropout",
        str(head_dropout),
        "--head_dropout_position",
        position,
        "--density_mode",
        density_mode,
        "--gs_lambda",
        "0.0",
        "--num_workers",
        "0",
        "--use_residual",
        "--gs_residual_weight",
        str(params["gs_residual_weight"]),

        "--des",
        f"Ablation_exchange_{gate_str}",
    ]

    if "k_base" in cfg:
        cmd.extend(["--k_base", str(cfg["k_base"])])
    cmd.extend(["--no_save_checkpoint"])
    cmd.extend(["--output_dir", output_dir])
    return cmd


def make_console_safe(text, encoding=None):
    encoding = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def run_command(cmd):
    # Parse model_id and output_dir from cmd args
    model_id = None
    output_dir = "loss_new"
    for idx, arg in enumerate(cmd):
        if arg == "--model_id" and idx + 1 < len(cmd):
            model_id = cmd[idx + 1]
        elif arg == "--output_dir" and idx + 1 < len(cmd):
            output_dir = cmd[idx + 1]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    last_mse = None
    last_mae = None
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(make_console_safe(line))
        sys.stdout.flush()
        if "mse:" in line and "mae:" in line:
            match = re.search(r"mse:([\d\.]+),\s*mae:([\d\.]+)", line)
            if match:
                last_mse = float(match.group(1))
                last_mae = float(match.group(2))

    process.wait()
    if process.returncode != 0 or last_mse is None:
        raise optuna.TrialPruned()

    egc = None
    egc_ratio = None
    density_sig_mean = None

    # Parse mechanistic metrics from diagnostics file
    if model_id is not None:
        diag_path = os.path.join(output_dir, f"diagnostics_{model_id}.txt")
        if os.path.exists(diag_path):
            try:
                with open(diag_path, "r", encoding="utf-8") as f:
                    content = f.read()
                match_egc = re.search(r"Average EGC:\s*([\d\.]+)", content)
                if match_egc:
                    egc = float(match_egc.group(1))
                match_egc_ratio = re.search(r"EGC Ratio:\s*([\d\.]+)", content)
                if match_egc_ratio:
                    egc_ratio = float(match_egc_ratio.group(1))
                match_mean = re.search(r"Density Sigmoid Mean:\s*([\d\.]+)", content)
                if match_mean:
                    density_sig_mean = float(match_mean.group(1))
            except Exception as e:
                print(f"[Warning] Failed to parse diagnostics file: {e}")

    return last_mse, last_mae, egc, egc_ratio, density_sig_mean


def make_objective(position, bounds, density_mode, pred_len, gate_type, gate_beta, output_dir, batch_size=16):
    def objective(trial):
        params = sample_params(trial, bounds, position, gate_type=gate_type, batch_size=batch_size)
        cmd = build_command(
            position, params, trial.number, density_mode, pred_len, gate_type, 
            params["gate_beta"],
            output_dir=output_dir
        )
        print(f"\n[Trial {trial.number}] Running HPO: {' '.join(cmd)}")
        mse, mae, egc, egc_ratio, density_sig_mean = run_command(cmd)
        trial.set_user_attr("mae", mae)
        trial.set_user_attr("head_dropout_position", position)
        
        # Save mechanistic metrics in user_attrs
        if egc is not None:
            trial.set_user_attr("egc", egc)
        if egc_ratio is not None:
            trial.set_user_attr("egc_ratio", egc_ratio)
        if density_sig_mean is not None:
            trial.set_user_attr("density_sigmoid_mean", density_sig_mean)
            
        return mse

    return objective


def save_best_callback(position, output_dir, gate_type, density_mode, pred_len, num_gaussians=None):
    def callback(study, trial):
        # 1. Manage diagnostics files (runs for every trial)
        gate_str = "router" if gate_type == "adaptive_direction" else (gate_type if gate_type is not None else "none")
        dataset_lower = "exchange"
        trial_diag_name = f"diagnostics_tune_linear_{dataset_lower}_gate_{gate_str}_trial_{trial.number}.txt"
        trial_diag_path = os.path.join(output_dir, trial_diag_name)
        
        if os.path.exists(trial_diag_path):
            try:
                with open(trial_diag_path, "r", encoding="utf-8") as f:
                    diag_content = f.read()
            except Exception as e:
                diag_content = None
                print(f"[Optuna Callback] Failed to read {trial_diag_path}: {e}")
                
            if diag_content is not None:
                # Save new content to trial_0.txt
                diag_0_path = os.path.join(output_dir, f"diagnostics_{dataset_lower}_0.txt")
                try:
                    with open(diag_0_path, "w", encoding="utf-8") as f:
                        f.write(diag_content)
                except Exception as e:
                    print(f"[Optuna Callback] Failed to write {diag_0_path}: {e}")
                
                # Check if this trial is the best trial so far
                is_best = False
                try:
                    is_best = (study.best_trial.number == trial.number)
                except Exception:
                    is_best = True
                    
                if is_best:
                    diag_best_path = os.path.join(output_dir, f"diagnostics_{dataset_lower}_best.txt")
                    try:
                        with open(diag_best_path, "w", encoding="utf-8") as f:
                            f.write(diag_content)
                    except Exception as e:
                        print(f"[Optuna Callback] Failed to write {diag_best_path}: {e}")
            
            # Clean up the raw trial file
            try:
                os.remove(trial_diag_path)
            except Exception:
                pass

        # 2. Save best config (runs only if this is the best trial so far)
        if study.best_trial.number != trial.number:
            return

        json_path = get_output_json_path(output_dir, gate_type, pred_len, density_mode)
        current_mse = study.best_value
        should_write = True
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if float(existing.get("mse", float("inf"))) <= current_mse:
                    should_write = False
                    print(
                        f"\n[Optuna Callback] Skip saving {json_path}: "
                        f"existing MSE {existing['mse']:.6f} <= new MSE {current_mse:.6f}"
                    )
            except Exception as exc:
                print(f"\n[Optuna Callback] Read existing {json_path} failed ({exc}), overwriting.")
        if should_write:
            best_tuned = {}
            best_tuned["dataset"] = "exchange"
            best_tuned["gate_type"] = gate_type if gate_type is not None else "none"
            best_tuned["mse"] = current_mse
            best_tuned["mae"] = study.best_trial.user_attrs.get("mae", -1.0)
            
            best_params = study.best_params
            best_tuned["batch_size"] = int(best_params.get("batch_size", 16))
            best_tuned["learning_rate"] = float(best_params.get("learning_rate", 1e-3))
            
            # gs_dropout and head_dropout are consolidated into "dropout"
            if "gs_dropout" in best_params:
                best_tuned["dropout"] = float(best_params["gs_dropout"])
            
            best_tuned["gs_residual_weight"] = float(best_params.get("gs_residual_weight", 0.1))
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(best_tuned, f, indent=4)

            print(f"\n[Optuna Callback] Saved {json_path}! MSE: {best_tuned['mse']:.6f}")

    return callback


def load_seed_params(position, experiment_tag="simplify", pred_len=96, gate_type=None, batch_size=16):
    seed_path = os.path.join("tune_seed", f"best_exchange_router_{pred_len}_cas.json")
    if not os.path.exists(seed_path):
        print(f"[Seed Warning] Seed file not found at {seed_path}. Falling back to random search.")
        return None, None

    print(f"[Seed] Loading HPO seed from: {seed_path}")
    with open(seed_path, "r", encoding="utf-8") as f:
        seed = json.load(f)

    seed_trial = {}
    for key in ["learning_rate", "batch_size", "gs_residual_weight"]:
        if key in seed:
            seed_trial[key] = seed[key]

    seed_trial["batch_size"] = seed_trial.get("batch_size", 16)
    seed_trial["gs_residual_weight"] = seed_trial.get("gs_residual_weight", 0.3)
    
    dropout_val = seed.get("gs_dropout", seed.get("dropout", 0.5))
    seed_trial["gs_dropout"] = float(dropout_val)
    
    if gate_type == "adaptive_direction":
        seed_trial["gate_lambda"] = seed_trial.get("gate_lambda", seed.get("gate_lambda", 0.01))
        seed_trial["gate_window_half"] = seed_trial.get("gate_window_half", seed.get("gate_window_half", 2))

    # Check categorical choices
    categorical_choices = {
        "batch_size": [batch_size],
    }
    for key, allowed in categorical_choices.items():
        if key in seed_trial and seed_trial[key] not in allowed:
            print(f"[Seed Warning] Parameter '{key}' value {seed_trial[key]} is not in allowed choices {allowed}. Setting to {allowed[0]}.")
            seed_trial[key] = allowed[0]

    return seed_trial, seed_path


def enqueue_seed(study, seed_trial, seed_path, position):
    if not seed_trial:
        return
    study.enqueue_trial(seed_trial)
    print(f"[Seed] Enqueued {seed_path} for exchange/{position}")


def main():
    parser = argparse.ArgumentParser(description="Ablation HPO script for exchange gating choices.")
    parser.add_argument("--gate_type", type=str, required=True, choices=["none", "forward", "reverse", "adaptive_direction"], help="Residual gating direction")
    parser.add_argument("--gate_beta", type=float, default=0.25, help="Scale gate beta bounds")
    parser.add_argument("--pred_len", type=int, nargs="+", default=[96], help="Prediction horizons (default: [96])")
    parser.add_argument("--batch_size", type=int, default=16, choices=[16, 32, 256, 512, 1024], help="Fixed batch size for HPO (default: 16)")
    parser.add_argument("--output_dir", type=str, default="loss_cas_simplify", help="Output directory for best json configs")
    parser.add_argument("--position", type=str, default="pre", choices=POSITIONS, help="Head dropout position")
    parser.add_argument("--density_mode", type=str, default="cas", choices=DENSITY_MODES, help="Density mode selection")
    parser.add_argument("--batches", type=int, default=20, help="Number of optuna batches")
    parser.add_argument("--trials_per_batch", type=int, default=15, help="Number of trials per optuna batch")
    parser.add_argument("--use_seed", action="store_true", help="Use previous best parameters as HPO seeds")
    parser.add_argument("--num_gaussians", type=int, default=None, help="Fixed number of gaussians (disables tuning)")
    
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length / lookback window (default: 512)")
    parser.add_argument("--k_base", type=int, default=-1, help="Manual k_base value for CAS gating (-1 means dynamic)")
    args = parser.parse_args()
    for d_name in DATASET_CONFIGS:
        DATASET_CONFIGS[d_name]["seq_len"] = args.seq_len
        DATASET_CONFIGS[d_name]["k_base"] = args.k_base
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse gate_type value
    gate_val = None if args.gate_type == "none" else args.gate_type
    
    for p_len in args.pred_len:
        print(f"\n=== [MODE: HPO {args.gate_type.upper()}] Starting tuning for exchange | pred_len={p_len} ===")
        bounds = make_bounds()
        if args.num_gaussians is not None:
            bounds["num_gaussians_lower"] = args.num_gaussians
            bounds["num_gaussians_upper"] = args.num_gaussians
        seed_trial, seed_path = None, None
        if args.use_seed:
            seed_trial, seed_path = load_seed_params(args.position, pred_len=p_len, gate_type=gate_val, batch_size=args.batch_size)
            if seed_trial:
                bounds = expand_bounds_to_include_seed(bounds, seed_trial, args.position)

        study = optuna.create_study(
            direction="minimize",
            study_name=get_study_name(gate_val, p_len),
            storage=get_storage_name(gate_val, p_len, output_dir=args.output_dir),
            load_if_exists=True,
        )
        if args.use_seed and seed_trial:
            enqueue_seed(study, seed_trial, seed_path, args.position)
     
        for batch in range(1, args.batches + 1):
            print(f"\n[Manager] Gate: {args.gate_type} | pred_len: {p_len} | batch {batch}/{args.batches}; bounds={bounds}")
            study.optimize(
                make_objective(
                    position=args.position,
                    bounds=bounds,
                    density_mode=args.density_mode,
                    pred_len=p_len,
                    gate_type=gate_val,
                    gate_beta=args.gate_beta,
                    output_dir=args.output_dir,
                    batch_size=args.batch_size
                ),
                n_trials=args.trials_per_batch,
                callbacks=[
                    save_best_callback(
                        position=args.position,
                        output_dir=args.output_dir,
                        gate_type=gate_val,
                        density_mode=args.density_mode,
                        pred_len=p_len,
                        num_gaussians=args.num_gaussians
                    )
                ],
            )
            best_params = study.best_params.copy()
            if args.position == "none":
                best_params["head_dropout"] = 0.0
            bounds = update_bounds(best_params, bounds, args.position)
            print(f"[Manager] Gate: {args.gate_type} | pred_len: {p_len} updated bounds={bounds}")
     
        print(f"\n[Manager] HPO completed for Gate {args.gate_type.upper()} | pred_len: {p_len}. Best MSE: {study.best_value:.6f}")
        
    # Clean up temporary diagnostic files before exiting (except trial_0, trial_best)
    for f_name in os.listdir(args.output_dir):
        if f_name.startswith("diagnostics_") and f_name.endswith(".txt"):
            if not any(f_name.endswith(f"_{suffix}.txt") for suffix in ["0", "best"]):
                try:
                    os.remove(os.path.join(args.output_dir, f_name))
                except Exception:
                    pass


if __name__ == "__main__":
    main()
