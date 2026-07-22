import argparse
import json
import os
import re
import subprocess
import sys

import optuna

# Basic dataset configuration for weather
DATASET_CONFIGS = {
    "weather": {
        "data_path": "weather.csv",
        "seq_len": 512,
        "batch_size": 256,
        "patience": 6,
        "dropout": 0.5,
        "head_dropout": 0.05,
        "learning_rate_lower": 1e-4,
        "learning_rate_upper": 1.5e-3,
        "extra_args": [],
    }
}

dataset_lower = "weather"
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
}

PARAM_KEYS = {
    "gs_dropout",
    "train_epochs",
    "learning_rate",
    "head_dropout",
    "batch_size",
    "fusion_init",
    "fusion_beta_max",
}


# Removed enforce_train_after_freeze


def get_output_json_path(output_dir, pred_len=96, density_mode=None):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{density_mode}" if density_mode in ["cas"] else ""
    return os.path.join(output_dir, f"best_weather_jet_{pred_len}{suffix}.json")


def get_storage_name(pred_len=96, output_dir=None):
    db_name = f"best_weather_jet_{pred_len}.db"
    if output_dir:
        return f"sqlite:///{os.path.join(output_dir, db_name)}"
    return f"sqlite:///{db_name}"


def get_study_name(pred_len=96):
    return f"best_weather_jet_{pred_len}_v1"


def make_bounds():
    return DEFAULT_BOUNDS.copy()


def sample_params(trial, bounds, position, batch_size=256):
    gs_dropout = trial.suggest_float(
        "gs_dropout",
        bounds["gs_dropout_lower"],
        bounds["gs_dropout_upper"],
        step=0.05,
    )

    batch_size = trial.suggest_categorical("batch_size", [batch_size])

    # Geometry-conditioned fusion parameters.
    fusion_init = trial.suggest_categorical(
        "fusion_init",
        [0.10, 0.20, 0.35, 0.50],
    )

    fusion_beta_max = trial.suggest_categorical(
        "fusion_beta_max",
        [0.50, 0.75, 1.00],
    )

    # Encoder requires:
    # 0 < fusion_init < fusion_beta_max
    if fusion_init >= fusion_beta_max:
        raise optuna.TrialPruned(
            "Invalid geometry fusion configuration: "
            f"fusion_init={fusion_init} must be smaller than "
            f"fusion_beta_max={fusion_beta_max}"
        )

    params = {
        "patch_len": 24,
        "stride": 12,
        "num_gaussians": bounds.get("num_gaussians_lower", 8),
        "d_model": 128,
        "gs_dropout": gs_dropout,
        "gs_weight_decay": 1e-4,
        "train_epochs": 30,
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1e-3, 1.2e-3, 1.5e-3, 1.8e-3, 2e-3]),
        "gs_lambda": 0.0,
        "batch_size": batch_size,
        "use_residual": True,

        # Keep fixed fusion coefficient only for compatibility
        # and later fixed-fusion ablation.
        "gs_residual_weight": 0.1,

        # Full model always uses geometry fusion.
        "fusion_mode": "geometry",
        "fusion_init": fusion_init,
        "fusion_beta_max": fusion_beta_max,

        # Keep structural choices fixed during this HPO.
        "fusion_hidden_dim": 16,
        "fusion_detach_geometry": 1,

        # Keep the corrected Jet geometry settings fixed.
        "jet_score_temperature": 0.01,
        "jet_max_shift_samples": 1.0,
    }

    if position == "none":
        params["head_dropout"] = 0.0
    else:
        params["head_dropout"] = trial.suggest_float(
            "head_dropout",
            bounds["head_dropout_lower"],
            bounds["head_dropout_upper"],
            step=0.05,
        )

    return params


def build_command(position, params, trial_number, density_mode="cas", pred_len=96, output_dir="loss_cas_simplify", jet_derivative_mode="exact"):
    if params.get("fusion_mode") != "geometry":
        raise RuntimeError(
            "This HPO script is intended for the full "
            "geometry-conditioned fusion model, but received "
            f"fusion_mode={params.get('fusion_mode')}"
        )

    cfg = DATASET_CONFIGS["weather"]
    head_dropout = 0.0 if position == "none" else params["head_dropout"]
    model_id = f"tune_weather_jet_trial_{trial_number}"
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
        "21",
        "--d_model",
        "128",
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

        "--fusion_mode",
        params["fusion_mode"],
        "--fusion_init",
        str(params["fusion_init"]),
        "--fusion_beta_max",
        str(params["fusion_beta_max"]),
        "--fusion_hidden_dim",
        str(params["fusion_hidden_dim"]),
        "--fusion_detach_geometry",
        str(params["fusion_detach_geometry"]),
        "--jet_score_temperature",
        str(params["jet_score_temperature"]),
        "--jet_max_shift_samples",
        str(params["jet_max_shift_samples"]),

        "--des",
        f"Ablation_weather_jet",
    ]

    if "k_base" in cfg:
        cmd.extend(["--k_base", str(cfg["k_base"])])
    cmd.extend(["--jet_derivative_mode", jet_derivative_mode])
    cmd.extend(["--no_save_checkpoint"])
    cmd.extend(["--output_dir", output_dir])

    if "--fusion_mode" not in cmd:
        raise RuntimeError(
            "--fusion_mode was not added to the run.py command"
        )
    fusion_mode_index = cmd.index("--fusion_mode")
    if cmd[fusion_mode_index + 1] != "geometry":
        raise RuntimeError(
            "HPO command must run with --fusion_mode geometry"
        )

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


def make_objective(position, bounds, density_mode, pred_len, output_dir, batch_size=256, jet_derivative_mode="exact"):
    def objective(trial):
        params = sample_params(trial, bounds, position, batch_size=batch_size)
        cmd = build_command(
            position, params, trial.number, density_mode, pred_len, 
            output_dir=output_dir,
            jet_derivative_mode=jet_derivative_mode
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
            
        # Health check penalties removed
            
        return mse

    return objective


def save_best_callback(position, output_dir, density_mode, pred_len, num_gaussians=None):
    def callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        # 1. Manage diagnostics files (runs for every trial)
        trial_diag_name = f"diagnostics_tune_{dataset_lower}_jet_trial_{trial.number}.txt"
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
        try:
            best_trial_num = study.best_trial.number
        except Exception:
            best_trial_num = trial.number

        if best_trial_num != trial.number:
            return

        json_path = get_output_json_path(output_dir, pred_len, density_mode)
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
            best_tuned["dataset"] = "weather"
            best_tuned["mse"] = current_mse
            best_tuned["mae"] = study.best_trial.user_attrs.get("mae", -1.0)
            
            best_params = study.best_params
            best_tuned["batch_size"] = int(best_params.get("batch_size", 256))
            best_tuned["learning_rate"] = float(best_params.get("learning_rate", 1e-3))
            
            if "gs_dropout" in best_params:
                best_tuned["gs_dropout"] = float(best_params["gs_dropout"])
            if "head_dropout" in best_params:
                best_tuned["head_dropout"] = float(best_params["head_dropout"])
            
            best_tuned["gs_residual_weight"] = 0.1
            best_tuned["fusion_mode"] = "geometry"
            best_tuned["fusion_init"] = float(best_params.get("fusion_init", 0.1))
            best_tuned["fusion_beta_max"] = float(best_params.get("fusion_beta_max", 0.5))
            best_tuned["fusion_hidden_dim"] = 16
            best_tuned["fusion_detach_geometry"] = 1
            best_tuned["jet_score_temperature"] = 0.01
            best_tuned["jet_max_shift_samples"] = 1.0
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(best_tuned, f, indent=4)

            print(f"\n[Optuna Callback] Saved {json_path}! MSE: {best_tuned['mse']:.6f}")

    return callback


def load_seed_params(position, experiment_tag="simplify", pred_len=96, batch_size=256):
    seed_path = os.path.join("tune_seed", f"best_weather_jet_{pred_len}_cas.json")
    if not os.path.exists(seed_path):
        print(f"[Seed Warning] Seed file not found at {seed_path}. Falling back to random search.")
        return None, None

    print(f"[Seed] Loading HPO seed from: {seed_path}")
    with open(seed_path, "r", encoding="utf-8") as f:
        seed = json.load(f)

    seed_trial = {}
    for key in ["learning_rate", "batch_size", "fusion_init", "fusion_beta_max"]:
        if key in seed:
            seed_trial[key] = seed[key]

    seed_trial["batch_size"] = seed_trial.get("batch_size", 256)
    seed_trial["fusion_init"] = float(seed_trial.get("fusion_init", 0.1))
    seed_trial["fusion_beta_max"] = float(seed_trial.get("fusion_beta_max", 0.5))

    if seed_trial["fusion_init"] >= seed_trial["fusion_beta_max"]:
        print(
            "[Seed Warning] Invalid fusion seed: "
            f"fusion_init={seed_trial['fusion_init']} >= "
            f"fusion_beta_max={seed_trial['fusion_beta_max']}. "
            "Resetting to 0.1 / 0.5."
        )
        seed_trial["fusion_init"] = 0.1
        seed_trial["fusion_beta_max"] = 0.5
    
    dropout_val = seed.get("gs_dropout", seed.get("dropout", 0.9))
    seed_trial["gs_dropout"] = float(dropout_val)
    if position != "none":
        seed_trial["head_dropout"] = float(seed.get("head_dropout", dropout_val))
    


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
    print(f"[Seed] Enqueued {seed_path} for weather/{position}")
def main():
    parser = argparse.ArgumentParser(description="Ablation HPO script for weather jet tuning.")
    parser.add_argument("--pred_len", type=int, nargs="+", default=[96], help="Prediction horizons (default: [96])")
    parser.add_argument("--batch_size", type=int, default=256, choices=[16, 32, 128, 256, 512, 1024], help="Fixed batch size for HPO (default: 256)")
    parser.add_argument("--output_dir", type=str, default="loss_cas_simplify", help="Output directory for best json configs")
    parser.add_argument("--position", type=str, default="pre", choices=POSITIONS, help="Head dropout position")
    parser.add_argument("--density_mode", type=str, default="cas", choices=DENSITY_MODES, help="Density mode selection")
    parser.add_argument("--batches", type=int, default=20, help="Number of optuna batches")
    parser.add_argument("--trials_per_batch", type=int, default=15, help="Number of trials per optuna batch")
    parser.add_argument("--use_seed", action="store_true", help="Use previous best parameters as HPO seeds")
    parser.add_argument("--num_gaussians", type=int, default=None, help="Fixed number of gaussians (disables tuning)")
    
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length / lookback window (default: 512)")
    parser.add_argument("--k_base", type=int, default=-1, help="Manual k_base value for CAS gating (-1 means dynamic)")
    parser.add_argument("--jet_derivative_mode", type=str, default="exact", choices=["exact", "centered_legacy"], help="Gaussian Jet derivative mode (exact or centered_legacy)")
    args = parser.parse_args()
    for d_name in DATASET_CONFIGS:
        DATASET_CONFIGS[d_name]["seq_len"] = args.seq_len
        DATASET_CONFIGS[d_name]["k_base"] = args.k_base
    
    # 确保输出文件夹存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    for p_len in args.pred_len:
        print(f"\n=== [MODE: HPO JET] Starting tuning for pred_len={p_len} ===")
        bounds = make_bounds()
        if args.num_gaussians is not None:
            bounds["num_gaussians_lower"] = args.num_gaussians
            bounds["num_gaussians_upper"] = args.num_gaussians
        seed_trial, seed_path = None, None
        if args.use_seed:
            seed_trial, seed_path = load_seed_params(args.position, pred_len=p_len, batch_size=args.batch_size)

        study = optuna.create_study(
            direction="minimize",
            study_name=get_study_name(p_len),
            storage=get_storage_name(p_len, output_dir=args.output_dir),
            load_if_exists=True,
        )
        if args.use_seed and seed_trial:
            enqueue_seed(study, seed_trial, seed_path, args.position)
     
        for batch in range(1, args.batches + 1):
            print(f"\n[Manager] Gate: jet | pred_len: {p_len} | batch {batch}/{args.batches}")
            study.optimize(
                make_objective(
                    position=args.position,
                    bounds=bounds,
                    density_mode=args.density_mode,
                    pred_len=p_len,
                    output_dir=args.output_dir,
                    batch_size=args.batch_size,
                    jet_derivative_mode=args.jet_derivative_mode
                ),
                n_trials=args.trials_per_batch,
                callbacks=[
                    save_best_callback(
                        position=args.position,
                        output_dir=args.output_dir,
                        density_mode=args.density_mode,
                        pred_len=p_len,
                        num_gaussians=args.num_gaussians
                    )
                ],
            )
     
        try:
            best_val = study.best_value
        except Exception:
            best_val = float("inf")
        print(f"\n[Manager] HPO completed for Gate JET | pred_len: {p_len}. Best MSE: {best_val:.6f}")
        
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
