import os

scripts = [
    "tune_gate_etth2.py",
    "tune_gate_ettm1.py",
    "tune_gate_ettm2.py",
    "tune_gate_weather.py",
    "tune_gate_electricity.py",
]

for script in scripts:
    if not os.path.exists(script):
        print(f"Warning: {script} does not exist in current directory.")
        continue
        
    with open(script, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Modify build_command return logic
    target_build = """    if gate_type == "adaptive_direction":
        cmd.extend(["--gate_lambda", str(params.get("gate_lambda", 0.0))])
        cmd.extend(["--gate_window_half", str(params.get("gate_window_half", 2))])
    cmd.extend(["--no_save_checkpoint"])
    cmd.extend(["--output_dir", output_dir])
    return cmd"""

    replacement_build = """    if gate_type == "adaptive_direction":
        cmd.extend(["--gate_lambda", str(params.get("gate_lambda", 0.0))])
        cmd.extend(["--gate_window_half", str(params.get("gate_window_half", 2))])
    if "k_base" in cfg:
        cmd.extend(["--k_base", str(cfg["k_base"])])
    cmd.extend(["--no_save_checkpoint"])
    cmd.extend(["--output_dir", output_dir])
    return cmd"""

    if target_build in content:
        content = content.replace(target_build, replacement_build)
    else:
        print(f"Error: Target build not found in {script}")

    # 2. Modify parser and DATASET_CONFIGS iteration logic
    target_parser = """    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length / lookback window (default: 512)")
    args = parser.parse_args()
    for d_name in DATASET_CONFIGS:
        DATASET_CONFIGS[d_name]["seq_len"] = args.seq_len"""

    replacement_parser = """    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length / lookback window (default: 512)")
    parser.add_argument("--k_base", type=int, default=-1, help="Manual k_base value for CAS gating (-1 means dynamic)")
    args = parser.parse_args()
    for d_name in DATASET_CONFIGS:
        DATASET_CONFIGS[d_name]["seq_len"] = args.seq_len
        DATASET_CONFIGS[d_name]["k_base"] = args.k_base"""

    if target_parser in content:
        content = content.replace(target_parser, replacement_parser)
    else:
        print(f"Error: Target parser not found in {script}")

    with open(script, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully patched {script}")
