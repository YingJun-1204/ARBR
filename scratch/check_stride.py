import os
import re
import json

def check_tune_files():
    tune_files = ["tune_gate_etth1.py", "tune_gate_etth2.py", "tune_gate_ettm1.py", "tune_gate_ettm2.py"]
    print("=== Checking stride and num_gaussians in tune_gate_*.py ===")
    for fname in tune_files:
        if not os.path.exists(fname):
            print(f"{fname} not found!")
            continue
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
        
        lines = content.splitlines()
        print(f"\n--- {fname} ---")
        for i, line in enumerate(lines, 1):
            if ("stride" in line or "num_gaussians" in line) and not line.strip().startswith("#"):
                print(f"L{i}: {line.strip()}")

def check_json_files():
    json_dir = "loss日志"
    print("\n=== Checking JSON configs in loss日志 ===")
    if not os.path.exists(json_dir):
        print(f"{json_dir} directory not found!")
        return
    for fname in sorted(os.listdir(json_dir)):
        if fname.endswith(".json"):
            path = os.path.join(json_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    print(f"\n{fname}:")
                    print(f"  dataset: {data.get('dataset')}")
                    print(f"  stride: {data.get('stride')}")
                    print(f"  patch_len: {data.get('patch_len')}")
                    print(f"  batch_size: {data.get('batch_size')}")
                    print(f"  num_gaussians: {data.get('num_gaussians')}")
                    print(f"  d_model: {data.get('d_model')}")
                    print(f"  d_ff: {data.get('d_ff')}")
                    print(f"  model_scale: {data.get('model_scale')}")
                    print(f"  use_residual: {data.get('use_residual')}")
                except Exception as e:
                    print(f"Error reading {fname}: {e}")

if __name__ == '__main__':
    check_tune_files()
    check_json_files()
