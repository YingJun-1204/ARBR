import os
import json
import glob
import re

json_files = glob.glob("loss_jetV15/best_*_*_*_cas.json")

json_data = {}
for f in json_files:
    filename = os.path.basename(f)
    match = re.match(r"best_(.+)_jet_(\d+)_cas\.json", filename)
    if match:
        dataset = match.group(1).lower()
        pred_len = match.group(2)
        with open(f, "r", encoding="utf-8") as fp:
            content = json.load(fp)
        json_data[(dataset, pred_len)] = content

param_keys = [
    "batch_size",
    "learning_rate",
    "gs_dropout",
    "head_dropout",
    "gs_residual_weight",
    "fusion_mode",
    "fusion_init",
    "fusion_beta_max",
    "fusion_hidden_dim",
    "fusion_detach_geometry",
    "jet_score_temperature",
    "jet_max_shift_samples",
]

sh_files = glob.glob("scripts/run_*.sh")
mismatch_count = 0

for sh_file in sorted(sh_files):
    filename = os.path.basename(sh_file)
    dataset = filename.replace("run_", "").replace(".sh", "").lower()
    
    with open(sh_file, "r", encoding="utf-8") as fp:
        sh_text = fp.read()
    
    blocks = re.split(r"echo\s+\"==================== Starting", sh_text)
    
    print(f"\n==================== {filename} ====================")
    for block in blocks[1:]:
        pred_len_match = re.search(r"--pred_len\s+(\d+)", block)
        if not pred_len_match:
            continue
        pred_len = pred_len_match.group(1)
        
        target_json = json_data.get((dataset, pred_len))
        if not target_json:
            print(f"  [MISSING JSON] {dataset} horizon {pred_len}")
            continue
            
        diffs = []
        for key in param_keys:
            if key not in target_json:
                continue
            json_val = target_json[key]
            
            sh_match = re.search(rf"--{key}\s+([^\s\\]+)", block)
            if sh_match:
                sh_val_str = sh_match.group(1).strip()
                try:
                    if isinstance(json_val, float):
                        sh_val = float(sh_val_str)
                        is_diff = abs(sh_val - json_val) > 1e-5
                    elif isinstance(json_val, int):
                        sh_val = int(sh_val_str)
                        is_diff = sh_val != json_val
                    else:
                        sh_val = sh_val_str
                        is_diff = str(sh_val) != str(json_val)
                except ValueError:
                    sh_val = sh_val_str
                    is_diff = str(sh_val) != str(json_val)
                
                if is_diff:
                    diffs.append((key, f"sh: {sh_val_str}", f"json: {json_val}"))
            else:
                diffs.append((key, "sh: [NOT FOUND]", f"json: {json_val}"))
        
        if diffs:
            mismatch_count += len(diffs)
            print(f"--- Dataset {dataset} Horizon {pred_len} ---")
            for d in diffs:
                print(f"    {d[0]}: {d[1]}  ===>  {d[2]}")
        else:
            print(f"--- Dataset {dataset} Horizon {pred_len}: All parameters match! ---")

print(f"\nTotal Mismatches Found: {mismatch_count}")
