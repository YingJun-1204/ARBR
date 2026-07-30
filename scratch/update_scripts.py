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

def format_val(val):
    if isinstance(val, float):
        # Round to 6 decimal places to avoid 0.35000000000000003
        r_val = round(val, 6)
        # Convert to string without trailing zeros if integer float (e.g. 1.0 -> 1 or 1.0 depending on preference, let's keep 1 or 1.0 clean)
        s = f"{r_val:g}"
        return s
    return str(val)

sh_files = glob.glob("scripts/run_*.sh")

total_updates = 0

for sh_file in sorted(sh_files):
    filename = os.path.basename(sh_file)
    dataset = filename.replace("run_", "").replace(".sh", "").lower()
    
    with open(sh_file, "r", encoding="utf-8") as fp:
        sh_text = fp.read()
    
    # We want to replace parameters in each command block
    # Pattern to match each horizon block or split by lines/blocks
    blocks = re.split(r"(echo\s+\"==================== Starting [^\"]+ ====================\")", sh_text)
    
    new_sh_text = blocks[0]
    
    for i in range(1, len(blocks), 2):
        echo_header = blocks[i]
        block_content = blocks[i+1]
        
        pred_len_match = re.search(r"--pred_len\s+(\d+)", block_content)
        if not pred_len_match:
            new_sh_text += echo_header + block_content
            continue
            
        pred_len = pred_len_match.group(1)
        target_json = json_data.get((dataset, pred_len))
        
        if not target_json:
            print(f"Warning: No JSON for {dataset} pred_len {pred_len}")
            new_sh_text += echo_header + block_content
            continue
        
        updated_block = block_content
        for key in param_keys:
            if key not in target_json:
                continue
            json_val = target_json[key]
            val_str = format_val(json_val)
            
            # Find exact pattern: --key value
            def replacer(match):
                global total_updates
                old_val = match.group(1)
                if old_val != val_str:
                    print(f"[{filename} horizon {pred_len}] {key}: {old_val} -> {val_str}")
                    total_updates += 1
                return f"--{key} {val_str}"
            
            updated_block = re.sub(rf"--{key}\s+([^\s\\]+)", replacer, updated_block)
            
        new_sh_text += echo_header + updated_block

    with open(sh_file, "w", encoding="utf-8", newline="\n") as fp:
        fp.write(new_sh_text)

print(f"\nTotal updates made across all scripts: {total_updates}")
