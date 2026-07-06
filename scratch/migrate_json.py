import json
import os

mappings = {
    "best_params_head_dropout_pre_sparse_etth1_pre_256.json": "best_nooc_etth1_256.json",
    "best_params_head_dropout_pre_sparse_etth2_pre_256.json": "best_nooc_etth2_256.json",
    "best_params_head_dropout_pre_sparse_ettm1_pre_256.json": "best_nooc_ettm1_256.json",
    "best_params_head_dropout_pre_sparse_ettm2_pre_256.json": "best_nooc_ettm2_256.json"
}

print("Starting parameter JSON files migration...")
for src, dst in mappings.items():
    if os.path.exists(src):
        try:
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["gs_lambda"] = 0.0
            # Ensure gs_lambda is explicitly 0.0
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            print(f"Successfully migrated {src} -> {dst} (gs_lambda reset to 0.0)")
        except Exception as e:
            print(f"Error migrating {src}: {e}")
    else:
        print(f"Legacy file {src} not found, skipping migration.")
print("Migration completed.")
