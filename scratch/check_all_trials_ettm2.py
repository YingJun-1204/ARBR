import sqlite3
import json

conn = sqlite3.connect("loss_mdagV12/best_ettm2_router_96.db")
cursor = conn.cursor()

# Get trials
cursor.execute("SELECT trial_id, number, state, value FROM trials ORDER BY value ASC")
trials = cursor.fetchall()

print(f"Total trials in DB: {len(trials)}")
print("\nTop 10 Trials by MSE:")
for t in trials[:10]:
    trial_id, number, state, value = t
    
    # Get parameters
    cursor.execute("SELECT name, value, distribution_json FROM trial_params WHERE trial_id=?", (trial_id,))
    params = cursor.fetchall()
    param_dict = {}
    for p in params:
        name, val, dist = p
        # Check if categorical
        try:
            dist_obj = json.loads(dist)
            if dist_obj["name"] == "CategoricalDistribution":
                # Categorical values are stored as index or value depending on optuna version, let's get the choices
                choices = dist_obj["choices"]
                param_dict[name] = choices[int(val)]
            else:
                param_dict[name] = val
        except Exception:
            param_dict[name] = val
            
    # Get user attributes (like MAE)
    cursor.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=?", (trial_id,))
    attrs = cursor.fetchall()
    attr_dict = {}
    for a in attrs:
        key, val_json = a
        attr_dict[key] = json.loads(val_json)
        
    print(f"Trial {number} | State: {state} | MSE: {value} | MAE: {attr_dict.get('mae', 'N/A')}")
    print(f"  Params: {param_dict}")
    print(f"  Attrs: {attr_dict}")
    print("-" * 50)

conn.close()
