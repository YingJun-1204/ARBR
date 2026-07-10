import optuna

db_path = "sqlite:///loss_mdagV12/best_ettm2_router_96.db"
study = optuna.load_study(study_name="best_ettm2_router_96_v1", storage=db_path)

print(f"Study: {study.study_name}")
print(f"Total trials: {len(study.trials)}")

# Get completed trials
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
completed_trials.sort(key=lambda t: t.value)

print(f"Completed trials: {len(completed_trials)}")
print("\nTop 10 completed trials:")
for idx, trial in enumerate(completed_trials[:10]):
    print(f"Rank {idx+1} | Trial {trial.number} | Value (MSE): {trial.value:.6f}")
    print(f"  Params: {trial.params}")
    print(f"  User Attrs: {trial.user_attrs}")
    print("-" * 50)
