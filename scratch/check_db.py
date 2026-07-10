import optuna

study = optuna.load_study(
    study_name="best_ettm2_router_96_v1",
    storage="sqlite:///loss_mdagV12/best_ettm2_router_96.db"
)

best_trial = study.best_trial
print("Best Trial Number:", best_trial.number)
print("Best Trial Value (MSE):", best_trial.value)
print("Best Trial Parameters:")
for k, v in best_trial.params.items():
    print(f"  {k}: {v}")

print("Best Trial User Attributes:")
for k, v in best_trial.user_attrs.items():
    print(f"  {k}: {v}")
