#!/usr/bin/env bash
set -euo pipefail

# Reproduce ARBD benchmark on Electricity across 4 horizons: 96, 192, 336, 720
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Hyperparameters for sensitivity analysis (default: rho=0.5, beta=0.5)
RHO="${RHO:-0.5}"
BETA="${BETA:-0.5}"

echo "================================================================================"
echo "                    Training ARBD: Electricity Benchmark Suite                   "
echo "================================================================================"

# ------------------------------------------------------------------------------
# 1. Horizon H = 96
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path electricity.csv \
  --model_id reproduce_electricity_96 \
  --model ARBD \
  --data custom \
  --features M \
  --pred_len 96 \
  --enc_in 321 \
  --batch_size 32 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.2 \
  --learning_rate 0.00025 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.25 \
  --local_weight_max 0.75 \
  --static_local_weight_init 0.35 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_Electricity_96 \
  "$@"

# ------------------------------------------------------------------------------
# 2. Horizon H = 192
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path electricity.csv \
  --model_id reproduce_electricity_192 \
  --model ARBD \
  --data custom \
  --features M \
  --pred_len 192 \
  --enc_in 321 \
  --batch_size 32 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.5 \
  --learning_rate 0.0004 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.2 \
  --local_weight_max 1.0 \
  --static_local_weight_init 0.1 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_Electricity_192 \
  "$@"

# ------------------------------------------------------------------------------
# 3. Horizon H = 336
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path electricity.csv \
  --model_id reproduce_electricity_336 \
  --model ARBD \
  --data custom \
  --features M \
  --pred_len 336 \
  --enc_in 321 \
  --batch_size 32 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.2 \
  --learning_rate 0.0004 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.55 \
  --local_weight_max 0.5 \
  --static_local_weight_init 0.35 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_Electricity_336 \
  "$@"

# ------------------------------------------------------------------------------
# 4. Horizon H = 720
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path electricity.csv \
  --model_id reproduce_electricity_720 \
  --model ARBD \
  --data custom \
  --features M \
  --pred_len 720 \
  --enc_in 321 \
  --batch_size 32 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.45 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.3 \
  --local_weight_max 0.75 \
  --static_local_weight_init 0.5 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_Electricity_720 \
  "$@"
