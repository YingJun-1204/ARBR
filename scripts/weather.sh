#!/usr/bin/env bash
set -euo pipefail

# Reproduce ARBD benchmark on Weather across 4 horizons: 96, 192, 336, 720
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Hyperparameters for sensitivity analysis (default: rho=0.5, beta=0.5)
RHO="${RHO:-0.5}"
BETA="${BETA:-0.5}"

echo "================================================================================"
echo "                    Training ARBD: Weather Benchmark Suite                   "
echo "================================================================================"

# ------------------------------------------------------------------------------
# 1. Horizon H = 96
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id reproduce_weather_96 \
  --model ARBD \
  --data custom \
  --features M \
  --pred_len 96 \
  --enc_in 21 \
  --batch_size 256 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.55 \
  --learning_rate 0.0012 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.35 \
  --local_weight_max 1.0 \
  --static_local_weight_init 0.1 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_Weather_96 \
  "$@"

# ------------------------------------------------------------------------------
# 2. Horizon H = 192
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id reproduce_weather_192 \
  --model ARBD \
  --data custom \
  --features M \
  --pred_len 192 \
  --enc_in 21 \
  --batch_size 256 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.5 \
  --learning_rate 0.0008 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.55 \
  --local_weight_max 1.0 \
  --static_local_weight_init 0.1 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_Weather_192 \
  "$@"

# ------------------------------------------------------------------------------
# 3. Horizon H = 336
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id reproduce_weather_336 \
  --model ARBD \
  --data custom \
  --features M \
  --pred_len 336 \
  --enc_in 21 \
  --batch_size 256 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.4 \
  --learning_rate 0.0015 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.65 \
  --local_weight_max 1.0 \
  --static_local_weight_init 0.1 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_Weather_336 \
  "$@"

# ------------------------------------------------------------------------------
# 4. Horizon H = 720
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id reproduce_weather_720 \
  --model ARBD \
  --data custom \
  --features M \
  --pred_len 720 \
  --enc_in 21 \
  --batch_size 256 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.5 \
  --learning_rate 0.0002 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.55 \
  --local_weight_max 1.0 \
  --static_local_weight_init 0.35 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_Weather_720 \
  "$@"
