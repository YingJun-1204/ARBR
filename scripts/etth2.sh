#!/usr/bin/env bash
set -euo pipefail

# Reproduce ARBD benchmark on ETTh2 across 4 horizons: 96, 192, 336, 720
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Hyperparameters for sensitivity analysis (default: rho=0.5, beta=0.5)
RHO="${RHO:-0.5}"
BETA="${BETA:-0.5}"

echo "================================================================================"
echo "                    Training ARBD: ETTh2 Benchmark Suite                   "
echo "================================================================================"

# ------------------------------------------------------------------------------
# 1. Horizon H = 96
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id reproduce_etth2_96 \
  --model ARBD \
  --data ETTh2 \
  --features M \
  --pred_len 96 \
  --enc_in 7 \
  --batch_size 256 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.9 \
  --learning_rate 0.0003 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.7 \
  --local_weight_max 0.75 \
  --static_local_weight_init 0.5 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_ETTh2_96 \
  "$@"

# ------------------------------------------------------------------------------
# 2. Horizon H = 192
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id reproduce_etth2_192 \
  --model ARBD \
  --data ETTh2 \
  --features M \
  --pred_len 192 \
  --enc_in 7 \
  --batch_size 256 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.85 \
  --learning_rate 0.0015 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.85 \
  --local_weight_max 0.75 \
  --static_local_weight_init 0.35 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_ETTh2_192 \
  "$@"

# ------------------------------------------------------------------------------
# 3. Horizon H = 336
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id reproduce_etth2_336 \
  --model ARBD \
  --data ETTh2 \
  --features M \
  --pred_len 336 \
  --enc_in 7 \
  --batch_size 256 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.85 \
  --learning_rate 0.0015 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.45 \
  --local_weight_max 0.75 \
  --static_local_weight_init 0.5 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_ETTh2_336 \
  "$@"

# ------------------------------------------------------------------------------
# 4. Horizon H = 720
# ------------------------------------------------------------------------------
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id reproduce_etth2_720 \
  --model ARBD \
  --data ETTh2 \
  --features M \
  --pred_len 720 \
  --enc_in 7 \
  --batch_size 256 \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.85 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.6 \
  --local_weight_max 0.75 \
  --static_local_weight_init 0.5 \
  --rho "${RHO}" \
  --beta "${BETA}" \
  --des reproduce_ETTh2_720 \
  "$@"
