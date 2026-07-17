#!/bin/bash
# 运行 ETTm2 数据集下的最优 Gaussian Jet 配置 (Horizon=96, 192, 336, 720, CAS Density Mode)
# 请在项目根目录下执行：sh scripts/run_ettm2_cas.sh

train_epochs=30
next_is_epochs=false
filtered_args=()

for arg in "$@"; do
  if [ "$next_is_epochs" = true ]; then
    train_epochs="$arg"
    next_is_epochs=false
  elif [ "$arg" = "--train_epochs" ]; then
    next_is_epochs=true
  elif [[ "$arg" == --train_epochs=* ]]; then
    train_epochs="${arg#*=}"
  else
    filtered_args+=("$arg")
  fi
done

echo "==================== Starting ETTm2 Horizon 96 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm2.csv \
  --model_id ettm2_jet_96 \
  --model SplatTS \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --enc_in 7 \
  --d_model 128 \
  --itr 1 \
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.35 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.35 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.2 \
  --des ETTm2_GaussianJet \
  "${filtered_args[@]}"

echo "==================== Starting ETTm2 Horizon 192 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm2.csv \
  --model_id ettm2_jet_192 \
  --model SplatTS \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --pred_len 192 \
  --enc_in 7 \
  --d_model 128 \
  --itr 1 \
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.9 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0008 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.9 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.3 \
  --des ETTm2_GaussianJet \
  "${filtered_args[@]}"

echo "==================== Starting ETTm2 Horizon 336 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm2.csv \
  --model_id ettm2_jet_336 \
  --model SplatTS \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --pred_len 336 \
  --enc_in 7 \
  --d_model 128 \
  --itr 1 \
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.65 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.001 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.65 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.6 \
  --des ETTm2_GaussianJet \
  "${filtered_args[@]}"

echo "==================== Starting ETTm2 Horizon 720 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm2.csv \
  --model_id ettm2_jet_720 \
  --model SplatTS \
  --data ETTm2 \
  --features M \
  --seq_len 512 \
  --pred_len 720 \
  --enc_in 7 \
  --d_model 128 \
  --itr 1 \
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.35 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.002 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.35 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.2 \
  --des ETTm2_GaussianJet \
  "${filtered_args[@]}"
