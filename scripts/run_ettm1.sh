#!/bin/bash
# 运行 ETTM1 数据集下的最优 Adaptive Geometry-Conditioned Gaussian Jet 配置 (V14)
# 请在项目根目录下执行：sh scripts/run_ettm1.sh

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

echo "==================== Starting ETTM1 Horizon 96 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --model_id ettm1_jet_96 \
  --model SplatTS \
  --data ETTm1 \
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
  --gs_dropout 0.45 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0008 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.45 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.1 \
  --fusion_mode geometry \
  --fusion_init 0.2 \
  --fusion_beta_max 1.0 \
  --fusion_hidden_dim 16 \
  --fusion_detach_geometry 1 \
  --jet_score_temperature 0.01 \
  --jet_max_shift_samples 1.0 \
  --des ETTm1_GaussianJet \
  "${filtered_args[@]}"

echo "==================== Starting ETTM1 Horizon 192 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --model_id ettm1_jet_192 \
  --model SplatTS \
  --data ETTm1 \
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
  --gs_dropout 0.25 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0002 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.6 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.1 \
  --fusion_mode geometry \
  --fusion_init 0.2 \
  --fusion_beta_max 0.5 \
  --fusion_hidden_dim 16 \
  --fusion_detach_geometry 1 \
  --jet_score_temperature 0.01 \
  --jet_max_shift_samples 1.0 \
  --des ETTm1_GaussianJet \
  "${filtered_args[@]}"

echo "==================== Starting ETTM1 Horizon 336 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --model_id ettm1_jet_336 \
  --model SplatTS \
  --data ETTm1 \
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
  --gs_dropout 0.25 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0012 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.85 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.1 \
  --fusion_mode geometry \
  --fusion_init 0.5 \
  --fusion_beta_max 1.0 \
  --fusion_hidden_dim 16 \
  --fusion_detach_geometry 1 \
  --jet_score_temperature 0.01 \
  --jet_max_shift_samples 1.0 \
  --des ETTm1_GaussianJet \
  "${filtered_args[@]}"

echo "==================== Starting ETTM1 Horizon 720 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --model_id ettm1_jet_720 \
  --model SplatTS \
  --data ETTm1 \
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
  --gs_dropout 0.7 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.002 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.25 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.1 \
  --fusion_mode geometry \
  --fusion_init 0.1 \
  --fusion_beta_max 0.5 \
  --fusion_hidden_dim 16 \
  --fusion_detach_geometry 1 \
  --jet_score_temperature 0.01 \
  --jet_max_shift_samples 1.0 \
  --des ETTm1_GaussianJet \
  "${filtered_args[@]}"
