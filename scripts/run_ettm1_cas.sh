#!/bin/bash
# 运行 ETTm1 数据集下的最优 Adaptive Direction Router 配置 (Horizon=96, 192, 336, 720, CAS Density Mode)
# 请在项目根目录下执行：sh scripts/run_ettm1_cas.sh

echo "==================== Starting ETTm1 Horizon 96 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --model_id ettm1_router_96 \
  --model SplatTS \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1 \
  --batch_size 1024 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 10 \
  --gs_dropout 0.35 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0012 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --dropout 0.0 \
  --head_dropout 0.35 \
  --head_dropout_position pre \
  --density_mode cas \
  --gs_lambda 0.0 \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.25 \
  --gate_type adaptive_direction \
  --gate_beta 0.15 \
  --gate_lambda 0.01 \
  --gate_window_half 3 \
  --des Ablation_ETTm1_router \
  "$@"

echo "==================== Starting ETTm1 Horizon 192 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --model_id ettm1_router_192 \
  --model SplatTS \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --pred_len 192 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1 \
  --batch_size 1024 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 10 \
  --gs_dropout 0.4 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.001 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --dropout 0.0 \
  --head_dropout 0.4 \
  --head_dropout_position pre \
  --density_mode cas \
  --gs_lambda 0.0 \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.2 \
  --gate_type adaptive_direction \
  --gate_beta 0.5 \
  --gate_lambda 0.03 \
  --gate_window_half 2 \
  --des Ablation_ETTm1_router \
  "$@"

echo "==================== Starting ETTm1 Horizon 336 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --model_id ettm1_router_336 \
  --model SplatTS \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --pred_len 336 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1 \
  --batch_size 1024 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 10 \
  --gs_dropout 0.6 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.001 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --dropout 0.0 \
  --head_dropout 0.6 \
  --head_dropout_position pre \
  --density_mode cas \
  --gs_lambda 0.0 \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.4 \
  --gate_type adaptive_direction \
  --gate_beta 0.1 \
  --gate_lambda 0.1 \
  --gate_window_half 2 \
  --des Ablation_ETTm1_router \
  "$@"

echo "==================== Starting ETTm1 Horizon 720 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm1.csv \
  --model_id ettm1_router_720 \
  --model SplatTS \
  --data ETTm1 \
  --features M \
  --seq_len 512 \
  --pred_len 720 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1 \
  --batch_size 1024 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 10 \
  --gs_dropout 0.75 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.001 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --dropout 0.0 \
  --head_dropout 0.75 \
  --head_dropout_position pre \
  --density_mode cas \
  --gs_lambda 0.0 \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.35 \
  --gate_type adaptive_direction \
  --gate_beta 0.45 \
  --gate_lambda 0.05 \
  --gate_window_half 3 \
  --des Ablation_ETTm1_router \
  "$@"
