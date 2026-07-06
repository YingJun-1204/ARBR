#!/bin/bash
# 运行 ETTm2 数据集下的最优 Adaptive Direction Router 配置 (Horizon=96, 192, 336, 720, CAS Density Mode)
# 请在项目根目录下执行：sh scripts/run_ettm2_cas.sh

echo "==================== Starting ETTm2 Horizon 96 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm2.csv \
  --model_id ettm2_router_96 \
  --model SplatTS \
  --data ETTm2 \
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
  --gs_dropout 0.9 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0015 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --dropout 0.0 \
  --head_dropout 0.9 \
  --head_dropout_position pre \
  --density_mode cas \
  --gs_lambda 0.0 \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.35 \
  --gate_type adaptive_direction \
  --gate_beta 0.1 \
  --gate_lambda 0.02 \
  --gate_window_half 4 \
  --des Ablation_ETTm2_router \
  "$@"

echo "==================== Starting ETTm2 Horizon 192 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm2.csv \
  --model_id ettm2_router_192 \
  --model SplatTS \
  --data ETTm2 \
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
  --gs_dropout 0.65 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.002 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --dropout 0.0 \
  --head_dropout 0.65 \
  --head_dropout_position pre \
  --density_mode cas \
  --gs_lambda 0.0 \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.2 \
  --gate_type adaptive_direction \
  --gate_beta 0.2 \
  --gate_lambda 0.08 \
  --gate_window_half 2 \
  --des Ablation_ETTm2_router \
  "$@"

echo "==================== Starting ETTm2 Horizon 336 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm2.csv \
  --model_id ettm2_router_336 \
  --model SplatTS \
  --data ETTm2 \
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
  --gs_dropout 0.3 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.002 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --dropout 0.0 \
  --head_dropout 0.3 \
  --head_dropout_position pre \
  --density_mode cas \
  --gs_lambda 0.0 \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.6 \
  --gate_type adaptive_direction \
  --gate_beta 0.5 \
  --gate_lambda 0.05 \
  --gate_window_half 3 \
  --des Ablation_ETTm2_router \
  "$@"

echo "==================== Starting ETTm2 Horizon 720 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTm2.csv \
  --model_id ettm2_router_720 \
  --model SplatTS \
  --data ETTm2 \
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
  --gs_dropout 0.4 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0001 \
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
  --gs_residual_weight 0.45 \
  --gate_type adaptive_direction \
  --gate_beta 0.5 \
  --gate_lambda 0.1 \
  --gate_window_half 4 \
  --des Ablation_ETTm2_router \
  "$@"
