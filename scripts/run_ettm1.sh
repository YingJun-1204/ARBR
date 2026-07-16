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
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.35 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0001 \
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
  --gs_residual_weight 0.55 \
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
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.55 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0005 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --dropout 0.0 \
  --head_dropout 0.55 \
  --head_dropout_position pre \
  --density_mode cas \
  --gs_lambda 0.0 \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.3 \
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
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.5 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0001 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --dropout 0.0 \
  --head_dropout 0.5 \
  --head_dropout_position pre \
  --density_mode cas \
  --gs_lambda 0.0 \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.35 \
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
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.8 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --dropout 0.0 \
  --head_dropout 0.8 \
  --head_dropout_position pre \
  --density_mode cas \
  --gs_lambda 0.0 \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.45 \
  --des Ablation_ETTm1_router \
  "$@"
