#!/bin/bash
# 运行 weather 数据集下的最优 Adaptive Direction Router 配置 (Horizon=96, 192, 336, 720, CAS Density Mode)
# 请在项目根目录下执行：sh scripts/run_weather_cas.sh

echo "==================== Starting weather Horizon 96 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id weather_router_96 \
  --model SplatTS \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1 \
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 10 \
  --gs_dropout 0.7 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0015 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --dropout 0.0 \
  --head_dropout 0.7 \
  --head_dropout_position pre \
  --density_mode cas \
  --gs_lambda 0.0 \
  --num_workers 0 \
  --use_residual \
  --gs_residual_weight 0.3 \
  --gate_type adaptive_direction \
  --gate_beta 0.15 \
  --gate_lambda 0.02 \
  --gate_window_half 2 \
  --des Ablation_weather_router \
  "$@"

echo "==================== Starting weather Horizon 192 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id weather_router_192 \
  --model SplatTS \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 192 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1 \
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 10 \
  --gs_dropout 0.75 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0008 \
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
  --gs_residual_weight 0.2 \
  --gate_type adaptive_direction \
  --gate_beta 0.1 \
  --gate_lambda 0.03 \
  --gate_window_half 2 \
  --des Ablation_weather_router \
  "$@"

echo "==================== Starting weather Horizon 336 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id weather_router_336 \
  --model SplatTS \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 336 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1 \
  --batch_size 256 \
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
  --gs_residual_weight 0.5 \
  --gate_type adaptive_direction \
  --gate_beta 0.5 \
  --gate_lambda 0.1 \
  --gate_window_half 4 \
  --des Ablation_weather_router \
  "$@"

echo "==================== Starting weather Horizon 720 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id weather_router_720 \
  --model SplatTS \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 720 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 128 \
  --d_ff 256 \
  --itr 1 \
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 10 \
  --gs_dropout 0.65 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0003 \
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
  --gs_residual_weight 0.5 \
  --gate_type adaptive_direction \
  --gate_beta 0.35 \
  --gate_lambda 0.01 \
  --gate_window_half 2 \
  --des Ablation_weather_router \
  "$@"
