#!/bin/bash
# 运行 ETTh2 数据集下的最优 Gaussian Jet 配置 (Horizon=96, 192, 336, 720, CAS Density Mode)
# 请在项目根目录下执行：sh scripts/run_etth2_cas.sh

echo "==================== Starting ETTh2 Horizon 96 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id etth2_jet_96 \
  --model SplatTS \
  --data ETTh2 \
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
  --gs_dropout 0.85 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.85 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.5 \
  --des ETTh2_GaussianJet \
  "$@"

echo "==================== Starting ETTh2 Horizon 192 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id etth2_jet_192 \
  --model SplatTS \
  --data ETTh2 \
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
  --gs_dropout 0.85 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0002 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.85 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.4 \
  --des ETTh2_GaussianJet \
  "$@"

echo "==================== Starting ETTh2 Horizon 336 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id etth2_jet_336 \
  --model SplatTS \
  --data ETTh2 \
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
  --gs_dropout 0.8 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0015 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.8 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.55 \
  --des ETTh2_GaussianJet \
  "$@"

echo "==================== Starting ETTh2 Horizon 720 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path ETTh2.csv \
  --model_id etth2_jet_720 \
  --model SplatTS \
  --data ETTh2 \
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
  --gs_dropout 0.8 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0005 \
  --lradj cosine \
  --train_epochs 30 \
  --patience 6 \
  --head_dropout 0.8 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.6 \
  --des ETTh2_GaussianJet \
  "$@"
