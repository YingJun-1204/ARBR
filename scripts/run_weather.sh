#!/bin/bash
# 运行 weather 数据集下的最优 Gaussian Jet 配置 (Horizon=96, 192, 336, 720, CAS Density Mode)
# 请在项目根目录下执行：sh scripts/run_weather_cas.sh

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

echo "==================== Starting weather Horizon 96 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id weather_jet_96 \
  --model SplatTS \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --enc_in 21 \
  --d_model 128 \
  --itr 1 \
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.7 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0012 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.7 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.5 \
  --des weather_GaussianJet \
  "${filtered_args[@]}"

echo "==================== Starting weather Horizon 192 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id weather_jet_192 \
  --model SplatTS \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 192 \
  --enc_in 21 \
  --d_model 128 \
  --itr 1 \
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.7 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0012 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.7 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.25 \
  --des weather_GaussianJet \
  "${filtered_args[@]}"

echo "==================== Starting weather Horizon 336 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id weather_jet_336 \
  --model SplatTS \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 336 \
  --enc_in 21 \
  --d_model 128 \
  --itr 1 \
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.75 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0012 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.75 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.35 \
  --des weather_GaussianJet \
  "${filtered_args[@]}"

echo "==================== Starting weather Horizon 720 ===================="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data \
  --data_path weather.csv \
  --model_id weather_jet_720 \
  --model SplatTS \
  --data custom \
  --features M \
  --seq_len 512 \
  --pred_len 720 \
  --enc_in 21 \
  --d_model 128 \
  --itr 1 \
  --batch_size 256 \
  --representation gs \
  --patch_len 24 \
  --stride 12 \
  --num_gaussians 8 \
  --gs_dropout 0.75 \
  --gs_weight_decay 1e-4 \
  --learning_rate 0.0005 \
  --lradj cosine \
  --train_epochs "$train_epochs" \
  --patience 6 \
  --head_dropout 0.75 \
  --head_dropout_position pre \
  --density_mode cas \
  --num_workers 0 \
  --gs_residual_weight 0.5 \
  --des weather_GaussianJet \
  "${filtered_args[@]}"
