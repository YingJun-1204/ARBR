#!/bin/bash
# 依次运行所有数据集下的最优 Adaptive Geometry-Conditioned Gaussian Jet 配置 (V15)
# 运行顺序: ETTh1 -> ETTh2 -> ETTm1 -> ETTm2 -> Weather -> Electricity -> Traffic
# 请在项目根目录下执行：bash scripts/run_all.sh 或 sh scripts/run_all.sh
# 支持消融实验参数 (如 --ablation_mode gaussian_only 或 --ablation_mode observation_only):
# 例如: bash scripts/run_all.sh --ablation_mode gaussian_only
#       bash scripts/run_all.sh --ablation_mode observation_only

start_line=0
if [ -f "result_long_term_forecast.txt" ]; then
    start_line=$(wc -l < result_long_term_forecast.txt | tr -d ' ')
fi

echo "================================================================="
echo " Starting All Dataset Benchmarks (V15)"
echo " Execution Order: ETTh1 -> ETTh2 -> ETTm1 -> ETTm2 -> Weather -> Electricity -> Traffic"
echo "================================================================="

echo ""
echo ">>> [1/7] Running ETTh1..."
bash scripts/run_etth1.sh "$@"

echo ""
echo ">>> [2/7] Running ETTh2..."
bash scripts/run_etth2.sh "$@"

echo ""
echo ">>> [3/7] Running ETTm1..."
bash scripts/run_ettm1.sh "$@"

echo ""
echo ">>> [4/7] Running ETTm2..."
bash scripts/run_ettm2.sh "$@"

echo ""
echo ">>> [5/7] Running Weather..."
bash scripts/run_weather.sh "$@"

echo ""
echo ">>> [6/7] Running Electricity..."
bash scripts/run_electricity.sh "$@"

echo ""
echo ">>> [7/7] Running Traffic..."
bash scripts/run_traffic.sh "$@"

echo ""
echo "================================================================="
echo " All Dataset Benchmarks Completed!"
echo "================================================================="

python scripts/summarize_results.py --start_line "$start_line"
