#!/bin/bash
# 自动测试 ETTh1 在 seq_len ∈ {96, 192, 336, 512, 720} 下的 4 个 Horizon 结果与平均值
# 使用方法：
# 1. 运行全部 seq_len (96, 192, 336, 512, 720) 与 4 个 Horizon (96, 192, 336, 720):
#    bash scripts/run_etth1_seqlen_benchmark.sh
# 2. 自定义 seq_len 列表或 epochs:
#    bash scripts/run_etth1_seqlen_benchmark.sh --seq_lens 96 512 --train_epochs 30

python -u scripts/run_etth1_seqlen_benchmark.py "$@"
