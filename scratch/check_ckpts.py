import os
import torch

ckpt_dir = "checkpoints"
for d in sorted(os.listdir(ckpt_dir)):
    p = os.path.join(ckpt_dir, d, "checkpoint.pth")
    if os.path.exists(p):
        sz = os.path.getsize(p) / 1024 / 1024
        print(f"Found checkpoint: {d} ({sz:.2f} MB)")
