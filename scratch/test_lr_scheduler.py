import sys
import os
import math
import torch
import torch.nn as nn
from torch import optim

# Mock args
class Args:
    def __init__(self, lr, train_epochs, freeze_epochs, lradj):
        self.learning_rate = lr
        self.train_epochs = train_epochs
        self.gs_freeze_epochs = freeze_epochs
        self.lradj = lradj

# Add current dir to path to import utils
sys.path.append(os.getcwd())
try:
    from utils.tools import adjust_learning_rate
except ImportError:
    # If called from a different location
    sys.path.append(os.path.join(os.getcwd(), ".."))
    from utils.tools import adjust_learning_rate

def test_cosine_after_unfreeze():
    base_lr = 0.001
    train_epochs = 30
    freeze_epochs = 10
    lradj = 'cosine_after_unfreeze'
    args = Args(base_lr, train_epochs, freeze_epochs, lradj)
    
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    
    lrs = []
    print(f"Testing {lradj} with base_lr={base_lr}, freeze={freeze_epochs}, total={train_epochs}")
    for epoch in range(1, train_epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        # print(f"Epoch {epoch}: {current_lr:.6f}")
    
    # Assertions
    # 1. During freeze, LR should be base_lr
    for i in range(freeze_epochs):
        assert abs(lrs[i] - base_lr) < 1e-9, f"Epoch {i+1} LR should be {base_lr}, got {lrs[i]}"
    
    # 2. At first epoch after freeze (epoch 11), LR should still be base_lr
    assert abs(lrs[freeze_epochs] - base_lr) < 1e-9, f"Epoch {freeze_epochs+1} LR should be {base_lr}, got {lrs[freeze_epochs]}"
    
    # 3. Last epoch should be approx min_lr (base_lr * 0.05)
    min_lr = base_lr * 0.05
    assert abs(lrs[-1] - min_lr) < 1e-6, f"Final LR should be approx {min_lr}, got {lrs[-1]}"
    
    # 4. Monotonic decrease after unfreeze
    for i in range(freeze_epochs, train_epochs - 1):
        assert lrs[i] >= lrs[i+1], f"LR should be non-increasing after unfreeze. Epoch {i+1} to {i+2} ({lrs[i]:.6f} -> {lrs[i+1]:.6f})"

    print("Test PASSED!")

if __name__ == "__main__":
    try:
        test_cosine_after_unfreeze()
    except Exception as e:
        print(f"Test FAILED: {e}")
        sys.exit(1)
