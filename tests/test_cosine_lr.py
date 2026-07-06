import unittest
import torch
import math
from utils.tools import adjust_learning_rate, dotdict

class TestCosineLR(unittest.TestCase):
    def test_new_cosine_lr_schedule(self):
        # Create a mock parameter and optimizer
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=0.001)
        
        args = dotdict({
            "lradj": "cosine",
            "learning_rate": 0.001,
            "train_epochs": 30
        })
        
        # Test Epoch 1 Start (represented by epoch=0)
        # Warmup step 1: 0.5 * base_lr = 0.0005
        adjust_learning_rate(optimizer, epoch=0, args=args)
        lr_val_0 = optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(lr_val_0, 0.0005, places=7)

        # Test Epoch 1 End / Epoch 2 Start (represented by epoch=1)
        # Warmup step 2: 1.0 * base_lr = 0.001
        adjust_learning_rate(optimizer, epoch=1, args=args)
        lr_val_1 = optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(lr_val_1, 0.001, places=7)

        # Test Epoch 2 End / Epoch 3 Start (represented by epoch=2)
        # Cosine annealing step 0: 1.0 * base_lr = 0.001
        adjust_learning_rate(optimizer, epoch=2, args=args)
        lr_val_2 = optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(lr_val_2, 0.001, places=7)

        # Test Mid-training Cosine Decay (represented by epoch=16, so t = 16 - 2 = 14, T = 28)
        # cos(14/28 * pi) = cos(0.5 * pi) = 0
        # lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + 0) = 1e-5 + 4.95e-4 = 5.05e-4
        adjust_learning_rate(optimizer, epoch=16, args=args)
        lr_val_16 = optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(lr_val_16, 0.000505, places=7)

        # Test Epoch 30 End (represented by epoch=30, so t = 30 - 2 = 28, T = 28)
        # cos(28/28 * pi) = cos(pi) = -1
        # lr = eta_min + 0.5 * (base_lr - eta_min) * (1 - 1) = eta_min = 1e-5
        adjust_learning_rate(optimizer, epoch=30, args=args)
        lr_val_30 = optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(lr_val_30, 0.00001, places=7)

if __name__ == "__main__":
    unittest.main()
