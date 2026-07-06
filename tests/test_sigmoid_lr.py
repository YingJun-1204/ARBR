import unittest
import torch
from utils.tools import adjust_learning_rate, dotdict

class TestSigmoidLR(unittest.TestCase):
    def test_sigmoid_lr_calculation(self):
        # 建立一个最小优化器
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([param], lr=0.001)
        
        # 模拟参数
        args = dotdict({
            "lradj": "sigmoid",
            "learning_rate": 0.001
        })
        
        # 测试 epoch 1
        adjust_learning_rate(optimizer, epoch=1, args=args)
        lr_val = optimizer.param_groups[0]['lr']
        # 应该发生调整（在未实现时，adjust_learning_rate 会直接 return 使得 lr_val 仍然是 0.001；
        # 而在 sigmoid 逻辑下，第一轮 warm-up 计算出来的学习率应该是非 0.001 的其他数值，我们可以用 assertNotEqual 甚至具体的断言来校验）
        self.assertNotEqual(lr_val, 0.001, "Learning rate should be adjusted under sigmoid strategy")
        self.assertGreaterEqual(lr_val, 0.0)

        # 测试 epoch 10
        adjust_learning_rate(optimizer, epoch=10, args=args)
        lr_val_10 = optimizer.param_groups[0]['lr']
        self.assertNotEqual(lr_val_10, 0.001)
        self.assertGreater(lr_val_10, 0.0)

if __name__ == "__main__":
    unittest.main()
