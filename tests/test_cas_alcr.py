import unittest
import torch
from models.gs_linear import Model

class TestCASALCR(unittest.TestCase):
    def test_cas_forward_and_properties(self):
        class Config:
            task_name = "long_term_forecast"
            seq_len = 336
            pred_len = 96
            c_out = 7
            enc_in = 7
            d_model = 128
            patch_len = 24
            stride = 12
            representation = "gs"
            num_gaussians = 10
            gs_dropout = 0.3
            dropout = 0.1
            head_dropout = 0.1
            subtract_last = False
            density_mode = "cas"
            use_residual = True
            gs_residual_weight = 0.3
            gamma_complement = 0.6
            gate_type = None

        configs = Config()
        model = Model(configs)
        self.assertEqual(model.splatting_residual.density_mode, "cas")
        self.assertTrue(hasattr(model.splatting_residual, "complexity_router"))
        
        x = torch.randn(4, 336, 7)
        out = model(x)
        self.assertEqual(out.shape, (4, 96, 7))
        
        # 验证正向属性是否正确生成
        self.assertTrue(hasattr(model.splatting_residual, "last_gate_probs"))
        self.assertTrue(hasattr(model.splatting_residual, "last_active_count"))
        self.assertEqual(model.splatting_residual.last_gate_probs.shape, (4 * 7, 10))
        
        # 检查级联概率是否单调递减
        probs = model.splatting_residual.last_gate_probs
        for i in range(probs.shape[1] - 1):
            self.assertTrue(torch.all(probs[:, i] >= probs[:, i+1] - 1e-6))

    def test_cas_gradient_flow(self):
        class Config:
            task_name = "long_term_forecast"
            seq_len = 336
            pred_len = 96
            c_out = 7
            enc_in = 7
            d_model = 128
            patch_len = 24
            stride = 12
            representation = "gs"
            num_gaussians = 10
            gs_dropout = 0.3
            dropout = 0.1
            head_dropout = 0.1
            subtract_last = False
            density_mode = "cas"
            use_residual = True
            gs_residual_weight = 0.3
            gamma_complement = 0.6
            gate_type = None

        configs = Config()
        model = Model(configs)
        x = torch.randn(2, 336, 7)
        out = model(x)
        loss = out.mean()
        loss.backward()
        
        # 验证 complexity_router 的梯度通路畅通，没有出现死锁
        for p in model.splatting_residual.complexity_router.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.any(p.grad != 0.0))

    def test_cas_global_spectral_anchor(self):
        class Config:
            task_name = "long_term_forecast"
            seq_len = 336
            pred_len = 96
            c_out = 7
            enc_in = 7
            d_model = 128
            patch_len = 24
            stride = 12
            representation = "gs"
            num_gaussians = 10
            gs_dropout = 0.3
            dropout = 0.1
            head_dropout = 0.1
            subtract_last = False
            density_mode = "cas"
            use_residual = True
            gs_residual_weight = 0.3
            gamma_complement = 0.6
            gate_type = None

        # 用例 A: 验证平滑数据集自适应锁定 K_base = 3
        configs_a = Config()
        model_a = Model(configs_a)
        
        # 验证初始化时 k_base = -1
        self.assertTrue(hasattr(model_a.splatting_residual, "k_base"))
        self.assertEqual(model_a.splatting_residual.k_base.item(), -1)
        
        # 输入平滑正弦信号
        t = torch.linspace(0, 10, 336).unsqueeze(0).unsqueeze(2).expand(2, -1, 7)
        x_smooth = torch.sin(t)
        _ = model_a(x_smooth)
        
        # 验证 k_base 成功自适应锁定为 3 (因为平滑正弦信号的 rho ≈ 1.0, Round(1.0 * 4) = 4, Clamp(4, 1, 3) = 3)
        self.assertEqual(model_a.splatting_residual.k_base.item(), 3)
        
        # 再次输入噪声信号，前 3 个核 (index 0, 1, 2) 必须保持恒为 1.0 (代表静态锁定底座)
        x_noise = torch.randn(2, 336, 7)
        _ = model_a(x_noise)
        probs_noise = model_a.splatting_residual.last_gate_probs
        self.assertTrue(torch.allclose(probs_noise[:, 0], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(probs_noise[:, 1], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(probs_noise[:, 2], torch.tensor(1.0)))
        
        # 用例 B: 验证纯噪声数据集自适应锁定 K_base = 1
        configs_b = Config()
        model_b = Model(configs_b)
        
        # 首先输入纯白噪声
        _ = model_b(x_noise)
        # 验证 k_base 自适应锁定为 1 (因为白噪声的 rho ≈ 0.0, Round(0.0 * 4) = 0, Clamp(0, 1, 3) = 1)
        self.assertEqual(model_b.splatting_residual.k_base.item(), 1)

if __name__ == "__main__":
    unittest.main()
