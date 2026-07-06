import unittest
import torch
import math
from models.gs_linear import Model

class TestModelSplatTSAdaptive(unittest.TestCase):
    def test_splatts_passes_density_mode(self):
        class Config:
            task_name = "long_term_forecast"
            seq_len = 336
            pred_len = 96
            c_out = 7
            enc_in = 7
            d_model = 256
            patch_len = 24
            stride = 24
            representation = "gs"
            gs_hidden_dim = 32
            num_gaussians = 5
            gs_dropout = 0.3
            dropout = 0.1
            head_dropout = 0.1
            subtract_last = False
            density_mode = "sparse"

        configs = Config()
        model = Model(configs)
        self.assertEqual(model.splatting_residual.density_mode, "sparse")
        
        x = torch.randn(16, 336, 7)
        out = model(x)
        self.assertEqual(out.shape, (16, 96, 7))

    def test_three_way_routing_initialization_and_forward(self):
        class Config:
            task_name = "long_term_forecast"
            seq_len = 336
            pred_len = 96
            c_out = 7
            enc_in = 7
            d_model = 256
            patch_len = 24
            stride = 24
            representation = "gs"
            gs_hidden_dim = 32
            num_gaussians = 5
            gs_dropout = 0.3
            dropout = 0.1
            head_dropout = 0.1
            subtract_last = False
            density_mode = "sparse"
            use_residual = True
            gate_type = "adaptive_direction"

        configs = Config()
        model = Model(configs)
        residual_module = model.splatting_residual

        # Assert route_router exists and is initialized to 0
        self.assertTrue(hasattr(residual_module, "route_router"))
        if isinstance(residual_module.route_router, torch.nn.Sequential):
            self.assertTrue(torch.all(residual_module.route_router[2].weight == 0.0))
            self.assertTrue(torch.all(residual_module.route_router[2].bias == 0.0))
        else:
            self.assertTrue(torch.all(residual_module.route_router.weight == 0.0))
            self.assertTrue(torch.all(residual_module.route_router.bias == 0.0))

        # Verify no legacy parameter exists
        self.assertFalse(hasattr(residual_module, "direction_router"))
        self.assertFalse(hasattr(residual_module, "strength_router"))
        self.assertFalse(hasattr(residual_module, "global_direction"))

        # Zero-initialize the learnable projection weights/biases for deterministic evaluation
        torch.nn.init.zeros_(residual_module.residual_router.u_proj.weight)
        torch.nn.init.zeros_(residual_module.residual_router.u_proj.bias)
        torch.nn.init.zeros_(residual_module.residual_router.d_proj.weight)
        torch.nn.init.zeros_(residual_module.residual_router.d_proj.bias)

        # Run forward pass
        x = torch.randn(2, 336, 7)
        _ = model(x)

        # Assert route probability output matches the zero-weight state [0.5, 0.25, 0.25]
        self.assertTrue(hasattr(residual_module, "last_gate_route_probs"))
        expected_probs = torch.zeros_like(residual_module.last_gate_route_probs)
        expected_probs[..., 0] = 0.5   # Neutral
        expected_probs[..., 1] = 0.25  # Forward
        expected_probs[..., 2] = 0.25  # Reverse
        self.assertTrue(torch.allclose(residual_module.last_gate_route_probs, expected_probs, atol=1e-5))

        # Assert direction is 0.25 - 0.25 = 0.0
        self.assertTrue(torch.allclose(residual_module.last_gate_direction, torch.tensor(0.0), atol=1e-5))

        # Assert strength is 1.0 - 0.5 = 0.5
        self.assertTrue(torch.allclose(residual_module.last_gate_strength, torch.tensor(0.5), atol=1e-5))

    def test_fixed_modes_compatibility(self):
        class Config:
            task_name = "long_term_forecast"
            seq_len = 336
            pred_len = 96
            c_out = 7
            enc_in = 7
            d_model = 256
            patch_len = 24
            stride = 24
            representation = "gs"
            gs_hidden_dim = 32
            num_gaussians = 5
            gs_dropout = 0.3
            dropout = 0.1
            head_dropout = 0.1
            subtract_last = False
            density_mode = "sparse"
            use_residual = True
            gate_type = "forward"

        configs = Config()
        
        # Case 1: fixed forward
        model_fwd = Model(configs)
        _ = model_fwd(torch.randn(2, 336, 7))
        res_fwd = model_fwd.splatting_residual
        self.assertTrue(torch.allclose(res_fwd.last_gate_strength, torch.tensor(1.0)))
        self.assertTrue(torch.allclose(res_fwd.last_gate_direction, torch.tensor(1.0)))
        self.assertTrue(torch.allclose(res_fwd.last_gate_route_probs[..., 1], torch.tensor(1.0)))

        # Case 2: fixed reverse
        configs.gate_type = "reverse"
        model_rev = Model(configs)
        _ = model_rev(torch.randn(2, 336, 7))
        res_rev = model_rev.splatting_residual
        self.assertTrue(torch.allclose(res_rev.last_gate_strength, torch.tensor(1.0)))
        self.assertTrue(torch.allclose(res_rev.last_gate_direction, torch.tensor(-1.0)))
        self.assertTrue(torch.allclose(res_rev.last_gate_route_probs[..., 2], torch.tensor(1.0)))

    def test_entropy_loss_gradient(self):
        route_probs = torch.tensor([[[0.1, 0.8, 0.1], [0.33, 0.33, 0.34]]], requires_grad=True)
        entropy = -(route_probs * torch.log(route_probs + 1e-8)).sum(dim=-1).mean()
        entropy.backward()
        
        self.assertIsNotNone(route_probs.grad)
        self.assertTrue(torch.nonzero(route_probs.grad).size(0) > 0)


if __name__ == "__main__":
    unittest.main()
