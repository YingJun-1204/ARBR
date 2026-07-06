import unittest
import torch
from layers.splatting_residual_encoder import SplattingResidualEncoder as SRSEventSplattingResidual

class TestPatchResidual(unittest.TestCase):
    def test_patch_residual_logic(self):
        seq_len = 512
        num_gaussians = 4
        d_model = 256
        patch_len = 24
        stride = 12
        patch_num = 42  # math.ceil((512 - 24) / 12) + 1 = 42

        # 1. Base HPO Layer (use_residual=False)
        layer_base = SRSEventSplattingResidual(
            seq_len=seq_len,
            d_model=d_model,
            num_gaussians=num_gaussians,
            density_mode="none",
            use_occlusion=False,
            patch_len=patch_len,
            stride=stride,
            use_residual=False
        )

        # 2. Residual Layer with weight=0.1
        layer_res = SRSEventSplattingResidual(
            seq_len=seq_len,
            d_model=d_model,
            num_gaussians=num_gaussians,
            density_mode="none",
            use_occlusion=False,
            patch_len=patch_len,
            stride=stride,
            use_residual=True,
            gs_residual_weight=0.1
        )

        # 3. Residual Layer with weight=0.0 (Should mathematically degrade to base layer)
        layer_zero = SRSEventSplattingResidual(
            seq_len=seq_len,
            d_model=d_model,
            num_gaussians=num_gaussians,
            density_mode="none",
            use_occlusion=False,
            patch_len=patch_len,
            stride=stride,
            use_residual=True,
            gs_residual_weight=0.0
        )

        # Share generator weights across all layers
        state_dict = layer_base.generator.state_dict()
        layer_res.generator.load_state_dict(state_dict)
        layer_zero.generator.load_state_dict(state_dict)

        # Set to eval mode to disable random Dropout behavior
        layer_base.eval()
        layer_res.eval()
        layer_zero.eval()

        # Dummy input: batch_size=2, channels=7
        x = torch.randn(2, 7, seq_len)

        # Run forward pass
        out_base = layer_base(x, patch_num=patch_num)
        out_res = layer_res(x, patch_num=patch_num)
        out_zero = layer_zero(x, patch_num=patch_num)

        # A. Shape Verification
        self.assertEqual(out_base.shape, out_res.shape)
        self.assertEqual(out_res.shape, out_zero.shape)
        self.assertEqual(out_res.shape, (14, patch_num, d_model))

        # B. Residual Activation Verification
        # Dynamic residual must lead to numerically distinct features compared to base layer
        self.assertFalse(torch.allclose(out_base, out_res, atol=1e-6))

        # C. Mathematical Degradation (weight=0.0) Verification
        # When weight is 0.0, the output must exactly match the base layer
        # Share projection weight to ensure absolute identity if initialized differently
        # (Though with weight=0.0, projection values are multiplied by 0.0, so it shouldn't matter)
        self.assertTrue(torch.allclose(out_base, out_zero, atol=1e-7))

    def test_adaptive_direction_diagnostics_states(self):
        seq_len = 512
        num_gaussians = 4
        d_model = 256
        patch_len = 24
        stride = 12
        patch_num = 42

        # 1. Layer with adaptive_direction gating
        layer_adaptive = SRSEventSplattingResidual(
            seq_len=seq_len,
            d_model=d_model,
            num_gaussians=num_gaussians,
            density_mode="none",
            use_occlusion=False,
            patch_len=patch_len,
            stride=stride,
            use_residual=True,
            gs_residual_weight=0.1,
            gate_type="adaptive_direction",
            gate_beta=0.25
        )
        layer_adaptive.eval()
        
        # Dummy Input
        x = torch.randn(2, 7, seq_len)
        
        # Run forward
        _ = layer_adaptive(x, patch_num=patch_num)
        
        # Verify cached diagnostic states exist and have correct shapes
        self.assertTrue(hasattr(layer_adaptive, "last_gate_direction"))
        self.assertTrue(hasattr(layer_adaptive, "last_gate_scale"))
        self.assertEqual(layer_adaptive.last_gate_direction.shape, (14, patch_num, 1))
        self.assertEqual(layer_adaptive.last_gate_scale.shape, (14, patch_num, 1))

        # 2. Layer with gate_type=None (Should have default placeholder values to avoid attribute errors)
        layer_none = SRSEventSplattingResidual(
            seq_len=seq_len,
            d_model=d_model,
            num_gaussians=num_gaussians,
            density_mode="none",
            use_occlusion=False,
            patch_len=patch_len,
            stride=stride,
            use_residual=True,
            gs_residual_weight=0.1,
            gate_type=None
        )
        layer_none.eval()
        _ = layer_none(x, patch_num=patch_num)
        self.assertTrue(hasattr(layer_none, "last_gate_direction"))
        self.assertTrue(hasattr(layer_none, "last_gate_scale"))

if __name__ == "__main__":
    unittest.main()
