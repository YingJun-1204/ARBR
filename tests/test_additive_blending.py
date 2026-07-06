import unittest
import torch
from layers.splatting_residual_encoder import SplattingResidualEncoder as SRSEventSplattingResidual

class TestAdditiveBlending(unittest.TestCase):
    def test_occlusion_flag_effect(self):
        seq_len = 336
        num_gaussians = 5
        d_model = 256
        d_latent = d_model
        
        # Test with occlusion=True (Legacy)
        layer_occ = SRSEventSplattingResidual(
            seq_len=seq_len,
            d_model=d_model,
            d_latent=d_latent,
            num_gaussians=num_gaussians,
            density_mode="none",
            use_occlusion=True
        )
        
        # Test with occlusion=False (Additive Blending, New Default)
        layer_add = SRSEventSplattingResidual(
            seq_len=seq_len,
            d_model=d_model,
            d_latent=d_latent,
            num_gaussians=num_gaussians,
            density_mode="none",
            use_occlusion=False
        )
        
        # Share weights between the two layers to isolate rendering logic difference
        layer_add.generator.load_state_dict(layer_occ.generator.state_dict())
        
        x = torch.randn(2, 7, seq_len)
        
        # Run forward pass
        out_occ = layer_occ(x, patch_num=48)
        out_add = layer_add(x, patch_num=48)
        
        # The output shape must be the same
        self.assertEqual(out_occ.shape, out_add.shape)
        self.assertEqual(out_add.shape, (14, 48, 256))
        
        # Outputs must differ numerically due to different rendering logic (additive vs sorted occlusion)
        self.assertFalse(torch.allclose(out_occ, out_add, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
