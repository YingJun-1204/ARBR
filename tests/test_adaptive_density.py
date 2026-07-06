import unittest
import torch
from layers.splatting_residual_encoder import SplattingResidualEncoder as SRSEventSplattingResidual

class TestAdaptiveDensity(unittest.TestCase):
    def test_density_mode_none(self):
        # density_mode == "none" should be identical to the original GS layer
        seq_len = 336
        num_gaussians = 5
        d_model = 256
        d_latent = d_model
        
        layer = SRSEventSplattingResidual(
            seq_len=seq_len,
            d_model=d_model,
            d_latent=d_latent,
            num_gaussians=num_gaussians,
            density_mode="none"
        )
        
        # original param projection output length: num_gaussians * (3 + d_latent)
        last_layer = layer.generator[-1]
        self.assertEqual(last_layer.out_features, num_gaussians * (3 + d_latent))
        
        x = torch.randn(8, 7, seq_len) # [B, C, L]
        out = layer(x, patch_num=48)
        self.assertEqual(out.shape, (56, 48, d_model)) # [B*C, patch_num, d_model]
        self.assertIsNone(layer.last_density_score)

    def test_density_mode_soft_and_sparse(self):
        seq_len = 336
        num_gaussians = 5
        d_model = 256
        d_latent = d_model
        
        for mode in ["soft", "sparse"]:
            layer = SRSEventSplattingResidual(
                seq_len=seq_len,
                d_model=d_model,
                d_latent=d_latent,
                num_gaussians=num_gaussians,
                density_mode=mode
            )
            
            # Dynamic parameters output length should be num_gaussians * (4 + d_latent)
            last_layer = layer.generator[-1]
            self.assertEqual(last_layer.out_features, num_gaussians * (4 + d_latent))
            
            x = torch.randn(8, 7, seq_len) # [B, C, L]
            out = layer(x, patch_num=48)
            self.assertEqual(out.shape, (56, 48, d_model))
            self.assertIsNotNone(layer.last_density_score)
            self.assertEqual(layer.last_density_score.shape, (56, num_gaussians))

if __name__ == "__main__":
    unittest.main()
