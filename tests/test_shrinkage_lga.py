import unittest
import torch
from layers.residual_router import AdaptiveResidualRouter

class TestShrinkageLGA(unittest.TestCase):
    def test_shrinkage_stabilizes_zero_local_variance(self):
        # Initialize router with shrinkage
        router = AdaptiveResidualRouter(d_model=256, gate_type="adaptive_direction", gate_window_half=2)
        
        # We will manually pass inputs to trigger the zero local variance issue
        # batch_channel = 1, patch_num = 5, seq_len = 336
        # Let's mock the arguments of forward:
        # forward(self, weights, rendered_event, res_proj, batch_channel, patch_num, x_seq_device)
        
        # Create weights shape (1, 5, 5)
        weights = torch.ones(1, 5, 5)
        # Create rendered_event shape (1, 5, 256)
        rendered_event = torch.randn(1, 5, 256)
        
        # We want res_proj to be such that for some patch, the neighbors are constant (local variance = 0),
        # but the patch itself is different.
        # res_proj shape: (1, 5, 256)
        res_proj = torch.zeros(1, 5, 256)
        # Set all patches to 0.0 except the center patch (index 2) which is 1.0
        # This will make neighbors (index 0, 1, 3, 4) all constant 0.0 -> local_var = 0.0
        # The center patch itself is 1.0, so diff_lga will be non-zero (1.0).
        res_proj[:, 2, :] = 1.0
        
        # Run forward pass
        gate_scale, gate_direction, gate_strength, gate_route_probs = router(
            weights=weights,
            rendered_event=rendered_event,
            res_proj=res_proj,
            batch_channel=1,
            patch_num=5,
            x_seq_device=torch.device("cpu")
        )
        
        # Retrieve the raw LGA distance
        raw_lga = router.last_raw_lga
        
        # Without shrinkage, local_var_detached is 1e-5.
        # The squared difference is (0 - 1)^2 = 1.
        # division is 1 / 1e-5 = 100,000.
        # With shrinkage (lambda = 0.15):
        # global_var is the variance of res_proj_norm over patch dimension.
        # res_proj_norm has mean ~0.2, std ~0.4.
        # global_var will be around 0.16.
        # robust_var = 0.85 * 0.0 + 0.15 * 0.16 = 0.024.
        # division is 1 / 0.024 = ~41.6.
        # Let's assert raw_lga is small and stable (e.g. < 500), whereas without shrinkage it would be > 10000.
        self.assertLess(raw_lga[0, 2].item(), 500.0)
        self.assertGreater(raw_lga[0, 2].item(), 0.0)

if __name__ == "__main__":
    unittest.main()
