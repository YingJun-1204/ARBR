import unittest
import torch
from layers.residual_router import AdaptiveResidualRouter

class TestSoftScaleNormalization(unittest.TestCase):
    def test_soft_scale_suppresses_flat_noise(self):
        router = AdaptiveResidualRouter(d_model=256, gate_type="adaptive_direction")
        
        # Test Case 1: Noise Sequence (Tiny Variance)
        # Sequence of length 10 with mean 2.0 and std 0.001
        # Shape: (batch_channel=1, patch_num=10, 1)
        noise_input = torch.tensor([2.0, 2.001, 1.999, 2.0, 2.0005, 1.9995, 2.0, 2.0, 2.001, 1.999]).unsqueeze(0).unsqueeze(-1)
        
        # Test Case 2: Signal Sequence (Healthy Variance)
        # Sequence of length 10 with mean 2.0 and std ~1.4
        signal_input = torch.tensor([2.0, 4.0, 0.0, 2.0, 3.0, 1.0, 2.0, 2.0, 4.0, 0.0]).unsqueeze(0).unsqueeze(-1)
        
        # Run zscore with soft scale (internally called via helper)
        z_noise = router._zscore_patch(noise_input)
        z_signal = router._zscore_patch(signal_input)
        
        # Verify noise is suppressed
        noise_std = z_noise.std(dim=1, keepdim=True, unbiased=False)
        # Standard deviation of normalized noise should be very small (< 0.05) instead of 1.0
        self.assertLess(noise_std.item(), 0.05)
        
        # Verify signal is preserved
        signal_std = z_signal.std(dim=1, keepdim=True, unbiased=False)
        # Standard deviation of normalized signal should be close to 1.0 (e.g. > 0.85)
        self.assertGreater(signal_std.item(), 0.85)
        self.assertLessEqual(signal_std.item(), 1.0)

if __name__ == "__main__":
    unittest.main()
