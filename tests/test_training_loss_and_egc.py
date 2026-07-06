import unittest
import torch
import torch.nn as nn
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

class TestTrainingLossAndEgc(unittest.TestCase):
    def test_joint_loss_computation(self):
        class Args:
            model = "SplatTS"
            task_name = "long_term_forecast"
            seq_len = 336
            pred_len = 96
            enc_in = 7
            dec_in = 7
            c_out = 7
            d_model = 256
            patch_len = 24
            stride = 24
            representation = "gs"
            gs_hidden_dim = 32
            num_gaussians = 5
            gs_dropout = 0.3
            head_dropout = 0.1
            dropout = 0.1
            subtract_last = False
            density_mode = "sparse"
            gs_lambda = 0.01
            use_multi_gpu = False
            use_gpu = False
            features = "M"
            gs_weight_decay = 0.01

        args = Args()
        exp = Exp_Long_Term_Forecast(args)
        exp.model = exp._build_model()
        
        # Forward pass
        batch_x = torch.randn(8, 336, 7)
        outputs = exp.model(batch_x)
        
        # Calculate regular MSE
        batch_y = torch.randn(8, 96, 7)
        criterion = nn.MSELoss()
        base_loss = criterion(outputs, batch_y)
        
        # Joint Loss Calculation Simulation
        loss = base_loss
        density_mode = getattr(exp.args, "density_mode", "none")
        if density_mode == "sparse":
            real_model = exp.model.module if isinstance(exp.model, nn.DataParallel) else exp.model
            splatting_module = getattr(real_model, "splatting_residual", None)
            if splatting_module is not None and getattr(splatting_module, "last_density_score", None) is not None:
                scores = splatting_module.last_density_score
                loss_density = torch.sigmoid(scores).mean()
                loss = loss + exp.args.gs_lambda * loss_density
                
        # With active density penalty and positive gs_lambda, the loss should change
        self.assertNotEqual(loss.item(), base_loss.item())
        self.assertGreater(loss.item(), base_loss.item())

if __name__ == "__main__":
    unittest.main()
