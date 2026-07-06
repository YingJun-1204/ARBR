import unittest
import os
import shutil
import torch
import torch.nn as nn
from utils.tools import EarlyStopping

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

class TestNoSaveCheckpoint(unittest.TestCase):
    def setUp(self):
        self.test_dir = "./test_checkpoints_temp"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        self.model = SimpleModel()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_checkpoint_to_disk_when_false(self):
        # Default behavior: no_save_checkpoint=False (writes to disk)
        early_stopping = EarlyStopping(patience=5, verbose=False)
        self.assertFalse(hasattr(early_stopping, "no_save_checkpoint") and early_stopping.no_save_checkpoint)
        
        # Trigger checkpoint save
        early_stopping(val_loss=1.0, model=self.model, path=self.test_dir)
        
        # Verify file is created on disk
        checkpoint_path = os.path.join(self.test_dir, "checkpoint.pth")
        self.assertTrue(os.path.exists(checkpoint_path))

    def test_save_checkpoint_to_ram_when_true(self):
        # New behavior: no_save_checkpoint=True (caches in RAM)
        early_stopping = EarlyStopping(patience=5, verbose=False, no_save_checkpoint=True)
        self.assertTrue(early_stopping.no_save_checkpoint)
        self.assertIsNone(early_stopping.best_model_state)
        
        # Trigger checkpoint save
        early_stopping(val_loss=1.0, model=self.model, path=self.test_dir)
        
        # Verify no folder or file is created on disk
        checkpoint_path = os.path.join(self.test_dir, "checkpoint.pth")
        self.assertFalse(os.path.exists(checkpoint_path))
        
        # Verify state is saved in CPU RAM
        self.assertIsNotNone(early_stopping.best_model_state)
        # Check that tensor location is CPU
        for name, tensor in early_stopping.best_model_state.items():
            self.assertEqual(tensor.device, torch.device("cpu"))

    def test_exp_train_memory_caching(self):
        from unittest.mock import MagicMock, patch
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        
        args = MagicMock()
        args.checkpoints = self.test_dir
        args.patience = 3
        args.train_epochs = 1
        args.no_save_checkpoint = True
        args.use_gpu = False
        args.use_multi_gpu = False
        args.model = "SplatTS"
        args.features = "M"
        args.gate_lambda = 0.0
        args.density_mode = "none"
        
        with patch('exp.exp_long_term_forecasting.Exp_Long_Term_Forecast._build_model') as mock_build_model, \
             patch('exp.exp_long_term_forecasting.Exp_Long_Term_Forecast._get_data') as mock_get_data, \
             patch('exp.exp_long_term_forecasting.adjust_learning_rate') as mock_adj_lr:
            
            # Setup mock model and data loader
            mock_model = MagicMock()
            mock_model.state_dict.return_value = {"param": torch.tensor([1.0])}
            mock_build_model.return_value = mock_model
            
            mock_loader = MagicMock()
            mock_loader.__len__.return_value = 1
            mock_loader.__iter__.return_value = iter([(torch.zeros(1, 10, 1), torch.zeros(1, 10, 1), None, None)])
            mock_get_data.return_value = (None, mock_loader)
            
            exp = Exp_Long_Term_Forecast(args)
            exp.device = torch.device("cpu")
            exp._select_optimizer = MagicMock()
            exp._select_criterion = MagicMock()
            exp.vali = MagicMock(return_value=0.5)
            
            # Run train
            exp.train("test_setting")
            
            # Verify no checkpoints directory was ever created
            self.assertFalse(os.path.exists(self.test_dir))

if __name__ == "__main__":
    unittest.main()
