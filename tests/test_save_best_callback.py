import os
import json
import unittest
from unittest.mock import MagicMock, patch
import sys

# Ensure workspace root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tune_gate_etth1 import save_best_callback

class TestSaveBestCallback(unittest.TestCase):
    def setUp(self):
        self.files_to_clean = [
            "best_params_head_dropout_weather_pre.json",
        ]
        self.backups = {}
        for f in self.files_to_clean:
            if os.path.exists(f):
                with open(f, "r", encoding="utf-8") as file:
                    self.backups[f] = file.read()
                os.remove(f)

    def tearDown(self):
        for f in self.files_to_clean:
            if os.path.exists(f):
                os.remove(f)
            if f in self.backups:
                with open(f, "w", encoding="utf-8") as file:
                    file.write(self.backups[f])

    @patch('tune_gate_etth1.get_output_json_path')
    def test_save_best_callback_comparison(self, mock_get_path):
        mock_get_path.return_value = "best_params_head_dropout_weather_pre.json"
        study = MagicMock()
        trial = MagicMock()
        
        trial.number = 5
        study.best_trial.number = 5
        study.best_value = 0.146
        study.best_params = {"patch_len": 16, "stride": 24, "gs_freeze_epochs": 5, "train_epochs": 10}
        study.best_trial.user_attrs = {"mae": 0.20}
        
        callback = save_best_callback(
            position="pre",
            output_dir="weather",
            gate_type="none",
            density_mode="none",
            pred_len=96
        )
        
        # Test case 1: JSON does not exist, should write
        callback(study, trial)
        self.assertTrue(os.path.exists("best_params_head_dropout_weather_pre.json"))
        with open("best_params_head_dropout_weather_pre.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["mse"], 0.146)
        # Check that fixed/unwanted parameters are not present
        self.assertNotIn("patch_len", data)
        self.assertNotIn("train_epochs", data)
        
        # Test case 2: JSON exists with better MSE (0.145), callback should NOT overwrite with poorer MSE (0.146)
        better_data = {"mse": 0.145, "mae": 0.19, "batch_size": 512}
        with open("best_params_head_dropout_weather_pre.json", "w", encoding="utf-8") as f:
            json.dump(better_data, f, indent=4)
            
        study.best_value = 0.146
        study.best_params = {"batch_size": 256, "gs_dropout": 0.5}
        study.best_trial.user_attrs = {"mae": 0.20}
        
        callback(study, trial)
        
        with open("best_params_head_dropout_weather_pre.json", "r", encoding="utf-8") as f:
            data_after = json.load(f)
        self.assertEqual(data_after["mse"], 0.145)
        self.assertEqual(data_after["batch_size"], 512)
        
        # Test case 3: JSON exists, callback gets a better MSE (0.140), should overwrite
        study.best_value = 0.140
        study.best_params = {"batch_size": 256, "gs_dropout": 0.5}
        study.best_trial.user_attrs = {"mae": 0.18}
        
        callback(study, trial)
        
        with open("best_params_head_dropout_weather_pre.json", "r", encoding="utf-8") as f:
            data_final = json.load(f)
        self.assertEqual(data_final["mse"], 0.140)
        self.assertEqual(data_final["batch_size"], 256)

if __name__ == "__main__":
    unittest.main()

