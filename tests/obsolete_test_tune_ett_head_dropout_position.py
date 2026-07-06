import unittest
from unittest.mock import patch

from tune_ett_head_dropout_position import (
    DEFAULT_BOUNDS,
    build_command,
    enforce_train_after_freeze,
    expand_bounds_to_include_seed,
    make_console_safe,
    get_output_json_path,
    get_seed_json_path,
    update_bounds,
    load_seed_params,
    run_post_tuning_evaluation,
)


class TestTuneEttHeadDropoutPosition(unittest.TestCase):
    def test_output_json_path_uses_dataset_and_batch_size(self):
        self.assertEqual(
            get_output_json_path("ETTh1", "pre"),
            "best_etth1_512.json",
        )
        self.assertEqual(
            get_output_json_path("ETTh1", "pre", experiment_tag="preonly"),
            "best_etth1_512.json",
        )
        self.assertEqual(
            get_output_json_path("ETTm2", "none"),
            "best_ettm2_512.json",
        )

    def test_build_command_uses_position_and_num_workers_zero(self):
        params = {
            "patch_len": 24,
            "stride": 12,
            "num_gaussians": 4,
            "gs_hidden_dim": 64,
            "gs_dropout": 0.3,
            "gs_weight_decay": 0.01,
            "gs_freeze_epochs": 7,
            "train_epochs": 12,
            "learning_rate": 0.0001,
            "dropout": 0.15,
            "head_dropout": 0.05,
        }

        cmd = build_command("ETTh1", "pre", params, trial_number=3)

        self.assertIn("--head_dropout_position", cmd)
        self.assertIn("pre", cmd)
        self.assertIn("--head_dropout", cmd)
        self.assertIn("0.05", cmd)
        self.assertIn("--dropout", cmd)
        self.assertIn("0.15", cmd)
        self.assertIn("--num_workers", cmd)
        self.assertIn("0", cmd)
        self.assertIn("headpos_pre", " ".join(cmd))

    def test_build_command_keeps_sparse_mode_and_gs_lambda_when_requested(self):
        params = {
            "patch_len": 24,
            "stride": 12,
            "num_gaussians": 4,
            "gs_hidden_dim": 64,
            "gs_dropout": 0.3,
            "gs_weight_decay": 0.01,
            "gs_freeze_epochs": 7,
            "train_epochs": 12,
            "learning_rate": 0.0001,
            "dropout": 0.15,
            "head_dropout": 0.05,
            "gs_lambda": 0.0002,
        }

        cmd = build_command("ETTh1", "pre", params, trial_number=3, density_mode="sparse")

        self.assertIn("--density_mode", cmd)
        self.assertEqual(cmd[cmd.index("--density_mode") + 1], "sparse")
        self.assertIn("--gs_lambda", cmd)
        self.assertEqual(cmd[cmd.index("--gs_lambda") + 1], "0.0002")

    def test_sparse_seed_path_uses_sparse_best_json(self):
        self.assertEqual(
            get_seed_json_path("ETTh1", "sparse"),
            "best_params_etth1_sparse.json",
        )
        self.assertEqual(
            get_seed_json_path("ETTh1", "none"),
            "best_params_etth1.json",
        )

    def test_none_position_forces_head_dropout_zero(self):
        params = {
            "patch_len": 24,
            "stride": 12,
            "num_gaussians": 4,
            "gs_hidden_dim": 64,
            "gs_dropout": 0.3,
            "gs_weight_decay": 0.01,
            "gs_freeze_epochs": 7,
            "train_epochs": 12,
            "learning_rate": 0.0001,
            "dropout": 0.15,
            "head_dropout": 0.2,
        }

        cmd = build_command("ETTh1", "none", params, trial_number=3)
        head_dropout_idx = cmd.index("--head_dropout") + 1

        self.assertEqual(cmd[head_dropout_idx], "0.0")
        self.assertIn("none", cmd)

    def test_enforce_train_epochs_at_least_five_after_freeze(self):
        params = {
            "gs_freeze_epochs": 20,
            "train_epochs": 20,
        }

        adjusted = enforce_train_after_freeze(params)

        self.assertEqual(adjusted["gs_freeze_epochs"], 20)
        self.assertEqual(adjusted["train_epochs"], 25)

    def test_build_command_enforces_train_after_freeze(self):
        params = {
            "patch_len": 24,
            "stride": 12,
            "num_gaussians": 4,
            "gs_hidden_dim": 64,
            "gs_dropout": 0.3,
            "gs_weight_decay": 0.01,
            "gs_freeze_epochs": 20,
            "train_epochs": 20,
            "learning_rate": 0.0001,
            "dropout": 0.15,
            "head_dropout": 0.05,
        }

        cmd = build_command("ETTh1", "pre", params, trial_number=3)
        freeze_epochs = int(cmd[cmd.index("--gs_freeze_epochs") + 1])
        train_epochs = int(cmd[cmd.index("--train_epochs") + 1])

        self.assertGreaterEqual(train_epochs, freeze_epochs + 5)

    def test_update_bounds_expands_when_best_hits_edges(self):
        bounds = DEFAULT_BOUNDS.copy()
        best_params = {
            "gs_dropout": bounds["gs_dropout_upper"],
            "dropout": bounds["dropout_upper"],
            "head_dropout": bounds["head_dropout_upper"],
            "gs_freeze_epochs": bounds["gs_freeze_epochs_upper"],
            "train_epochs": bounds["train_epochs_upper"],
            "learning_rate": bounds["learning_rate_upper"],
            "gs_lambda": bounds["gs_lambda_upper"],
        }

        updated = update_bounds(best_params, bounds, "pre")

        self.assertGreater(updated["gs_dropout_upper"], DEFAULT_BOUNDS["gs_dropout_upper"])
        self.assertGreater(updated["dropout_upper"], DEFAULT_BOUNDS["dropout_upper"])
        self.assertGreater(updated["head_dropout_upper"], DEFAULT_BOUNDS["head_dropout_upper"])
        self.assertGreater(updated["gs_freeze_epochs_upper"], DEFAULT_BOUNDS["gs_freeze_epochs_upper"])
        self.assertGreater(updated["train_epochs_upper"], DEFAULT_BOUNDS["train_epochs_upper"])
        self.assertGreater(updated["learning_rate_upper"], DEFAULT_BOUNDS["learning_rate_upper"])
        self.assertGreater(updated["gs_lambda_upper"], DEFAULT_BOUNDS["gs_lambda_upper"])

    def test_expand_bounds_to_include_sparse_seed_values(self):
        bounds = DEFAULT_BOUNDS.copy()
        seed = {
            "gs_dropout": 0.56,
            "gs_lambda": 0.003,
            "train_epochs": 62,
            "learning_rate": 0.001,
            "dropout": 0.2,
            "head_dropout": 0.1,
        }

        expanded = expand_bounds_to_include_seed(bounds, seed, "pre")

        self.assertGreaterEqual(expanded["gs_dropout_upper"], 0.56)
        self.assertGreaterEqual(expanded["gs_lambda_upper"], 0.003)
        self.assertGreaterEqual(expanded["train_epochs_upper"], 62)

    def test_update_bounds_skips_head_dropout_for_none_position(self):
        bounds = DEFAULT_BOUNDS.copy()
        best_params = {"head_dropout": bounds["head_dropout_upper"]}

        updated = update_bounds(best_params, bounds, "none")

        self.assertEqual(updated["head_dropout_upper"], DEFAULT_BOUNDS["head_dropout_upper"])

    def test_make_console_safe_replaces_unencodable_characters(self):
        text = "prefix ڳ suffix"

        safe = make_console_safe(text, encoding="gbk")

        self.assertEqual(safe, "prefix ? suffix")

    @patch("tune_ett_head_dropout_position.get_output_json_path")
    def test_load_seed_params_prefers_latest_hpo_result(self, mock_get_path):
        import os
        import json
        
        latest_file = "best_params_head_dropout_test_tag_etth1_pre.json"
        mock_get_path.return_value = latest_file
        old_seed_file = "best_params_etth1_sparse.json"
        
        # 确保清理
        for f in (latest_file, old_seed_file):
            if os.path.exists(f):
                os.remove(f)
                
        try:
            # 写入最新 HPO JSON 文件
            latest_data = {
                "patch_len": 16,
                "stride": 16,
                "gs_freeze_epochs": 10,
                "train_epochs": 20,
                "gs_lambda": 1.2e-5,
            }
            with open(latest_file, "w", encoding="utf-8") as f:
                json.dump(latest_data, f)
                
            # 运行测试：应该优先读取 latest_file
            seed, path = load_seed_params("ETTh1", "pre", "sparse", "test_tag")
            self.assertEqual(path, latest_file)
            self.assertEqual(seed["patch_len"], 16)
            self.assertEqual(seed["gs_freeze_epochs"], 10)
        finally:
            if os.path.exists(latest_file):
                os.remove(latest_file)

    @patch("tune_ett_head_dropout_position.get_output_json_path")
    def test_load_seed_params_falls_back_to_old_seed_when_latest_missing(self, mock_get_path):
        import os
        import json
        
        latest_file = "best_params_head_dropout_test_tag_etth1_pre.json"
        mock_get_path.return_value = latest_file
        old_seed_file = "best_params_etth1_sparse.json"
        
        for f in (latest_file, old_seed_file):
            if os.path.exists(f):
                os.remove(f)
                
        try:
            # 写入旧版本种子文件
            old_data = {
                "patch_len": 24,
                "stride": 8,
                "gs_freeze_epochs": 5,
                "train_epochs": 15,
                "gs_lambda": 2.5e-5,
            }
            with open(old_seed_file, "w", encoding="utf-8") as f:
                json.dump(old_data, f)
                
            # 运行测试：找不到最新 HPO 参数文件，应该回退到旧版种子文件
            seed, path = load_seed_params("ETTh1", "pre", "sparse", "test_tag")
            self.assertEqual(path, old_seed_file)
            self.assertEqual(seed["patch_len"], 24)
            self.assertEqual(seed["gs_freeze_epochs"], 5)
        finally:
            if os.path.exists(old_seed_file):
                os.remove(old_seed_file)

    @patch("tune_ett_head_dropout_position.get_output_json_path")
    @patch("tune_ett_head_dropout_position.get_results_json_path")
    def test_run_post_tuning_evaluation_saves_results(self, mock_get_results, mock_get_path):
        import os
        import json
        
        best_params_file = "best_params_head_dropout_test_tag_etth1_pre.json"
        results_file = "results_head_dropout_test_tag_etth1_pre_best_horizons.json"
        mock_get_path.return_value = best_params_file
        mock_get_results.return_value = results_file
        
        # 确保清理
        for f in (best_params_file, results_file):
            if os.path.exists(f):
                os.remove(f)
                
        try:
            # 写入模拟最佳参数
            best_data = {
                "patch_len": 16,
                "stride": 16,
                "num_gaussians": 3,
                "gs_hidden_dim": 64,
                "gs_dropout": 0.3,
                "gs_weight_decay": 0.01,
                "gs_freeze_epochs": 5,
                "train_epochs": 15,
                "learning_rate": 1e-4,
                "dropout": 0.1,
                "head_dropout": 0.05,
                "gs_lambda": 1e-5,
            }
            with open(best_params_file, "w", encoding="utf-8") as f:
                json.dump(best_data, f)
                
            # Mock 掉 run_command 避免跑真实的神经网络
            with patch("tune_ett_head_dropout_position.run_command") as mock_run:
                mock_run.return_value = (0.35, 0.40) # mse, mae
                
                run_post_tuning_evaluation("ETTh1", "pre", "test_tag", "sparse")
                
            # 验证结果文件生成
            self.assertTrue(os.path.exists(results_file))
            with open(results_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
                
            self.assertEqual(payload["dataset"], "ETTh1")
            self.assertEqual(payload["head_dropout_position"], "pre")
            self.assertEqual(len(payload["results"]), 4)
            for res in payload["results"]:
                self.assertIn(res["horizon"], [96, 192, 336, 720])
                self.assertEqual(res["mse"], 0.35)
                self.assertEqual(res["mae"], 0.40)
        finally:
            for f in (best_params_file, results_file):
                if os.path.exists(f):
                    os.remove(f)



    def test_sample_params_uses_fixed_num_gaussians_per_dataset(self):
        from unittest.mock import MagicMock
        from tune_ett_head_dropout_position import sample_params
        
        trial = MagicMock()
        trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        trial.suggest_int.side_effect = lambda name, low, high: low
        trial.suggest_float.side_effect = lambda name, low, high, **kwargs: low
        
        bounds = DEFAULT_BOUNDS.copy()
        
        # Test ETTh1 -> 4
        params = sample_params(trial, bounds, "pre", "ETTh1")
        self.assertEqual(params["num_gaussians"], 4)
        
        # Test ETTh2 -> 3
        params = sample_params(trial, bounds, "pre", "ETTh2")
        self.assertEqual(params["num_gaussians"], 3)
        
        # Test ETTm1 -> 5
        params = sample_params(trial, bounds, "pre", "ETTm1")
        self.assertEqual(params["num_gaussians"], 5)
        
        # Test ETTm2 -> 4
        params = sample_params(trial, bounds, "pre", "ETTm2")
        self.assertEqual(params["num_gaussians"], 4)
        
        # Test Weather -> 6
        params = sample_params(trial, bounds, "pre", "Weather")
        self.assertEqual(params["num_gaussians"], 6)
        
        # Verify trial.suggest_categorical was not called with "num_gaussians"
        for call in trial.suggest_categorical.call_args_list:
            self.assertNotEqual(call[0][0], "num_gaussians")


if __name__ == "__main__":
    unittest.main()
