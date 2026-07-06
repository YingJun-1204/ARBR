import unittest

from run_head_dropout_pre_sparse_horizons import (
    build_eval_command,
    get_best_json_path,
    get_results_json_path,
    parse_horizons,
)


class TestRunHeadDropoutPreSparseHorizons(unittest.TestCase):
    def test_best_json_path_matches_pre_sparse_tuning_outputs(self):
        self.assertEqual(
            get_best_json_path("ETTh1", "pre", "pre_sparse"),
            "best_params_head_dropout_pre_sparse_etth1_pre.json",
        )

    def test_results_json_path_is_separate_from_existing_sparse_results(self):
        self.assertEqual(
            get_results_json_path("ETTm2", "pre", "pre_sparse"),
            "results_head_dropout_pre_sparse_ettm2_pre_best_horizons.json",
        )

    def test_parse_horizons_accepts_comma_separated_values(self):
        self.assertEqual(parse_horizons("96,192,336,720"), [96, 192, 336, 720])

    def test_build_eval_command_preserves_pre_sparse_parameters(self):
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

        cmd = build_eval_command("ETTh1", 192, params, "pre", "sparse", "pre_sparse")

        self.assertEqual(cmd[cmd.index("--pred_len") + 1], "192")
        self.assertEqual(cmd[cmd.index("--head_dropout_position") + 1], "pre")
        self.assertEqual(cmd[cmd.index("--density_mode") + 1], "sparse")
        self.assertEqual(cmd[cmd.index("--gs_lambda") + 1], "0.0002")
        self.assertEqual(cmd[cmd.index("--dropout") + 1], "0.15")
        self.assertEqual(cmd[cmd.index("--head_dropout") + 1], "0.05")
        self.assertEqual(cmd[cmd.index("--num_workers") + 1], "0")


if __name__ == "__main__":
    unittest.main()
