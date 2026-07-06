import unittest

import torch

from layers.common_blocks import FlattenHead
from models.gs_linear import Model


class TestFlattenHeadDropoutPosition(unittest.TestCase):
    def _head(self, position):
        head = FlattenHead(
            n_vars=1,
            nf=4,
            target_window=2,
            head_dropout=1.0,
            head_dropout_position=position,
        )
        with torch.no_grad():
            head.head.weight.fill_(1.0)
            head.head.bias.copy_(torch.tensor([1.0, -1.0]))
        head.train()
        return head

    def test_post_position_drops_final_prediction(self):
        head = self._head("post")
        x = torch.ones(1, 1, 4)

        out = head(x)

        self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_pre_position_drops_head_input_not_prediction(self):
        head = self._head("pre")
        x = torch.ones(1, 1, 4)

        out = head(x)

        expected = torch.tensor([[1.0, -1.0]])
        self.assertTrue(torch.equal(out, expected))

    def test_none_position_ignores_head_dropout(self):
        head = self._head("none")
        x = torch.ones(1, 1, 4)

        out = head(x)

        expected = torch.tensor([[5.0, 3.0]])
        self.assertTrue(torch.equal(out, expected))

    def test_invalid_position_rejected(self):
        with self.assertRaises(ValueError):
            FlattenHead(
                n_vars=1,
                nf=4,
                target_window=2,
                head_dropout=0.1,
                head_dropout_position="bad",
            )

    def test_splatts_passes_head_dropout_position(self):
        class Config:
            task_name = "long_term_forecast"
            seq_len = 336
            pred_len = 96
            c_out = 7
            enc_in = 7
            d_model = 256
            patch_len = 24
            stride = 24
            representation = "gs"
            gs_hidden_dim = 32
            num_gaussians = 5
            gs_dropout = 0.3
            dropout = 0.1
            head_dropout = 0.1
            head_dropout_position = "pre"
            subtract_last = False
            density_mode = "none"

        model = Model(Config())

        self.assertEqual(model.head_srs.head_dropout_position, "pre")


if __name__ == "__main__":
    unittest.main()
