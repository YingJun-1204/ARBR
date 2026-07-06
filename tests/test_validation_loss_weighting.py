import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


class _ZeroForecastModel:
    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, batch_x, is_training=False):
        return torch.zeros(batch_x.shape[0], 2, 1)


class TestValidationLossWeighting(unittest.TestCase):
    def test_vali_weights_batches_by_number_of_elements(self):
        exp = object.__new__(Exp_Long_Term_Forecast)
        exp.model = _ZeroForecastModel()
        exp.device = torch.device("cpu")
        exp.args = SimpleNamespace(pred_len=2, features="M")

        batch1 = (
            torch.zeros(2, 2, 1),
            torch.zeros(2, 2, 1),
            torch.zeros(1),
            torch.zeros(1),
        )
        batch2 = (
            torch.zeros(1, 2, 1),
            torch.full((1, 2, 1), 3.0),
            torch.zeros(1),
            torch.zeros(1),
        )

        loss = exp.vali(None, [batch1, batch2], nn.MSELoss())

        self.assertAlmostEqual(loss, 3.0)


if __name__ == "__main__":
    unittest.main()
