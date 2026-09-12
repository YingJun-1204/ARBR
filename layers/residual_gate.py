import math

import torch
import torch.nn as nn


def _initial_logit(weight_init, weight_max):
    weight_init = float(weight_init)
    weight_max = float(weight_max)
    if weight_max <= 0.0:
        raise ValueError(f"local_weight_max must be positive, got {weight_max}")
    if not 0.0 < weight_init < weight_max:
        raise ValueError(
            "static_local_weight_init must satisfy "
            f"0 < weight < local_weight_max, got {weight_init} and {weight_max}"
        )
    probability = weight_init / weight_max
    return math.log(probability / (1.0 - probability))


class StaticResidualGate(nn.Module):
    """Learnable dataset-shared weight for the Linear correction."""

    def __init__(self, weight_init=0.5, weight_max=0.75):
        super().__init__()
        self.weight_max = float(weight_max)
        self.raw_weight = nn.Parameter(
            torch.tensor(
                _initial_logit(weight_init, weight_max),
                dtype=torch.float32,
            )
        )

    def forward(self):
        return self.weight_max * torch.sigmoid(self.raw_weight)
