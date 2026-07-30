import math
import torch
import torch.nn as nn


class GeometryConditionedResidualGate(nn.Module):
    """
    Produce a static global learnable residual injection weight:
    beta = beta_max * sigmoid(b)

    where b is a dataset-shared learnable scalar parameter.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        beta_init: float = 0.1,
        beta_max: float = 0.5,
    ):
        super().__init__()

        if beta_max <= 0.0:
            raise ValueError(
                f"beta_max must be positive, got {beta_max}"
            )

        if not 0.0 < beta_init < beta_max:
            raise ValueError(
                "beta_init must satisfy 0 < beta_init < beta_max, "
                f"got beta_init={beta_init}, beta_max={beta_max}"
            )

        self.beta_max = float(beta_max)

        init_ratio = beta_init / beta_max
        b_init = math.log(init_ratio / (1.0 - init_ratio))

        self.b = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))

    def forward(
        self,
        confidence: torch.Tensor = None,
        delta: torch.Tensor = None,
        max_shift_local: float = None,
    ):
        beta = self.beta_max * torch.sigmoid(self.b)
        return beta, None
