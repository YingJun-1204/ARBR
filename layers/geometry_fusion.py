import math
import torch
import torch.nn as nn


class GeometryConditionedResidualGate(nn.Module):
    """
    Produce a sample- and patch-dependent residual injection weight from
    field-derived geometric statistics.

    Inputs
    ------
    confidence:
        Explicit-field coverage confidence, shape (N, P, 1).
    delta:
        Local normalized displacement, shape (N, P, 1).
    max_shift_local:
        Maximum absolute local displacement allowed by the Jet.

    Output
    ------
    beta:
        Residual injection weight, shape (N, P, 1), bounded in
        (0, beta_max).
    features:
        The normalized geometric features used by the gate.
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        beta_init: float = 0.1,
        beta_max: float = 0.5,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )

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
        base_logit = math.log(
            init_ratio / (1.0 - init_ratio)
        )

        self.register_buffer(
            "base_logit",
            torch.tensor(base_logit, dtype=torch.float32),
            persistent=False,
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Make the initial gate exactly equal to beta_init for all
        # samples and patches.
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(
        self,
        confidence: torch.Tensor,
        delta: torch.Tensor,
        max_shift_local: float,
    ):
        if confidence.ndim != 3 or confidence.shape[-1] != 1:
            raise ValueError(
                "confidence must have shape (N, P, 1), "
                f"got {tuple(confidence.shape)}"
            )

        if delta.shape != confidence.shape:
            raise ValueError(
                "delta and confidence must have the same shape, "
                f"got delta={tuple(delta.shape)}, "
                f"confidence={tuple(confidence.shape)}"
            )

        if max_shift_local <= 0.0:
            raise ValueError(
                "max_shift_local must be positive, "
                f"got {max_shift_local}"
            )

        eps = 1e-6

        # Larger uncertainty means the explicit field provides weaker
        # local support, so more observation-level evidence may be useful.
        uncertainty = (
            1.0 - confidence
        ).clamp(0.0, 1.0)

        # Normalize displacement to a bounded and configuration-independent
        # quantity.
        shift_magnitude = (
            delta.abs() / (max_shift_local + eps)
        ).clamp(0.0, 1.0)

        geometry_features = torch.cat(
            [
                uncertainty,
                shift_magnitude,
            ],
            dim=-1,
        )

        gate_offset = self.gate_mlp(
            geometry_features
        )

        base_logit = self.base_logit.to(
            device=gate_offset.device,
            dtype=gate_offset.dtype,
        )

        beta = self.beta_max * torch.sigmoid(
            base_logit + gate_offset
        )

        return beta, geometry_features
