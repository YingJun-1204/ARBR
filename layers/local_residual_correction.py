import torch
import torch.nn as nn

from layers.local_residual import LocalResidualProjection
from layers.residual_gate import StaticResidualGate


class CalibratedLocalResidualCorrection(nn.Module):
    """Calibrated Local Residual Correction (CLRC) - Module 3 of ARBD.

    Acts as a lightweight, auxiliary complementary pathway to ARBD:
      Z = G + lambda * L
    where L is a local patch-level linear projection and lambda in (0, lambda_max)
    is a dataset-shared bounded scalar gate.
    """

    def __init__(
        self,
        seq_len: int,
        patch_len: int,
        stride: int,
        d_model: int,
        weight_init: float = 0.5,
        weight_max: float = 0.75,
        has_residual_gate: bool = True,
    ):
        super().__init__()
        self.local_projection = LocalResidualProjection(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
        )
        self.residual_gate = None
        if has_residual_gate:
            self.residual_gate = StaticResidualGate(
                weight_init=weight_init,
                weight_max=weight_max,
            )

    @property
    def patch_num(self):
        return self.local_projection.patch_num

    def forward(self, x_seq: torch.Tensor, radial_repr: torch.Tensor):
        """Applies calibrated local residual correction to the radial representation.

        Args:
            x_seq: [N, L, 1] normalized univariate patch sequences
            radial_repr: [N, P, D] rendered radial representation

        Returns:
            fused_repr: [N, P, D]
            local_repr: [N, P, D]
            local_weight: scalar tensor
        """
        local_repr = self.local_projection(x_seq)
        if self.residual_gate is not None:
            local_weight = self.residual_gate()
            fused_repr = radial_repr + local_weight * local_repr
        else:
            local_weight = torch.tensor(1.0, device=x_seq.device, dtype=x_seq.dtype)
            fused_repr = radial_repr + local_repr
        return fused_repr, local_repr, local_weight
