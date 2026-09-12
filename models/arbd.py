import math
import torch
import torch.nn as nn
from layers.arbd_encoder import ARBDEncoder
from layers.common_blocks import FlattenHead
from layers.RevIN import RevIN


class Model(nn.Module):
    """Adaptive Radial Basis Decomposition (ARBD) Network.

    Paper Model: ARBD
    Consists of:
    - Channel-Independent RevIN normalization
    - ARBDEncoder: ARG (Module 1) + RCG & Renderer (Module 2) + CLRC (Module 3)
    - FlattenHead projection to forecasting horizon
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.dim = int(configs.d_model)
        self.patch_len = int(configs.patch_len)
        self.stride = int(getattr(configs, "stride", self.patch_len))
        self.num_gaussians = int(getattr(configs, "num_gaussians", 8))
        if self.num_gaussians <= 0:
            raise ValueError("num_gaussians must be positive")

        self.num_patches_srs = math.ceil(
            (self.seq_len - self.patch_len) / self.stride
        ) + 1

        self.norm = RevIN(
            configs.enc_in,
            affine=True,
            subtract_last=getattr(configs, "subtract_last", False),
        )

        self.arbd_encoder = ARBDEncoder(
            seq_len=self.seq_len,
            d_model=self.dim,
            num_gaussians=self.num_gaussians,
            gs_dropout=getattr(configs, "gs_dropout", 0.3),
            patch_len=self.patch_len,
            stride=self.stride,
            local_weight_max=getattr(configs, "local_weight_max", 0.75),
            static_local_weight_init=getattr(
                configs,
                "static_local_weight_init",
                0.5,
            ),
            rho=getattr(configs, "rho", 0.5),
            beta=getattr(configs, "beta", 0.5),
        )

        self.head_srs = FlattenHead(
            nf=self.dim * self.num_patches_srs,
            target_window=self.pred_len,
            head_dropout=getattr(configs, "head_dropout", 0.0),
            mode=getattr(configs, "head_mode", "linear"),
            head_dropout_position=getattr(
                configs,
                "head_dropout_position",
                "pre",
            ),
        )

    @property
    def splatting_residual(self):
        return self.arbd_encoder

    def forward(self, x):
        batch_size, _, channels = x.shape
        x_norm = self.norm(x, "norm")
        x_repr = self.arbd_encoder(
            x_norm.permute(0, 2, 1),
            self.num_patches_srs,
        )
        x_repr = x_repr.reshape(
            batch_size,
            channels,
            self.num_patches_srs,
            self.dim,
        )
        x_repr = x_repr.permute(0, 1, 3, 2).reshape(
            -1,
            self.dim,
            self.num_patches_srs,
        )
        pred = self.head_srs(x_repr)
        pred = pred.reshape(
            batch_size,
            channels,
            self.pred_len,
        ).permute(0, 2, 1)
        return self.norm(pred, "denorm")
