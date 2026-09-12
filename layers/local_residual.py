import math

import torch.nn as nn


class LocalResidualProjection(nn.Module):
    """Project local observation patches into the model dimension."""

    def __init__(self, seq_len, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.patch_num = math.ceil(
            (int(seq_len) - self.patch_len) / self.stride
        ) + 1

        padding = (
            self.patch_len
            + (self.patch_num - 1) * self.stride
            - int(seq_len)
        )
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.projection = nn.Linear(self.patch_len, d_model, bias=False)

    def extract_patches(self, x_seq):
        if x_seq.ndim == 2:
            x_seq = x_seq.unsqueeze(1)

        patches = self.padding_patch_layer(x_seq).unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )
        return patches.reshape(-1, self.patch_num, self.patch_len)

    def forward(self, x_seq, return_patches=False):
        patches = self.extract_patches(x_seq)
        residual = self.projection(patches)
        if return_patches:
            return residual, patches
        return residual
