import math

import torch.nn as nn


class PatchLinearRepresentation(nn.Module):
    def __init__(self, seq_len, patch_len, stride, d_model, dropout=0.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = math.ceil((seq_len - patch_len) / stride) + 1
        padding = patch_len + (self.patch_num - 1) * stride - seq_len

        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.projection = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, patch_num=None):
        B, C, L = x_seq.shape
        x_seq = self.padding_patch_layer(x_seq)
        patches = x_seq.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(B * C, self.patch_num, self.patch_len)
        return self.dropout(self.projection(patches))
