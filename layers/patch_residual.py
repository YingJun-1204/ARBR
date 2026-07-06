import torch
import torch.nn as nn
import math

class PatchResidualProjection(nn.Module):
    def __init__(self, seq_len, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = math.ceil((seq_len - patch_len) / stride) + 1
        padding = patch_len + (self.patch_num - 1) * stride - seq_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.projection = nn.Linear(patch_len, d_model, bias=False)

    def forward(self, x_seq, batch_channel):
        if len(x_seq.shape) == 2:
            x_seq_temp = x_seq.unsqueeze(1)
        else:
            x_seq_temp = x_seq
        
        x_padded = self.padding_patch_layer(x_seq_temp)
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(batch_channel, self.patch_num, self.patch_len)
        res_proj = self.projection(patches)
        return res_proj
