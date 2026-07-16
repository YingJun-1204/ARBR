import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def inverse_softplus(x):
    return torch.where(x > 20.0, x, torch.log(torch.exp(x) - 1.0))

class GaussianJetProjection(nn.Module):
    def __init__(
        self,
        seq_len,
        patch_len,
        stride,
        d_model,
        num_implicit_gaussians=4,
        jet_scale_init=0.1,
        sigma_init=0.20,
        learnable_mu=True,
        learnable_sigma=True,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = math.ceil((seq_len - patch_len) / stride) + 1
        padding = patch_len + (self.patch_num - 1) * stride - seq_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.d_model = d_model
        
        # 1. 基础线性层，负责拟合高频尖峰与非周期细节
        self.base_projection = nn.Linear(patch_len, d_model, bias=False)
        
        # 2. Gaussian Jet 隐式高斯基底参数
        self.num_implicit_gaussians = num_implicit_gaussians
        
        g_mu_tensor = torch.linspace(0, 1, num_implicit_gaussians)
        if learnable_mu:
            self.g_mu = nn.Parameter(g_mu_tensor)
        else:
            self.register_buffer("g_mu", g_mu_tensor)
            
        g_sigma_tensor = inverse_softplus(torch.full((num_implicit_gaussians,), sigma_init))
        if learnable_sigma:
            self.g_sigma = nn.Parameter(g_sigma_tensor)
        else:
            self.register_buffer("g_sigma", g_sigma_tensor)
            
        self.gaussian_up = nn.Linear(num_implicit_gaussians, d_model, bias=False)
        nn.init.zeros_(self.gaussian_up.weight)
        
        # sigmoid(adapter_logit) = jet_scale_init
        adapter_logit_val = math.log(jet_scale_init / (1.0 - jet_scale_init))
        self.adapter_logit = nn.Parameter(torch.tensor(adapter_logit_val))
        
        # 诊断属性（detach 后存储）
        self.last_phi = None
        self.last_psi = None
        self.last_q0 = None
        self.last_q1 = None
        self.last_jet_norm = None
        self.last_base_norm = None
        self.last_adapter_scale = None

    def _extract_patches(self, x_seq):
        if len(x_seq.shape) == 2:
            x_seq_temp = x_seq.unsqueeze(1)
        else:
            x_seq_temp = x_seq
        
        x_padded = self.padding_patch_layer(x_seq_temp)
        patches = x_padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        batch_channel = x_seq_temp.shape[0] * x_seq_temp.shape[1]
        patches = patches.reshape(batch_channel, self.patch_num, self.patch_len)
        return patches

    def _build_gaussian_jet(self, device, dtype):
        t = torch.linspace(0, 1, self.patch_len, device=device, dtype=dtype)
        mu = self.g_mu.view(-1, 1).to(device=device, dtype=dtype)
        sigma = (F.softplus(self.g_sigma) + 0.03).view(-1, 1).to(device=device, dtype=dtype)
        
        d_mesh = t.view(1, -1) - mu
        eps = 1e-5
        
        # 零阶 Gaussian basis
        phi = torch.exp(-(d_mesh**2) / (2 * sigma**2 + eps))
        phi = phi / (phi.sum(dim=-1, keepdim=True) + eps)
        
        # 一阶 Gaussian derivative basis
        psi = (d_mesh / (sigma**2 + eps)) * phi
        psi = psi - psi.mean(dim=-1, keepdim=True)
        
        return phi, psi

    def forward(self, x_seq, delta, confidence=None):
        patches = self._extract_patches(x_seq)
        base = self.base_projection(patches)
        
        phi, psi = self._build_gaussian_jet(patches.device, patches.dtype)
        
        q0 = torch.einsum("npl,kl->npk", patches, phi)
        q1 = torch.einsum("npl,kl->npk", patches, psi)
        
        q_jet = q0 + delta * q1
        jet = self.gaussian_up(q_jet)
        
        if confidence is None:
            confidence = 1.0
            
        scale = torch.sigmoid(self.adapter_logit)
        
        self.last_phi = phi.detach()
        self.last_psi = psi.detach()
        self.last_q0 = q0.detach()
        self.last_q1 = q1.detach()
        self.last_jet_norm = torch.norm(jet.detach(), p=2, dim=-1, keepdim=True)
        self.last_base_norm = torch.norm(base.detach(), p=2, dim=-1, keepdim=True)
        self.last_adapter_scale = scale.detach()
        
        return base + scale * confidence * jet
