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
        jet_derivative_mode="exact",
        ablation_mode="none",
    ):
        super().__init__()
        if jet_derivative_mode not in {"exact", "centered_legacy"}:
            raise ValueError(
                "jet_derivative_mode must be either 'exact' or 'centered_legacy', "
                f"got {jet_derivative_mode}"
            )
        self.jet_derivative_mode = jet_derivative_mode
        self.ablation_mode = ablation_mode
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = math.ceil((seq_len - patch_len) / stride) + 1
        padding = patch_len + (self.patch_num - 1) * stride - seq_len
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        # 1. 基础线性层，负责拟合高频尖峰与非周期细节
        self.base_projection = nn.Linear(patch_len, d_model, bias=False)
        
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
        
        # Zero-order epsilon-stabilized normalized Gaussian basis.
        gaussian_denom = 2.0 * sigma.square() + eps

        phi_raw = torch.exp(-d_mesh.square() / gaussian_denom)
        phi = phi_raw / (phi_raw.sum(dim=-1, keepdim=True) + eps)

        if self.jet_derivative_mode == "exact":
            # Exact derivative of the normalized Gaussian basis
            # with respect to its center mu.
            center_score = 2.0 * d_mesh / gaussian_denom
            normalized_score_mean = (phi * center_score).sum(dim=-1, keepdim=True)
            psi = phi * (center_score - normalized_score_mean)
        elif self.jet_derivative_mode == "centered_legacy":
            # Original derivative-inspired implementation,
            # retained only for controlled comparison.
            psi = (d_mesh / (sigma.square() + eps)) * phi
            psi = psi - psi.mean(dim=-1, keepdim=True)
        else:
            raise RuntimeError(
                "Unexpected jet derivative mode: "
                f"{self.jet_derivative_mode}"
            )
        
        return phi, psi

    def forward(self, x_seq, delta, confidence=None):
        patches = self._extract_patches(x_seq)
        base = self.base_projection(patches)
        
        if self.ablation_mode == "wo_jet":
            return base
            
        phi, psi = self._build_gaussian_jet(patches.device, patches.dtype)
        
        q0 = torch.einsum("npl,kl->npk", patches, phi)
        q1 = torch.einsum("npl,kl->npk", patches, psi)
        
        q_jet = q0 + delta * q1
        jet = self.gaussian_up(q_jet)
        
        if confidence is None:
            confidence = 1.0
            
        scale = torch.sigmoid(self.adapter_logit)
        
        return base + scale * confidence * jet
