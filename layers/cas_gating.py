import torch
import torch.nn as nn
import torch.nn.functional as F

class CASGating(nn.Module):
    def __init__(self, d_model, num_gaussians, patch_len=16, stride=None, k_base=-1):
        super().__init__()
        self.tau = 0.5
        self.patch_len = patch_len
        self.stride = stride
        self.num_gaussians = num_gaussians
        self.d_model = d_model
        init_k = k_base if k_base is not None else -1
        self.register_buffer('k_base', torch.tensor(init_k, dtype=torch.long))
        if init_k != -1:
            print(f"[CASGating Init] Pre-resolved k_base locked to: {init_k}")
        self.complexity_router = nn.Sequential(
            nn.Linear(d_model + 1, 32),
            nn.GELU(),
            nn.Linear(32, num_gaussians)
        )

    def forward(self, x_flat):
        batch_channel = x_flat.shape[0]
        
        # ALCR physical prior: smooth low frequency
        x_seq_temp = x_flat.unsqueeze(1) # (batch_channel, 1, L)
        patch_len = self.patch_len if self.patch_len is not None else 16
        x_low = F.avg_pool1d(x_seq_temp, kernel_size=patch_len, stride=1, padding=patch_len // 2)
        if x_low.shape[-1] != x_seq_temp.shape[-1]:
            x_low = x_low[..., :x_seq_temp.shape[-1]]
        
        # Compute and lock k_base once dynamically
        if self.k_base.item() == -1:
            energy_low = torch.sum(x_low ** 2, dim=-1)
            energy_total = torch.sum(x_seq_temp ** 2, dim=-1) + 1e-5
            rho_mean = torch.mean(energy_low / energy_total).item()
            stride_val = self.stride if self.stride is not None else (patch_len // 2)
            k_val = int(round(rho_mean * self.num_gaussians * (stride_val / patch_len)))
            max_k = max(1, int(round(self.num_gaussians * (stride_val / patch_len))))
            k_val = max(1, min(max_k, k_val))
            self.k_base.fill_(k_val)
            print(f"[Dataset-Specific Spectral Anchor] rho_mean: {rho_mean:.4f} -> Locked k_base to: {k_val}")
        
        p_prior = torch.full((batch_channel, self.num_gaussians), 0.01, device=x_flat.device)
        p_prior[:, :self.k_base.item()] = 1.0
        
        # Second-order difference for curvature
        diff2 = x_low[..., 2:] - 2 * x_low[..., 1:-1] + x_low[..., :-2]
        curvature = torch.sum(torch.abs(diff2), dim=-1) # (batch_channel, 1)
        
        # Global average features
        x_mean_feat = x_flat.mean(dim=-1, keepdim=True).expand(-1, self.d_model)
        router_in = torch.cat([x_mean_feat, curvature], dim=-1)
        
        gate_logits = self.complexity_router(router_in)
        p_i = (1.0 - 0.01) * torch.sigmoid(gate_logits) + 0.01
        p_i_eff = torch.max(p_i, p_prior)
        pi_i = torch.cumprod(p_i_eff, dim=1) # (batch_channel, num_gaussians)
        
        # STE gating
        gate_hard = (pi_i > self.tau).float()
        gate_effective = pi_i + (gate_hard - pi_i).detach()
        
        return gate_effective, pi_i
