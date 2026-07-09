import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalGaussianSplatting(nn.Module):
    def __init__(self, seq_len, d_model, num_gaussians, gs_dropout=0.3, in_features=None, density_mode="none", use_occlusion=False):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.d_model = d_model
        self.d_latent = d_model
        self.density_mode = density_mode
        self.use_occlusion = use_occlusion

        generator_in = in_features if in_features is not None else seq_len
        num_params_per_gaussian = 3 + self.d_latent

        self.generator = nn.Sequential(
            nn.Linear(generator_in, 128),
            nn.GELU(),
            nn.Dropout(gs_dropout),
            nn.Linear(128, num_gaussians * num_params_per_gaussian),
        )

    def forward(self, x_flat, patch_num, x_seq_device, gate_effective=None):
        batch_channel = x_flat.shape[0]
        raw_params = self.generator(x_flat)

        raw_params = raw_params.view(batch_channel, self.num_gaussians, 3 + self.d_latent)
        mu = torch.sigmoid(raw_params[:, :, 0])
        sigma = F.softplus(raw_params[:, :, 1]) + 1e-5
        alpha = torch.sigmoid(raw_params[:, :, 2])
        c = raw_params[:, :, 3:]
        last_density_score = None
        
        if self.density_mode == "cas" and gate_effective is not None:
            alpha_effective = alpha * gate_effective
        else:
            alpha_effective = alpha

        t_queries = torch.linspace(0, 1, patch_num, device=x_seq_device)
        t_expanded = t_queries.unsqueeze(0).unsqueeze(2)
        d = t_expanded - mu.unsqueeze(1)

        if self.use_occlusion:
            d_for_sort = torch.where(d >= 0, d, torch.full_like(d, float("inf")))
            sorted_idx = torch.argsort(d_for_sort, dim=2, descending=False)

            sorted_d = d.gather(2, sorted_idx)
            sorted_sigma = sigma.unsqueeze(1).expand(-1, patch_num, -1).gather(2, sorted_idx)
            sorted_alpha = alpha_effective.unsqueeze(1).expand(-1, patch_num, -1).gather(2, sorted_idx)

            c_expanded = c.unsqueeze(1).expand(-1, patch_num, -1, -1)
            sorted_idx_c = sorted_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_model)
            sorted_c = c_expanded.gather(2, sorted_idx_c)

            valid_mask = sorted_d >= 0
            weights = sorted_alpha * torch.exp(-(sorted_d**2) / (2 * sorted_sigma**2))
            weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

            one_minus_weights = 1.0 - weights
            shifted_transmittance = torch.cat(
                [
                    torch.ones(batch_channel, patch_num, 1, device=x_seq_device),
                    one_minus_weights[:, :, :-1],
                ],
                dim=2,
            )
            transmittance = torch.cumprod(shifted_transmittance, dim=2)
            rendered_event = torch.sum(sorted_c * (weights * transmittance).unsqueeze(-1), dim=2)
        else:
            weights = alpha_effective.unsqueeze(1) * torch.exp(-(d**2) / (2 * sigma.unsqueeze(1)**2))
            rendered_event = torch.sum(c.unsqueeze(1) * weights.unsqueeze(-1), dim=2)

        return rendered_event, weights, mu, sigma, alpha_effective, last_density_score
