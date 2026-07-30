import torch
import torch.nn as nn
import torch.nn.functional as F

def inverse_softplus(x):
    return torch.where(x > 20.0, x, torch.log(torch.exp(x) - 1.0))

class TemporalGaussianSplatting(nn.Module):
    def __init__(
        self,
        input_length,
        d_model,
        num_gaussians,
        gs_dropout=0.3,
        density_mode="none",
        use_occlusion=False,
        jet_sigma_init=0.20,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.d_model = d_model
        self.d_latent = d_model
        self.density_mode = density_mode
        self.use_occlusion = use_occlusion

        num_params_per_gaussian = 1 + self.d_latent

        self.generator = nn.Sequential(
            nn.Linear(input_length, 128),
            nn.GELU(),
            nn.Dropout(gs_dropout),
            nn.Linear(128, num_gaussians * num_params_per_gaussian),
        )

        init_sigma_val = max(1e-4, float(jet_sigma_init))
        raw_sigma_tensor = inverse_softplus(torch.full((num_gaussians,), init_sigma_val - 1e-5))
        self.raw_sigma = nn.Parameter(raw_sigma_tensor)

    def forward(self, x_flat, query_positions, gate_effective=None):
        batch_channel = x_flat.shape[0]
        raw_params = self.generator(x_flat)

        raw_params = raw_params.view(batch_channel, self.num_gaussians, 1 + self.d_latent)
        mu = torch.sigmoid(raw_params[:, :, 0])
        c = raw_params[:, :, 1:]

        sigma_shared = F.softplus(self.raw_sigma) + 1e-5
        sigma = sigma_shared.unsqueeze(0).expand(batch_channel, -1)
        
        if self.density_mode == "cas" and gate_effective is not None:
            alpha_effective = gate_effective
        else:
            alpha_effective = torch.ones_like(mu)

        if query_positions.ndim != 1:
            raise ValueError(
                "query_positions must be a 1-D tensor, "
                f"got shape {tuple(query_positions.shape)}"
            )

        t_queries = query_positions.to(
            device=mu.device,
            dtype=mu.dtype,
        )

        patch_num = t_queries.numel()
        t_expanded = t_queries.view(1, patch_num, 1)
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
                    torch.ones(batch_channel, patch_num, 1, device=t_queries.device),
                    one_minus_weights[:, :, :-1],
                ],
                dim=2,
            )
            transmittance = torch.cumprod(shifted_transmittance, dim=2)
            rendered_event = torch.sum(sorted_c * (weights * transmittance).unsqueeze(-1), dim=2)
        else:
            weights = alpha_effective.unsqueeze(1) * torch.exp(-(d**2) / (2 * sigma.unsqueeze(1)**2))
            rendered_event = torch.sum(c.unsqueeze(1) * weights.unsqueeze(-1), dim=2)

        return rendered_event, mu, sigma, alpha_effective
