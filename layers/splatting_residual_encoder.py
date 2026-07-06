import torch
import torch.nn as nn
import math
from layers.cas_gating import CASGating
from layers.gaussian_splatting import TemporalGaussianSplatting
from layers.patch_residual import PatchResidualProjection
from layers.residual_router import AdaptiveResidualRouter

class SplattingResidualEncoder(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model=256,
        d_latent=None,
        num_gaussians=8,
        gs_dropout=0.3,
        in_features=None,
        density_mode="none",
        use_occlusion=False,
        patch_len=None,
        stride=None,
        use_residual=False,
        gs_residual_weight=0.1,
        gate_type=None,
        gate_beta=0.25,
        gate_window_half=2,
        gamma_complement=0.6,
        k_base=-1,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.d_model = d_model
        self.d_latent = d_latent if d_latent is not None else d_model
        self.density_mode = density_mode
        self.use_occlusion = use_occlusion
        self.use_residual = use_residual
        self.gs_residual_weight = gs_residual_weight
        self.gate_type = gate_type
        self.gate_beta = gate_beta
        self.patch_len = patch_len
        self.stride = stride
        self.gate_window_half = gate_window_half
        self.gamma_complement = gamma_complement

        # 1. CAS Gating
        if self.density_mode == "cas":
            self.cas_gating = CASGating(
                d_model=d_model,
                num_gaussians=num_gaussians,
                patch_len=patch_len,
                stride=stride,
                gamma_complement=gamma_complement,
                k_base=k_base,
            )
        else:
            self.cas_gating = None

        # 2. Residual Gating/Router
        if self.use_residual:
            self.residual_router = AdaptiveResidualRouter(
                d_model=d_model,
                gate_type=gate_type,
                gate_beta=gate_beta,
                gate_window_half=gate_window_half
            )
        else:
            self.residual_router = None

        # 3. Patch Residual
        if self.use_residual:
            self.patch_residual = PatchResidualProjection(
                seq_len=seq_len,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model
            )
            self.patch_num = self.patch_residual.patch_num
        else:
            self.patch_residual = None
            if stride is not None and patch_len is not None:
                self.patch_num = math.ceil((seq_len - patch_len) / stride) + 1
            else:
                self.patch_num = None

        # 4. Gaussian Splatting
        self.gaussian_splatting = TemporalGaussianSplatting(
            seq_len=seq_len,
            d_model=d_model,
            num_gaussians=num_gaussians,
            gs_dropout=gs_dropout,
            in_features=in_features,
            density_mode=density_mode,
            use_occlusion=use_occlusion
        )

        # States required by outer tests & modules
        self.last_gate_direction = None
        self.last_gate_strength = None
        self.last_gate_route_probs = None
        self.last_gate_scale = None
        self.last_density_score = None
        self.last_raw_lga = None
        self.last_z_lga = None
        self.last_energy = None
        
        # CAS specific states
        self.last_gate_probs = None
        self.last_active_count = None
        self.last_p_i = None
        self.last_pi_i = None
        self.last_sigma = None
        self.last_gate_hard = None

    @property
    def generator(self):
        # For compatibility with tests that access generator
        return self.gaussian_splatting.generator

    @property
    def route_router(self):
        # For compatibility with tests checking route_router properties
        if self.residual_router is not None and hasattr(self.residual_router, 'route_router'):
            return self.residual_router.route_router
        return None

    @property
    def complexity_router(self):
        # For compatibility with tests checking complexity_router properties
        if self.cas_gating is not None:
            return self.cas_gating.complexity_router
        return None

    @property
    def k_base(self):
        if self.cas_gating is not None:
            return self.cas_gating.k_base
        raise AttributeError("k_base is only available in CAS density mode")

    def forward(self, x_seq, patch_num, is_flat=False):
        if not is_flat:
            B, C, L = x_seq.shape
            x_flat = x_seq.reshape(B * C, L)
            batch_channel = B * C
        else:
            batch_channel = x_seq.shape[0]
            x_flat = x_seq.flatten(start_dim=1)

        # Forward CAS Gating
        gate_effective = None
        if self.density_mode == "cas" and self.cas_gating is not None:
            gate_effective, pi_i, gate_hard, p_i = self.cas_gating(x_flat, x_seq)
            self.last_gate_probs = pi_i
            self.last_p_i = p_i
            self.last_pi_i = pi_i
            self.last_gate_hard = gate_hard

        # Forward Gaussian Splatting
        rendered_event, weights, mu, sigma, alpha_effective, last_density_score = self.gaussian_splatting(
            x_flat, patch_num, x_seq.device, gate_effective
        )
        
        self.last_density_score = last_density_score
        if self.density_mode == "cas":
            self.last_active_count = self.last_gate_hard.sum(dim=-1).mean()
            self.last_sigma = sigma

        # Forward Residual
        if self.use_residual and self.gs_residual_weight != 0.0:
            res_proj = self.patch_residual(x_seq, batch_channel)
            
            gate_scale, gate_direction, gate_strength, gate_route_probs = self.residual_router(
                weights, rendered_event, res_proj, batch_channel, patch_num, x_seq.device
            )
            
            self.last_gate_direction = gate_direction
            self.last_gate_strength = gate_strength
            self.last_gate_route_probs = gate_route_probs
            
            # Copy diagnostic attributes if available
            self.last_raw_lga = getattr(self.residual_router, "last_raw_lga", None)
            self.last_z_lga = getattr(self.residual_router, "last_z_lga", None)
            self.last_energy = getattr(self.residual_router, "last_energy", None)
            
            if self.gate_type not in ["forward", "reverse", "adaptive_direction"] and self.density_mode == "cas":
                active_ratio = gate_effective.sum(dim=-1, keepdim=True) / self.num_gaussians
                scale = 1.5 - self.gamma_complement * active_ratio
                self.last_gate_scale = scale.unsqueeze(1).expand(-1, patch_num, 1)
            else:
                self.last_gate_scale = gate_scale
            
            rendered_event = rendered_event + self.gs_residual_weight * self.last_gate_scale * res_proj
        else:
            self.last_gate_direction = torch.zeros(batch_channel, patch_num, 1, device=x_seq.device)
            self.last_gate_strength = torch.zeros(batch_channel, patch_num, 1, device=x_seq.device)
            self.last_gate_scale = torch.ones(batch_channel, patch_num, 1, device=x_seq.device)
            
            probs = torch.zeros(batch_channel, patch_num, 3, device=x_seq.device)
            probs[..., 0] = 1.0
            self.last_gate_route_probs = probs
            
            self.last_raw_lga = None
            self.last_z_lga = None
            self.last_energy = None

        return rendered_event
