import torch
import torch.nn as nn
import math
from layers.cas_gating import CASGating
from layers.gaussian_splatting import TemporalGaussianSplatting
from layers.patch_residual import GaussianJetProjection

class SplattingResidualEncoder(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model=256,
        num_gaussians=8,
        gs_dropout=0.3,
        in_features=None,
        density_mode="none",
        use_occlusion=False,
        patch_len=None,
        stride=None,
        use_residual=False,
        gs_residual_weight=0.1,
        k_base=-1,
        num_implicit_gaussians=4,
        jet_max_shift_samples=1.0,
        jet_score_temperature=1.0,
        jet_density_tau=1.0,
        jet_detach_geometry=True,
        jet_scale_init=0.1,
        jet_sigma_init=0.2,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.d_model = d_model
        self.density_mode = density_mode
        self.use_occlusion = use_occlusion
        self.use_residual = use_residual
        self.gs_residual_weight = gs_residual_weight
        self.patch_len = patch_len
        self.stride = stride
        
        self.num_implicit_gaussians = num_implicit_gaussians
        self.jet_max_shift_samples = jet_max_shift_samples
        self.jet_score_temperature = jet_score_temperature
        self.jet_density_tau = jet_density_tau
        self.jet_detach_geometry = jet_detach_geometry
        self.jet_scale_init = jet_scale_init
        self.jet_sigma_init = jet_sigma_init

        # 1. CAS Gating
        if self.density_mode == "cas":
            self.cas_gating = CASGating(
                d_model=d_model,
                num_gaussians=num_gaussians,
                patch_len=patch_len,
                stride=stride,
                k_base=k_base,
            )
        else:
            self.cas_gating = None

        # 2. Patch Residual
        if self.use_residual:
            self.patch_residual = GaussianJetProjection(
                seq_len=seq_len,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                num_implicit_gaussians=num_implicit_gaussians,
                jet_scale_init=jet_scale_init,
                sigma_init=jet_sigma_init,
            )
            self.patch_num = self.patch_residual.patch_num
        else:
            self.patch_residual = None
            if stride is not None and patch_len is not None:
                self.patch_num = math.ceil((seq_len - patch_len) / stride) + 1
            else:
                self.patch_num = None

        # 3. Gaussian Splatting
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
        
        # CAS specific states
        self.last_gate_probs = None
        self.last_active_count = None
        self.last_p_i = None
        self.last_pi_i = None
        self.last_sigma = None
        self.last_gate_hard = None

        # Jet specific diagnostic states
        self.last_score_shift = None
        self.last_score_confidence = None
        self.last_score_weights = None

    @property
    def generator(self):
        return self.gaussian_splatting.generator

    @property
    def complexity_router(self):
        if self.cas_gating is not None:
            return self.cas_gating.complexity_router
        return None

    @property
    def k_base(self):
        if self.cas_gating is not None:
            return self.cas_gating.k_base
        raise AttributeError("k_base is only available in CAS density mode")

    def _compute_gaussian_score_shift(
        self,
        mu,
        sigma,
        alpha_effective,
        patch_num,
        device,
    ):
        t = torch.linspace(
            0,
            1,
            patch_num,
            device=device,
            dtype=mu.dtype,
        )
        
        distance = (
            t.view(1, patch_num, 1)
            - mu.unsqueeze(1)
        )
        
        eps = 1e-5
        score_weights = (
            alpha_effective.unsqueeze(1)
            * torch.exp(
                -(distance ** 2)
                / (2 * sigma.unsqueeze(1) ** 2 + eps)
            )
        )
        
        precision = 1.0 / (
            sigma.unsqueeze(1) ** 2 + eps
        )
        
        weighted_precision = score_weights * precision
        
        mean_shift = (
            weighted_precision
            * (
                mu.unsqueeze(1)
                - t.view(1, patch_num, 1)
            )
        ).sum(dim=-1, keepdim=True) / (
            weighted_precision.sum(
                dim=-1,
                keepdim=True,
            )
            + eps
        )
        
        max_shift = (
            self.jet_max_shift_samples
            / max(self.patch_len - 1, 1)
        )
        
        delta = max_shift * torch.tanh(
            self.jet_score_temperature
            * mean_shift
            / (max_shift + eps)
        )
        
        density = score_weights.sum(
            dim=-1,
            keepdim=True,
        )
        
        confidence = density / (
            density + self.jet_density_tau
        )
        
        return delta, confidence, score_weights

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
        rendered_event, mu, sigma, alpha_effective = self.gaussian_splatting(
            x_flat, patch_num, x_seq.device, gate_effective
        )
        
        if self.density_mode == "cas":
            self.last_active_count = self.last_gate_hard.sum(dim=-1).mean()
            self.last_sigma = sigma

        # Forward Residual
        if self.use_residual and self.gs_residual_weight != 0.0:
            delta, confidence, score_weights = self._compute_gaussian_score_shift(
                mu=mu,
                sigma=sigma,
                alpha_effective=alpha_effective,
                patch_num=patch_num,
                device=x_seq.device,
            )
            
            if self.jet_detach_geometry:
                delta = delta.detach()
                confidence = confidence.detach()
                score_weights = score_weights.detach()
            
            res_proj = self.patch_residual(
                x_seq=x_seq,
                delta=delta,
                confidence=confidence,
            )
            
            rendered_event = rendered_event + self.gs_residual_weight * res_proj
            
            # Compatibility diagnostic attributes
            eps = 1e-5
            max_shift = self.jet_max_shift_samples / max(self.patch_len - 1, 1)
            self.last_gate_direction = (delta / (max_shift + eps)).detach()
            self.last_gate_strength = confidence.detach()
            self.last_gate_scale = torch.ones_like(confidence)
            
            self.last_score_shift = delta.detach()
            self.last_score_confidence = confidence.detach()
            self.last_score_weights = score_weights.detach()
            
            neutral = 1.0 - confidence
            forward_dir = confidence * torch.relu(self.last_gate_direction)
            reverse_dir = confidence * torch.relu(-self.last_gate_direction)
            normalizer = neutral + forward_dir + reverse_dir + eps
            
            self.last_gate_route_probs = torch.cat(
                [
                    neutral / normalizer,
                    forward_dir / normalizer,
                    reverse_dir / normalizer,
                ],
                dim=-1,
            )
        else:
            self.last_gate_direction = torch.zeros(batch_channel, patch_num, 1, device=x_seq.device)
            self.last_gate_strength = torch.zeros(batch_channel, patch_num, 1, device=x_seq.device)
            self.last_gate_scale = torch.ones(batch_channel, patch_num, 1, device=x_seq.device)
            
            probs = torch.zeros(batch_channel, patch_num, 3, device=x_seq.device)
            probs[..., 0] = 1.0
            self.last_gate_route_probs = probs

        return rendered_event
