import math
import torch
import torch.nn as nn

from layers.adaptive_radial_geometry import AdaptiveRadialGeometry
from layers.radial_coefficient_generation import (
    RadialCoefficientGeneration,
    RadialFeatureRenderer,
)
from layers.local_residual_correction import CalibratedLocalResidualCorrection


class ARBDEncoder(nn.Module):
    """Adaptive Radial Basis Decomposition (ARBD) Encoder.

    Unifies:
    1. AdaptiveRadialGeometry (ARG): (x, q) -> (mu(x), sigma(x)) -> B_rad
    2. RadialCoefficientGeneration (RCG): features -> V(x)
    3. RadialFeatureRenderer (Renderer): (B_rad, V) -> G = B_rad @ V
    4. CalibratedLocalResidualCorrection (CLRC): (G, x) -> Z = G + lambda * L
    """

    @staticmethod
    def _build_patch_centers(seq_len, patch_len, stride, patch_num):
        starts = torch.arange(patch_num, dtype=torch.float32).unsqueeze(1)
        starts = starts * float(stride)
        offsets = torch.arange(patch_len, dtype=torch.float32).unsqueeze(0)
        sampled = (starts + offsets).clamp(max=float(seq_len - 1))
        return sampled.mean(dim=1) / float(seq_len - 1)

    @staticmethod
    def _remap_legacy_state_dict(state_dict, prefix, *args):
        legacy_prefix = prefix + "radial_renderer."
        new_prefix = prefix + "rcg."
        for key in list(state_dict.keys()):
            if key.startswith(legacy_prefix):
                new_key = new_prefix + key[len(legacy_prefix):]
                state_dict[new_key] = state_dict.pop(key)

    def __init__(
        self,
        seq_len,
        d_model=256,
        num_gaussians=8,
        gs_dropout=0.3,
        patch_len=None,
        stride=None,
        local_weight_max=0.75,
        static_local_weight_init=0.5,
        rho=0.5,
        beta=0.5,
        **kwargs,
    ):
        super().__init__()
        self._register_load_state_dict_pre_hook(self._remap_legacy_state_dict)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.num_gaussians = int(num_gaussians)
        self.d_model = int(d_model)
        self.rho = float(rho)
        self.beta = float(beta)
        self.feature_eps = 1e-6

        if self.rho <= 0:
            raise ValueError("rho must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")

        if self.seq_len <= 1:
            raise ValueError("seq_len must be greater than 1")
        if self.patch_len <= 1:
            raise ValueError("patch_len must be greater than 1")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.num_gaussians <= 0:
            raise ValueError("num_gaussians must be positive")

        self.patch_num = math.ceil(
            (self.seq_len - self.patch_len) / self.stride
        ) + 1
        if self.patch_num <= 0:
            raise ValueError("invalid patch configuration")

        # ----------------------------------------------------------------------
        # Strict PRNG sequence matching validated baseline:
        # Draw 1: local_projection (inside CalibratedLocalResidualCorrection)
        # Draw 2: compatibility_draw
        # Draw 3: feature_extractor (inside AdaptiveRadialGeometry)
        # Draw 4: geometry_head (inside AdaptiveRadialGeometry)
        # Draw 5: coefficient_head (inside RadialCoefficientGeneration)
        # ----------------------------------------------------------------------

        # 1. Module 3: Calibrated Local Residual Correction (CLRC)
        self.local_correction = CalibratedLocalResidualCorrection(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            weight_init=static_local_weight_init,
            weight_max=local_weight_max,
            has_residual_gate=True,
        )
        if self.local_correction.patch_num != self.patch_num:
            raise RuntimeError("Gaussian and Linear patch grids do not match")

        # 2. Historical compatibility draw to preserve exact RNG alignment
        initialization_compatibility_draw = torch.empty(d_model, 4)
        nn.init.kaiming_uniform_(
            initialization_compatibility_draw,
            a=math.sqrt(5),
        )

        # 3. Module 1: Adaptive Radial Geometry (ARG)
        self.radial_geometry = AdaptiveRadialGeometry(
            input_length=self.seq_len,
            num_gaussians=self.num_gaussians,
            gs_dropout=gs_dropout,
            rho=self.rho,
            beta=self.beta,
        )

        # 4. Module 2: Radial Coefficient Generation (RCG)
        self.rcg = RadialCoefficientGeneration(
            num_gaussians=self.num_gaussians,
            d_model=self.d_model,
            feature_dim=128,
        )

        # 5. Parameter-free Bilinear Renderer: G = B_rad @ V
        self.renderer = RadialFeatureRenderer()

        self.register_buffer(
            "patch_centers",
            self._build_patch_centers(
                self.seq_len,
                self.patch_len,
                self.stride,
                self.patch_num,
            ),
            persistent=False,
        )

        # Diagnostics preservation
        self.last_mu = None
        self.last_sigma = None
        self.last_basis = None
        self.last_radial_coefficients = None
        self.last_radial_repr = None
        self.last_local_repr = None
        self.last_residual_weight = None
        self.last_gaussian_norm = None
        self.last_linear_norm = None
        self.last_residual_ratio = None

    def get_geometry_params(self):
        """Returns geometry deformation parameters for optimizer (weight_decay=0.0)."""
        if self.radial_geometry is not None:
            return self.radial_geometry.get_geometry_params()
        return []

    def get_non_geometry_params(self):
        geom_param_ids = {id(p) for p in self.get_geometry_params()}
        return [p for p in self.parameters() if id(p) not in geom_param_ids]

    def forward(self, x_seq, patch_num, is_flat=False):
        if patch_num != self.patch_num:
            raise ValueError(
                f"patch count mismatch: {patch_num} != {self.patch_num}"
            )

        x_flat = (
            x_seq.flatten(0, 1)
            if not is_flat
            else x_seq.flatten(start_dim=1)
        )

        # 1. ARG (Module 1): (x, q) -> features, mu, sigma, B_rad
        features, mu, sigma, radial_basis = self.radial_geometry(
            x_flat, query_positions=self.patch_centers
        )

        # 2. RCG (Module 2): features -> V
        radial_coefficients = self.rcg(features)

        # 3. Renderer (Bilinear Coupling): (B_rad, V) -> G = B_rad @ V
        radial_repr = self.renderer(radial_basis, radial_coefficients)

        # Diagnostics preservation
        self.last_mu = mu.detach()
        self.last_sigma = sigma.detach()
        self.last_basis = radial_basis.detach()
        self.last_radial_coefficients = radial_coefficients.detach()
        self.last_radial_repr = radial_repr.detach()

        # 4. CLRC (Module 3): G, x -> Z = G + lambda * L
        fused_repr, local_repr, local_weight = self.local_correction(
            x_seq=x_seq,
            radial_repr=radial_repr,
        )

        self.last_local_repr = local_repr.detach()
        self.last_residual_weight = local_weight.detach()

        radial_norm = torch.linalg.vector_norm(
            radial_repr.float(), dim=-1, keepdim=True
        )
        residual_norm = torch.linalg.vector_norm(
            (local_weight * local_repr).float(), dim=-1, keepdim=True
        )
        self.last_gaussian_norm = radial_norm.detach()
        self.last_linear_norm = torch.linalg.vector_norm(
            local_repr.float(), dim=-1, keepdim=True
        ).detach()
        self.last_residual_ratio = (
            residual_norm / (radial_norm + residual_norm + self.feature_eps)
        ).detach()

        return fused_repr
