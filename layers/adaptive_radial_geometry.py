import math
import torch
import torch.nn as nn


class AdaptiveRadialGeometry(nn.Module):
    """Adaptive Radial Geometry (ARG) - Module 1 of ARBD.

    Determines WHERE and AT WHAT TEMPORAL SCALE each radial component operates:
      x_flat -> mu_k(x), sigma_k(x)
    """

    def __init__(
        self,
        input_length: int,
        num_gaussians: int = 8,
        gs_dropout: float = 0.3,
        sigma_min: float = 0.05,
        sigma_max: float = 0.40,
        rho: float = 0.5,
        beta: float = 0.5,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.num_gaussians = int(num_gaussians)
        self.input_length = int(input_length)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.beta = float(beta)
        self.feature_dim = int(feature_dim)

        if self.num_gaussians <= 0:
            raise ValueError("num_gaussians must be positive")

        # 1. Non-symmetric temporal center anchors
        center_anchors = (
            torch.arange(self.num_gaussians, dtype=torch.float32) + 0.5
        ) / float(self.num_gaussians)
        self.register_buffer("center_anchors", center_anchors)

        # 2. Log-spaced multi-scale scale anchors
        if self.num_gaussians == 1:
            scale_anchors = torch.tensor(
                [(self.sigma_min + self.sigma_max) / 2.0],
                dtype=torch.float32,
            )
        else:
            scale_step = torch.arange(
                self.num_gaussians, dtype=torch.float32
            ) / float(self.num_gaussians - 1)
            scale_anchors = self.sigma_min * (
                (self.sigma_max / self.sigma_min) ** scale_step
            )
        self.register_buffer("scale_anchors", scale_anchors)

        # 3. Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_length, self.feature_dim),
            nn.GELU(),
            nn.Dropout(gs_dropout),
        )

        # 4. Decoupled geometry head with zero-initialization (starts from exact anchors)
        self.geometry_head = nn.Linear(self.feature_dim, self.num_gaussians * 2)
        nn.init.zeros_(self.geometry_head.weight)
        nn.init.zeros_(self.geometry_head.bias)

    def get_geometry_params(self):
        """Returns geometry deformation parameters (bound to weight_decay=0.0 in optimizer)."""
        return list(self.geometry_head.parameters())

    @staticmethod
    def evaluate_basis(
        mu: torch.Tensor,
        sigma: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluates continuous Gaussian radial basis matrix B_rad on query positions.

        Args:
            mu: [N, K] radial centers
            sigma: [N, K] radial temporal scales
            query_positions: [P] normalized query timestamps

        Returns:
            radial_basis: [N, P, K]
        """
        queries = query_positions.to(device=mu.device, dtype=mu.dtype)
        distances = queries.view(1, -1, 1) - mu.unsqueeze(1)
        radial_basis = torch.exp(
            -(distances.square())
            / (2.0 * sigma.unsqueeze(1).square() + 1e-8)
        )
        return radial_basis

    def forward(self, x_flat: torch.Tensor, query_positions: torch.Tensor = None):
        """Computes shared features, adaptive radial geometry (mu, sigma), and optional B_rad.

        Args:
            x_flat: [N, L] flattened sequence tensor where N = B * C.
            query_positions: Optional [P] query timestamps for basis evaluation.

        Returns:
            If query_positions is provided:
                (features, mu, sigma, radial_basis)
            Otherwise:
                (features, mu, sigma)
        """
        batch_channel = x_flat.shape[0]
        features = self.feature_extractor(x_flat)

        geom_raw = self.geometry_head(features).view(
            batch_channel,
            self.num_gaussians,
            2,
        )
        delta_mu = geom_raw[:, :, 0]
        delta_s = geom_raw[:, :, 1]

        center_anchors = self.center_anchors.to(
            device=delta_mu.device, dtype=delta_mu.dtype
        )
        scale_anchors = self.scale_anchors.to(
            device=delta_s.device, dtype=delta_s.dtype
        )

        mu = center_anchors.unsqueeze(0) + (
            self.rho / float(self.num_gaussians)
        ) * torch.tanh(delta_mu)

        sigma = scale_anchors.unsqueeze(0) * torch.exp(
            self.beta * torch.tanh(delta_s)
        )

        if query_positions is not None:
            radial_basis = self.evaluate_basis(mu, sigma, query_positions)
            return features, mu, sigma, radial_basis

        return features, mu, sigma
