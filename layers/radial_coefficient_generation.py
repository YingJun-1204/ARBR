import math
import torch
import torch.nn as nn


class RadialCoefficientGeneration(nn.Module):
    """Radial Coefficient Generation (RCG) - Module 2 of ARBD.

    Encodes WHAT latent content is carried by each radial component:
      features h(x) -> V_k(x) in R^D
    """

    def __init__(
        self,
        num_gaussians: int = 8,
        d_model: int = 128,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.num_gaussians = int(num_gaussians)
        self.d_model = int(d_model)
        self.feature_dim = int(feature_dim)

        self.coefficient_head = nn.Linear(
            self.feature_dim, self.num_gaussians * self.d_model
        )

    def compute_coefficients(self, features: torch.Tensor) -> torch.Tensor:
        """Computes radial coefficients V_k.

        Args:
            features: [N, feature_dim]

        Returns:
            radial_coefficients: [N, K, D]
        """
        batch_channel = features.shape[0]
        return self.coefficient_head(features).view(
            batch_channel,
            self.num_gaussians,
            self.d_model,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Generates radial coefficients V from shared features h(x).

        Args:
            features: [N, feature_dim]

        Returns:
            radial_coefficients: [N, K, D]
        """
        return self.compute_coefficients(features)


class RadialFeatureRenderer(nn.Module):
    """Parameter-free Bilinear Pairing / Renderer of ARBD.

    Renders continuous temporal representation by combining query-aligned radial basis
    with latent content coefficients:
      G = B_rad @ V
    where B_rad in R^{N x P x K} and V in R^{N x K x D} -> G in R^{N x P x D}.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        radial_basis: torch.Tensor,
        radial_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """Renders the radial representation via batch matrix multiplication.

        Args:
            radial_basis: [N, P, K] evaluated radial basis responses
            radial_coefficients: [N, K, D] latent content coefficients

        Returns:
            radial_repr: [N, P, D] rendered radial representation G
        """
        # [N, P, K] @ [N, K, D] -> [N, P, D]
        return torch.bmm(radial_basis, radial_coefficients)
