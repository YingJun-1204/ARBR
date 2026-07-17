import torch
import torch.nn as nn
import math
from layers.cas_gating import CASGating
from layers.gaussian_splatting import TemporalGaussianSplatting
from layers.patch_residual import GaussianJetProjection
from layers.geometry_fusion import GeometryConditionedResidualGate

class SplattingResidualEncoder(nn.Module):
    @staticmethod
    def _build_patch_centers(
        seq_len: int,
        patch_len: int,
        stride: int,
        patch_num: int,
    ) -> torch.Tensor:
        """
        Build normalized temporal centroids of the actual patches produced
        by ReplicationPad1d followed by unfold.

        Each center is expressed in the global input coordinate [0, 1].
        Right-padded samples are mapped to the final valid input index.
        """
        starts = (
            torch.arange(patch_num, dtype=torch.float32)
            .unsqueeze(1)
            * float(stride)
        )

        offsets = torch.arange(
            patch_len,
            dtype=torch.float32,
        ).unsqueeze(0)

        sampled_indices = starts + offsets

        # Match right-side replication padding:
        # indices beyond seq_len - 1 correspond to the final observation.
        effective_indices = sampled_indices.clamp(
            max=float(seq_len - 1)
        )

        centers = effective_indices.mean(dim=1)
        centers = centers / float(seq_len - 1)

        return centers

    def __init__(
        self,
        seq_len,
        d_model=256,
        num_gaussians=8,
        gs_dropout=0.3,
        density_mode="none",
        use_occlusion=False,
        patch_len=None,
        stride=None,
        use_residual=False,
        gs_residual_weight=0.1,
        k_base=-1,
        num_implicit_gaussians=4,
        jet_max_shift_samples=1.0,
        jet_score_temperature=0.01,
        jet_density_tau=1.0,
        jet_detach_geometry=True,
        jet_scale_init=0.1,
        jet_sigma_init=0.2,
        fusion_mode="fixed",
        fusion_hidden_dim=16,
        fusion_init=0.1,
        fusion_beta_max=0.5,
        fusion_detach_geometry=True,
        ablation_mode="none",
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)

        self.patch_num = (
            math.ceil(
                (self.seq_len - self.patch_len) / self.stride
            )
            + 1
        )

        if self.seq_len <= 1:
            raise ValueError(
                f"seq_len must be greater than 1, got {self.seq_len}"
            )

        if self.patch_len <= 1:
            raise ValueError(
                f"patch_len must be greater than 1, got {self.patch_len}"
            )

        if self.stride <= 0:
            raise ValueError(
                f"stride must be positive, got {self.stride}"
            )

        if self.patch_num <= 0:
            raise ValueError(
                "Invalid patch configuration: "
                f"seq_len={self.seq_len}, "
                f"patch_len={self.patch_len}, "
                f"stride={self.stride}"
            )

        if fusion_mode not in {"fixed", "geometry"}:
            raise ValueError(
                "fusion_mode must be either 'fixed' or 'geometry', "
                f"got {fusion_mode}"
            )

        if fusion_beta_max <= 0.0:
            raise ValueError(
                "fusion_beta_max must be positive, "
                f"got {fusion_beta_max}"
            )

        if fusion_mode == "geometry":
            if not 0.0 < fusion_init < fusion_beta_max:
                raise ValueError(
                    "For geometry fusion, fusion_init must satisfy "
                    "0 < fusion_init < fusion_beta_max, "
                    f"got fusion_init={fusion_init}, "
                    f"fusion_beta_max={fusion_beta_max}"
                )

        self.num_gaussians = num_gaussians
        self.density_mode = density_mode
        self.ablation_mode = ablation_mode
        
        self.fusion_mode = fusion_mode
        self.fusion_init = float(fusion_init)
        self.fusion_beta_max = float(fusion_beta_max)
        self.fusion_detach_geometry = bool(fusion_detach_geometry)

        if fusion_mode == "fixed":
            residual_capacity = float(gs_residual_weight)
        else:
            residual_capacity = float(fusion_beta_max)

        self.use_residual = bool(use_residual and residual_capacity > 0.0)
        self.gs_residual_weight = float(gs_residual_weight)

        self.jet_max_shift_samples = jet_max_shift_samples
        self.jet_score_temperature = jet_score_temperature
        self.jet_density_tau = jet_density_tau
        self.jet_detach_geometry = jet_detach_geometry

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
                ablation_mode=ablation_mode,
            )
        else:
            self.patch_residual = None

        # 2.5 Geometry Fusion Gate
        if self.use_residual and self.fusion_mode == "geometry":
            self.fusion_gate = GeometryConditionedResidualGate(
                hidden_dim=fusion_hidden_dim,
                beta_init=fusion_init,
                beta_max=fusion_beta_max,
            )
        else:
            self.fusion_gate = None

        self.last_fusion_gate = None
        self.last_fusion_features = None
        self.last_delta = None
        self.last_confidence = None

        self._last_fusion_gate_mean = None
        self._last_fusion_gate_std = None
        self._last_fusion_gate_min = None
        self._last_fusion_gate_max = None
        self._last_fusion_low_ratio = None
        self._last_fusion_high_ratio = None
        self._last_geometry_uncertainty_mean = None
        self._last_geometry_shift_ratio_mean = None
        self.collect_diagnostics = False

        # 3. Gaussian Splatting
        self.gaussian_splatting = TemporalGaussianSplatting(
            input_length=seq_len,
            d_model=d_model,
            num_gaussians=num_gaussians,
            gs_dropout=gs_dropout,
            density_mode=density_mode,
            use_occlusion=use_occlusion
        )

        # CAS specific states
        self.last_gate_probs = None

        patch_centers = self._build_patch_centers(
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            patch_num=self.patch_num,
        )

        self.register_buffer(
            "patch_centers",
            patch_centers,
            persistent=False,
        )

        if self.use_residual:
            if self.patch_residual.patch_num != self.patch_num:
                raise RuntimeError(
                    "Patch-grid mismatch between "
                    "SplattingResidualEncoder and GaussianJetProjection: "
                    f"{self.patch_num} vs {self.patch_residual.patch_num}"
                )

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
        query_positions,
    ):
        """
        Compute field-derived displacement in the local normalized
        coordinate of each patch.

        mu, sigma and query_positions use the global input coordinate.
        The returned delta uses the local patch coordinate expected by
        GaussianJetProjection.
        """
        t = query_positions.to(
            device=mu.device,
            dtype=mu.dtype,
        )

        if t.ndim != 1:
            raise ValueError(
                "query_positions must be one-dimensional, "
                f"got shape {tuple(t.shape)}"
            )

        patch_num = t.numel()
        eps = 1e-5

        t_expanded = t.view(1, patch_num, 1)

        distance_global = (
            t_expanded - mu.unsqueeze(1)
        )

        score_weights = (
            alpha_effective.unsqueeze(1)
            * torch.exp(
                -(distance_global ** 2)
                / (
                    2 * sigma.unsqueeze(1) ** 2
                    + eps
                )
            )
        )

        precision_global = 1.0 / (
            sigma.unsqueeze(1) ** 2 + eps
        )

        weighted_precision = (
            score_weights * precision_global
        )

        mean_shift_global = (
            weighted_precision
            * (
                mu.unsqueeze(1)
                - t_expanded
            )
        ).sum(
            dim=-1,
            keepdim=True,
        ) / (
            weighted_precision.sum(
                dim=-1,
                keepdim=True,
            )
            + eps
        )

        global_to_local = (
            float(self.seq_len - 1)
            / float(self.patch_len - 1)
        )

        mean_shift_local = (
            mean_shift_global
            * global_to_local
        )

        max_shift_local = (
            self.jet_max_shift_samples
            / float(self.patch_len - 1)
        )

        delta_local = (
            max_shift_local
            * torch.tanh(
                self.jet_score_temperature
                * mean_shift_local
                / (max_shift_local + eps)
            )
        )

        density = score_weights.sum(
            dim=-1,
            keepdim=True,
        )

        confidence = density / (
            density + self.jet_density_tau
        )

        return delta_local, confidence

    def _build_constant_fusion_weight(
        self,
        reference: torch.Tensor,
        value: float,
    ) -> torch.Tensor:
        """
        Build a broadcastable fusion weight of shape (N, P, 1).
        """
        return reference.new_full(
            (
                reference.shape[0],
                reference.shape[1],
                1,
            ),
            float(value),
        )

    def _compute_fusion_weight(
        self,
        rendered_event,
        delta,
        confidence,
    ):
        """
        Compute the residual injection weight.

        Geometry fusion is intentionally disabled for the wo_guidance
        ablation, which falls back to a constant initial weight.
        """
        if self.fusion_mode == "fixed":
            beta = self._build_constant_fusion_weight(
                reference=rendered_event,
                value=self.gs_residual_weight,
            )
            return beta, None

        if self.fusion_gate is None:
            raise RuntimeError(
                "fusion_gate is not initialized while "
                "fusion_mode='geometry'"
            )

        # wo_guidance must remove both Jet guidance and field-aware
        # fusion guidance.
        if self.ablation_mode == "wo_guidance":
            beta = self._build_constant_fusion_weight(
                reference=rendered_event,
                value=self.fusion_init,
            )
            return beta, None

        if not torch.is_tensor(delta):
            raise TypeError(
                "delta must be a tensor for geometry fusion"
            )

        if confidence is None:
            raise ValueError(
                "confidence must not be None for geometry fusion"
            )

        delta_for_fusion = delta
        confidence_for_fusion = confidence

        if self.fusion_detach_geometry:
            delta_for_fusion = (
                delta_for_fusion.detach()
            )
            confidence_for_fusion = (
                confidence_for_fusion.detach()
            )

        max_shift_local = (
            self.jet_max_shift_samples
            / float(self.patch_len - 1)
        )

        beta, geometry_features = self.fusion_gate(
            confidence=confidence_for_fusion,
            delta=delta_for_fusion,
            max_shift_local=max_shift_local,
        )

        return beta, geometry_features

    def forward(self, x_seq, patch_num, is_flat=False):
        if patch_num != self.patch_num:
            raise ValueError(
                "Patch count mismatch: "
                f"external patch_num={patch_num}, "
                f"encoder patch_num={self.patch_num}, "
                f"seq_len={self.seq_len}, "
                f"patch_len={self.patch_len}, "
                f"stride={self.stride}"
            )

        if not is_flat:
            x_flat = x_seq.flatten(0, 1)
        else:
            x_flat = x_seq.flatten(start_dim=1)

        # Forward CAS Gating
        gate_effective = None
        if self.density_mode == "cas" and self.cas_gating is not None:
            gate_effective, pi_i = self.cas_gating(x_flat)
            self.last_gate_probs = pi_i

        query_positions = self.patch_centers

        # Forward Gaussian Splatting
        rendered_event, mu, sigma, alpha_effective = self.gaussian_splatting(
            x_flat=x_flat,
            query_positions=query_positions,
            gate_effective=gate_effective,
        )

        # Forward Residual
        if self.use_residual:
            if self.ablation_mode == "wo_guidance":
                delta_raw = 0.0
                confidence_raw = None
            else:
                delta_raw, confidence_raw = self._compute_gaussian_score_shift(
                    mu=mu,
                    sigma=sigma,
                    alpha_effective=alpha_effective,
                    query_positions=query_positions,
                )
                


            # Store geometry statistics for diagnostics.
            if torch.is_tensor(delta_raw):
                self.last_delta = delta_raw.detach()
            else:
                self.last_delta = None

            if torch.is_tensor(confidence_raw):
                self.last_confidence = confidence_raw.detach()
            else:
                self.last_confidence = None

            # Geometry used by Gaussian Jet.
            delta_for_jet = delta_raw
            confidence_for_jet = confidence_raw

            if self.ablation_mode != "wo_guidance" and self.jet_detach_geometry:
                delta_for_jet = delta_for_jet.detach()
                if torch.is_tensor(confidence_for_jet):
                    confidence_for_jet = confidence_for_jet.detach()

            res_proj = self.patch_residual(
                x_seq=x_seq,
                delta=delta_for_jet,
                confidence=confidence_for_jet,
            )

            # Geometry-conditioned route-level coupling.
            fusion_weight, fusion_features = self._compute_fusion_weight(
                rendered_event=rendered_event,
                delta=delta_raw,
                confidence=confidence_raw,
            )

            self.last_fusion_gate = fusion_weight.detach()
            if fusion_features is not None:
                self.last_fusion_features = fusion_features.detach()
            else:
                self.last_fusion_features = None

            rendered_event = rendered_event + fusion_weight * res_proj

            if self.collect_diagnostics and self.fusion_mode == "geometry" and self.last_fusion_gate is not None:
                self._last_fusion_gate_mean = self.last_fusion_gate.mean().item()
                self._last_fusion_gate_std = self.last_fusion_gate.std().item()
                self._last_fusion_gate_min = self.last_fusion_gate.min().item()
                self._last_fusion_gate_max = self.last_fusion_gate.max().item()
                self._last_fusion_low_ratio = (self.last_fusion_gate < 0.05 * self.fusion_beta_max).float().mean().item()
                self._last_fusion_high_ratio = (self.last_fusion_gate > 0.95 * self.fusion_beta_max).float().mean().item()
                if self.last_fusion_features is not None:
                    self._last_geometry_uncertainty_mean = self.last_fusion_features[..., 0].mean().item()
                    self._last_geometry_shift_ratio_mean = self.last_fusion_features[..., 1].mean().item()
            else:
                self._last_fusion_gate_mean = None
                self._last_fusion_gate_std = None
                self._last_fusion_gate_min = None
                self._last_fusion_gate_max = None
                self._last_fusion_low_ratio = None
                self._last_fusion_high_ratio = None
                self._last_geometry_uncertainty_mean = None
                self._last_geometry_shift_ratio_mean = None
        else:
            self.last_delta = None
            self.last_confidence = None
            self.last_fusion_gate = None
            self.last_fusion_features = None

        return rendered_event
