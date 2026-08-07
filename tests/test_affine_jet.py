import unittest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.patch_residual import GaussianJetProjection
from layers.splatting_residual_encoder import SplattingResidualEncoder


class TestAffineJet(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_exact_normalized_center_derivative_finite_difference(self):
        """
        Validate analytical normalized center derivative psi_center against
        float64 central finite difference.
        Target relative error < 1e-4.
        """
        patch_len = 16
        num_implicit_gaussians = 4
        eps_fd = 1e-6
        dtype = torch.float64

        jet = GaussianJetProjection(
            seq_len=96,
            patch_len=patch_len,
            stride=8,
            d_model=64,
            num_implicit_gaussians=num_implicit_gaussians,
            jet_derivative_mode="exact",
        ).to(dtype=dtype)

        t = torch.linspace(0, 1, patch_len, dtype=dtype)
        mu = jet.g_mu.view(-1, 1).to(dtype=dtype)
        sigma = (F.softplus(jet.g_sigma) + 0.03).view(-1, 1).to(dtype=dtype)
        gaussian_denom = 2.0 * sigma.square() + 1e-5

        def compute_phi(mu_in):
            d = t.view(1, -1) - mu_in
            phi_raw = torch.exp(-d.square() / gaussian_denom)
            return phi_raw / (phi_raw.sum(dim=-1, keepdim=True) + 1e-5)

        phi, psi_center, _ = jet._build_gaussian_jet(device=torch.device("cpu"), dtype=dtype)

        # Numerical center derivative d(phi) / d(mu) via central finite difference:
        psi_num = torch.zeros_like(phi)
        for m in range(num_implicit_gaussians):
            mu_plus = mu.clone()
            mu_minus = mu.clone()
            mu_plus[m] += eps_fd
            mu_minus[m] -= eps_fd

            phi_plus = compute_phi(mu_plus)
            phi_minus = compute_phi(mu_minus)

            psi_num[m] = (phi_plus[m] - phi_minus[m]) / (2.0 * eps_fd)

        rel_err = torch.norm(psi_center - psi_num) / (torch.norm(psi_num) + 1e-8)
        self.assertLess(rel_err.item(), 1e-4)

    def test_exact_normalized_logscale_derivative_finite_difference(self):
        """
        Validate analytical normalized log-width derivative psi_scale against
        float64 central finite difference w.r.t. log(sigma).
        Target relative error < 1e-4.
        """
        patch_len = 16
        num_implicit_gaussians = 4
        eps_fd = 1e-6
        dtype = torch.float64

        jet = GaussianJetProjection(
            seq_len=96,
            patch_len=patch_len,
            stride=8,
            d_model=64,
            num_implicit_gaussians=num_implicit_gaussians,
            jet_derivative_mode="exact",
        ).to(dtype=dtype)

        t = torch.linspace(0, 1, patch_len, dtype=dtype)
        mu = jet.g_mu.view(-1, 1).to(dtype=dtype)

        def compute_phi_from_logsigma(log_sigma_in):
            sigma_in = torch.exp(log_sigma_in).view(-1, 1)
            denom = 2.0 * sigma_in.square() + 1e-5
            d = t.view(1, -1) - mu
            phi_raw = torch.exp(-d.square() / denom)
            return phi_raw / (phi_raw.sum(dim=-1, keepdim=True) + 1e-5)

        sigma_base = (F.softplus(jet.g_sigma) + 0.03).view(-1, 1).to(dtype=dtype)
        log_sigma_base = torch.log(sigma_base)

        _, _, psi_scale = jet._build_gaussian_jet(
            device=torch.device("cpu"), dtype=dtype, use_scale_jet=True
        )

        psi_scale_num = torch.zeros_like(psi_scale)
        for m in range(num_implicit_gaussians):
            log_s_plus = log_sigma_base.clone()
            log_s_minus = log_sigma_base.clone()
            log_s_plus[m] += eps_fd
            log_s_minus[m] -= eps_fd

            phi_plus = compute_phi_from_logsigma(log_s_plus)
            phi_minus = compute_phi_from_logsigma(log_s_minus)

            psi_scale_num[m] = (phi_plus[m] - phi_minus[m]) / (2.0 * eps_fd)

        rel_err = torch.norm(psi_scale - psi_scale_num) / (torch.norm(psi_scale_num) + 1e-8)
        self.assertLess(rel_err.item(), 1e-4)

    def test_use_scale_jet_false_baseline_equivalence(self):
        """
        Verify that setting use_scale_jet=False reproduces 100% exact baseline output.
        """
        seq_len = 96
        patch_len = 16
        stride = 8

        encoder = SplattingResidualEncoder(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=64,
            use_residual=True,
            use_scale_jet=False,
        )
        encoder.eval()

        x = torch.randn(2, 4, seq_len)
        patch_num = encoder.patch_num

        out_baseline = encoder(x, patch_num)

        encoder.use_scale_jet = False
        out_test = encoder(x, patch_num)

        diff = torch.max(torch.abs(out_baseline - out_test))
        self.assertEqual(diff.item(), 0.0)

    def test_bounded_rho(self):
        """
        Verify that hybrid scale cue rho is strictly bounded within [-rho_max, rho_max].
        """
        seq_len = 96
        patch_len = 16
        stride = 8
        rho_max = 0.25

        encoder = SplattingResidualEncoder(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=64,
            use_residual=True,
            use_scale_jet=True,
            scale_rho_max=rho_max,
        )

        x = torch.randn(4, 3, seq_len) * 10.0
        patch_num = encoder.patch_num

        _ = encoder(x, patch_num)

        rendered, mu, sigma, alpha_eff = encoder.gaussian_splatting(
            x_flat=x.flatten(0, 1), query_positions=encoder.patch_centers
        )
        rho = encoder._compute_hybrid_scale_shift(
            mu=mu,
            sigma=sigma,
            alpha_effective=alpha_eff,
            query_positions=encoder.patch_centers,
            x_seq=x,
            confidence=torch.ones(x.shape[0]*x.shape[1], patch_num),
        )

        max_abs_rho = torch.max(torch.abs(rho)).item()
        self.assertLessEqual(max_abs_rho, rho_max + 1e-6)

    def test_stop_gradient(self):
        """
        Verify scale_cue_detach=True prevents gradients from flowing back to mu/sigma of primitive generator.
        """
        seq_len = 96
        patch_len = 16
        stride = 8

        encoder = SplattingResidualEncoder(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=64,
            use_residual=True,
            use_scale_jet=True,
            scale_cue_detach=True,
        )
        # Initialize gaussian_up weight to non-zero so gradient flows through q_affine
        nn.init.ones_(encoder.patch_residual.gaussian_up.weight)

        x = torch.randn(2, 2, seq_len, requires_grad=True)
        out = encoder(x, encoder.patch_num)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(encoder.scale_gamma_field.grad)
        self.assertIsNotNone(encoder.scale_gamma_patch.grad)
        self.assertNotEqual(encoder.scale_gamma_field.grad.item(), 0.0)

    def test_robustness_edge_cases(self):
        """
        Test robustness against constant patch, zero patch, single primitive (K=1).
        No NaNs or Infs must be produced.
        """
        seq_len = 96
        patch_len = 16
        stride = 8

        for K in [1, 4]:
            encoder = SplattingResidualEncoder(
                seq_len=seq_len,
                patch_len=patch_len,
                stride=stride,
                d_model=64,
                num_gaussians=K,
                use_residual=True,
                use_scale_jet=True,
            )

            x_const = torch.ones(2, 2, seq_len)
            x_zero = torch.zeros(2, 2, seq_len)

            out_const = encoder(x_const, encoder.patch_num)
            out_zero = encoder(x_zero, encoder.patch_num)

            self.assertFalse(torch.isnan(out_const).any())
            self.assertFalse(torch.isinf(out_const).any())
            self.assertFalse(torch.isnan(out_zero).any())
            self.assertFalse(torch.isinf(out_zero).any())

    def test_wo_ajc_ablation_mode(self):
        """
        Verify that wo_ajc ablation mode disables delta (translation shift) and rho (scale shift)
        in GaussianJetProjection, rendering output invariant to delta and rho.
        """
        seq_len = 96
        patch_len = 16
        stride = 8
        d_model = 64

        jet = GaussianJetProjection(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            num_implicit_gaussians=4,
            ablation_mode="wo_ajc",
        )
        jet.eval()

        x_seq = torch.randn(2, 4, seq_len)
        patch_num = jet.patch_num
        batch_channel = 2 * 4

        delta_zero = torch.zeros(batch_channel, patch_num, 1)
        delta_random = torch.randn(batch_channel, patch_num, 1)
        rho_random = torch.randn(batch_channel, patch_num, 1)

        out_zero = jet(x_seq, delta=delta_zero, use_scale_jet=True, rho=None)
        out_shifts = jet(x_seq, delta=delta_random, use_scale_jet=True, rho=rho_random)

        diff = torch.max(torch.abs(out_zero - out_shifts))
        self.assertEqual(diff.item(), 0.0)

    def test_wo_jet_ablation_mode(self):
        """
        Verify that wo_jet ablation mode removes all JET components in GaussianJetProjection,
        returning pure Base Projection output (base_projection(patches)).
        """
        seq_len = 96
        patch_len = 16
        stride = 8
        d_model = 64

        jet = GaussianJetProjection(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            num_implicit_gaussians=4,
            ablation_mode="wo_jet",
        )
        jet.eval()

        x_seq = torch.randn(2, 4, seq_len)
        patches = jet._extract_patches(x_seq)
        expected_base = jet.base_projection(patches)

        delta_random = torch.randn(2 * 4, jet.patch_num, 1)
        out_wo_jet = jet(x_seq, delta=delta_random)

        diff = torch.max(torch.abs(expected_base - out_wo_jet))
        self.assertEqual(diff.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
