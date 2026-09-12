import math
import torch
import numpy as np


def compute_effective_rank(basis: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute the effective rank of basis matrix Phi [B, P, K] using singular value entropy.
    
    r_eff = exp(- sum(p_i * ln(p_i))) where p_i = s_i / sum(s_j)
    """
    if basis.ndim == 2:
        basis = basis.unsqueeze(0)  # [1, P, K]
    
    # basis: [B, P, K]
    # SVD per sample
    svs = torch.linalg.svdvals(basis.float())  # [B, min(P, K)]
    sv_sum = svs.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = svs / sv_sum
    entropy = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)  # [B]
    r_eff = torch.exp(entropy)
    return r_eff.mean().item()


def compute_basis_cosine_similarity(basis: torch.Tensor) -> float:
    """Compute the mean pairwise cosine similarity between basis columns across samples.
    
    basis: [B, P, K]
    """
    if basis.ndim == 2:
        basis = basis.unsqueeze(0)
    
    B, P, K = basis.shape
    if K <= 1:
        return 1.0
    
    # Normalize along temporal query dimension P: [B, P, K]
    norm = torch.linalg.vector_norm(basis.float(), dim=1, keepdim=True).clamp_min(1e-8)
    normalized_basis = basis / norm
    
    # Gram matrix: [B, K, K]
    gram = torch.bmm(normalized_basis.transpose(1, 2), normalized_basis)
    
    # Extract upper triangle indices
    triu_indices = torch.triu_indices(K, K, offset=1)
    pairwise_cos = gram[:, triu_indices[0], triu_indices[1]]  # [B, K*(K-1)/2]
    return pairwise_cos.mean().item()


def compute_center_metrics(mu: torch.Tensor, rho: float = 0.5) -> dict:
    """Compute center statistics and per-input adaptivity.
    
    mu: [B, K]
    """
    if mu.ndim == 1:
        mu = mu.unsqueeze(0)
    
    K = mu.shape[-1]
    
    # 1. Within-sample std across kernels: E_x[std_k(mu_k(x))] (separation)
    std_per_sample = mu.std(dim=-1, unbiased=True if K > 1 else False)
    mean_center_std = float(std_per_sample.mean().item())
    
    # 2. Across-sample std for each kernel: std_x(mu_k(x)) (sample adaptivity)
    std_per_kernel = mu.std(dim=0, unbiased=False).cpu().tolist() if mu.shape[0] > 1 else [0.0] * K
    mean_kernel_adaptivity = float(np.mean(std_per_kernel))
    
    # 3. Normalized center adaptivity: A_mu = mean(std_x(mu_k)) / (rho / K)
    rho_over_k = rho / float(K) if K > 0 else 1.0
    norm_center_adaptivity = float(mean_kernel_adaptivity / rho_over_k)
    
    # 4. P95 - P05 span for each kernel
    if mu.shape[0] >= 20:
        p95 = torch.quantile(mu.float(), 0.95, dim=0)
        p05 = torch.quantile(mu.float(), 0.05, dim=0)
        p95_p05_span = (p95 - p05).cpu().tolist()
    else:
        p95_p05_span = [0.0] * K
    
    # Per-kernel mean across samples
    mean_per_kernel = mu.mean(dim=0).cpu().tolist()
    
    return {
        "mean_sample_center_std": mean_center_std,
        "per_kernel_center_mean": mean_per_kernel,
        "per_kernel_center_std": std_per_kernel,
        "mean_kernel_center_adaptivity": mean_kernel_adaptivity,
        "norm_center_adaptivity": norm_center_adaptivity,
        "per_kernel_p95_p05_span": p95_p05_span,
        "mean_p95_p05_span": float(np.mean(p95_p05_span)),
        "center_min": float(mu.min().item()),
        "center_max": float(mu.max().item()),
    }


def compute_scale_metrics(sigma: torch.Tensor) -> dict:
    """Compute scale statistics and per-input adaptivity.
    
    sigma: [B, K]
    """
    if sigma.ndim == 1:
        sigma = sigma.unsqueeze(0)
    
    K = sigma.shape[-1]
    mean_per_kernel = sigma.mean(dim=0).cpu().tolist()
    std_per_kernel = sigma.std(dim=0, unbiased=False).cpu().tolist() if sigma.shape[0] > 1 else [0.0] * K
    
    # Across-sample std of log(sigma_k): E_k[std_x(log(sigma_k(x)))]
    log_sigma = torch.log(sigma.clamp(min=1e-8))
    std_log_scale_per_kernel = log_sigma.std(dim=0, unbiased=False).cpu().tolist() if sigma.shape[0] > 1 else [0.0] * K
    mean_log_scale_adaptivity = float(np.mean(std_log_scale_per_kernel))
    
    return {
        "per_kernel_scale_mean": mean_per_kernel,
        "per_kernel_scale_std": std_per_kernel,
        "per_kernel_log_scale_std": std_log_scale_per_kernel,
        "mean_kernel_scale_adaptivity": mean_log_scale_adaptivity,
        "scale_min": float(sigma.min().item()),
        "scale_max": float(sigma.max().item()),
    }


def evaluate_phase1a_metrics(model, data_loader, device="cuda"):
    """Evaluate Phase 1A diagnostic metrics over a dataloader."""
    model.eval()
    all_ranks = []
    all_cosims = []
    all_mus = []
    all_sigmas = []
    
    # Find splatting module
    real_model = model.module if hasattr(model, "module") else model
    splatting_encoder = getattr(real_model, "splatting_residual", None)
    if splatting_encoder is None or getattr(splatting_encoder, "gaussian_splatting", None) is None:
        raise ValueError("Model does not contain a gaussian_splatting module")
        
    gs_module = splatting_encoder.gaussian_splatting
    queries = splatting_encoder.patch_centers
    
    with torch.no_grad():
        for batch_x, _, _, _ in data_loader:
            batch_x = batch_x.float().to(device)
            # x_norm through RevIN if exists
            if hasattr(real_model, "norm"):
                batch_x_norm = real_model.norm(batch_x, "norm")
            else:
                batch_x_norm = batch_x
            
            # shape: [B, N_vars, L] -> [B*N_vars, L]
            x_flat = batch_x_norm.permute(0, 2, 1).flatten(0, 1)
            
            # forward through gaussian_splatting
            res = gs_module(
                x_flat,
                query_positions=queries,
                return_contributions=True,
            )
            if len(res) == 4:
                _, basis, mu, sigma = res
            elif len(res) == 2:
                _, basis = res
                mu = gs_module.last_mu if hasattr(gs_module, "last_mu") else torch.zeros(x_flat.shape[0], gs_module.num_gaussians)
                sigma = gs_module.last_sigma if hasattr(gs_module, "last_sigma") else torch.ones(x_flat.shape[0], gs_module.num_gaussians)
            else:
                raise RuntimeError(f"Unexpected return format from gaussian_splatting: {len(res)}")
            
            all_ranks.append(compute_effective_rank(basis))
            all_cosims.append(compute_basis_cosine_similarity(basis))
            all_mus.append(mu.detach().cpu())
            all_sigmas.append(sigma.detach().cpu())
            
    mus_tensor = torch.cat(all_mus, dim=0)
    sigmas_tensor = torch.cat(all_sigmas, dim=0)
    
    center_metrics = compute_center_metrics(mus_tensor)
    scale_metrics = compute_scale_metrics(sigmas_tensor)
    
    K = mus_tensor.shape[-1] if mus_tensor.shape[-1] > 0 else 1
    mean_rank = float(np.mean(all_ranks))
    
    return {
        "effective_rank": mean_rank,
        "norm_effective_rank": float(mean_rank / K),
        "cosine_similarity": float(np.mean(all_cosims)),
        "center_std": center_metrics["mean_sample_center_std"],
        "center_adaptivity": center_metrics["mean_kernel_center_adaptivity"],
        "norm_center_adaptivity": center_metrics["norm_center_adaptivity"],
        "scale_adaptivity": scale_metrics["mean_kernel_scale_adaptivity"],
        "mean_p95_p05_span": center_metrics["mean_p95_p05_span"],
        "per_kernel_center_std": center_metrics["per_kernel_center_std"],
        "per_kernel_p95_p05_span": center_metrics["per_kernel_p95_p05_span"],
        "center_means": center_metrics["per_kernel_center_mean"],
        "center_range": [center_metrics["center_min"], center_metrics["center_max"]],
        "scale_means": scale_metrics["per_kernel_scale_mean"],
        "scale_range": [scale_metrics["scale_min"], scale_metrics["scale_max"]],
    }


