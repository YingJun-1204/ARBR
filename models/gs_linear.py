import math
import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.splatting_residual_encoder import SplattingResidualEncoder
from layers.common_blocks import FlattenHead
from layers.representations import PatchLinearRepresentation

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.dim = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = getattr(configs, "stride", configs.patch_len)
        self.num_patches_srs = math.ceil((self.seq_len - self.patch_len) / self.stride) + 1
        self.representation_name = getattr(configs, "representation", "gs")
        ablation_mode = getattr(configs, "ablation_mode", "none")
        if ablation_mode == "linear_only":
            self.representation_name = "patch_linear"

        self.norm = RevIN(
            configs.enc_in,
            affine=True,
            subtract_last=getattr(configs, "subtract_last", False),
        )

        k_base = getattr(configs, "k_base", -1)
        density_mode = getattr(configs, "density_mode", "none")
        if self.representation_name == "gs" and density_mode == "cas" and k_base == -1:
            dataset_name = getattr(configs, "data", "").lower()
            data_path = getattr(configs, "data_path", "").lower()
            
            # Map dataset to its pre-computed rho_mean on the training set
            rho_map = {
                "etth1": 0.342331,
                "etth2": 0.429279,
                "ettm1": 0.651924,
                "ettm2": 0.575884,
                "weather": 0.755071,
                "electricity": 0.097540,
                "traffic": 0.187379
            }
            
            ds_key = None
            if "etth1" in dataset_name or "etth1" in data_path:
                ds_key = "etth1"
            elif "etth2" in dataset_name or "etth2" in data_path:
                ds_key = "etth2"
            elif "ettm1" in dataset_name or "ettm1" in data_path:
                ds_key = "ettm1"
            elif "ettm2" in dataset_name or "ettm2" in data_path:
                ds_key = "ettm2"
            elif "weather" in dataset_name or "weather" in data_path:
                ds_key = "weather"
            elif "electricity" in dataset_name or "electricity" in data_path:
                ds_key = "electricity"
            elif "traffic" in dataset_name or "traffic" in data_path:
                ds_key = "traffic"
                
            if ds_key is not None:
                rho_mean = rho_map[ds_key]
                num_gaussians = getattr(configs, "num_gaussians", 8)
                stride_val = self.stride
                patch_len = self.patch_len
                
                k_val = int(round(rho_mean * num_gaussians * (stride_val / patch_len)))
                max_k = max(1, int(round(num_gaussians * (stride_val / patch_len))))
                k_base = max(1, min(max_k, k_val))
                print(f"[Model Init] Auto-resolved k_base to {k_base} for {ds_key} (full-set rho_mean: {rho_mean:.6f}, num_gaussians: {num_gaussians})")

        if self.representation_name == "gs":
            use_residual = getattr(configs, "use_residual", False)
            if ablation_mode == "gaussian_only":
                use_residual = False
            gs_residual_weight = getattr(configs, "gs_residual_weight", 0.1)
            fusion_mode = getattr(configs, "fusion_mode", "fixed")
            fusion_init = getattr(configs, "fusion_init", -1.0)
            if fusion_init < 0.0:
                fusion_init = gs_residual_weight

            if ablation_mode == "wo_adaptive_fusion":
                fusion_mode = "fixed"

            self.splatting_residual = SplattingResidualEncoder(
                seq_len=configs.seq_len,
                d_model=configs.d_model,
                num_gaussians=getattr(configs, "num_gaussians", 8),
                gs_dropout=getattr(configs, "gs_dropout", 0.3),
                density_mode=getattr(configs, "density_mode", "none"),
                use_occlusion=getattr(configs, "use_occlusion", False),
                patch_len=self.patch_len,
                stride=self.stride,
                use_residual=use_residual,
                gs_residual_weight=gs_residual_weight,
                k_base=k_base,
                num_implicit_gaussians=getattr(configs, "num_implicit_gaussians", 4),
                jet_max_shift_samples=getattr(configs, "jet_max_shift_samples", 1.0),
                jet_score_temperature=getattr(configs, "jet_score_temperature", 0.01),
                jet_density_tau=getattr(configs, "jet_density_tau", 1.0),
                jet_detach_geometry=bool(getattr(configs, "jet_detach_geometry", 1)),
                jet_scale_init=getattr(configs, "jet_scale_init", 0.1),
                jet_sigma_init=getattr(configs, "jet_sigma_init", 0.2),
                fusion_mode=fusion_mode,
                fusion_hidden_dim=getattr(configs, "fusion_hidden_dim", 16),
                fusion_init=fusion_init,
                fusion_beta_max=getattr(configs, "fusion_beta_max", 0.5),
                fusion_detach_geometry=bool(getattr(configs, "fusion_detach_geometry", 1)),
                ablation_mode=ablation_mode,
            )
            self.representation = self.splatting_residual
        elif self.representation_name == "patch_linear":
            self.patch_linear = PatchLinearRepresentation(
                seq_len=configs.seq_len,
                patch_len=self.patch_len,
                stride=self.stride,
                d_model=configs.d_model,
                dropout=0.0,
            )
            self.representation = self.patch_linear
        else:
            raise ValueError(f"Unsupported representation: {self.representation_name}")

        head_nf = configs.d_model * self.num_patches_srs
        self.head_srs = FlattenHead(
            nf=head_nf,
            target_window=configs.pred_len,
            head_dropout=getattr(configs, "head_dropout", 0.0),
            mode=getattr(configs, "head_mode", "linear"),
            head_dropout_position=getattr(configs, "head_dropout_position", "post"),
        )

    def forward(self, x):
        B, T, C = x.shape
        x_norm = self.norm(x, "norm")
        x_repr_in = x_norm.permute(0, 2, 1)

        enc_out = self.representation(x_repr_in, self.num_patches_srs)
        
        enc_out = enc_out.reshape(B, C, self.num_patches_srs, self.dim)
        enc_out = enc_out.permute(0, 1, 3, 2).reshape(-1, self.dim, self.num_patches_srs)

        pred = self.head_srs(enc_out)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return self.norm(pred, "denorm")
