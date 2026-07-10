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
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.patch_len = configs.patch_len
        self.stride = getattr(configs, "stride", configs.patch_len)
        self.num_patches_srs = math.ceil((self.seq_len - self.patch_len) / self.stride) + 1
        self.representation_name = getattr(configs, "representation", "gs")

        self.norm = RevIN(
            configs.enc_in,
            affine=True,
            subtract_last=getattr(configs, "subtract_last", False),
        )

        in_features = configs.seq_len
        
        k_base = getattr(configs, "k_base", -1)
        if k_base == -1:
            dataset_name = getattr(configs, "data", "").lower()
            data_path = getattr(configs, "data_path", "").lower()
            
            # Map dataset to its pre-computed rho_mean on the training set
            rho_map = {
                "etth1": 0.342331,
                "etth2": 0.429279,
                "ettm1": 0.651924,
                "ettm2": 0.575884,
                "weather": 0.755071,
                "electricity": 0.097540
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
            self.splatting_residual = SplattingResidualEncoder(
                seq_len=configs.seq_len,
                d_model=configs.d_model,
                num_gaussians=getattr(configs, "num_gaussians", 5),
                gs_dropout=getattr(configs, "gs_dropout", 0.3),
                in_features=in_features,
                density_mode=getattr(configs, "density_mode", "none"),
                use_occlusion=getattr(configs, "use_occlusion", False),
                patch_len=self.patch_len,
                stride=self.stride,
                use_residual=getattr(configs, "use_residual", False),
                gs_residual_weight=getattr(configs, "gs_residual_weight", 0.1),
                gate_type=getattr(configs, "gate_type", None),
                gate_beta=getattr(configs, "gate_beta", 0.25),
                gate_window_half=getattr(configs, "gate_window_half", 2),
                gamma_complement=getattr(configs, "gamma_complement", 0.6),
                k_base=k_base,
                residual_mode=getattr(configs, "residual_mode", "gaussian_jet"),
                num_implicit_gaussians=getattr(configs, "num_implicit_gaussians", 4),
                jet_max_shift_samples=getattr(configs, "jet_max_shift_samples", 1.0),
                jet_score_temperature=getattr(configs, "jet_score_temperature", 1.0),
                jet_density_tau=getattr(configs, "jet_density_tau", 1.0),
                jet_detach_geometry=bool(getattr(configs, "jet_detach_geometry", 1)),
                jet_scale_init=getattr(configs, "jet_scale_init", 0.1),
                jet_sigma_init=getattr(configs, "jet_sigma_init", 0.2),
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
            n_vars=configs.enc_in,
            nf=head_nf,
            target_window=configs.pred_len,
            head_dropout=getattr(configs, "head_dropout", configs.dropout),
            mode=getattr(configs, "head_mode", "linear"),
            head_dropout_position=getattr(configs, "head_dropout_position", "post"),
        )

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, masks=None, is_training=False, target=None, epoch=None, max_epochs=None):
        B, T, C = x.shape
        x_norm = self.norm(x, "norm")
        x_repr_in = x_norm.permute(0, 2, 1)

        enc_out = self.representation(x_repr_in, self.num_patches_srs)
        
        enc_out = enc_out.reshape(B, C, self.num_patches_srs, self.dim)
        enc_out = enc_out.permute(0, 1, 3, 2).reshape(-1, self.dim, self.num_patches_srs)

        pred = self.head_srs(enc_out)
        pred = pred.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return self.norm(pred, "denorm")
