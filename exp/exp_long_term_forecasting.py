from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def resolve_min_checkpoint_epoch():
    return 1


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_splatting_module(self):
        real_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return getattr(real_model, "splatting_residual", None)

    def _select_optimizer(self):
        splatting_params = []
        base_params = []

        for name, param in self.model.named_parameters():
            if "splatting_residual" in name:
                splatting_params.append(param)
            else:
                base_params.append(param)

        if len(splatting_params) > 0:
            model_optim = optim.AdamW(
                [
                    {"params": base_params, "weight_decay": 1e-4},
                    {"params": splatting_params, "weight_decay": self.args.gs_weight_decay},
                ],
                lr=self.args.learning_rate,
            )
            print(
                "\t[Optimizer] Applied differentiated weight decay ({}) for splatting module".format(
                    self.args.gs_weight_decay
                )
            )
        else:
            model_optim = optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=1e-4,
            )
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = 0.0
        total_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                outputs = self.model(batch_x, is_training=False)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss += loss.item() * pred.numel()
                total_count += pred.numel()

        total_loss = total_loss / total_count if total_count else 0.0
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        # total_params = sum(p.numel() for p in self.model.parameters())
        # # 统计可训练参数量
        # trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)    

        path = os.path.join(self.args.checkpoints, setting)
        if not getattr(self.args, "no_save_checkpoint", False):
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, 
            verbose=True, 
            no_save_checkpoint=getattr(self.args, "no_save_checkpoint", False)
        )

        best_val_loss = float("inf")
        best_val_epoch = 0
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 确保 warmup 从第 1 个 epoch 生效
        adjust_learning_rate(model_optim, 0, self.args)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()



            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(
                    batch_x,
                    is_training=True,
                    epoch=epoch,
                    max_epochs=self.args.train_epochs,
                )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y)

                # --- Density Penalty Joint Loss Optimization ---
                density_mode = getattr(self.args, "density_mode", "none")
                if density_mode == "cas":
                    real_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    splatting_module = getattr(real_model, "splatting_residual", None)
                    if splatting_module is not None and getattr(splatting_module, "last_gate_probs", None) is not None:
                        num_gaussians = splatting_module.num_gaussians
                        
                        # 1. 计算 Sigmoid Warmup 因子 (在前几个 epoch 从 0 平滑且可导地增长到 1)
                        import math
                        e_mid = 5.0
                        s = 1.2
                        warmup_factor = 1.0 / (1.0 + math.exp(-(epoch - e_mid) / s))
                        
                        # 构建前低后高的非对称惩罚权重 (前3个核几近无损，后续单调递增)
                        weights_list = [0.0] * min(3, num_gaussians)
                        remaining = num_gaussians - len(weights_list)
                        if remaining > 0:
                            weights_list += torch.linspace(0.01, 0.2, remaining).tolist()
                        lambda_weights = torch.tensor(weights_list, device=loss.device)
                        
                        gate_probs = splatting_module.last_gate_probs # (batch_channel, num_gaussians)
                        loss_capacity = (gate_probs * lambda_weights.unsqueeze(0)).mean()
                        loss = loss + (self.args.gs_lambda * warmup_factor) * loss_capacity
                # ------------------------------------------------


                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 200 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1,
                            epoch + 1,
                            loss.item(),
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))

                    if getattr(self.args, "debug_mode", False):
                        try:
                            real_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                            head_grad_norm = 0.0
                            if hasattr(real_model, "head_srs"):
                                for p in real_model.head_srs.parameters():
                                    if p.grad is not None:
                                        head_grad_norm += p.grad.data.norm(2).item() ** 2
                                head_grad_norm = head_grad_norm**0.5
                            else:
                                head_grad_norm = -1.0

                            w_grad_msg = "N/A"
                            splatting_module = self._get_splatting_module()
                            if splatting_module is not None:
                                generator_weight = splatting_module.generator[-1].weight
                                if generator_weight.grad is not None:
                                    w_grad_avg = generator_weight.grad.abs().mean().item()
                                    w_grad_msg = f"GS_Generator: {w_grad_avg:.6f}"

                            print(
                                "\t[Grad Probe] Head Grad Norm: {:.6f} | w_grad: {}".format(
                                    head_grad_norm,
                                    w_grad_msg,
                                )
                            )
                        except Exception as e:
                            print("\t[Grad Probe] Error: {}".format(e))

                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # total_params = sum(p.numel() for p in self.model.parameters())
            # # 统计可训练参数量
            # trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)   
            # print(total_params,trainable_params)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    vali_loss,
                    test_loss,
                )
            )

            if vali_loss < best_val_loss:
                best_val_loss = vali_loss
                best_val_epoch = epoch + 1

            early_stopping(vali_loss, self.model, path)


            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        print("\n" + "=" * 30)
        print(f"Final Diagnosis: Best Vali Epoch: {best_val_epoch}")
        print("=" * 30 + "\n")

        if getattr(self.args, "no_save_checkpoint", False):
            if early_stopping.best_model_state is not None:
                self.model.load_state_dict(early_stopping.best_model_state)
            else:
                print("Warning: No best model state found in memory.")
        else:
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth")))

        gate_dir_list = []
        gate_scale_list = []
        gate_strength_list = []
        gate_route_prob_list = []
        # CAS-specific lists
        p_i_list = []
        pi_i_list = []
        sigma_list = []
        all_probs = None
        all_strengths = None
        preds = []
        trues = []
        inputs = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x, is_training=False)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    B, T, C = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, C)).reshape(B, T, C)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, C)).reshape(B, T, C)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:]

                # --- Calculate CAS mechanics ---
                real_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                splatting_module = getattr(real_model, "splatting_residual", None)
                if splatting_module is not None and getattr(self.args, "density_mode", "none") == "cas":
                    if getattr(splatting_module, "last_p_i", None) is not None:
                        p_i_list.append(splatting_module.last_p_i.detach().cpu().numpy())
                        pi_i_list.append(splatting_module.last_pi_i.detach().cpu().numpy())
                        sigma_list.append(splatting_module.last_sigma.detach().cpu().numpy())
                # --------------------------------------------------

                # --- Collect Gate Diagnostics ---
                if splatting_module is not None:
                    if getattr(splatting_module, "last_gate_direction", None) is not None:
                        gate_dir_list.append(splatting_module.last_gate_direction.detach().cpu().numpy())
                    if getattr(splatting_module, "last_gate_scale", None) is not None:
                        gate_scale_list.append(splatting_module.last_gate_scale.detach().cpu().numpy())
                    if getattr(splatting_module, "last_gate_strength", None) is not None:
                        gate_strength_list.append(splatting_module.last_gate_strength.detach().cpu().numpy())
                    if getattr(splatting_module, "last_gate_route_probs", None) is not None:
                        gate_route_prob_list.append(splatting_module.last_gate_route_probs.detach().cpu().numpy())

                # ---------------------------------

                inputs.append(batch_x)
                preds.append(outputs)
                trues.append(batch_y)

        inputs = np.concatenate(inputs, axis=0)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))

        # --- Collect Gaussian and Gate diagnostics for file log ---
        diag_lines = []


        # --- Write Gate Diagnostics to File (Skip Console Print) ---
        if len(gate_dir_list) > 0:
            all_dirs = np.concatenate(gate_dir_list, axis=0)
            all_scales = np.concatenate(gate_scale_list, axis=0)
            if len(gate_strength_list) > 0:
                all_strengths = np.concatenate(gate_strength_list, axis=0)
            if len(gate_route_prob_list) > 0:
                all_probs = np.concatenate(gate_route_prob_list, axis=0)
            
            dir_mean = np.mean(all_dirs)
            dir_std = np.std(all_dirs)
            dir_min, dir_max = np.min(all_dirs), np.max(all_dirs)
            abs_dir_mean = np.mean(np.abs(all_dirs))
            
            scale_mean = np.mean(all_scales)
            scale_std = np.std(all_scales)
            
            total_elements = all_dirs.size
            
            diag_lines.append("\n" + "=" * 50)
            diag_lines.append("      [Adaptive Gating Direction Diagnostics]      ")
            diag_lines.append("-" * 50)
            diag_lines.append(f"Direction -> mean: {dir_mean:.4f} | std: {dir_std:.4f} | abs_mean: {abs_dir_mean:.4f}")
            diag_lines.append(f"Direction -> min/max: {dir_min:.4f} / {dir_max:.4f}")
            
            if all_probs is not None:
                neutral_probs = all_probs[..., 0]
                fwd_probs = all_probs[..., 1]
                rev_probs = all_probs[..., 2]
                
                neutral_prob_mean, neutral_prob_std = np.mean(neutral_probs), np.std(neutral_probs)
                fwd_prob_mean, fwd_prob_std = np.mean(fwd_probs), np.std(fwd_probs)
                rev_prob_mean, rev_prob_std = np.mean(rev_probs), np.std(rev_probs)
                
                max_prob = np.max(all_probs, axis=-1)
                max_prob_mean, max_prob_std = np.mean(max_prob), np.std(max_prob)
                uncertain_ratio = np.sum(max_prob < 0.45) / total_elements
                
                argmax_route = np.argmax(all_probs, axis=-1)
                argmax_neutral = np.sum(argmax_route == 0) / total_elements
                argmax_fwd = np.sum(argmax_route == 1) / total_elements
                argmax_rev = np.sum(argmax_route == 2) / total_elements
                
                if uncertain_ratio > 0.50:
                    decision = "Uncertain / Neutral Gating Strategy (High uncertainty across patches)"
                elif argmax_neutral > 0.70:
                    decision = "Mostly Neutral / No Density Gating"
                elif argmax_fwd > 0.70:
                    decision = "Forward Dominant"
                    if uncertain_ratio > 0.30:
                        decision += " (with Moderate Uncertainty)"
                elif argmax_rev > 0.70:
                    decision = "Reverse Dominant"
                    if uncertain_ratio > 0.30:
                        decision += " (with Moderate Uncertainty)"
                else:
                    decision = "Mixed Gating"
                    
                diag_lines.append("Route Probabilities:")
                diag_lines.append(f"  Neutral -> mean: {neutral_prob_mean:.4f} | std: {neutral_prob_std:.4f}")
                diag_lines.append(f"  Forward -> mean: {fwd_prob_mean:.4f} | std: {fwd_prob_std:.4f}")
                diag_lines.append(f"  Reverse -> mean: {rev_prob_mean:.4f} | std: {rev_prob_std:.4f}")
                diag_lines.append(f"Route Confidence -> max_prob mean: {max_prob_mean:.4f} | std: {max_prob_std:.4f}")
                diag_lines.append(f"Uncertain ratio  -> max_prob < 0.45: {uncertain_ratio:.2%}")
                diag_lines.append("Route Argmax Ratios:")
                diag_lines.append(f"  Neutral: {argmax_neutral:.2%} | Forward: {argmax_fwd:.2%} | Reverse: {argmax_rev:.2%}")
                
            diag_lines.append("-" * 50)
            
            if all_strengths is not None:
                str_mean = np.mean(all_strengths)
                str_std = np.std(all_strengths)
                str_min, str_max = np.min(all_strengths), np.max(all_strengths)
                active_ratio = np.sum(all_strengths > 0.5) / total_elements
                weak_ratio = np.sum(all_strengths < 0.1) / total_elements
                mid_ratio = np.sum((all_strengths >= 0.1) & (all_strengths <= 0.5)) / total_elements
                
                eff_dirs = all_strengths * all_dirs
                eff_dir_mean = np.mean(eff_dirs)
                eff_dir_std = np.std(eff_dirs)
                eff_abs_dir_mean = np.mean(np.abs(eff_dirs))
                eff_dir_min = np.min(eff_dirs)
                eff_dir_max = np.max(eff_dirs)
                
                eff_pos_ratio = np.sum(eff_dirs > 0.05) / total_elements
                eff_neg_ratio = np.sum(eff_dirs < -0.05) / total_elements
                eff_near_zero_ratio = np.sum(np.abs(eff_dirs) <= 0.05) / total_elements
                
                diag_lines.append(f"Strength  -> mean: {str_mean:.4f} | std: {str_std:.4f} | min/max: {str_min:.4f} / {str_max:.4f}")
                diag_lines.append(f"Strength ratios -> Active(>0.5): {active_ratio:.2%} | Weak(<0.1): {weak_ratio:.2%} | Mid: {mid_ratio:.2%}")
                diag_lines.append("-" * 50)
                diag_lines.append("      [Effective Gating (Strength * Direction)]      ")
                diag_lines.append("-" * 50)
                diag_lines.append(f"Eff_Dir   -> mean: {eff_dir_mean:.4f} | std: {eff_dir_std:.4f} | abs_mean: {eff_abs_dir_mean:.4f}")
                diag_lines.append(f"Eff_Dir   -> min/max: {eff_dir_min:.4f} / {eff_dir_max:.4f}")
                diag_lines.append(f"Eff_Dir ratios -> Positive(>0.05): {eff_pos_ratio:.2%}")
                diag_lines.append(f"                  Negative(<-0.05): {eff_neg_ratio:.2%}")
                diag_lines.append(f"                  Near Zero(<=0.05): {eff_near_zero_ratio:.2%}")
                
            diag_lines.append(f"Scale     -> mean: {scale_mean:.4f} | std: {scale_std:.4f}")
            diag_lines.append(f"Decision  -> Current Strategy: {decision}")
            diag_lines.append("=" * 50 + "\n")
            

            
        # --- CAS Gating Diagnostics ---
        if len(p_i_list) > 0:
            all_p_i = np.concatenate(p_i_list, axis=0)
            all_pi_i = np.concatenate(pi_i_list, axis=0)
            all_sigma = np.concatenate(sigma_list, axis=0)
            
            mean_p_i = np.mean(all_p_i, axis=0)
            mean_pi_i = np.mean(all_pi_i, axis=0)
            
            sample_egc = np.sum(all_pi_i, axis=1)
            egc_mean = np.mean(sample_egc)
            egc_min = np.min(sample_egc)
            egc_max = np.max(sample_egc)
            
            active_mask = all_pi_i > 0.5
            active_sigmas = all_sigma[active_mask]
            truncated_sigmas = all_sigma[~active_mask]
            
            mean_sigma_active = np.mean(active_sigmas) if active_sigmas.size > 0 else 0.0
            mean_sigma_truncated = np.mean(truncated_sigmas) if truncated_sigmas.size > 0 else 0.0
            
            diag_lines.append("\n" + "=" * 50)
            diag_lines.append("      [Mechanistic Metric (CAS / Gating Cascade)]      ")
            diag_lines.append("-" * 50)
            
            p_i_str = " | ".join([f"{val:.4f}" for val in mean_p_i])
            diag_lines.append(f"Independent Prob (p_i):\n  [ {p_i_str} ]")
            
            pi_i_str = " | ".join([f"{val:.4f}" for val in mean_pi_i])
            diag_lines.append(f"Cascaded Prob (pi_i):\n  [ {pi_i_str} ]")
            
            diag_lines.append("-" * 50)
            diag_lines.append(f"Effective Gaussian Count (EGC) Summary:")
            diag_lines.append(f"  Mean EGC : {egc_mean:.4f}")
            diag_lines.append(f"  Min EGC  : {egc_min:.4f}")
            diag_lines.append(f"  Max EGC  : {egc_max:.4f}")
            
            diag_lines.append("-" * 50)
            diag_lines.append(f"Gaussian Bandwidth (Sigma) Bifurcation:")
            diag_lines.append(f"  Active Nuclei Sigma (pi_i > 0.5)   : {mean_sigma_active:.4f}")
            diag_lines.append(f"  Truncated Nuclei Sigma (pi_i <= 0.5) : {mean_sigma_truncated:.4f}")
            
            if mean_sigma_active > mean_sigma_truncated * 1.2:
                status = "SUCCESS (Active sigma is significantly larger)"
            else:
                status = "WEAK / NO BIFURCATION"
            diag_lines.append(f"  Bifurcation Status : {status}")
            diag_lines.append("=" * 50 + "\n")
            
        # --- Write all diagnostics to file ---
        if len(diag_lines) > 0:
            diag_text = "\n".join(diag_lines)
            output_dir = getattr(self.args, "output_dir", "loss日志")
            os.makedirs(output_dir, exist_ok=True)
            diag_file = os.path.join(output_dir, f"diagnostics_{self.args.model_id}.txt")
            try:
                with open(diag_file, "w", encoding="utf-8") as f:
                    f.write(diag_text)
            except Exception as e:
                # Fallback to console print if writing fails
                print(f"[Warning] Failed to write diagnostics to file: {e}")
                print(diag_text)
        # -----------------------------
        f = open("result_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n\n")
        f.close()

        return
