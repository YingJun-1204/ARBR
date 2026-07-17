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

                outputs = self.model(batch_x)

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

                outputs = self.model(batch_x)

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

        splatting_module = self._get_splatting_module()
        if splatting_module is not None:
            splatting_module.collect_diagnostics = getattr(self.args, "diag", False)


        preds = []
        trues = []


        fusion_gate_means = []
        fusion_gate_stds = []
        fusion_gate_mins = []
        fusion_gate_maxs = []
        fusion_low_ratios = []
        fusion_high_ratios = []
        geom_uncertainty_means = []
        geom_shift_ratio_means = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)

                splatting_module = self._get_splatting_module()
                if splatting_module is not None and getattr(self.args, "diag", False):

                    
                    if getattr(splatting_module, "_last_fusion_gate_mean", None) is not None:
                        fusion_gate_means.append(splatting_module._last_fusion_gate_mean)
                        fusion_gate_stds.append(splatting_module._last_fusion_gate_std)
                        fusion_gate_mins.append(splatting_module._last_fusion_gate_min)
                        fusion_gate_maxs.append(splatting_module._last_fusion_gate_max)
                        fusion_low_ratios.append(splatting_module._last_fusion_low_ratio)
                        fusion_high_ratios.append(splatting_module._last_fusion_high_ratio)
                        if getattr(splatting_module, "_last_geometry_uncertainty_mean", None) is not None:
                            geom_uncertainty_means.append(splatting_module._last_geometry_uncertainty_mean)
                            geom_shift_ratio_means.append(splatting_module._last_geometry_shift_ratio_mean)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    B, T, C = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, C)).reshape(B, T, C)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, C)).reshape(B, T, C)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                preds.append(outputs)
                trues.append(batch_y)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))



        # Print Geometry-Conditioned Fusion Gate Diagnostics
        if getattr(self.args, "diag", False) and len(fusion_gate_means) > 0:
            print("=" * 50)
            print("  [Geometry-Conditioned Fusion Gate Diagnostics on Test Set]  ")
            print("-" * 50)
            print(f"  Fusion Gate Mean    : {np.mean(fusion_gate_means):.6f}")
            print(f"  Fusion Gate Std     : {np.mean(fusion_gate_stds):.6f}")
            print(f"  Fusion Gate Min     : {np.min(fusion_gate_mins):.6f}")
            print(f"  Fusion Gate Max     : {np.max(fusion_gate_maxs):.6f}")
            print(f"  Fusion Low Ratio    : {np.mean(fusion_low_ratios):.2%}")
            print(f"  Fusion High Ratio   : {np.mean(fusion_high_ratios):.2%}")
            if len(geom_uncertainty_means) > 0:
                print(f"  Geom Uncertainty Mn : {np.mean(geom_uncertainty_means):.6f}")
                print(f"  Geom Shift Ratio Mn : {np.mean(geom_shift_ratio_means):.6f}")
            print("=" * 50 + "\n")

        # -----------------------------
        f = open("result_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n\n")
        f.close()

        if splatting_module is not None:
            splatting_module.collect_diagnostics = False

        return
