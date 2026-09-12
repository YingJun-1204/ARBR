from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import json
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
        return optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=0.0,
        )

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion, return_mae=False):
        total_loss = 0.0
        total_mae = 0.0
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
                if return_mae:
                    total_mae += torch.abs(pred - true).sum().item()
                total_count += pred.numel()

        total_loss = total_loss / total_count if total_count else 0.0
        total_mae = total_mae / total_count if total_count else 0.0
        self.model.train()
        if return_mae:
            return total_loss, total_mae
        return total_loss

    def train(self, setting):
        if self.args.use_gpu and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass
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
                            if splatting_module is not None and getattr(splatting_module, "gaussian_splatting", None) is not None:
                                gs = splatting_module.gaussian_splatting
                                geom_grad = gs.geometry_head.weight.grad.abs().mean().item() if gs.geometry_head.weight.grad is not None else 0.0
                                val_grad = gs.value_head.weight.grad.abs().mean().item() if gs.value_head.weight.grad is not None else 0.0
                                w_grad_msg = f"Geom: {geom_grad:.6f}, Val: {val_grad:.6f}"

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
            print_mae = getattr(self.args, "print_mae", False)
            if print_mae:
                vali_loss, vali_mae = self.vali(vali_data, vali_loader, criterion, return_mae=True)
                test_loss, test_mae = self.vali(test_data, test_loader, criterion, return_mae=True)
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} Test MAE: {5:.7f}".format(
                        epoch + 1,
                        train_steps,
                        train_loss,
                        vali_loss,
                        test_loss,
                        test_mae,
                    )
                )
            else:
                vali_loss = self.vali(vali_data, vali_loader, criterion, return_mae=False)
                test_loss = self.vali(test_data, test_loader, criterion, return_mae=False)
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
            checkpoint_path = os.path.join(
                "./checkpoints/" + setting,
                "checkpoint.pth",
            )
            self.model.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
            )

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)

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

        with open("result_long_term_forecast.txt", "a") as result_file:
            result_file.write(setting + "  \n")
            result_file.write("mse:{}, mae:{}".format(mse, mae))
            result_file.write("\n\n")

        # ----------------------------------------------------------------------
        # Measure efficiency metrics ONLY if explicitly enabled
        # ----------------------------------------------------------------------
        if getattr(self.args, "eval_efficiency", 0) == 1 or os.environ.get("EVAL_EFFICIENCY", "0") == "1":
            dataset_name = self.args.data
            if dataset_name == "custom":
                dataset_name = os.path.splitext(os.path.basename(self.args.data_path))[0].capitalize()
            name_map = {
                "etth1": "ETTh1",
                "etth2": "ETTh2",
                "ettm1": "ETTm1",
                "ettm2": "ETTm2",
                "weather": "Weather",
                "electricity": "Electricity",
                "traffic": "Traffic",
            }
            norm_ds = name_map.get(dataset_name.lower(), dataset_name)
            params_m = sum(p.numel() for p in self.model.parameters()) / 1e6

            real_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            try:
                from torch.utils.flop_counter import FlopCounterMode
                sample_x = torch.randn(1, self.args.seq_len, self.args.enc_in, device=self.device)
                with FlopCounterMode(display=False) as fc:
                    real_model(sample_x)
                flops_g = fc.get_total_flops() / 1e9
            except Exception:
                flops_g = 0.0

            try:
                if len(test_loader) > 0:
                    timing_batch = next(iter(test_loader))[0].float().to(self.device)
                else:
                    timing_batch = torch.randn(self.args.batch_size, self.args.seq_len, self.args.enc_in, device=self.device)
                real_model.eval()
                with torch.no_grad():
                    for _ in range(5):
                        _ = real_model(timing_batch)
                    if self.args.use_gpu and torch.cuda.is_available():
                        torch.cuda.synchronize(self.device)
                        s_evt = torch.cuda.Event(enable_timing=True)
                        e_evt = torch.cuda.Event(enable_timing=True)
                        num_timing_runs = 30
                        s_evt.record()
                        for _ in range(num_timing_runs):
                            _ = real_model(timing_batch)
                        e_evt.record()
                        torch.cuda.synchronize(self.device)
                        latency_ms = s_evt.elapsed_time(e_evt) / num_timing_runs
                    else:
                        t0 = time.time()
                        num_timing_runs = 10
                        for _ in range(num_timing_runs):
                            _ = real_model(timing_batch)
                        latency_ms = ((time.time() - t0) / num_timing_runs) * 1000.0
            except Exception:
                latency_ms = 0.0

            if self.args.use_gpu and torch.cuda.is_available():
                memory_g = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
            else:
                memory_g = 0.0

            print(
                f"[Efficiency] Setting: {setting} | Horizon: {self.args.pred_len} | "
                f"Params/M: {params_m:.4f} | FLOPS/G: {flops_g:.4f} | "
                f"Latency/ms: {latency_ms:.4f} | Memory/G: {memory_g:.4f}"
            )

            eff_file = "result_efficiency.json"
            all_eff = {}
            if os.path.exists(eff_file):
                try:
                    with open(eff_file, "r", encoding="utf-8") as f:
                        all_eff = json.load(f)
                except Exception:
                    all_eff = {}
            ds_dict = all_eff.setdefault(norm_ds, {})
            ds_dict[str(self.args.pred_len)] = {
                "params_m": round(params_m, 4),
                "flops_g": round(flops_g, 4),
                "latency_ms": round(latency_ms, 4),
                "memory_g": round(memory_g, 4),
                "setting": setting,
            }
            try:
                with open(eff_file, "w", encoding="utf-8") as f:
                    json.dump(all_eff, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

            try:
                with open("result_efficiency.txt", "a", encoding="utf-8") as f:
                    f.write(f"{setting}\n")
                    f.write(
                        f"Horizon: {self.args.pred_len}, Params/M: {params_m:.4f}, "
                        f"FLOPS/G: {flops_g:.4f}, Latency/ms: {latency_ms:.4f}, "
                        f"Memory/G: {memory_g:.4f}\n\n"
                    )
            except Exception:
                pass

        return
