# LGA Gating Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the Local Geometry Attention (LGA) Mahalanobis distance features into the adaptive residual router of the time-series forecasting model, utilizing local temporal manifold geometry to distinguish random high-frequency noise from structural trend changes.

**Architecture:** We will compute the expected local Mahalanobis distance of the residual projection `res_proj` within a sliding temporal window of half-width `gate_window_half`. The local feature variance is computed with detached gradients (for numerical stability). This distance is appended to the existing 4 features (Z-score density, std, diff-std, energy) to form a 5D feature. A 2-layer MLP (hidden size defined by `gate_router_hidden`) will then map the 5D feature to 3 routing logits (Neutral, Forward, Reverse).

**Tech Stack:** PyTorch, Python, Optuna (HPO)

---

## Proposed Changes

### Task 1: Update CLI Arguments & Config Passing

**Files:**
- Modify: [run.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/run.py)
- Modify: [models/gs_linear.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/models/gs_linear.py)
- Modify: [layers/splatting_residual_encoder.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/layers/splatting_residual_encoder.py)

- [ ] **Step 1: Add `--gate_window_half` CLI argument to run.py**
  Add `--gate_window_half` argument of type `int`, default `2`, choices `[1, 2, 3, 4]`, and pass it through.
- [ ] **Step 2: Update models/gs_linear.py to pass `gate_window_half`**
  Ensure the model instantiation passes `gate_window_half` from `configs` (default to `2` if not present) into `SplattingResidualEncoder`.
- [ ] **Step 3: Update layers/splatting_residual_encoder.py constructor**
  Update `SplattingResidualEncoder.__init__` to accept `gate_window_half` (default `2`), store it as `self.gate_window_half`, and forward both `gate_router_hidden` and `gate_window_half` to `AdaptiveResidualRouter`.

---

### Task 2: Implement Local Geometry Attention (LGA) Gating Router in layers/residual_router.py

**Files:**
- Modify: [layers/residual_router.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/layers/residual_router.py)

- [ ] **Step 1: Update AdaptiveResidualRouter __init__**
  Update `AdaptiveResidualRouter.__init__` to accept `gate_router_hidden` (default `16`) and `gate_window_half` (default `2`).
- [ ] **Step 2: Reconstruct route_router to a 2-layer MLP**
  If `gate_type == "adaptive_direction"`, initialize `self.route_router` as a `nn.Sequential` with:
  1. `nn.Linear(5, self.gate_router_hidden)`
  2. `nn.GELU()`
  3. `nn.Linear(self.gate_router_hidden, 3)`
  Initialize the weights and biases of the final linear layer (`self.route_router[2]`) to zeros so it starts with a neutral output.
- [ ] **Step 3: Implement sliding window Mahalanobis distance calculation**
  In the `adaptive_direction` path in `forward`:
  1. Compute local Z-score Mahalanobis distance of `res_proj` (shape `[batch_channel, patch_num, d_model]`) over a temporal sliding window of half-width `W = gate_window_half`.
  2. Pad the sequence edges with replication/mirroring padding (`torch.nn.functional.pad` with `mode='replicate'` or custom slicing) to ensure boundary patches have $2W$ neighbors.
  3. Compute the local feature variance with detached gradients: $\sigma^2_{t, d} = \text{Var}(\text{neighbors}) + 1e-5$. Use `.detach()` on the variance denominator to avoid NaN gradients.
  4. Compute the expected Mahalanobis distance: $\bar{d}_G^2(t) = \frac{1}{2W} \sum_{j \in \mathcal{N}(t)} \sum_{d=1}^{d\_model} \frac{(x_{j, d} - x_{t, d})^2}{\sigma^2_{t, d}}$.
  5. Apply Z-score standardization to the computed distance across the patches in the sequence using `self._zscore_patch`.
- [ ] **Step 4: Concatenate features and compute routing logits**
  Concatenate `z_density`, `z_std`, `z_diff`, `z_energy`, and the Z-scored `z_lga` Mahalanobis distance to construct a 5D feature tensor. Pass the detached features to `self.route_router` to get the routing logits.

---

### Task 3: Update HPO Tuning Scripts

**Files:**
- Modify: [tune_gate_etth1.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/tune_gate_etth1.py)
- Modify: [tune_gate_etth2.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/tune_gate_etth2.py)
- Modify: [tune_gate_ettm1.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/tune_gate_ettm1.py)
- Modify: [tune_gate_ettm2.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/tune_gate_ettm2.py)

- [ ] **Step 1: Add gate_window_half bounds**
  Add `"gate_window_half_lower": 1` and `"gate_window_half_upper": 4` to `DEFAULT_BOUNDS` or parameters.
- [ ] **Step 2: Suggest gate_window_half in Optuna objective**
  Suggest the parameter using `trial.suggest_int("gate_window_half", gate_window_half_lower, gate_window_half_upper)`.
- [ ] **Step 3: Update build_command**
  Pass `--gate_window_half` into the run command generated by `build_command`.
- [ ] **Step 4: Save gate_window_half in callbacks**
  Include `"gate_window_half"` in the best JSON configuration saved during tuning callbacks.

---

### Task 4: Verification & Dry-run

- [ ] **Step 1: Run dry-run validation command**
  Run a single-epoch training command on ETTh1 to verify that:
  - The model initializes correctly.
  - The forward/backward passes complete without NaNs.
  - Diagnostics are correctly printed to `loss_cas_simplify`.
  Command:
  `cmd /d /c "chcp 65001 > nul && call D:\Conda\Scripts\activate.bat demo && python run.py --task_name long_term_forecast --is_training 1 --root_path ./data --data_path ETTh1.csv --model_id verify_lga --model yzy --data ETTh1 --features M --seq_len 512 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --itr 1 --batch_size 128 --representation pure_gs --num_gaussians 10 --density_mode cas --use_residual --gs_residual_weight 0.4441 --gate_type adaptive_direction --gate_beta 0.4589 --gate_router_hidden 16 --gate_window_half 2 --train_epochs 1 --num_workers 0"`
- [ ] **Step 2: Verify output files and log check**
  Check that the training completed and the diagnostics file `loss_cas_simplify/diagnostics_verify_lga.txt` was created successfully.

---

## Verification Plan

### Automated Tests
- Run the full suite of unit tests to verify no regressions:
  `cmd /d /c "chcp 65001 > nul && call D:\Conda\Scripts\activate.bat demo && python -m unittest discover -s tests"`
- Run the dry-run training script command above and verify loss convergence.
