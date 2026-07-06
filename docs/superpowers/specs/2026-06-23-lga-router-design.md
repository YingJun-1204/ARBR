# Design Document: Local Geometry Attention (LGA) Gating Router

This design document specifies the integration of the Local Geometry Attention (LGA) Mahalanobis distance features into the adaptive residual router of the time-series forecasting model. This design enhances the router's ability to distinguish high-frequency random noise from structural trend changes by modeling local temporal manifold geometry.

## 1. System Architecture & Module Boundaries

The LGA feature computation will reside inside `AdaptiveResidualRouter` in [layers/residual_router.py](file:///c:/Users/19331/Desktop/Code/agent/Gemini/demo-PureGS+Patchlinear/layers/residual_router.py). The overall data flow is as follows:

1.  **Encoder**: `SplattingResidualEncoder` receives parameters `gate_router_hidden` and `gate_window_half` and instantiates `AdaptiveResidualRouter`.
2.  **Router**: `AdaptiveResidualRouter` processes the residual projection `res_proj` to compute local Z-score Mahalanobis distance over a temporal sliding window.
3.  **Feature Fusion**: The Mahalanobis distance is concatenated with `z_density`, `z_std`, `z_diff`, and `z_energy` to form a 5D feature map.
4.  **MLP Routing Network**: A 2-layer MLP (hidden dimension defined by `gate_router_hidden`) projects the 5D feature to 3 logits representing (Neutral, Forward, Reverse) routing pathways.

---

## 2. Mathematical Formulation

Let $x_t \in \mathbb{R}^{d\_model}$ be the Z-score standardized residual projection of patch $t$ in a sequence of $P$ patches.
The sliding window neighborhood is defined as $\mathcal{N}(t) = \{ t - W, \dots, t + W \} \setminus \{t\}$, where $W$ is the half-window size parameter `gate_window_half`.

### Step 1: Local Feature Variance (Denominator Detachment)
To prevent gradient explosion when the local window is extremely flat, the local feature variance is computed with detached gradients:
$$\sigma^2_{t, d} = \text{Var}\Big( \{ x_{j, d} \Big\}_{j \in \mathcal{N}(t)} \Big).\text{detach}() + \epsilon \quad (\epsilon = 1e-5)$$

### Step 2: Expected Local Mahalanobis Distance
$$\bar{d}_G^2(t) = \frac{1}{2W} \sum_{j \in \mathcal{N}(t)} \sum_{d=1}^{d\_model} \frac{(x_{j, d} - x_{t, d})^2}{\sigma^2_{t, d}}$$
Boundary conditions are handled by mirroring/replication padding at the sequence edges to ensure $2W$ neighbors are always present.

### Step 3: MLP Routing Network
The router input feature vector $F_t$ is constructed as:
$$F_t = \Big[ z_{\text{density}}(t), \, z_{\text{std}}(t), \, z_{\text{diff}}(t), \, z_{\text{energy}}(t), \, \bar{d}_G^2(t) \Big]$$
The Softmax route logits are computed as:
$$\text{logits}_t = W_2 \cdot \text{GELU}(W_1 \cdot F_t + b_1) + b_2$$
Where $W_1 \in \mathbb{R}^{\text{hidden} \times 5}$ and $W_2 \in \mathbb{R}^{3 \times \text{hidden}}$, with $\text{hidden}$ being the `gate_router_hidden` hyperparameter.

---

## 3. Interface and Configuration Changes

### CLI Arguments (`run.py` & Tuning Scripts)
*   Add `--gate_window_half` (type `int`, default `2`, choices `[1, 2, 3, 4]`).
*   Retrieve and pass `gate_window_half` to the model config.

### Optuna HPO Search Bounds (`tune_gate_*.py`)
*   Add bounds: `"gate_window_half_lower": 1`, `"gate_window_half_upper": 4`.
*   Suggest parameter:
    `gate_window_half = trial.suggest_int("gate_window_half", bounds["gate_window_half_lower"], bounds["gate_window_half_upper"])`
*   Pass `gate_window_half` parameter in `build_command`.
*   In `save_best_callback`, append the optimal `gate_window_half` to the best JSON configuration.

---

## 4. Verification Plan

### Dry-run Command
Run a 1-epoch execution on `ETTh1` with local sliding window $W=2$:
`python run.py --task_name long_term_forecast --is_training 1 --root_path ./data --data_path ETTh1.csv --model_id verify_lga --model yzy --data ETTh1 --features M --seq_len 512 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 --d_model 128 --d_ff 256 --itr 1 --batch_size 128 --representation pure_gs --num_gaussians 10 --density_mode cas --use_residual --gs_residual_weight 0.4441 --gate_type adaptive_direction --gate_beta 0.4589 --gate_router_hidden 16 --gate_window_half 2 --train_epochs 1 --num_workers 0`

Verify that:
1.  The code runs and training/validation loss decreases without `NaN` gradients.
2.  Diagnostics are logged correctly in `loss_cas_simplify/diagnostics_verify_lga.txt`.
