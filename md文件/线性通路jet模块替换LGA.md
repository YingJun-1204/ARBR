## 结论：你这次 HPO 实际运行的是 `gaussian_jet`，没有使用 LGA

你执行的是：

```bash
python .\run_adaptive_tuning.py \
  --batch_size 256 \
  --batches 1 \
  --trials_per_batch 120 \
  --datasets etth1 etth2 ettm1 ettm2 weather \
  --output_dir loss_mdagV13_256 \
  --pred_len 96 192 336 720
```

完整调用链是：

```text
run_adaptive_tuning.py
        ↓
tune_gate_etth1.py / tune_gate_etth2.py / ...
        ↓
run.py
        ↓
Model → SplattingResidualEncoder
```

我检查了你本次上传的全部 HPO 脚本：

- `run_adaptive_tuning.py` 没有 `residual_mode` 参数；
- 五个 `tune_gate_*.py` 都没有向 `run.py` 传入 `--residual_mode`；
- `run.py` 中默认值明确是：

```python
parser.add_argument(
    "--residual_mode",
    type=str,
    default="gaussian_jet",
    choices=["legacy_dynamic_shell", "gaussian_jet"],
)
```



而且 `apply_variant_configs()` 只保证 `representation` 存在，并不会修改 `residual_mode`：

```python
def apply_variant_configs(args):
    if not hasattr(args, "representation"):
        args.representation = "gs"
    return args
```



因此，这次实验的最终配置是：

```text
residual_mode = gaussian_jet
```

不是：

```text
residual_mode = legacy_dynamic_shell
```

------

# 1. 为什么可以确定没有使用 LGA？

在 `SplattingResidualEncoder` 中，LGA 所属的 `AdaptiveResidualRouter` 只在下面的条件成立时实例化：

```python
if self.use_residual and self.residual_mode == "legacy_dynamic_shell":
    self.residual_router = AdaptiveResidualRouter(
        d_model=d_model,
        gate_type=gate_type,
        gate_beta=gate_beta,
        gate_window_half=gate_window_half
    )
else:
    self.residual_router = None
```

你运行的是 `gaussian_jet`，所以实际结果是：

```python
self.residual_router = None
```

同时，残差投影会选择：

```python
self.patch_residual = GaussianJetProjection(...)
```

而不是旧版的：

```python
self.patch_residual = PatchResidualProjection(...)
```



在前向传播中，Gaussian Jet 分支结束时还明确写了：

```python
self.last_raw_lga = None
self.last_z_lga = None
self.last_energy = None
```

也就是说：

- 没有计算局部 Mahalanobis 距离；
- 没有产生 `z_lga`；
- 没有调用 `AdaptiveResidualRouter`；
- `gate_window_half` 没有进入任何有效运算。

所以答案是确定的：

> **你这批 HPO 没有使用 LGA。**

------

# 2. 但调参脚本确实在搜索 `gate_window_half`

你的各数据集 HPO 脚本都在执行：

```python
params["gate_window_half"] = trial.suggest_int(
    "gate_window_half",
    1,
    4
)
```

随后又传给了 `run.py`：

```python
cmd.extend([
    "--gate_window_half",
    str(params.get("gate_window_half", 2))
])
```

例如 ETTh1 脚本就是这样构造命令的。

问题在于：

```text
Optuna 采样了 gate_window_half
        ↓
命令行传给 run.py
        ↓
run.py 保存到 args
        ↓
模型接收到 gate_window_half
        ↓
但 gaussian_jet 分支不读取它
```

所以它属于：

> **被采样、被记录、被保存，但不影响模型输出的无效 HPO 参数。**

这也解释了为什么五个数据集、不同 horizon 的最优 `gate_window_half` 看起来毫无稳定规律：它只是某个最优 trial 附带的随机取值，而不是性能真正选择出来的窗口。

------

# 3. `legacy_dynamic_shell` 是什么？

`legacy_dynamic_shell` 是你旧版的动态高斯残差路径，结构大致为：

```text
显式 Gaussian Splatting
        ↓
提取显式高斯密度、特征变化、残差能量等状态
        ↓
LGA 检测局部几何偏离
        ↓
AdaptiveResidualRouter 判断
neutral / forward / reverse
        ↓
生成 gate_direction 和 gate_scale
        ↓
动态移动隐式高斯坐标并调制线性投影
        ↓
与显式高斯特征融合
```

它由两个主要组件组成：

1. `AdaptiveResidualRouter`
2. `PatchResidualProjection`

------

## 3.1 LGA 在旧模式中的作用

旧版路由器先从残差投影中得到一维局部表示：

```python
res_proj_1d = torch.mean(
    res_proj,
    dim=-1,
    keepdim=True
)
```

然后使用：

```python
W = self.gate_window_half
```

取当前位置前后各 (W) 个 patch，计算局部方差以及 Mahalanobis 式几何距离：

[
d_t^2 =
\frac{1}{2W}
\sum_{j\in\mathcal N_t}
\frac{(z_j-z_t)^2}
{\widetilde{\sigma}_t^2+\epsilon}.
]

得到 `z_lga` 后，它会和：

- `z_diff`：特征变化；
- `z_energy`：残差能量；
- `z_density`：显式高斯覆盖密度；
- `z_std`：显式特征标准差；

一起进入路由器。

因此，`gate_window_half` 只对这个 LGA 计算有效。

------

## 3.2 旧模式如何决定方向？

路由器计算：

```python
u_t = torch.sigmoid(self.u_proj(u_in))
d_t = torch.sigmoid(self.d_proj(d_in))
```

然后构造三个概率：

[
p_{\mathrm{neutral}}=1-u_t,
]

[
p_{\mathrm{forward}}=u_t(1-d_t),
]

[
p_{\mathrm{reverse}}=u_td_t.
]

方向定义为：

# [ \mathrm{direction}

## p_{\mathrm{forward}}

p_{\mathrm{reverse}}.
]

再利用 `gate_beta` 调节残差幅度：

[
s_t =
\operatorname{clip}
\left(
1+\beta\cdot\mathrm{direction}_t,
0.5,1.5
\right).
]

所以在 `legacy_dynamic_shell` 中：

- `gate_window_half`：控制 LGA 邻域；
- `gate_beta`：控制动态残差缩放幅度；
- `gate_lambda`：约束三路概率的熵；
- `gate_type=adaptive_direction`：启用完整路由机制。

------

## 3.3 旧模式怎样修正隐式分支？

`PatchResidualProjection` 根据路由方向移动局部时间坐标：

```python
delta_t = gate_direction * self.shift_scaler
t_queries = t_base - delta_t.unsqueeze(-1)
```

然后围绕可学习的隐式高斯中心构造高斯权重：

# [ G_k(t-\Delta t)

\exp
\left[
-\frac{(t-\Delta t-\mu_k)^2}{2\sigma_k^2}
\right].
]

这些高斯权重形成动态调制掩码，作用于基础线性投影权重。

最终融合形式近似为：

[
Z =
Z_{\mathrm{GS}}
+
\lambda_r
\cdot s_t
\cdot Z_{\mathrm{shell}}.
]

这就是“dynamic shell”这个名字的来源：隐式高斯壳会根据路由方向动态平移和调制。

------

# 4. `gaussian_jet` 是什么？

Gaussian Jet 是现在实际运行的路径。

它不再经过 LGA 和三路可学习路由器，而是直接从显式高斯参数推导局部平移信息：

```text
显式 Gaussian Splatting 输出
μ、σ、α
        ↓
根据高斯中心和当前 patch 位置计算 score shift
        ↓
得到 delta 和 confidence
        ↓
delta 作用于 Gaussian Jet 一阶导数基底
        ↓
形成隐式高斯残差
        ↓
与显式高斯表示融合
```

------

## 4.1 平移方向从哪里来？

Gaussian Jet 使用显式高斯的：

- 中心 (\mu_i)；
  -尺度 (\sigma_i)；
- 有效强度 (\alpha_i)。

对每个 patch 位置 (t)，计算：

# [ w_i(t)

\alpha_i
\exp
\left(
-\frac{(t-\mu_i)^2}{2\sigma_i^2}
\right).
]

同时用精度：

[
\pi_i=\frac{1}{\sigma_i^2+\epsilon}
]

加权，得到平均位移：

# [ \bar{\Delta}_t

\frac{
\sum_i
w_i(t)\pi_i(\mu_i-t)
}{
\sum_i w_i(t)\pi_i+\epsilon
}.
]

再将其限制在最大允许位移内：

# [ \delta_t

\Delta_{\max}
\tanh
\left(
\frac{\tau\bar{\Delta}*t}
{\Delta*{\max}+\epsilon}
\right).
]

覆盖置信度则是：

[
c_t=
\frac{\sum_iw_i(t)}
{\sum_iw_i(t)+\tau_d}.
]

这个过程完全没有 `gate_window_half`，也没有 LGA。

------

## 4.2 Gaussian Jet 如何表示平移？

Gaussian Jet 构造两类基函数。

零阶高斯基：

# [ \phi_k(t)

\exp
\left(
-\frac{(t-\mu_k)^2}{2\sigma_k^2}
\right).
]

一阶导数基：

[
\psi_k(t)
\propto
\frac{t-\mu_k}{\sigma_k^2}
\phi_k(t).
]

代码中：

```python
q0 = torch.einsum("npl,kl->npk", patches, phi)
q1 = torch.einsum("npl,kl->npk", patches, psi)

q_jet = q0 + delta * q1
```

这里本质上利用了一阶近似：

[
G(t-\delta)
\approx
G(t)
+
\delta G'(t).
]

因此：

- (q_0) 描述局部高斯内容；
- (q_1) 描述对平移方向的响应；
- (\delta q_1) 对局部平移进行一阶修正。

最终输出：

# [ Z_{\mathrm{jet}}

Z_{\mathrm{base}}
+
s_{\mathrm{jet}}
\cdot c_t
\cdot
Z_{\mathrm{GaussianJet}}.
]

代码对应：

```python
return base + scale * confidence * jet
```



------

# 5. 两种模式的核心区别

| 比较项                              | `legacy_dynamic_shell`                | `gaussian_jet`                              |
| ----------------------------------- | ------------------------------------- | ------------------------------------------- |
| 是否使用 LGA                        | 是                                    | 否                                          |
| 是否实例化 `AdaptiveResidualRouter` | 是                                    | 否                                          |
| 平移方向来源                        | LGA、残差能量、显式密度等经过学习路由 | 直接由显式高斯 (\mu,\sigma,\alpha) 解析计算 |
| 路由形式                            | neutral / forward / reverse           | 连续位移 (\delta) 和置信度                  |
| 隐式表示                            | 动态高斯掩码调制线性权重              | 零阶高斯基 + 一阶导数基                     |
| `gate_window_half`                  | 有效                                  | 无效                                        |
| `gate_beta`                         | 有效                                  | 无效                                        |
| `gate_lambda`                       | 有效                                  | 仍可能有效                                  |
| 方向是否依赖局部邻域                | 是                                    | 不直接依赖 LGA 邻域                         |
| 机制复杂度                          | 较高                                  | 更简洁                                      |
| 理论形式                            | 学习式离散方向路由                    | 显式高斯引导的一阶连续平移修正              |

------

# 6. 你当前 HPO 中哪些参数实际有效？

这是现在最值得注意的部分。

## `gate_window_half`：无效

只在 LGA 中使用，而你没有运行 LGA。

因此：

```text
gate_window_half = 1 / 2 / 3 / 4
```

不会改变 Gaussian Jet 模型。

------

## `gate_beta`：无效

`gate_beta` 只在 `AdaptiveResidualRouter` 中用于：

```python
scale = torch.clamp(
    1.0 + self.gate_beta * direction,
    0.5,
    1.5
)
```

Gaussian Jet 分支没有读取 `self.gate_beta`，而是：

```python
self.last_gate_scale = torch.ones_like(confidence)
```

所以本次 HPO 搜索的 `gate_beta` 也是无效维度。

------

## `gate_type=adaptive_direction`：不再控制模型路由，但会控制损失逻辑

Gaussian Jet 分支不根据 `gate_type` 选择方向，它始终从显式高斯计算 `delta`。

不过训练脚本中有：

```python
if getattr(self.args, "gate_type", None) == "adaptive_direction":
    ...
    loss = loss + gate_lambda_eff * entropy
```

因此 `gate_type=adaptive_direction` 会启用熵损失，但它不再启用 LGA 路由器。

------

## `gate_lambda`：仍然有效

Gaussian Jet 虽然没有真正的 LGA 路由概率，但代码根据：

- `confidence`；
- 正向位移；
- 反向位移；

构造了兼容性的三路概率：

```python
neutral = 1.0 - confidence
forward_dir = confidence * relu(direction)
reverse_dir = confidence * relu(-direction)
```

再形成：

```python
self.last_gate_route_probs
```

训练代码仍然对这些概率计算熵惩罚。

所以：

> `gate_lambda` 在当前 Gaussian Jet 下仍会影响训练损失，但它约束的是由 Gaussian Jet 几何量构造的概率，而不是 LGA 路由器输出。

------

## 其他有效 HPO 参数

当前实际有效的主要参数包括：

- `learning_rate`
- `gs_dropout`
- `head_dropout`
- `gs_residual_weight`
- `gate_lambda`
- 以及固定的 Gaussian Jet 参数默认值

当前没有被 HPO 搜索、但真正控制 Gaussian Jet 的参数包括：

- `num_implicit_gaussians`
- `jet_max_shift_samples`
- `jet_score_temperature`
- `jet_density_tau`
- `jet_scale_init`
- `jet_sigma_init`

------

# 7. 这批 HPO 结果是否还能用？

可以用，但需要区分两个层面。

## 模型性能结果仍然有效

`gate_window_half` 和 `gate_beta` 没有参与模型计算，不会破坏训练，也不会让结果错误。

每个最优 trial 仍然是由以下有效参数决定的：

- 学习率；
- dropout；
- 残差权重；
- gate entropy 权重等。

所以已有 MSE/MAE 不需要直接作废。

------

## 但“最优 gate_window_half”没有意义

你不能在论文中写：

> ETTh1 的最优窗口是 4，ETTh2 的最优窗口是 1。

因为这些值没有真正作用于模型。

同理，也不能把各数据集的“最优 `gate_beta`”解释成模型对路由强度的偏好。

它们只是 Optuna trial 中附带保存的死参数。

------

# 8. 建议你马上修改调参脚本

对于当前 `gaussian_jet` 主模型，应从 HPO 中删除：

```python
gate_window_half
gate_beta
```

`sample_params()` 中删除：

```python
gate_beta = trial.suggest_float(...)
```

以及：

```python
params["gate_window_half"] = trial.suggest_int(...)
```

构建命令时也删除：

```python
"--gate_beta", ...
"--gate_window_half", ...
```

并建议在命令中显式写明：

```python
"--residual_mode",
"gaussian_jet",
```

虽然它目前已经是默认值，但显式传入更安全，避免未来修改 `run.py` 默认值后实验悄悄改变。

还可以在 `run.py` 启动时打印：

```python
print(f"Residual mode: {args.residual_mode}")
print(f"Gate type: {args.gate_type}")
```

------

# 9. 论文机制命名也需要同步检查

你当前输出目录叫：

```text
loss_mdagV13_256
```

调参描述里也一直使用：

```text
Adaptive Gating
router
LGA
MDAG
```

但你实际模型已经不是 LGA-based adaptive router，而是：

> **explicit Gaussian-guided continuous Gaussian Jet correction**

如果论文仍把当前模型描述为“LGA 检测局部偏离，再通过 MDAG 选择 forward/reverse”，那会和实际代码不一致。

当前代码更准确的叙事是：

1. 显式 Gaussian Splatting 建模全局平滑结构；
2. 显式高斯的中心、尺度和强度解析地产生局部 score shift；
3. 隐式 Gaussian Jet 使用零阶与一阶高斯基进行连续平移修正；
4. 显隐式两条高斯路径协同完成抗平移表征。

所以，从当前代码来看：

> `legacy_dynamic_shell` 是旧版 LGA/MDAG 路由机制；`gaussian_jet` 是你现在真正运行的、由显式高斯直接引导隐式高斯导数修正的新机制。

最关键的清理动作是：**从 Gaussian Jet 的 HPO 空间中删除 `gate_window_half` 和 `gate_beta`，并重新审视论文中是否还应继续使用 LGA/Adaptive Router 的叙事。**