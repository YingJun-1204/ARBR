这一次的目标需要明确调整为：

> **让模型在现有平移测试脚本的扰动方式下获得稳定、可量化的低退化性能，而不只是在线性通路中形式化地加入一个高斯模块。**

当前测试更接近“整段输入右移并进行边界复制”，而现有增强主要是局部片段平移，两者并不一致；旧残差实现还保留动态权重和双 Pass，现有 HPO 也仍然在搜索旧路由参数。

建议 Antigravity 按以下规划执行。

# Splat-Guided Gaussian Canonicalization 实现规划

## 一、修改背景

当前模型虽然尝试在线性通路中加入高斯平移适配器，但在现有平移测试脚本下，抗平移效果仍不理想，主要存在以下问题。

### 1. 训练扰动与测试扰动不一致

当前已有的 `local_shift_augment()` 只随机移动输入中的局部片段，并为每个 patch 生成局部位移标签。

而现有平移测试脚本采用的是整段输入平移，即所有时间位置同时向右移动若干步，并用左侧边界值填充。

因此，目前模型即使使用局部平移增强，也没有真正学习测试脚本中的全局滞后扰动。

------

### 2. 当前线性通路仍可能运行旧动态高斯壳

目前可见的 `patch_residual.py` 仍然显式构造：

```python
M_modulator  # [B*C, P, D, L]
W_dynamic    # [B*C, P, D, L]
```

并继续使用 `shift_scaler` 和动态高斯 mask。

编码器中也仍然存在：

```python
get_base_projection()
AdaptiveResidualRouter
第二次动态投影
```

即旧的双 Pass 计算路径。

虽然顶层 `gs_linear.py` 已经传递了 `residual_mode="gaussian_jet"` 等新参数，但底层代码需要确认是否真正接入了对应实现。

------

### 3. 一阶 Gaussian Jet 不适合测试中的较大平移

一阶展开：

[
\phi(t-\delta)
\approx
\phi(t)+\delta\phi'(t)
]

只适合一至两个采样点的小平移。

当测试平移为 4、8，甚至 16 步时，一阶近似误差会快速增大。继续提高 Jet 阶数虽然可以缓解，但不能解决跨 patch 数据迁移以及序列边界信息丢失问题。

------

### 4. Gaussian score 不能直接等同于真实平移量

显式高斯密度的 score 或 mean-shift 表示查询位置朝局部密度峰移动的方向，但它不天然等于输入相对原始序列发生的真实位移。

因此不能直接令：

# [ \hat{s}

\operatorname{GaussianScore}
]

而应把显式高斯几何作为先验特征，再通过一个轻量校准器预测真实位移。

------

### 5. 当前预测头仍绑定绝对 patch 位置

当前 `FlattenHead` 将全部 patch 表示直接展平后输入静态线性层。

这种结构对残余 patch 位置偏移较敏感。即使输入端已经进行了大部分校正，只要仍有少量跨 patch 偏差，预测头就可能放大这种误差。

------

## 二、最终敲定方案

最终建议采用：

# Splat-Guided Gaussian Canonicalization

中文名称：

> **显式高斯引导的隐式高斯规范化**

模块简称可以使用：

```text
SGGC
```

核心流程为：

```text
平移后的输入
    │
    ├─ 显式高斯参数探测
    │       │
    │       └─ 高斯几何特征
    │
    ├─ 轻量位移校准器
    │       │
    │       └─ 预测全局位移和置信度
    │
    ├─ 隐式 Gaussian Transport
    │       │
    │       └─ 将完整序列反向运输到规范时间坐标
    │
    ├─ 边界缺失补全
    │
    └─ 在规范化序列上执行
            ├─ 显式 Gaussian Splatting
            └─ 普通线性 patch 投影
```

该方案不再依赖一阶 Taylor 展开完成主要平移校正，而是直接在完整输入序列上执行精确的可微高斯运输。

显式通路负责提供几何先验，隐式高斯负责完成连续时间运输，两者形成真正的协作关系。

------

# 三、整体实现思路

## 3.1 训练增强必须与测试脚本严格对齐

新增整段输入平移增强：

```python
global_shift_augment()
```

扰动方式必须与测试脚本保持一致。

当平移量 (s>0) 时：

[
x^{(s)}*t=
\begin{cases}
x_0, & t<s,\
x*{t-s}, & t\geq s.
\end{cases}
]

对应代码逻辑：

```python
x_shifted[:, s:, :] = x_clean[:, :-s, :]
x_shifted[:, :s, :] = x_clean[:, :1, :]
```

增强函数应返回：

```python
x_shifted
shift_target_samples
shift_mask
valid_transport_mask
```

其中：

```python
shift_target_samples.shape == [B, 1]
```

目标直接使用采样点单位，例如：

```text
0、1、2、4、8、16
```

不要在增强阶段转换成：

```python
s / (patch_len - 1)
```

也不要使用含义不清晰的 `correction_shift`。

统一规定：

> `shift_target_samples > 0` 表示原始内容向右移动了对应采样点数。

Gaussian Transport 在恢复时应从：

```python
source_position = target_position + predicted_shift
```

进行采样。

------

## 3.2 使用混合平移训练分布

建议默认训练位移集合：

```python
train_shift_values = [0, 1, 2, 4, 8]
```

如果测试脚本包含 16 步，则加入：

```python
train_shift_values = [0, 1, 2, 4, 8, 16]
```

建议采样概率：

```text
shift 0:  30%
shift 1:  15%
shift 2:  15%
shift 4:  15%
shift 8:  20%
shift 16: 5%
```

这样不会让模型只针对大平移训练，同时能够覆盖测试脚本的全部扰动。

推荐使用 curriculum：

```text
Epoch 1–5:    [0, 1, 2]
Epoch 6–15:   [0, 1, 2, 4]
Epoch 16+:    [0, 1, 2, 4, 8]
最后 20% epoch: 少量加入 16
```

局部平移增强可以保留，但应作为次要增强：

```text
global shift probability = 0.6
local shift probability = 0.1～0.2
```

------

## 3.3 显式高斯参数探测

对输入 `x_norm` 先执行一次轻量的显式高斯参数探测，得到：

```python
mu_probe
sigma_probe
alpha_probe
```

该探测仅用于生成位移估计特征，不直接作为最终显式通路输出。

第一版可以复用现有 Gaussian generator，但探测 Pass 使用：

```python
with torch.no_grad():
    probe_params = gaussian_splatting.extract_parameters(x_flat)
```

这样：

- 不保存探测 Pass 的反向图；
- 不显著增加显存；
- 位移校准器仍可正常训练；
- 最终显式通路在规范化输入上只进行一次有梯度的正式计算。

从探测参数提取以下几何特征：

```python
weighted_mu
weighted_mu_var
weighted_sigma
left_mass
right_mass
mass_imbalance
total_alpha
```

例如：

# [ \bar{\mu}

\frac{\sum_k\alpha_k\mu_k}
{\sum_k\alpha_k+\epsilon}
]

# [ v_\mu

\frac{
\sum_k\alpha_k(\mu_k-\bar{\mu})^2
}{
\sum_k\alpha_k+\epsilon
}
]

# [ r_{\mathrm{imbalance}}

\frac{
\rho_{\mathrm{right}}-\rho_{\mathrm{left}}
}{
\rho_{\mathrm{right}}+\rho_{\mathrm{left}}+\epsilon
}
]

------

## 3.4 增加轻量位移校准器

新增：

```python
class SplatGuidedShiftEstimator(nn.Module):
```

它不直接把 Gaussian score 当成位移，而是融合：

1. 显式高斯几何特征；
2. 输入的轻量时序特征；
3. 左侧复制边界特征；
4. 局部一阶差分特征。

建议输入时序特征使用：

```python
channel_mean = x_norm.mean(dim=-1)
channel_std = x_norm.std(dim=-1)
```

得到：

```python
temporal_summary.shape == [B, 2, T]
```

再经过轻量 Conv1d：

```python
Conv1d(2, 16, kernel_size=5, padding=2)
GELU
Conv1d(16, 16, kernel_size=5, padding=2)
AdaptiveAvgPool1d(1)
```

与显式高斯几何特征拼接后，通过小 MLP 输出：

```python
predicted_shift_samples  # [B, 1]
shift_confidence         # [B, 1]
```

形式为：

# [ \hat{s}

s_{\max}
\tanh(h_\theta(f_{\mathrm{temporal}},f_{\mathrm{GS}}))
]

# [ c

\sigma(g_\theta(f_{\mathrm{temporal}},f_{\mathrm{GS}}))
]

其中：

```python
max_shift_samples = max(train_shift_values)
```

位移预测必须使用采样点单位，而不是归一化时间单位。

------

## 3.5 在完整序列上执行 Gaussian Transport

这是本次修改最重要的部分。

不能再只在单个 patch 内移动高斯基底，因为测试中的全局平移会造成数据跨 patch 边界移动。

Gaussian Transport 必须在 patch 提取之前作用于完整序列：

```python
x_norm  # [B, T, C]
```

对于目标位置 (t)，根据预测位移 (\hat{s}) 计算源位置：

[
u_t=t+\hat{s}
]

再使用归一化 Gaussian interpolation：

# [ \widetilde{x}_t

\frac{
\sum_{j\in\mathcal N(u_t)}
\exp\left(
-\frac{(j-u_t)^2}{2\tau^2}
\right)x_j
}{
\sum_{j\in\mathcal N(u_t)}
\exp\left(
-\frac{(j-u_t)^2}{2\tau^2}
\right)+\epsilon
}
]

建议：

```text
transport_radius = 3
transport_sigma = 0.65～0.90
```

每个目标位置只访问附近：

```text
2 × radius + 1
```

个采样点。

复杂度为：

[
O(BCTR)
]

其中 (R) 是小常数，不包含 `d_model`，也不会出现：

```text
[B*C, P, D, L]
```

动态权重张量。

------

## 3.6 置信度控制的恒等退化

为了避免 clean 输入被错误移动，使用：

# [ x_{\mathrm{canonical}}

c\cdot
\mathcal T_{\hat{s}}^G(x)
+
(1-c)\cdot x
]

其中 (c) 是 shift confidence。

对于 clean 输入：

```text
predicted_shift ≈ 0
confidence ≈ 0
```

模型退化为恒等输入。

对于明显平移输入：

```text
confidence → 1
```

主要使用 Gaussian Transport 后的序列。

------

## 3.7 增加边界缺失补全

现有测试脚本将输入向右移动时，序列最后 (s) 个真实观测会丢失。

单纯的反向运输无法恢复这些不存在于输入中的值，因此需要增加一个轻量边界补全器：

```python
class GaussianBoundaryCompleter(nn.Module):
```

建议使用共享线性自回归结构：

```python
context_len = 32
max_shift_samples = 16

tail_predictor = nn.Sequential(
    nn.Linear(context_len, 64),
    nn.GELU(),
    nn.Linear(64, max_shift_samples),
)
```

按每个变量独立使用：

```text
输入：运输后最后 context_len 个有效位置
输出：最多 max_shift_samples 个尾部补全值
```

根据预测位移，只替换运输结果中无有效源坐标的位置。

例如预测位移为 8，则只使用前 8 个补全值。

边界补全器必须通过训练增强中的 clean 序列进行直接监督。

------

## 3.8 在规范化序列上重新运行双通路

得到：

```python
x_canonical
```

之后，模型的正式双路都使用该输入。

### 显式通路

```python
rendered_event = gaussian_splatting(x_canonical)
```

这样最终的：

```text
mu、sigma、alpha、c
```

来自规范坐标，而不是平移后的原始坐标。

### 隐式线性通路

先从 `x_canonical` 提取 patch：

```python
patches = unfold(x_canonical)
```

再执行一次普通线性投影：

```python
res_proj = base_projection(patches)
```

隐式高斯思想体现在输入进入 Linear 之前已经通过动态 Gaussian Transport 回到规范坐标。

不再构造：

```python
M_modulator
W_dynamic
g_coeff
shift_scaler
```

也不再执行第一次基础投影和旧路由器。

------

## 3.9 预测头前增加轻量 Gaussian Patch Mixer

为了抑制运输后剩余的少量 patch 索引偏差，在 `FlattenHead` 前加入一个低成本的 patch 轴高斯混合层：

```python
class GaussianPatchMixer(nn.Module):
```

输入：

```python
[B*C, d_model, patch_num]
```

使用固定或半可学习的 Gaussian kernel：

```text
kernel_size = 3
sigma = 0.8
```

沿 patch 维进行 depthwise Conv1d：

```python
smoothed = depthwise_gaussian_conv(enc_out)

eta = sigmoid(mixer_logit)

enc_out = enc_out + eta * (smoothed - enc_out)
```

建议：

```text
eta 初始值 = 0.1
```

这样模型初始接近原始 `FlattenHead`，但能够降低少量跨 patch 漂移的影响。

当前 `FlattenHead` 的输入输出接口不变。

------

# 四、训练损失设计

最终训练目标建议为：

# [ L

L_{\mathrm{forecast}}
+
\lambda_sL_{\mathrm{shift}}
+
\lambda_cL_{\mathrm{canonical}}
+
\lambda_bL_{\mathrm{boundary}}
+
\lambda_pL_{\mathrm{consistency}}
]

## 4.1 预测损失

对 clean 和 shifted 样本都使用相同真实未来：

# [ L_{\mathrm{forecast}}

\operatorname{MSE}(\hat y,y)
]

一个 batch 中直接混合 clean 与 shifted 样本，保持一次主 forward。

------

## 4.2 位移监督损失

使用采样点单位监督预测位移：

# [ L_{\mathrm{shift}}

\operatorname{SmoothL1}(\hat{s},s)
]

只对增强样本计算。

同时增加 shift presence BCE：

# [ L_{\mathrm{presence}}

\operatorname{BCE}(c,\mathbb 1[s\neq0])
]

可以合并到 `L_shift` 中。

------

## 4.3 规范化重建损失

因为训练时同时拥有 clean 输入和 shifted 输入，可以直接监督运输结果：

# [ L_{\mathrm{canonical}}

\frac{
\sum_t m_t
\left|
x^{\mathrm{canonical}}_t-x^{\mathrm{clean}}_t
\right|^2
}{
\sum_tm_t+\epsilon
}
]

其中 (m_t) 表示源位置仍然有效的部分。

这个损失比单纯依赖最终 forecasting MSE 更直接地训练 Gaussian Transport。

------

## 4.4 边界补全损失

对因右移丢失的最后 (s) 个位置计算：

# [ L_{\mathrm{boundary}}

\frac{
\sum_t(1-m_t)
\left|
x^{\mathrm{canonical}}_t-x^{\mathrm{clean}}_t
\right|^2
}{
\sum_t(1-m_t)+\epsilon
}
]

它专门训练尾部补全器。

------

## 4.5 预测一致性损失

每隔若干 step，在少量样本上同时计算 clean 与 shifted 预测：

# [ L_{\mathrm{consistency}}

## \left| f(x^{\mathrm{shift}})

\operatorname{sg}[f(x^{\mathrm{clean}})]
\right|_2^2
]

其中 `sg` 表示 stop-gradient。

为了控制训练时间：

```text
consistency_interval = 4
consistency_batch_ratio = 0.25
```

即每四个 step，只抽取四分之一个 batch 做额外 clean forward。

平均额外计算约为 6.25%。

------

## 4.6 推荐初始权重

```text
forecast_weight       = 1.0
shift_loss_weight     = 0.10
presence_loss_weight  = 0.02
canonical_loss_weight = 0.05
boundary_loss_weight  = 0.10
consistency_weight    = 0.05
```

训练前 3 个 epoch 可以使用：

```text
shift_loss_weight     = 0.20
canonical_loss_weight = 0.10
boundary_loss_weight  = 0.20
```

先让规范化模块学会工作，再逐步恢复正式权重。

------

# 五、具体文件修改规划

## 5.1 `utils/augmentations.py`

### 保留

保留现有：

```python
local_shift_augment()
```

它仍可用于局部鲁棒性实验。现有函数会返回 patch 级 `shift_target` 和 `shift_mask`。

### 新增

新增：

```python
def global_shift_augment(
    batch_x,
    shift_values,
    probability,
    positive_only=True,
    shared_across_channels=True,
    boundary_mode="replicate",
):
```

返回：

```python
batch_x_shifted
shift_target_samples
shift_presence_target
valid_transport_mask
aug_info
```

### 重要要求

- 平移逻辑必须和测试脚本完全相同；
- target 使用采样点单位；
- 明确正号表示内容向右移动；
- 不再使用含义模糊的 `correction_shift`；
- 支持每个样本独立采样 shift；
- 默认所有通道共享同一 shift；
- `shift=0` 必须保留一部分样本。

------

## 5.2 `layers/gaussian_splatting.py`

### 修改目的

支持无梯度参数探测和规范化输入上的正式渲染。

### 将现有 forward 拆为

```python
def extract_parameters(
    self,
    x_flat,
    gate_effective=None,
):
```

返回：

```python
mu
sigma
alpha
alpha_effective
c
```

新增：

```python
def render_from_parameters(
    self,
    mu,
    sigma,
    alpha_effective,
    c,
    patch_num,
    device,
):
```

返回：

```python
rendered_event
weights
last_density_score
```

最终 `forward()` 仍然组合调用这两个方法，保持外部兼容。

------

## 5.3 新增 `layers/gaussian_canonicalizer.py`

建议新建文件，避免继续把所有逻辑堆在 `patch_residual.py` 中。

包含三个类。

### 类一

```python
class SplatGuidedShiftEstimator(nn.Module):
```

负责：

- 读取 probe Gaussian 参数；
- 计算显式几何描述；
- 提取轻量 temporal Conv 特征；
- 输出 `predicted_shift_samples`；
- 输出 `shift_confidence`。

### 类二

```python
class GaussianSequenceTransport(nn.Module):
```

负责：

- 根据预测位移生成源坐标；
- 使用局部 Gaussian interpolation；
- 返回运输序列；
- 返回 valid mask；
- 不生成与 `d_model` 相关的高维动态张量。

### 类三

```python
class GaussianBoundaryCompleter(nn.Module):
```

负责：

- 根据运输后的有效历史；
- 预测最多 `max_shift_samples` 个尾部点；
- 仅替换 invalid tail；
- 输出补全后的 canonical sequence。

### 包装类

```python
class SplatGuidedGaussianCanonicalizer(nn.Module):
```

统一执行：

```python
probe geometry
→ shift estimation
→ Gaussian transport
→ boundary completion
→ confidence blending
```

返回：

```python
x_canonical
aux_dict
```

`aux_dict` 包含：

```python
predicted_shift_samples
shift_confidence
transported_x
valid_mask
completed_tail
geometry_features
```

------

## 5.4 `layers/patch_residual.py`

### 删除旧逻辑

删除或移动到 legacy 类：

```python
g_coeff
g_mu
g_sigma
shift_scaler
get_base_projection
M_modulator
W_dynamic
```

当前这些逻辑仍会生成高维动态矩阵。

### 新主类

```python
class GaussianTransportProjection(nn.Module):
```

它只负责：

```python
padding
patch extraction
base_projection
```

输入已经是 canonical sequence：

```python
def forward(self, x_canonical):
    patches = extract_patches(x_canonical)
    return self.base_projection(patches)
```

只执行一次 Linear。

### Legacy 兼容

旧类重命名为：

```python
class LegacyDynamicGaussianProjection(nn.Module):
```

仅用于消融，不作为默认路径。

------

## 5.5 `layers/splatting_residual_encoder.py`

### 删除默认主路径中的旧路由流程

Gaussian canonicalization 模式下不得调用：

```python
AdaptiveResidualRouter
get_base_projection
gate_direction
gate_scale
W_dynamic
```

当前旧编码器仍然采用两阶段基础投影和动态投影。

### 新 forward 流程

```python
# 1. 对原输入进行无梯度显式高斯探测
with torch.no_grad():
    probe_params = self.gaussian_splatting.extract_parameters(
        x_raw_flat
    )

# 2. 预测 shift 并执行 Gaussian canonicalization
x_canonical, canonical_aux = self.canonicalizer(
    x_seq,
    probe_params,
)

# 3. 在 canonical 输入上正式计算显式高斯
rendered_event, ... = self.gaussian_splatting(
    x_canonical,
    ...
)

# 4. 在 canonical 输入上进行一次线性 patch projection
res_proj = self.patch_residual(x_canonical)

# 5. 双路融合
output = (
    rendered_event
    + self.gs_residual_weight * res_proj
)
```

### 新增状态

```python
self.last_predicted_shift
self.last_shift_confidence
self.last_transport_valid_ratio
self.last_canonical_error
self.last_boundary_ratio
self.last_canonical_sequence
```

训练时不要长期保留计算图，诊断状态必须使用 `detach()`。

### residual mode

支持：

```text
linear
legacy_dynamic_shell
gaussian_canonicalization
```

默认：

```python
residual_mode = "gaussian_canonicalization"
```

------

## 5.6 `layers/common_blocks.py`

当前 `FlattenHead` 直接将所有 patch 表示展平。

新增：

```python
class GaussianPatchMixer(nn.Module):
```

并在 Flatten 前应用。

接口：

```python
enc_out = self.patch_mixer(enc_out)
pred = self.head_srs(enc_out)
```

保持 `FlattenHead` 本身不变，以减少代码侵入。

支持配置：

```python
use_gaussian_patch_mixer=True
patch_mixer_kernel=3
patch_mixer_sigma=0.8
patch_mixer_init=0.1
```

------

## 5.7 `models/gs_linear.py`

### 新增模块

初始化：

```python
self.gaussian_patch_mixer = GaussianPatchMixer(...)
```

### 修改 forward

模型 forward 应支持：

```python
def forward(
    self,
    x,
    ...,
    return_aux=False,
):
```

内部：

```python
enc_out, encoder_aux = self.representation(
    x_repr_in,
    self.num_patches_srs,
    return_aux=return_aux,
)
```

再执行：

```python
enc_out = self.gaussian_patch_mixer(enc_out)
```

默认仍只返回：

```python
pred
```

训练需要辅助损失时返回：

```python
pred, aux_dict
```

现有顶层已经在传递 Gaussian Jet 相关参数，需要将这些旧参数替换成 canonicalization 参数。

建议新增：

```python
canonical_max_shift_samples
transport_radius
transport_sigma
shift_estimator_hidden
boundary_context_len
canonical_confidence_blend
use_gaussian_patch_mixer
patch_mixer_sigma
```

删除默认主路径依赖的：

```python
jet_max_shift_samples
jet_score_temperature
jet_density_tau
jet_detach_geometry
```

这些参数可以保留在 legacy Jet 消融模式中。

------

## 5.8 `exp/exp_long_term_forecasting.py`

### 接入全局平移增强

训练循环中，在 RevIN 和模型 forward 之前执行：

```python
batch_x_clean = batch_x

batch_x_model, shift_target, shift_presence, valid_mask, aug_info = (
    global_shift_augment(...)
)
```

主 forward：

```python
outputs, aux = self.model(
    batch_x_model,
    return_aux=True,
)
```

### 计算联合损失

```python
loss_forecast
loss_shift
loss_presence
loss_canonical
loss_boundary
loss_consistency
```

### 一次主 forward 的要求

不要对整个 batch 同时分别运行 clean 和 shifted 两次。

主 batch 中直接混合 clean 与 shifted 样本。

一致性 forward 只在：

```text
每 4 step
25% batch
```

上执行。

### 保存训练诊断

每个 epoch 记录：

```text
forecast loss
shift MAE
shift sign accuracy
shift confidence clean mean
shift confidence shifted mean
canonical reconstruction MSE
boundary reconstruction MSE
augmentation ratio
mean sampled shift
```

------

## 5.9 `run.py`

新增参数：

```python
--residual_mode gaussian_canonicalization

--shift_aug_probability 0.6
--train_shift_values 0 1 2 4 8
--shift_positive_only 1
--shift_curriculum 1

--canonical_max_shift_samples 8
--transport_radius 3
--transport_sigma 0.75

--shift_loss_weight 0.10
--presence_loss_weight 0.02
--canonical_loss_weight 0.05
--boundary_loss_weight 0.10
--consistency_weight 0.05
--consistency_interval 4
--consistency_batch_ratio 0.25

--boundary_context_len 32
--use_gaussian_patch_mixer 1
--patch_mixer_sigma 0.8
```

如果测试包含 shift 16，则：

```python
--canonical_max_shift_samples 16
--train_shift_values 0 1 2 4 8 16
```

------

## 5.10 `utils/print_args.py`

增加打印分组：

```text
Shift Robustness
Gaussian Canonicalization
Robustness Loss
```

避免训练时参数实际未生效却无法察觉。

------

## 5.11 五个 HPO 脚本

需要同步修改：

```text
tune_gate_etth1.py
tune_gate_etth2.py
tune_gate_ettm1.py
tune_gate_ettm2.py
tune_gate_weather.py
```

当前脚本仍主要搜索：

```text
gate_beta
gate_lambda
gate_window_half
```

并将 clean MSE 作为唯一优化目标。

Gaussian canonicalization 模式下应删除这些旧路由参数的 HPO。

### 新增搜索参数

建议第一轮只搜索：

```python
shift_aug_probability = [0.4, 0.6, 0.8]

shift_loss_weight = [
    0.05,
    0.10,
    0.20,
]

canonical_loss_weight = [
    0.02,
    0.05,
    0.10,
]

boundary_loss_weight = [
    0.05,
    0.10,
    0.20,
]

transport_sigma = [
    0.50,
    0.75,
    1.00,
]

patch_mixer_init = [
    0.0,
    0.1,
    0.2,
]
```

继续搜索：

```python
learning_rate
gs_dropout
gs_residual_weight
```

固定：

```text
transport_radius = 3
boundary_context_len = 32
canonical_max_shift_samples = 测试最大 shift
```

------

## 5.12 HPO 目标不能只使用 clean MSE

每个 HPO trial 训练完成后，至少在验证集上测试：

```text
shift = 0
shift = 1
shift = 4
shift = 8
```

定义：

# [ D_s

\frac{
\operatorname{MSE}_s-\operatorname{MSE}_0
}{
\operatorname{MSE}_0
}
]

建议 Optuna 目标：

# [ J

\operatorname{MSE}*0
+
\lambda*{\mathrm{mean}}
\operatorname{mean}(D_1,D_4,D_8)
+
\lambda_{\mathrm{worst}}
\max(D_1,D_4,D_8)
]

初始建议：

```text
lambda_mean  = 0.10
lambda_worst = 0.05
```

也可以使用约束式选择：

```text
clean MSE 不得比当前最佳 clean 模型恶化超过 1%
```

在满足 clean MSE 约束的 trial 中，选择平均 shift degradation 最小的模型。

保存 JSON 时增加：

```python
mse_shift_0
mse_shift_1
mse_shift_4
mse_shift_8
degradation_shift_1
degradation_shift_4
degradation_shift_8
mean_degradation
worst_degradation
shift_mae
```

------

## 5.13 现有平移测试脚本

扰动逻辑原则上保持不变，以保证修改前后结果可比。

只修改以下内容。

### checkpoint 必须严格加载

```python
model.load_state_dict(
    state_dict,
    strict=True,
)
```

禁止使用：

```python
strict=False
```

### 启动时检查模型路径

必须打印：

```text
residual_mode
canonicalizer class
transport sigma
max shift
boundary completer enabled
patch mixer enabled
```

并断言：

```python
assert residual_mode == "gaussian_canonicalization"
```

### 增加机制诊断

对每个实际 shift 输出：

```text
true shift
predicted shift mean
predicted shift MAE
shift sign accuracy
shift confidence
transport valid ratio
canonical reconstruction proxy
MSE
MSE degradation
```

通过这些指标可以判断：

- 位移估计是否错误；
- 运输是否错误；
- 边界补全是否失效；
- 还是最终预测头仍然敏感。

------

# 六、代码实现顺序

## 第一阶段：彻底清理主计算路径

先确保：

```text
Gaussian canonicalization 模式
不实例化 AdaptiveResidualRouter
不调用 get_base_projection
不生成 M_modulator
不生成 W_dynamic
不包含 shift_scaler
```

模型启动时打印实际类名和参数名。

这是继续实验前的硬性条件。

------

## 第二阶段：实现测试对齐的 global shift augmentation

单独测试：

```text
shift 1
shift 2
shift 4
shift 8
```

生成结果必须逐元素与测试脚本一致。

不要先改模型，先确认训练增强和测试腐蚀完全相同。

------

## 第三阶段：实现位移监督

先只训练：

```text
SplatGuidedShiftEstimator
```

验证集上要求：

```text
shift MAE < 0.5 sample，针对 shift 1/2
shift MAE < 1.0 sample，针对 shift 4/8
符号准确率 > 95%
clean confidence < 0.2
shifted confidence > 0.7
```

达不到时，不继续测试完整预测性能。

------

## 第四阶段：实现 Gaussian Transport

用人工输入验证：

```text
正弦波
单峰脉冲
趋势序列
周期序列
```

将 clean 序列右移后，再输入真实 shift 执行运输。

在有效位置上应满足：

```text
transported MSE ≈ 0
```

整数 shift 下误差应非常小。

------

## 第五阶段：加入边界补全

分别验证：

```text
shift 4
shift 8
shift 16
```

补全前后最后若干点的 MSE 应明显下降。

如果边界补全不工作，大 shift 的预测退化不会明显改善。

------

## 第六阶段：接入双路模型

先固定真实 shift 作为 transport 输入，测试模型上限。

即训练和测试时暂时令：

```python
predicted_shift = shift_target
```

如果真实 shift 条件下抗平移仍然不好，说明问题在：

- Transport；
- 边界补全；
- 双路融合；
- Head。

如果真实 shift 条件下效果很好，而预测 shift 条件下不好，问题只在 shift estimator。

这是必须做的 oracle 消融。

------

## 第七阶段：加入 Gaussian Patch Mixer

只在 canonicalization 已经有效后加入。

比较：

```text
canonicalization only
canonicalization + patch mixer
```

避免同时修改太多组件后无法判断来源。

------

## 第八阶段：修改 HPO

最后才开始大规模 HPO。

不要在模型路径和辅助监督尚未通过单元测试前运行几百个 Optuna trial。

------

# 七、必须增加的单元测试

## 7.1 增强和测试脚本一致性

同一个输入、同一个 shift，验证：

```python
train_augment_output == test_shift_output
```

------

## 7.2 位移符号测试

原信号峰值从位置 10 右移到位置 14：

```text
target shift = +4
source position = target position + 4
```

运输结果应恢复到原峰位置 10。

------

## 7.3 恒等测试

当：

```text
predicted shift = 0
confidence = 0
```

必须满足：

```python
x_canonical == x_input
```

------

## 7.4 Oracle shift 测试

使用真实 shift 进行运输，测试：

```text
shift 1/2/4/8
```

有效区域重建误差必须接近零。

------

## 7.5 边界补全测试

验证缺失尾部位置只由 completer 填充，其他有效位置不被覆盖。

------

## 7.6 梯度测试

确认以下模块有有效梯度：

```text
shift_estimator
transport_sigma（若可学习）
boundary_completer
base_projection
Gaussian generator 正式 Pass
GaussianPatchMixer gate
```

probe Pass 不应保存梯度。

------

## 7.7 显存测试

确认运行图中不存在：

```text
[B*C, P, D, L]
```

主要动态运输张量只能与：

```text
B、C、T、transport_radius
```

相关。

------

# 八、默认配置建议

以当前：

```text
seq_len = 512
patch_len = 24
stride = 12
d_model = 128
```

为例，第一版建议：

```text
residual_mode = gaussian_canonicalization

train_shift_values = [0, 1, 2, 4, 8]
shift_aug_probability = 0.6
shift_positive_only = True

canonical_max_shift_samples = 8
transport_radius = 3
transport_sigma = 0.75

boundary_context_len = 32

shift_loss_weight = 0.10
presence_loss_weight = 0.02
canonical_loss_weight = 0.05
boundary_loss_weight = 0.10
consistency_weight = 0.05

consistency_interval = 4
consistency_batch_ratio = 0.25

use_gaussian_patch_mixer = True
patch_mixer_sigma = 0.8
patch_mixer_init = 0.1
```

如果正式测试包含 shift 16，则修改为：

```text
canonical_max_shift_samples = 16
train_shift_values = [0, 1, 2, 4, 8, 16]
boundary_context_len = 48
```

------

# 九、验收标准

完成后必须同时满足以下条件。

## 结构验收

1. 主模式真正使用 `SplatGuidedGaussianCanonicalizer`；
2. 不再调用旧 `AdaptiveResidualRouter`；
3. 不再执行两次线性投影；
4. 不再生成完整动态线性权重；
5. 显式和线性通路都使用 canonical sequence；
6. checkpoint 使用 `strict=True` 正常加载。

## 位移估计验收

```text
shift 1/2 的平均误差 < 0.5 sample
shift 4/8 的平均误差 < 1.0 sample
clean shift confidence < 0.2
shifted confidence > 0.7
```

## 运输验收

在使用 oracle shift 时：

```text
有效区域重建 MSE 接近 0
```

## 预测性能验收

相对 clean MSE：

```text
shift 1 degradation 目标：< 1%
shift 2 degradation 目标：< 2%
shift 4 degradation 目标：< 5%
shift 8 degradation 目标：< 10%
```

shift 16 因为最新 16 个观测完全丢失，无法保证接近零退化，建议目标为：

```text
显著优于原模型和 augmentation-only baseline
```

## 效率验收

相对普通双路模型：

```text
峰值显存目标：≤ 1.25×
训练时间目标：≤ 1.40×
```

探测 Pass 应使用 `no_grad`，一致性 forward 只在少量 step 和样本上执行。

------

# 十、论文消融需要保留的模式

最终至少支持：

```text
A. clean baseline
B. shift augmentation only
C. supervised shift estimator only
D. Gaussian Transport without explicit GS features
E. explicit-GS-guided Gaussian Transport
F. E + boundary completion
G. F + Gaussian Patch Mixer
H. oracle shift upper bound
```

最关键的对比是：

```text
纯数据增强
vs.
普通插值运输
vs.
隐式 Gaussian Transport
vs.
显式高斯引导的 Gaussian Transport
```

这样才能证明性能提升不是简单来自“训练时见过平移”，而是显式高斯几何、隐式高斯运输和边界补全共同带来的。

这版方案的核心变化是：**不再要求一个低秩 Jet 去抵消完整 Linear 的平移误差，而是在进入两条预测通路前，先用显式高斯引导的 Gaussian Transport 将整个输入恢复到规范坐标。**这样更直接对应现有测试脚本，也能处理跨 patch 和 4、8 步以上的平移。