## 结论

**还没有完全清理完毕。**

旧的 LGA/Legacy **执行主干已经删除干净**：`AdaptiveResidualRouter`、`PatchResidualProjection`、`gate_type`、`gate_beta`、`gate_window_half`、`residual_mode` 都已经不在当前模型路径中，Encoder 现在只实例化 `GaussianJetProjection`。 

但外围接口、诊断字段和实验脚本仍有一批明确残留。另外，实际前向测试还发现了一个 `patch_linear` 分支错误。

## 仍需删除的明确残留

| 残留                       | 位置                    | 判断                    |
| ------------------------ | --------------------- | --------------------- |
| `gamma_complement` 参数和属性 | `cas_gating`          | 完全未读取，应删除             |
| `route_router` property  | Encoder               | 旧测试兼容接口，始终返回 `None`   |
| `last_raw_lga`           | Encoder               | 完全无数据，只反复赋值 `None`    |
| `last_z_lga`             | Encoder               | 同上                    |
| `last_energy`            | Encoder               | 同上                    |
| `raw_lga_list`           | 实验脚本                  | 创建后从未写入或读取            |
| `z_lga_list`             | 实验脚本                  | 同上                    |
| `energy_list`            | 实验脚本                  | 同上                    |
| `d_latent` 参数和属性         | Encoder               | 当前 Encoder 中从未使用      |
| `weights` 返回值            | Gaussian → Encoder    | 旧 Router 曾需要，现在解包后未使用 |
| `last_density_score`     | Gaussian、Encoder、实验脚本 | 永远是 `None`，相关逻辑不可达    |

### 1. `gamma_complement` 仍然存在

`CASGating` 仍定义并保存：

```python
def __init__(
    self,
    d_model,
    num_gaussians,
    patch_len=16,
    stride=None,
    gamma_complement=0.6,
    k_base=-1,
):
    ...
    self.gamma_complement = gamma_complement
```

但 forward 中完全没有使用。Encoder 也已经不再传入它，所以这是纯粹的旧代码残留。

应改为：

```python
def __init__(
    self,
    d_model,
    num_gaussians,
    patch_len=16,
    stride=None,
    k_base=-1,
):
```

同时删除：

```python
self.gamma_complement = gamma_complement
```

### 2. Encoder 中仍保留 LGA 诊断遗迹

以下字段已经不可能产生有效数据：

```python
self.last_raw_lga = None
self.last_z_lga = None
self.last_energy = None
```

forward 的两个分支中又反复将它们设为 `None`。这些赋值没有任何作用，应全部删除。

下面的 property 也属于旧测试兼容残留：

```python
@property
def route_router(self):
    return None
```

除非现有测试仍强制访问 `route_router`，否则也应该删除。

### 3. 实验脚本仍有 LGA 空列表

测试函数中仍然创建：

```python
raw_lga_list = []
z_lga_list = []
energy_list = []
```

但后续没有向它们写入数据，也没有读取，直接删除即可。

---

## `density_mode="sparse"` 已经成为不可达代码

这是目前最大的一组死代码。

`TemporalGaussianSplatting` 中：

```python
last_density_score = None
...
return rendered_event, weights, mu, sigma, alpha_effective, last_density_score
```

无论输入和配置是什么，`last_density_score` 都恒为 `None`。

Encoder 只是接收并继续保存：

```python
self.last_density_score = last_density_score
```

所以它仍然恒为 `None`。

因此实验脚本中的以下训练逻辑永远不会执行实际惩罚：

```python
if density_mode == "sparse":
    if splatting_module.last_density_score is not None:
        ...
```

测试阶段基于 `last_density_score` 计算 EGC 的代码也永远不会执行。

如果当前项目只支持：

```text
density_mode = none
density_mode = cas
```

建议一次性删除：

* Gaussian 中的 `last_density_score`
* Encoder 中的 `self.last_density_score`
* Gaussian 返回值中的 `last_density_score`
* 实验脚本中的 `"sparse"` 损失分支
* 测试阶段基于 `last_density_score` 的 EGC 统计
* `egc_list` 和 `density_sig_list` 相关通用诊断

CAS 已经拥有独立的 `p_i`、`pi_i`、`gate_hard` 和 `last_active_count`，无需依赖这套失效的 sparse 接口。

---

## `weights` 也是旧 Router 留下的返回接口

Gaussian Splatting 内部当然仍需要 `weights` 完成渲染，但现在 Encoder 接收它之后没有任何使用：

```python
rendered_event, weights, mu, sigma, alpha_effective, last_density_score = ...
```

过去 `weights` 会传给 LGA Router；Router 删除后，这个跨模块返回值已经多余。 

可以把接口收缩为：

```python
return rendered_event, mu, sigma, alpha_effective
```

Encoder 对应改成：

```python
rendered_event, mu, sigma, alpha_effective = self.gaussian_splatting(
    x_flat,
    patch_num,
    x_seq.device,
    gate_effective,
)
```

注意只是停止**返回** `weights`，Gaussian Splatting 内部计算渲染时仍然要保留它。

---

## Jet 诊断仍在使用旧的“Gate/Route”话语

下面这些字段不是死代码，因为实验脚本仍在使用：

```python
last_gate_direction
last_gate_strength
last_gate_scale
last_gate_route_probs
```

但它们已经不再来自一个 gating router，而是由：

* Gaussian score shift；
* density confidence；
* shift 正负方向；

人工转换出来的诊断量。

实验日志仍写成：

```text
Adaptive Gating Direction Diagnostics
Route Probabilities
Effective Gating
Current Strategy
```

这容易让代码、实验日志和论文叙事产生错位。

建议改名，例如：

```text
Gaussian Score-Shift Diagnostics
Directional Confidence Decomposition
Effective Score Shift
Shift Pattern
```

字段也可以逐步改为：

```python
last_shift_direction
last_shift_confidence
last_directional_components
```

严格来说，这部分是**语义残留**，不是不可达代码。如果暂时需要兼容旧日志，可以保留；但不能称为完全清理。

---

## 另外发现两个实际问题

### 1. `patch_linear` 分支目前无法运行

主模型统一调用：

```python
enc_out = self.representation(
    x_repr_in,
    self.num_patches_srs,
)
```

但 `PatchLinearRepresentation.forward` 只接收：

```python
def forward(self, x_seq):
```

 

实际前向测试结果是：

```text
TypeError:
PatchLinearRepresentation.forward() takes 2 positional arguments
but 3 were given
```

建议让两个 representation 的接口统一。最小修改是：

```python
class PatchLinearRepresentation(nn.Module):
    ...

    def forward(self, x_seq, patch_num=None):
        B, C, L = x_seq.shape
        ...
```

这里 `patch_num` 可以忽略，因为该模块已经自行计算 `self.patch_num`。

另一种方式是在主模型中显式分支，但统一接口更干净。

### 2. `num_gaussians` 默认值不一致

自动计算 `k_base` 时使用：

```python
num_gaussians = getattr(configs, "num_gaussians", 8)
```

实际创建 Encoder 时却使用：

```python
num_gaussians = getattr(configs, "num_gaussians", 5)
```



当配置没有显式给出 `num_gaussians` 时，会出现：

* 按 8 个 Gaussian 计算 `k_base`；
* 实际只创建 5 个 Gaussian。

应当在前面只解析一次：

```python
num_gaussians = getattr(configs, "num_gaussians", 5)
```

后面全部使用该局部变量。

同时，`k_base` 的自动计算最好只在真正需要 CAS 时运行：

```python
density_mode = getattr(configs, "density_mode", "none")
num_gaussians = getattr(configs, "num_gaussians", 5)

if (
    self.representation_name == "gs"
    and density_mode == "cas"
    and k_base == -1
):
    ...
```

否则在 `patch_linear` 或 `density_mode="none"` 下计算并打印 `k_base` 都没有意义。

---

## 其他低优先级死字段

以下字段在上传代码中也没有实际读取：

```python
# SplattingResidualEncoder
d_latent
self.d_latent

# Model
self.task_name
self.n_vars

# FlattenHead
self.n_vars
```

其中 `task_name` 和 forward 的扩展参数可能是上层预测框架的统一接口，可以为了框架兼容保留；但 `Encoder.d_latent` 基本可以确定删除，因为底层 Gaussian 模块直接使用 `d_model` 作为 latent 维度。   

## 最终判断

当前状态可以描述为：

> **旧 LGA 的模型计算路径已经清理完成，但旧接口、旧诊断字段和旧实验逻辑尚未完全清理。**

至少还应处理这几组，才能称为真正清理完毕：

```text
gamma_complement
route_router
last_raw_lga / last_z_lga / last_energy
raw_lga_list / z_lga_list / energy_list
d_latent（Encoder 层）
weights 跨模块返回值
last_density_score 及整个 sparse 分支
旧 Gate/Route 诊断命名
```

此外必须修复：

```text
patch_linear forward 参数不匹配
num_gaussians 默认值 8/5 不一致
```

语法检查全部通过；`gs + none`、`gs + cas`、启用或关闭 Jet 残差的前向和反向均可运行。当前唯一直接触发运行错误的是 `patch_linear` 分支。
