## 结论

当前版本的**模型主干已经基本符合预期**：

* `gamma_complement` 已删除；
* Legacy Router 和旧残差投影已删除；
* `d_latent` 已从 Encoder 接口移除；
* Gaussian Splatting 不再向 Encoder 返回无用的 `weights`；
* `num_gaussians` 默认值已经统一为 8；
* `PatchLinearRepresentation.forward()` 已兼容统一调用接口；
* GS+CAS+Jet 前向、反向均能运行，最小测试中所有可训练参数都有梯度；
* `patch_linear` 分支也已通过前向测试。   

但**清理还没有彻底完成**，主要残留在实验诊断代码和运行参数中。

---

# 1. `d_ff` 是否无效？

**是。对于当前 `SplatTS/gs_linear` 模型，`d_ff` 是无效的模型超参数。**

它只出现在两处：

```python
parser.add_argument("--d_ff", type=int, default=256)
```

以及实验名称：

```python
setting = "..._dm{}_df{}_...".format(
    ...
    args.d_model,
    args.d_ff,
    ...
)
```

模型构造、Gaussian Generator、Jet、Head、优化器和损失函数中均没有读取 `d_ff`。

我还做了实际对照测试：

```text
d_ff = 4
d_ff = 64

模型 state_dict 最大差异：0.0
相同输入的输出最大差异：0.0
```

因此它的实际效果只有：

> 改变 checkpoint 目录和实验名称中的 `_df256_` 字段。

这会产生一个麻烦：两个计算完全相同的模型，仅仅因为 `d_ff` 不同，就被保存为两个不同实验。

### 建议

从 ETTh1 脚本中删除：

```bash
--d_ff 256
```

并从 `setting` 中删除：

```python
_df{}
args.d_ff
```

如果 `run.py` 还要兼容其他真正使用 FFN 的模型，可以保留 parser 参数，但不要把它当作 SplatTS 的有效超参数。

---

# 2. ETTh1 脚本中的其他无效参数

## 完全不影响当前模型计算

| 参数        | 当前状态                        | 建议                       |
| --------- | --------------------------- | ------------------------ |
| `d_ff`    | 只影响实验名称                     | 从 SplatTS 脚本和 setting 删除 |
| `dec_in`  | 没有被模型或实验读取                  | 从该脚本删除                   |
| `c_out`   | 只赋给未使用的 `self.n_vars`       | 删除脚本参数和死属性               |
| `dropout` | `head_dropout` 已单独存在，因此它不生效 | 从该脚本删除                   |

---

## 2.1 `dec_in` 完全无效

当前模型不使用 Decoder，也不读取 `configs.dec_in`。模型 forward 虽然保留了 `x_dec`，但根本没有使用：

```python
def forward(
    self,
    x,
    x_mark_enc=None,
    x_dec=None,
    x_mark_dec=None,
    ...
):
```

计算只依赖 `x`。

实际测试中，把：

```text
dec_in = 1
```

改成：

```text
dec_in = 99
```

模型参数和输出均完全相同。

因此 ETTh1 脚本中的：

```bash
--dec_in 7
```

可以删除。

---

## 2.2 `c_out` 当前也不影响输出通道数

`c_out` 唯一进入模型的位置是：

```python
self.n_vars = configs.c_out
```

但 `self.n_vars` 后续没有被使用。实际输出通道数由输入的：

```python
B, T, C = x.shape
```

中的 `C` 决定：

```python
pred = pred.reshape(B, C, self.pred_len)
```



因此改变 `c_out`：

* 不改变参数量；
* 不改变 Head；
* 不改变输出形状；
* 不改变预测结果。

实测 `c_out=1` 与 `c_out=99` 的输出差异也是 `0.0`。

建议删除：

```python
self.n_vars = configs.c_out
```

ETTh1 脚本中的：

```bash
--c_out 7
```

也可以删除。若为了兼容其他模型保留 parser 参数，需要明确它对 SplatTS 无效。

---

## 2.3 `dropout` 对当前 SplatTS 无效

当前模型写的是：

```python
head_dropout=getattr(
    configs,
    "head_dropout",
    configs.dropout,
)
```

但 `run.py` 永远都会创建 `args.head_dropout`：

```python
parser.add_argument(
    "--head_dropout",
    type=float,
    default=0.1,
)
```

所以 `configs.dropout` 永远不会成为实际的 Head dropout。Gaussian Generator 使用的是独立的 `gs_dropout`。 

ETTh1 脚本又明确设置了：

```bash
--dropout 0.0
--head_dropout 0.9
```

实际生效的只有：

```bash
--head_dropout 0.9
```

我将 `dropout` 从 `0.0` 改成 `0.9`，固定 `head_dropout` 后，模型参数和输出仍完全相同。

因此可以删除：

```bash
--dropout 0.0
```

如果希望保留全局 dropout 的回退逻辑，应把 `head_dropout` 默认值改成 `None`，否则当前的回退永远不可达。

---

# 3. 两个“参数有效，但当前写法无效果”的项目

## 3.1 `--use_residual` 在脚本中是多余的

当前 parser 定义为：

```python
parser.add_argument(
    "--use_residual",
    action="store_true",
    default=True,
)
```

这意味着：

* 不写 `--use_residual`：值是 `True`；
* 写了 `--use_residual`：值还是 `True`。

所以 ETTh1 脚本中的：

```bash
--use_residual
```

不会改变任何东西。

更严重的是，目前命令行没有正常关闭 Residual 的方式。

建议使用：

```python
parser.add_argument(
    "--use_residual",
    action=argparse.BooleanOptionalAction,
    default=True,
)
```

这样可以使用：

```bash
--use_residual
--no-use_residual
```

如果项目希望默认关闭，则改为：

```python
parser.add_argument(
    "--use_residual",
    action="store_true",
    default=False,
)
```

当前 ETTh1 脚本可以暂时删除 `--use_residual`，因为默认已经开启。

---

## 3.2 `gs_lambda=0.0` 会完全关闭容量正则项

CAS 训练损失中确实使用了：

```python
loss = loss + (
    self.args.gs_lambda * warmup_factor
) * loss_capacity
```

所以 `gs_lambda` 本身不是无效参数。

但 ETTh1 脚本设置：

```bash
--gs_lambda 0.0
```

意味着整个容量惩罚严格为零。

需要区分：

* CAS gating 仍然参与前向和预测损失训练；
* 只是额外的 CAS capacity regularization 被关闭。

而 `run.py` 的默认值本身也是 `0.0`，所以脚本显式写这一行是冗余的。

如果实验本来就要关闭正则，可以删除该行；如果本来希望使用 CAS 容量控制，那么当前脚本并没有启用它。

---

# 4. 当前代码仍残留的死代码

## 4.1 `last_density_score` 还没有完全清理

`TemporalGaussianSplatting` 仍保留：

```python
last_density_score = None
```

但它既没有被更新，也没有被返回。

这一行是纯死代码，可以直接删除。

实验脚本中则仍然存在：

```python
egc_list = []
density_sig_list = []
```

以及：

```python
if (
    splatting_module is not None
    and getattr(
        splatting_module,
        "last_density_score",
        None,
    ) is not None
):
    ...
```

但当前 Encoder 已经没有 `last_density_score` 属性，所以这个分支永远不会进入。后面的 Gaussian Density 诊断块也永远不会生成。 

应删除整组：

```text
egc_list
density_sig_list
last_density_score 检查
Mechanistic Metric (Gaussian Density) 日志块
```

CAS 自己的 `p_i / pi_i / sigma` 诊断可以继续保留。

---

## 4.2 LGA 空列表仍然存在

实验脚本仍然初始化：

```python
raw_lga_list = []
z_lga_list = []
energy_list = []
```

但之后完全没有使用。

这三行应删除。说明上一轮的 LGA 实验端残留还没有真正清掉。

---

## 4.3 日志仍然使用旧 Router 术语

测试日志仍然写：

```text
Adaptive Gating Direction Diagnostics
Route Probabilities
Effective Gating
Current Strategy
```

但当前这些数值实际来自 Gaussian score shift 和 density confidence，而不是 Adaptive Direction Router。 

建议改成：

```text
Gaussian Score-Shift Diagnostics
Directional Confidence Components
Effective Score Shift
Shift Pattern
```

`last_gate_direction` 等属性如果暂时为了兼容测试可以保留，但论文版代码最好改成：

```text
last_shift_direction
last_shift_confidence
last_directional_components
```

---

# 5. ETTh1 脚本的命名仍然是旧项目语义

当前脚本仍有：

```text
Adaptive Direction Router
etth1_router_96
Ablation_ETTh1_router
```



现在项目已经是 Gaussian Jet，这些命名会让：

* checkpoint 名称；
* 日志文件；
* 实验表；
* 论文复现实验；

继续显示成 Router。

建议统一改成：

```bash
--model_id etth1_jet_96
--des ETTh1_GaussianJet
```

脚本注释改为：

```bash
# ETTh1 Gaussian Jet + CAS configuration
```

这不是数值计算错误，但属于应清理的语义残留。

---

# 6. `k_base` 仍会在不需要时计算

当前 `gs_linear` 只要：

```python
k_base == -1
```

就根据数据集计算 `k_base`，无论：

* `representation` 是否为 `gs`；
* `density_mode` 是否为 `cas`。



对于当前 ETTh1 脚本，因为确实使用：

```text
representation = gs
density_mode = cas
```

所以结果符合预期。

但更完整的写法应为：

```python
density_mode = getattr(
    configs,
    "density_mode",
    "none",
)
num_gaussians = getattr(
    configs,
    "num_gaussians",
    8,
)

if (
    self.representation_name == "gs"
    and density_mode == "cas"
    and k_base == -1
):
    ...
```

否则运行 `patch_linear` baseline 时，也会无意义地打印 CAS 的 `k_base` 信息。

---

# 7. `run.py` 会静默吞掉拼错的参数

当前使用：

```python
args, unknown = parser.parse_known_args()
```

但没有检查 `unknown`。

这意味着类似：

```bash
--gate_bate 0.5
--jet_density_tua 1.0
```

不会报错，而是被悄悄忽略。这正是此前残留参数问题很难发现的原因之一。

如果不需要接受外部未知参数，直接改为：

```python
args = parser.parse_args()
```

如果必须使用 `parse_known_args()`，至少加上：

```python
args, unknown = parser.parse_known_args()

if unknown:
    parser.error(
        "Unrecognized arguments: "
        + " ".join(unknown)
    )
```

这是非常建议修复的一项。

---

# 8. 哪些“看起来没直接引用”的参数不要删除

以下参数虽然在当前几个模型文件里不一定直接出现，但会通过外部模块使用，不应仅凭 grep 删除：

```text
root_path
data_path
batch_size
num_workers
lradj
checkpoints
```

`Exp_Long_Term_Forecast` 会把完整的 `args` 交给 `data_provider`，并把它交给 `adjust_learning_rate`；这些参数通常由数据加载器和学习率工具读取。

---

# 推荐的 ETTh1 脚本精简结果

从当前脚本中可以安全删除：

```bash
--dec_in 7
--c_out 7
--d_ff 256
--dropout 0.0
--use_residual
--gs_lambda 0.0
```

其中前四个对当前模型计算无影响；后两个分别是“默认已开启”和“默认已为零”。

为了论文复现，反而建议显式补上真正有效的 Jet 参数：

```bash
--num_implicit_gaussians 4
--jet_max_shift_samples 1.0
--jet_score_temperature 1.0
--jet_density_tau 1.0
--jet_detach_geometry 1
--jet_scale_init 0.1
--jet_sigma_init 0.2
--k_base -1
```

这些参数当前虽然没有写进 ETTh1 脚本，但默认值确实会影响 Jet 结构或计算。  

**最终判断：模型核心修复已经正确，但实验端的 `last_density_score`、LGA 空列表和旧 Router 日志尚未清理；`d_ff`、`dec_in`、`c_out`、`dropout` 对当前 SplatTS 都是无效参数。**
