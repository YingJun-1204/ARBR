明白，这里只修复**第一个问题：当前内部 HPO 脚本没有显式启用 `geometry fusion`，导致实际仍在调 `fixed fusion`**。其他流程保持原样，不改数据库、不改 objective、不改外层调度逻辑。

## 修复后的目标

你仍然使用原命令：

```powershell
python .\run_adaptive_tuning.py `
  --batch_size 256 `
  --batches 1 `
  --trials_per_batch 180 `
  --datasets etth1 etth2 ettm1 ettm2 weather `
  --output_dir loss_jetV14 `
  --pred_len 96 192 336 720
```

外层 `run_adaptive_tuning.py` 不需要改。它只负责依次调用各数据集的 tuning 脚本，本身不直接构造 `run.py` 的模型参数。

需要统一修改：

```text
tune_gate_etth1.py
tune_gate_etth2.py
tune_gate_ettm1.py
tune_gate_ettm2.py
tune_gate_weather.py
```

若以后运行 Electricity，再对 `tune_gate_electricity.py` 做同样修改。

------

# 一、当前问题的准确位置

当前各 tuning 脚本搜索的是：

```python
gs_residual_weight = trial.suggest_float(...)
```

而 `build_command()` 只向 `run.py` 传递：

```text
--use_residual
--gs_residual_weight ...
```

没有传递：

```text
--fusion_mode geometry
--fusion_init
--fusion_beta_max
--fusion_hidden_dim
--fusion_detach_geometry
```

因此 `run.py` 会使用默认：

```python
fusion_mode = "fixed"
```

实际 HPO 的仍然是：

[
E+\lambda I,
]

而不是第二优先级的新模型：

[
E+\beta_jI.
]



------

# 二、修复原则

本次只做以下变化：

1. 所有 HPO trial 强制运行 `fusion_mode=geometry`；
2. 不再搜索在 geometry 模式下不起主要作用的 `gs_residual_weight`；
3. 将 `gs_residual_weight` 固定为 `0.1`，用于兼容固定融合消融；
4. 新增对 `fusion_init` 和 `fusion_beta_max` 的搜索；
5. 第一轮固定：
   - `fusion_hidden_dim=16`
   - `fusion_detach_geometry=1`
   - `jet_score_temperature=0.01`
   - `jet_max_shift_samples=1.0`
6. 将最佳 geometry 参数保存到最终 JSON；
7. 外层 HPO 命令保持不变。

之所以不能只追加：

```text
--fusion_mode geometry
```

是因为当前 `gs_residual_weight` 搜索范围为 (0.2\sim0.6)，而 `fusion_init=-1` 会继承它。与此同时默认 `fusion_beta_max=0.5`，可能产生：

[
\texttt{fusion_init}\geq
\texttt{fusion_beta_max},
]

从而触发 encoder 的合法性检查。

因此必须显式搜索并传入 `fusion_init`。

------

# 三、修改 `DEFAULT_BOUNDS`

以 `tune_gate_etth1.py` 为模板。

当前：

```python
DEFAULT_BOUNDS = {
    "learning_rate_lower": 1e-4,
    "learning_rate_upper": 1.5e-3,
    "gs_dropout_lower": 0.2,
    "gs_dropout_upper": 0.9,
    "head_dropout_lower": 0.2,
    "head_dropout_upper": 0.9,
    "gs_residual_weight_lower": 0.2,
    "gs_residual_weight_upper": 0.6,
}
```

改成：

```python
DEFAULT_BOUNDS = {
    "learning_rate_lower": 1e-4,
    "learning_rate_upper": 1.5e-3,
    "gs_dropout_lower": 0.2,
    "gs_dropout_upper": 0.9,
    "head_dropout_lower": 0.2,
    "head_dropout_upper": 0.9,
}
```

删除：

```python
"gs_residual_weight_lower"
"gs_residual_weight_upper"
```

因为 geometry 模式的实际融合系数是：

```text
fusion_init
fusion_beta_max
```

不是 `gs_residual_weight`。

------

# 四、修改 `PARAM_KEYS`

当前：

```python
PARAM_KEYS = {
    "gs_dropout",
    "train_epochs",
    "learning_rate",
    "head_dropout",
    "batch_size",
    "gs_residual_weight",
    "gate_lambda",
}
```

改为：

```python
PARAM_KEYS = {
    "gs_dropout",
    "train_epochs",
    "learning_rate",
    "head_dropout",
    "batch_size",
    "fusion_init",
    "fusion_beta_max",
}
```

其中：

- 删除 `gs_residual_weight`；
- 删除当前搜索中无实际作用的 `gate_lambda`；
- 加入新的 geometry fusion 参数。

------

# 五、重写 `sample_params()`

当前函数开头搜索：

```python
gs_residual_weight = trial.suggest_float(...)
```

应删除。

推荐完整改成：

```python
def sample_params(
    trial,
    bounds,
    position,
    batch_size=256,
):
    gs_dropout = trial.suggest_float(
        "gs_dropout",
        bounds["gs_dropout_lower"],
        bounds["gs_dropout_upper"],
        step=0.05,
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [batch_size],
    )

    # Geometry-conditioned fusion parameters.
    fusion_init = trial.suggest_categorical(
        "fusion_init",
        [0.05, 0.10, 0.20],
    )

    fusion_beta_max = trial.suggest_categorical(
        "fusion_beta_max",
        [0.20, 0.50, 1.00],
    )

    # Encoder requires:
    # 0 < fusion_init < fusion_beta_max
    if fusion_init >= fusion_beta_max:
        raise optuna.TrialPruned(
            "Invalid geometry fusion configuration: "
            f"fusion_init={fusion_init} must be smaller than "
            f"fusion_beta_max={fusion_beta_max}"
        )

    params = {
        "patch_len": 24,
        "stride": 12,
        "num_gaussians": bounds.get(
            "num_gaussians_lower",
            8,
        ),
        "d_model": 128,
        "gs_dropout": gs_dropout,
        "gs_weight_decay": 1e-4,
        "train_epochs": 30,
        "learning_rate": trial.suggest_categorical(
            "learning_rate",
            [
                1e-4,
                2e-4,
                3e-4,
                5e-4,
                8e-4,
                1e-3,
                1.2e-3,
                1.5e-3,
                1.8e-3,
                2e-3,
            ],
        ),
        "gs_lambda": 0.0,
        "batch_size": batch_size,
        "use_residual": True,

        # Keep fixed fusion coefficient only for compatibility
        # and later fixed-fusion ablation.
        "gs_residual_weight": 0.1,

        # Full model always uses geometry fusion.
        "fusion_mode": "geometry",
        "fusion_init": fusion_init,
        "fusion_beta_max": fusion_beta_max,

        # Keep structural choices fixed during this HPO.
        "fusion_hidden_dim": 16,
        "fusion_detach_geometry": 1,

        # Keep the corrected Jet geometry settings fixed.
        "jet_score_temperature": 0.01,
        "jet_max_shift_samples": 1.0,
    }

    if position == "none":
        params["head_dropout"] = 0.0
    else:
        params["head_dropout"] = trial.suggest_float(
            "head_dropout",
            bounds["head_dropout_lower"],
            bounds["head_dropout_upper"],
            step=0.05,
        )

    return params
```

这会让原来的 180 个 trial 搜索：

```text
learning_rate
gs_dropout
head_dropout
fusion_init
fusion_beta_max
```

而不是继续搜索旧的固定残差系数。

------

# 六、修改 `build_command()`

当前命令中有：

```python
"--use_residual",
"--gs_residual_weight",
str(params["gs_residual_weight"]),
```

保留这两项，但 `gs_residual_weight` 固定为 `0.1`。

紧接着加入：

```python
"--fusion_mode",
params["fusion_mode"],

"--fusion_init",
str(params["fusion_init"]),

"--fusion_beta_max",
str(params["fusion_beta_max"]),

"--fusion_hidden_dim",
str(params["fusion_hidden_dim"]),

"--fusion_detach_geometry",
str(params["fusion_detach_geometry"]),

"--jet_score_temperature",
str(params["jet_score_temperature"]),

"--jet_max_shift_samples",
str(params["jet_max_shift_samples"]),
```

修改后的对应片段应为：

```python
"--num_workers",
"0",

"--use_residual",

"--gs_residual_weight",
str(params["gs_residual_weight"]),

# Geometry-conditioned residual fusion.
"--fusion_mode",
params["fusion_mode"],

"--fusion_init",
str(params["fusion_init"]),

"--fusion_beta_max",
str(params["fusion_beta_max"]),

"--fusion_hidden_dim",
str(params["fusion_hidden_dim"]),

"--fusion_detach_geometry",
str(params["fusion_detach_geometry"]),

# Explicitly freeze Jet geometry settings during fusion HPO.
"--jet_score_temperature",
str(params["jet_score_temperature"]),

"--jet_max_shift_samples",
str(params["jet_max_shift_samples"]),

"--des",
f"Ablation_ETTh1_jet",
```

不同数据集只修改最后的 `des` 名称，其他部分完全一致。

------

# 七、增加强制检查，防止再次误跑 fixed fusion

在 `build_command()` 开头增加：

```python
if params.get("fusion_mode") != "geometry":
    raise RuntimeError(
        "This HPO script is intended for the full "
        "geometry-conditioned fusion model, but received "
        f"fusion_mode={params.get('fusion_mode')}"
    )
```

在返回命令之前再检查一次：

```python
if "--fusion_mode" not in cmd:
    raise RuntimeError(
        "--fusion_mode was not added to the run.py command"
    )

fusion_mode_index = cmd.index("--fusion_mode")

if cmd[fusion_mode_index + 1] != "geometry":
    raise RuntimeError(
        "HPO command must run with "
        "--fusion_mode geometry"
    )
```

这样以后即使有人误删了一项参数，HPO 也会直接停止，而不会静默调成 fixed 模型。

------

# 八、修改最佳 JSON 保存逻辑

当前 callback 只保存：

```python
best_tuned["gs_residual_weight"] = ...
```



改成：

```python
best_tuned["gs_residual_weight"] = 0.1

best_tuned["fusion_mode"] = "geometry"

best_tuned["fusion_init"] = float(
    best_params.get(
        "fusion_init",
        0.1,
    )
)

best_tuned["fusion_beta_max"] = float(
    best_params.get(
        "fusion_beta_max",
        0.5,
    )
)

best_tuned["fusion_hidden_dim"] = 16

best_tuned["fusion_detach_geometry"] = 1

best_tuned["jet_score_temperature"] = 0.01

best_tuned["jet_max_shift_samples"] = 1.0
```

最终 JSON 至少应包含：

```json
{
    "fusion_mode": "geometry",
    "fusion_init": 0.1,
    "fusion_beta_max": 0.5,
    "fusion_hidden_dim": 16,
    "fusion_detach_geometry": 1,
    "jet_score_temperature": 0.01,
    "jet_max_shift_samples": 1.0,
    "gs_residual_weight": 0.1
}
```

否则 HPO 得到了最佳 gate，但后续主实验读取 JSON 时无法复现。

------

# 九、修改 `load_seed_params()`，仅在使用 `--use_seed` 时需要

你当前命令没有 `--use_seed`，所以这项不影响本轮直接运行。但为了脚本完整，建议一起修复。

当前加载：

```python
for key in [
    "learning_rate",
    "batch_size",
    "gs_residual_weight",
]:
```

改为：

```python
for key in [
    "learning_rate",
    "batch_size",
    "fusion_init",
    "fusion_beta_max",
]:
    if key in seed:
        seed_trial[key] = seed[key]
```

增加默认值：

```python
seed_trial["fusion_init"] = float(
    seed_trial.get(
        "fusion_init",
        0.1,
    )
)

seed_trial["fusion_beta_max"] = float(
    seed_trial.get(
        "fusion_beta_max",
        0.5,
    )
)
```

删除：

```python
seed_trial["gs_residual_weight"]
```

因为该参数不再属于 Optuna 搜索空间。

增加合法性处理：

```python
if (
    seed_trial["fusion_init"]
    >= seed_trial["fusion_beta_max"]
):
    print(
        "[Seed Warning] Invalid fusion seed: "
        f"fusion_init={seed_trial['fusion_init']} >= "
        f"fusion_beta_max={seed_trial['fusion_beta_max']}. "
        "Resetting to 0.1 / 0.5."
    )

    seed_trial["fusion_init"] = 0.1
    seed_trial["fusion_beta_max"] = 0.5
```

------

# 十、五个脚本必须保持一致

Antigravity 不应只修改 ETTh1。需要将同一逻辑同步到：

```text
tune_gate_etth1.py
tune_gate_etth2.py
tune_gate_ettm1.py
tune_gate_ettm2.py
tune_gate_weather.py
```

当前这些脚本的 `sample_params()` 和 `build_command()` 基本是复制结构，因此应做一致修改。

不要出现某些数据集运行 geometry、某些数据集仍运行 fixed 的情况。

------

# 十一、外层 `run_adaptive_tuning.py` 不需要修改

外层脚本继续传：

```text
pred_len
batches
trials_per_batch
position
density_mode
output_dir
num_gaussians
batch_size
seq_len
k_base
```

即可。内部 tuning 脚本负责在构造 `run.py` 命令时强制加入 geometry 参数。

所以你原来的启动命令保持不变。

------

# 十二、修复后的验收测试

正式运行 180 trials 前，先跑：

```powershell
python .\run_adaptive_tuning.py `
  --batch_size 256 `
  --batches 1 `
  --trials_per_batch 1 `
  --datasets etth1 `
  --output_dir loss_jetV14_sanity `
  --pred_len 96
```

控制台打印的子命令必须包含：

```text
--fusion_mode geometry
--fusion_init 0.05/0.1/0.2
--fusion_beta_max 0.2/0.5/1.0
--fusion_hidden_dim 16
--fusion_detach_geometry 1
--jet_score_temperature 0.01
--jet_max_shift_samples 1.0
--gs_residual_weight 0.1
```

不得出现：

```text
--fusion_mode fixed
```

也不得依赖：

```text
--fusion_init -1
```

Optuna trial 参数应至少包含：

```text
fusion_init
fusion_beta_max
learning_rate
gs_dropout
head_dropout
```

最终 JSON 应明确包含：

```text
fusion_mode = geometry
fusion_init
fusion_beta_max
fusion_hidden_dim = 16
fusion_detach_geometry = 1
```

------

# 可直接交给 Antigravity 的精简指令

```text
请只修复当前 HPO 脚本默认运行 fixed fusion 的问题，不修改外层
run_adaptive_tuning.py，也不修改数据库、objective 或训练流程。

需要统一修改：
- tune_gate_etth1.py
- tune_gate_etth2.py
- tune_gate_ettm1.py
- tune_gate_ettm2.py
- tune_gate_weather.py

要求：

1. 所有完整模型 HPO trial 必须显式传入：
       --fusion_mode geometry

2. 不再搜索 gs_residual_weight。将其固定为：
       gs_residual_weight = 0.1
   该参数只用于固定融合兼容和后续消融。

3. 在 sample_params 中新增：
       fusion_init ∈ {0.05, 0.10, 0.20}
       fusion_beta_max ∈ {0.20, 0.50, 1.00}

4. 当：
       fusion_init >= fusion_beta_max
   时使用 optuna.TrialPruned() 拒绝非法组合。

5. 第一轮 HPO 固定：
       fusion_hidden_dim = 16
       fusion_detach_geometry = 1
       jet_score_temperature = 0.01
       jet_max_shift_samples = 1.0

6. build_command 必须显式传入：
       --fusion_mode geometry
       --fusion_init <trial value>
       --fusion_beta_max <trial value>
       --fusion_hidden_dim 16
       --fusion_detach_geometry 1
       --jet_score_temperature 0.01
       --jet_max_shift_samples 1.0
       --gs_residual_weight 0.1

7. 在 build_command 中增加断言，若 fusion_mode 不是 geometry，
   立即抛出 RuntimeError，禁止静默运行 fixed 模型。

8. 最佳配置 JSON 必须保存：
       fusion_mode
       fusion_init
       fusion_beta_max
       fusion_hidden_dim
       fusion_detach_geometry
       jet_score_temperature
       jet_max_shift_samples
       gs_residual_weight

9. PARAM_KEYS 删除 gs_residual_weight，加入：
       fusion_init
       fusion_beta_max

10. 若支持 --use_seed，seed loader 不再加载 gs_residual_weight，
    改为加载 fusion_init 和 fusion_beta_max；缺失时使用 0.1 和 0.5。

11. 五个数据集脚本必须保持完全相同的 geometry fusion 搜索逻辑。

12. 不修改 run_adaptive_tuning.py。修复后原有外层启动命令应直接可用。

13. 增加一轮 1-trial sanity test，确认打印出的 run.py 子命令明确包含：
       --fusion_mode geometry
    并确认最终 JSON 保存了 fusion 参数。
```

完成这项修复后，你原来的外层 HPO 命令就能真正搜索第二优先级后的 **geometry-conditioned fusion 模型**，而不再静默退回固定残差融合。