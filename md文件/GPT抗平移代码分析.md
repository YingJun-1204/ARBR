我认为 antigravity 的判断**基本正确**：你现在这个测试脚本的 `Forecasting Accuracy` 不能证明“抗平移能力”，它更像是在测试 **模型在相邻真实滑动窗口上的普通预测稳定性**。

你的脚本里对 $\Delta t$ 的处理是：

```python
seq_x_0, _, _, _ = test_data[j]
seq_x_dt, seq_y_dt, _, _ = test_data[j + delta_t]
```

然后用 `seq_x_dt` 预测，并和 `seq_y_dt` 比较 MSE/MAE。

这意味着：
**输入窗口平移了，target 也跟着平移了。**

所以模型面对的是一个完全正常的测试样本：

```text
x[j + Δt]  ->  y[j + Δt]
```

而不是一个“被扰动的输入”。时间序列数据本来就是 stride=1 滑动窗口切出来的，训练和测试里到处都是这种相邻窗口。所以 MixLinear 也几乎不变，这是合理的。这个结果不能说明 MixLinear 抗平移很强，也不能说明 SplatTS 的局部平移鲁棒性优势。

------

## 1. 当前脚本测到的到底是什么？

当前 `Forecasting Accuracy` 测的是：

> 模型在真实时间推进后的正常窗口上，预测是否稳定。

它回答的是：

```text
如果我把测试样本从第 j 个换成第 j+Δt 个，模型预测准不准？
```

而不是：

```text
如果同一个样本的输入局部错位了，模型还能不能预测原来的未来？
```

所以这个实验天然会让很多模型都表现稳定，尤其是数据平滑、窗口重叠很高时。

------

## 2. `seq_y_0` 代替 `seq_y_dt` 是否正确？

这要看你的实验目标。

如果你想测 **equivariance**，那么用 `seq_y_dt` 是对的，因为输入和目标一起移动，符合物理时间推进。

如果你想测 **robustness to local temporal misalignment**，那么应该用 `seq_y_0`，但不能简单地把整个输入替换成 `test_data[j + Δt]`。那样会变成：

```text
用未来偏移后的完整窗口 x[j+Δt] 去预测原来的 y[j]
```

这个确实会让模型崩，不一定有意义。

更合理的做法是：

```text
原始样本：
x[j] -> y[j]

抗平移测试：
corrupt_shift(x[j]) -> y[j]
```

也就是：**target 保持 clean，不动；只对输入窗口内部做局部平移扰动。**

这才是你论文里要讲的“局部平移鲁棒性”。

------

## 3. 你真正需要的实验设计

你应该构造一个新的 shifted input：

```python
seq_x_shifted = local_shift(seq_x_0, delta_t)
target = seq_y_0
```

然后评估：

```python
model(seq_x_shifted) vs seq_y_0
```

这样才是在问：

> 输入历史里局部模式发生错位、延迟或相位偏移时，模型还能否预测原本的未来？

这和真实场景更接近，比如传感器局部延迟、局部事件滞后、局部相位漂移。

------

## 4. 什么叫 local shift，建议怎么做？

不要直接把整个 `seq_x` 全局换成 `seq_x[j+Δt]`。建议做 **窗口内部局部扰动**。

比如对输入序列中间一段做平移：

```text
原始输入:
[正常区域 A][局部区域 B][正常区域 C]

扰动输入:
[正常区域 A][B 内部向右/向左平移 Δt][正常区域 C]
```

空出来的位置可以用：

- replication padding；
- linear interpolation；
- neighboring values；
- circular shift，但这个稍微不自然；
- mask 后用原始边界值填充。

论文里最好固定一种简单、可复现的方式，例如：

```text
We apply a localized temporal displacement to a randomly selected segment of the input window while keeping the forecasting target unchanged.
```

这样才是“输入错位但未来真实值不变”的 robustness test。

------

## 5. 那 Equivariance Error 还有用吗？

有用，但它不是主证据。

当前的 `Equivariance Error` 是比较：

```python
outputs_0[:, delta_t:, :]
outputs_dt[:, :overlap_len, :]
```

也就是原始输入预测和真实平移输入预测在物理时间对齐后的差异。

它说明的是：

> 模型对相邻真实窗口的预测轨迹是否平滑、一致。

这个可以作为 **diagnostic metric**，但它不能证明“抗局部扰动”。因为它没有破坏输入，只是在看相邻真实窗口。

所以论文里可以这样处理：

| 指标                                                         | 是否作为主表     | 说明                 |
| ------------------------------------------------------------ | ---------------- | -------------------- |
| `Forecasting Accuracy` with `seq_x_dt -> seq_y_dt`           | 不建议           | 只是正常滑动窗口预测 |
| `Equivariance Error`                                         | 可放附录         | 预测轨迹一致性诊断   |
| `Forecasting Accuracy` with `local_shift(seq_x_0) -> seq_y_0` | **应该作为主表** | 真正的局部平移鲁棒性 |

------

## 6. 你补 LGA、MixLinear、PhaseFormer、WPMixer 是否还合理？

**合理，但前提是要用新的 local-shift perturbation 测试。**

如果继续用当前脚本，所有模型都可能看起来很好，这个表没有区分度。

如果改成：

```text
local_shift(seq_x_0) -> seq_y_0
```

那么补这几个模型就非常合理：

| 大类                                | 模型        | 为什么选它                                  |
| ----------------------------------- | ----------- | ------------------------------------------- |
| Local geometry / geometry attention | LGA         | 对比几何建模方法                            |
| Linear mixer                        | MixLinear   | 对比最新线性混合模型                        |
| Fourier operator                    | PhaseFormer | 对比频域/相位类模型                         |
| Wavelet/time-frequency              | WPMixer     | 对比局部时频模型                            |
| Ours                                | SplatTS     | 连续 Gaussian + adaptive residual alignment |

这个组合很适合正文，不会太臃肿。

但我仍然建议你加一个 **PatchTST**。因为你的故事里有“rigid patch boundary sensitivity”，PatchTST 是最典型的 patch-based baseline。最终表格可以是：

```text
SplatTS, LGA, PatchTST, MixLinear, PhaseFormer, WPMixer
```

这样审稿人会觉得覆盖面更完整。

------

## 7. 接下来怎么改实验路线？

我建议你现在暂停用当前脚本做 `Forecasting Accuracy` 主表，改成下面这个路线：

```text
Step 1: 固定 clean test sample: seq_x_0, seq_y_0
Step 2: 对 seq_x_0 内部做 local shift，得到 seq_x_shifted
Step 3: 用 model(seq_x_shifted) 预测
Step 4: 和 seq_y_0 计算 MSE / MAE
Step 5: 对 SplatTS, LGA, PatchTST, MixLinear, PhaseFormer, WPMixer 使用相同扰动
Step 6: 报告不同 Δt 下的 MSE/MAE 或 degradation %
```

正文主表最好报 degradation percentage：

```latex
\mathrm{Deg}(\Delta t)
=
\frac{
\mathrm{MSE}_{\Delta t} - \mathrm{MSE}_{0}
}{
\mathrm{MSE}_{0}
}
\times 100\%.
```

这样一眼就能看出谁抗平移。

------

## 8. 最终判断

antigravity 的分析是对的：**当前 Forecasting Accuracy 稳定，是因为你测试的是合法滑动窗口，不是扰动输入。**

你论文里真正需要的是：

```text
输入发生局部错位，target 保持不变。
```

也就是：

```text
local_shift(seq_x_0) -> seq_y_0
```

而不是：

```text
seq_x_dt -> seq_y_dt
```

所以，LGA / MixLinear / PhaseFormer / WPMixer 的补充实验依然很合理，但要先把测试协议改成真正的 **localized phase-shift perturbation**。