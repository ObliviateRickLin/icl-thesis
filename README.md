# moe-icl: Experiments Mapping (aligned with `main.tex`)

这份 README 只做一件事：把 `main.tex` 的 **experiments 部分**和代码一一对齐。

## Experiments at a glance

| `main.tex` section | Experiment IDs | What changes | Core metrics |
|---|---|---|---|
| Results I: Architecture Scaling Effects | `S13-S24` | 架构宽度/深度/GPT2 preset（任务固定 NLR, `sigma=0.1`） | RMSE, SpeedCP width, coverage |
| Results II: Task Complexity and Signal-to-Noise Effects | `S52-S67` | 任务族（NLR/NQR/2NN/NDT）和噪声（`sigma in {0.1,0.25,0.5,1.0}`） | RMSE, SpeedCP width, coverage, final-L box/coverage |
| Results III: Input Dimensionality Effects | `S69-S84` | 维度扫描（`d in {10,20,40,100}`） | final-L width/coverage, width distribution |

---

## 1) Experimental Setup (from `main.tex`)

对应 `main.tex` 的 `Experimental Setup` 三小节：

- Transformer architectures
- Regression tasks + noise
- Conformal evaluation protocol

代码落点：

- 训练主入口: [train.py](src/train.py)
- 任务定义（NLR/NQR/2NN/NDT + noise）: [tasks.py](src/tasks.py)
- 课程学习（维度、点数）: [curriculum.py](src/curriculum.py)
- 通用评估聚合（mean/bootstrap）: [eval.py](src/eval.py)
- SpeedCP/CondConf实现: [speedcp_conformal.py](src/uq/speedcp_conformal.py), [conditional_conformal.py](src/uq/conditional_conformal.py)

配置入口（全部实验配置目录）：

- [src/conf/gpt/](src/conf/gpt)

---

## 2) Results I: Architecture Scaling Effects (`S13-S24`)

这部分固定任务为 noisy linear regression (`sigma=0.1`)，比较架构能力。

### Involved experiments

- Width scaling: `S13-S16`
- Depth scaling: `S17-S20`
- GPT-2 presets: `S21-S24`

配置文件（直接点开）：

- [S13_gpt2_w32_d6_nlr80x40.yaml](src/conf/gpt/S13_gpt2_w32_d6_nlr80x40.yaml)
- [S14_gpt2_w64_d6_nlr80x40.yaml](src/conf/gpt/S14_gpt2_w64_d6_nlr80x40.yaml)
- [S15_gpt2_w128_d6_nlr80x40.yaml](src/conf/gpt/S15_gpt2_w128_d6_nlr80x40.yaml)
- [S16_gpt2_w256_d6_nlr80x40.yaml](src/conf/gpt/S16_gpt2_w256_d6_nlr80x40.yaml)
- [S17_gpt2_w64_d2_nlr80x40.yaml](src/conf/gpt/S17_gpt2_w64_d2_nlr80x40.yaml)
- [S18_gpt2_w64_d4_nlr80x40.yaml](src/conf/gpt/S18_gpt2_w64_d4_nlr80x40.yaml)
- [S19_gpt2_w64_d8_nlr80x40.yaml](src/conf/gpt/S19_gpt2_w64_d8_nlr80x40.yaml)
- [S20_gpt2_w64_d12_nlr80x40.yaml](src/conf/gpt/S20_gpt2_w64_d12_nlr80x40.yaml)
- [S21_gpt2_tiny_nlr80x40.yaml](src/conf/gpt/S21_gpt2_tiny_nlr80x40.yaml)
- [S22_gpt2_small_nlr80x40.yaml](src/conf/gpt/S22_gpt2_small_nlr80x40.yaml)
- [S23_gpt2_medium_nlr80x40.yaml](src/conf/gpt/S23_gpt2_medium_nlr80x40.yaml)
- [S24_gpt2_large_nlr80x40.yaml](src/conf/gpt/S24_gpt2_large_nlr80x40.yaml)

评估脚本：

- RMSE/误差曲线： [eval_icl_lr2x_ci.py](src/eval_icl_lr2x_ci.py)（family: `s_nlr80_series`）
- SpeedCP宽度与覆盖率： [eval_icl_lr2x_speedcp.py](src/eval_icl_lr2x_speedcp.py)

---

## 3) Results II: Task Complexity and Signal-to-Noise Effects (`S52-S67`)

这部分固定架构（`w256 d12`），扫任务和噪声。

### Involved experiments

- NLR: `S52-S55`
- NQR: `S56-S59`
- 2NN: `S60-S63`
- NDT: `S64-S67`

配置文件（每族给一个入口）：

- [S52_gpt2_w256_d12_nlr80x40_noise01.yaml](src/conf/gpt/S52_gpt2_w256_d12_nlr80x40_noise01.yaml)
- [S56_gpt2_w256_d12_nqr200x40_noise01.yaml](src/conf/gpt/S56_gpt2_w256_d12_nqr200x40_noise01.yaml)
- [S60_gpt2_w256_d12_n2nn200x40_noise01.yaml](src/conf/gpt/S60_gpt2_w256_d12_n2nn200x40_noise01.yaml)
- [S64_gpt2_w256_d12_ndt200x40_noise01.yaml](src/conf/gpt/S64_gpt2_w256_d12_ndt200x40_noise01.yaml)
- 其余噪声级别同名前缀：`noise025 / noise05 / noise10`（位于 [src/conf/gpt/](src/conf/gpt)）

评估脚本：

- RMSE/误差趋势： [eval_icl_curve.py](src/eval_icl_curve.py)
- SpeedCP width/coverage： [eval_icl_lr2x_speedcp.py](src/eval_icl_lr2x_speedcp.py)
- 额外 mixed-noise 评估工具： [eval_icl_mix_noise2.py](src/eval_icl_mix_noise2.py), [eval_icl_mix_noise2_ci.py](src/eval_icl_mix_noise2_ci.py), [eval_icl_mix_noise2_conformal.py](src/eval_icl_mix_noise2_conformal.py)

SNR/噪声在代码中的位置：

- 噪声参数：`training.tasks[*].kwargs.noise_std`（见上面各 `S52-S67` 配置）
- 噪声注入实现： [tasks.py](src/tasks.py)
  - `NoisyLinearRegression`
  - `NoisyQuadraticRegression`
  - `NoisyRelu2nnRegression`
  - `NoisyDecisionTree`

---

## 4) Results III: Input Dimensionality Effects (`S69-S84`)

这部分看不同输入维度下的宽度/覆盖率变化。

### Involved experiments

- NLR dims: `S69`, `S73-S75`
- NQR dims: `S70`, `S76-S78`
- 2NN dims: `S71`, `S79-S81`
- NDT dims: `S72`, `S82-S84`

配置文件（代表项）：

- [S69_gpt2_w512_d12_nlr201x100.yaml](src/conf/gpt/S69_gpt2_w512_d12_nlr201x100.yaml)
- [S70_gpt2_w512_d12_nqr501x100.yaml](src/conf/gpt/S70_gpt2_w512_d12_nqr501x100.yaml)
- [S71_gpt2_w512_d12_n2nn501x100.yaml](src/conf/gpt/S71_gpt2_w512_d12_n2nn501x100.yaml)
- [S72_gpt2_w512_d12_ndt501x100.yaml](src/conf/gpt/S72_gpt2_w512_d12_ndt501x100.yaml)
- 其余同组维度配置： [src/conf/gpt/](src/conf/gpt)

评估脚本：

- RMSE/误差曲线： [eval_icl_curve.py](src/eval_icl_curve.py)
- final-L coverage/width（SpeedCP）： [eval_icl_lr2x_speedcp.py](src/eval_icl_lr2x_speedcp.py)

---

## 5) Commands you actually run

训练（任意实验配置）：

```bash
python src/train.py --config src/conf/gpt/<YOUR_EXPERIMENT>.yaml
```

Results I（S13-S24）示例：

```bash
python src/eval_icl_lr2x_ci.py --results-dir ../results --family s_nlr80_series
python src/eval_icl_lr2x_speedcp.py --results-dir ../results --family s_nlr80_series
```

Results II（S52-S67）示例：

```bash
python src/eval_icl_curve.py --results-dir ../results --exps S52_gpt2_w256_d12_nlr80x40_noise01
python src/eval_icl_lr2x_speedcp.py --run-dir ../results/S52_gpt2_w256_d12_nlr80x40_noise01/<run_uuid>
```

Results III（S69-S84）示例：

```bash
python src/eval_icl_curve.py --results-dir ../results --exps S69_gpt2_w512_d12_nlr201x100
python src/eval_icl_lr2x_speedcp.py --run-dir ../results/S69_gpt2_w512_d12_nlr201x100/<run_uuid>
```

---

## 6) Why SNR is consistent in Experiment III

在现在做的这套 **4x4（4 种任务 x 4 个 max-dim）** 里，如果我们把 SNR 理解为

$$
\text{SNR} := \frac{\operatorname{Var}(y_{\text{clean}})}{\operatorname{Var}(\epsilon)}
= \frac{\operatorname{Var}(y_{\text{clean}})}{\sigma^2},
\quad \epsilon \sim \mathcal N(0,\sigma^2)
$$

那么结论是：

- **在同一种实验种类（同一个 task）下，当前 SNR 是常数（不随 dim / L 变化）**
- **并且可以直接算出来（主要由 `noise_std` 决定）**

原因是：`xs` 由高斯采样（`torch.randn(...)`），并且这四类任务的生成都被显式做了“方差归一化”到 O(1) 量级。

### 1) `noisy_linear_regression`（`S69/S73/S74/S75`）

- 配置里 `noise_std=0.1` 且 `normalize_w=True`（例如 [S69_gpt2_w512_d12_nlr201x100.yaml](src/conf/gpt/S69_gpt2_w512_d12_nlr201x100.yaml)）
- 线性回归在 `normalize_w=True` 时使用 `scale / sqrt(n_dims)` 缩放（见 [tasks.py](src/tasks.py)）
- 然后加噪声 `+ N(0, noise_std^2)`

在 `xs ~ N(0,1), w ~ N(0,1), scale=1` 下：

- $\operatorname{Var}(y_{\text{clean}}) \approx 1$
- $\operatorname{Var}(\epsilon) = \sigma^2$

所以：

$$
\text{SNR} \approx \frac{1}{\sigma^2}
$$

当前这组 `sigma=0.1`，即 **SNR 约为 100（约 20 dB）**。

### 2) `noisy_quadratic_regression`（`S70/S76/S77/S78`）

- 配置同样 `normalize_w=True`、`noise_std=0.1`（例如 [S70_gpt2_w512_d12_nqr501x100.yaml](src/conf/gpt/S70_gpt2_w512_d12_nqr501x100.yaml)）
- 二次回归内部有 `1/sqrt(3)` 的对齐缩放，并在 `normalize_w=True` 时再除以 `sqrt(n_dims)`（见 [tasks.py](src/tasks.py)）
- 最后加噪声（同 noisy 体系）

这套缩放让 $\operatorname{Var}(y_{\text{clean}}) \approx 1$，因此同样有：

$$
\text{SNR} \approx \frac{1}{\sigma^2}
$$

`sigma=0.1` 时仍约为 **100**。

### 3) `noisy_relu_2nn_regression`（`S71/S79/S80/S81`）

- 配置里 `hidden_layer_size=4`、`normalize_w=True`、`noise_std=0.1`（例如 [S71_gpt2_w512_d12_n2nn501x100.yaml](src/conf/gpt/S71_gpt2_w512_d12_n2nn501x100.yaml)）
- 2NN 实现会在 `normalize_w=True` 时做 `xs / sqrt(n_dims)`，输出再乘 `sqrt(2/hidden_layer_size)` 稳定方差（见 [tasks.py](src/tasks.py)）
- 最后加 `noise_std` 高斯噪声

因此该任务也被设计为 $\operatorname{Var}(y_{\text{clean}})$ 近似常数（约 1）：

$$
\text{SNR} \approx \frac{1}{\sigma^2}
$$

`sigma=0.1` 时约为 **100**。

### 4) `noisy_decision_tree`（`S72/S82/S83/S84`）

- 配置里 `depth=4`、`noise_std=0.1`（例如 [S72_gpt2_w512_d12_ndt501x100.yaml](src/conf/gpt/S72_gpt2_w512_d12_ndt501x100.yaml)）
- 决策树目标值 `target_tensor` 来自 `torch.randn(...)`（方差近似 1，见 [tasks.py](src/tasks.py)）
- 然后加噪声 `+ N(0, noise_std^2)`

所以也有：

$$
\text{SNR} \approx \frac{1}{\sigma^2}
$$

`sigma=0.1` 时约为 **100**。

### 什么时候 SNR 不是常数

主要两种开关会打破“跨 dim 的常数 SNR”：

1. 关闭 `normalize_w`：线性/二次/2NN 的信号方差会随维度增长（不再除以 $\sqrt{d}$）
2. 打开 `renormalize_ys`：会对 `ys_b_noisy` 做按 std 的重标定，使输出尺度显式带上维度依赖

### 结论（当前 4x4）

对每一种 task（NLR / NQR / 2NN / NDT），当前配置都把 **信号方差稳定在约 1**，同时噪声方差是 $\sigma^2$（这套实验基本是 `sigma=0.1`），所以：

- **SNR 是常数**
- **可近似为：`SNR ~= 1 / noise_std^2`**
- 现在 `noise_std=0.1`，即 **SNR 约为 100（约 20 dB）**
