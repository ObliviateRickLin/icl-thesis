# moe-icl (src-only)

这份 README 按 `main.tex` 的实验结构来写，目标是回答两件事：
1. 每一部分实验到底看哪些代码。
2. SNR 在这个仓库里具体是什么意思、由哪个参数控制、怎么读结果。

## 1. 当前仓库范围

当前分支只保留了 `src/` 主代码和 `src/conf/gpt/` 配置，`scaling_report/` 已删除。

核心目录：

- `src/train.py`: 训练入口（Quinine 配置驱动）
- `src/models.py`, `src/base_models.py`, `src/moe.py`: 模型定义（GPT / MoE）
- `src/tasks.py`: 任务定义（线性、二次、2NN、决策树 + noisy 版本）
- `src/samplers.py`: 输入采样（高斯）
- `src/curriculum.py`: 维度与上下文长度课程学习
- `src/eval.py`: 通用评估（mean/std/bootstrap）
- `src/uq/conditional_conformal.py`: CondConf 封装
- `src/uq/speedcp_conformal.py`: SpeedCP 封装
- `src/conf/gpt/*.yaml`: S01-S84、E 系列配置

## 2. main.tex 章节 -> 代码对照总表

| main.tex 实验部分 | 配置文件 | 训练代码 | 评估代码（RMSE/误差） | 评估代码（覆盖率/区间宽度） |
|---|---|---|---|---|
| Experiment One: Dense Transformer Scaling (S01-S12) | `src/conf/gpt/S01_...yaml` 到 `S12_...yaml` | `src/train.py` | `src/eval_icl_curve.py`, `src/eval_icl_lr2x_ci.py` | `src/eval_icl_lr2x_conformal.py`, `src/eval_icl_lr2x_condconf.py`, `src/eval_icl_lr2x_speedcp.py` |
| Experiment Two: UQ with CondConf + s_proj | 同上（通常基于 S01-S12 结果） | 无新增训练 | 误差仍用上面 RMSE 脚本 | 重点看 `eval_icl_lr2x_condconf.py`（`--x-features s_proj`）与 `eval_icl_lr2x_conformal.py` |
| Architecture Scaling (S13-S24, noisy linear) | `src/conf/gpt/S13_...yaml` 到 `S24_...yaml` | `src/train.py` | `src/eval_icl_lr2x_ci.py`（family=`s_nlr80_series`）或 `src/eval_icl_curve.py` | `src/eval_icl_lr2x_speedcp.py` / `src/eval_icl_lr2x_condconf.py` |
| Noise / SNR 系列（S25-S39、S52-S67） | `src/conf/gpt/S25_...` 到 `S67_...` | `src/train.py` | `src/eval_icl_mix_noise2.py`, `src/eval_icl_mix_noise2_ci.py`, `src/eval_icl_curve.py` | `src/eval_icl_mix_noise2_conformal.py`, `src/eval_icl_lr2x_speedcp.py` |
| Dimension 扫描（S69-S84） | `src/conf/gpt/S69_...` 到 `S84_...` | `src/train.py` | `src/eval_icl_curve.py`（按 exp 列表跑） | `src/eval_icl_lr2x_speedcp.py`（通常按 run-dir 单独评估） |

## 3. 分实验详细说明

### 3.1 Experiment One（S01-S12）

你要看的最小代码集合：

- 配置：`src/conf/gpt/S01_gpt2_w32_d6_lr.yaml` 到 `src/conf/gpt/S12_gpt2_large_lr.yaml`
- 任务定义：`src/tasks.py` 里的 `LinearRegression`
- 训练：`src/train.py`
- ICL 曲线/CI：`src/eval_icl_lr2x_ci.py`（含 `s_series`）
- 通用曲线：`src/eval_icl_curve.py`

建议先读顺序：

1. `S01/S12` 配置（看 `model`、`training.tasks`、`curriculum`）
2. `tasks.py::LinearRegression.evaluate`
3. `train.py`（`train_step` + `Curriculum` 的使用）
4. `eval_icl_lr2x_ci.py`（怎么从 per-position 指标得到 ICL length 曲线）

### 3.2 Experiment Two（Split CP vs CondConf + s_proj）

你要看的最小代码集合：

- Split conformal 基线：`src/eval_icl_lr2x_conformal.py`
- CondConf 主脚本：`src/eval_icl_lr2x_condconf.py`
- CondConf 封装：`src/uq/conditional_conformal.py`
- SpeedCP 版本：`src/eval_icl_lr2x_speedcp.py` + `src/uq/speedcp_conformal.py`

`s_proj` 的实现位置：

- `src/eval_icl_lr2x_condconf.py` 中 `_s_proj(...)`
- `src/eval_icl_lr2x_speedcp.py` 中 `_s_proj(...)`

对应 main.tex 的重点参数：

- `--x-features s_proj`
- `--phi linear`
- `--alpha 0.05`
- `--calib-frac 0.5`
- `--diagnostic-bins 10`（条件分箱诊断）

### 3.3 S13-S24（Architecture + noisy linear）

你要看的最小代码集合：

- 配置：`src/conf/gpt/S13_...` 到 `S24_...`
- noisy 任务：`src/tasks.py::NoisyLinearRegression`
- RMSE 曲线：`src/eval_icl_lr2x_ci.py`（`s_nlr80_series`）
- 区间宽度/覆盖率：`src/eval_icl_lr2x_speedcp.py` 或 `src/eval_icl_lr2x_condconf.py`

### 3.4 S52-S67（Task x Noise）

任务家族和代码对应：

- NLR: `NoisyLinearRegression`
- NQR: `NoisyQuadraticRegression`
- 2NN: `NoisyRelu2nnRegression`
- NDT: `NoisyDecisionTree`

都在 `src/tasks.py`。

配置对应：

- NLR: `S52-S55`
- NQR: `S56-S59`
- 2NN: `S60-S63`
- NDT: `S64-S67`

### 3.5 S69-S84（维度扫描）

你要看的最小代码集合：

- 配置：`src/conf/gpt/S69_...` 到 `S84_...`
- 训练仍是 `src/train.py`
- 曲线评估建议从 `src/eval_icl_curve.py` 开始
- 覆盖率/宽度走 `src/eval_icl_lr2x_speedcp.py`（常见做法是逐 run-dir 评估）

## 4. SNR 在这个项目里的定义（重点）

### 4.1 参数对应

在本项目中，噪声强度由配置里的：

- `training.tasks[*].kwargs.noise_std`

控制，并在 `src/tasks.py` 的 noisy 任务中以：

- `ys_noisy = ys_clean + N(0, noise_std^2)`

实现。

也就是：

- `sigma = noise_std`
- `y = f(x) + epsilon, epsilon ~ N(0, sigma^2)`

### 4.2 与 SNR 的关系

常用定义（功率 SNR）：

- `SNR_power = Var(signal) / Var(noise)`

在你这套配置里（尤其线性回归），大量实验使用 `normalize_w: True`，并且 `x ~ N(0, I)`，会让信号尺度大体稳定在同一量级，因此可以近似把：

- `SNR_power` 看作与 `1 / sigma^2` 同量级。

对应关系（近似）：

- `sigma=0.1` -> `SNR_power ~ 100`（高 SNR）
- `sigma=0.25` -> `SNR_power ~ 16`
- `sigma=0.5` -> `SNR_power ~ 4`
- `sigma=1.0` -> `SNR_power ~ 1`（低 SNR）

这和 main.tex 里的结论一致：sigma 越大，RMSE floor 越高，区间越难收缩。

### 4.3 混合噪声实验怎么对应

两种模式：

1. 多任务混合噪声（配置里直接写多个 task，不同 `noise_std`）
   - 例如 `S32_gpt2_w64_d12_mixnoise4_80x40.yaml`
2. 单 task 内部随机噪声级别（代码内采样）
   - `src/tasks.py::NoisyQuadraticRegressionMix4`

### 4.4 一个易踩坑

`src/tasks.py` 里的 noisy 任务都支持 `renormalize_ys`。如果打开它，会改变输出尺度，从而改变“你以为的 SNR”。
当前 S 系列配置默认是按 `noise_std` 直接加噪，通常不启用该选项。

## 5. 最小可复现实验路径（按 main.tex）

### 5.1 训练一个 S01

```bash
python src/train.py --config src/conf/gpt/S01_gpt2_w32_d6_lr.yaml
```

### 5.2 画 S01-S12 的 ICL 误差曲线（CI）

```bash
python src/eval_icl_lr2x_ci.py --results-dir ../results --family s_series --prefer-model-step 400000
```

### 5.3 跑 CondConf（s_proj）

```bash
python src/eval_icl_lr2x_condconf.py \
  --results-dir ../results \
  --family s_series \
  --alpha 0.05 \
  --x-features s_proj \
  --phi linear \
  --calib-frac 0.5 \
  --num-eval-examples 6400 \
  --diagnostic-bins 10
```

## 6. 当前分支的已知状态

当前 `src/` 中以下评估脚本依赖 `ckpt_utils.py`：

- `eval_icl_lr2x_ci.py`
- `eval_icl_lr2x_conformal.py`
- `eval_icl_lr2x_condconf.py`
- `eval_icl_lr2x_speedcp.py`
- `eval_icl_mix_noise2.py`
- `eval_icl_mix_noise2_ci.py`
- `eval_icl_mix_noise2_conformal.py`

如果你保持“只留 src 且已删 `ckpt_utils.py`”的状态，上述脚本会先报 `ModuleNotFoundError`，需要先恢复该工具文件后再跑。
