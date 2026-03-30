# ICL-Thesis

**Conditional Conformal Prediction for In-Context Learning: Architecture Scaling, Task Complexity, and Signal-to-Noise Effects**

*Master's Thesis, Department of Statistics, UCLA, 2026*

---

## Overview

This repository contains all code for the thesis, organized into two independent modules:

| Module | Description | Key Question |
|--------|-------------|--------------|
| [`train_icl/`](#train_icl) | Train GPT-2 style transformers from scratch on synthetic regression tasks | How do architecture, task complexity, noise, and dimensionality affect ICL and conformal prediction? |
| [`pretrain_icl/`](#pretrain_icl) | Evaluate pretrained LLMs (Nemotron-120B, Mistral-Large, Qwen3) on the same tasks via vLLM | What regression mechanism does pretrained ICL implement? |

## Repository Structure

```
icl-thesis/
├── train_icl/                    # Part I: Trained Transformers
│   ├── train.py                  #   Training entry point
│   ├── models.py                 #   GPT-2 decoder architectures
│   ├── tasks.py                  #   Task definitions (NLR, NQR, 2NN, DT)
│   ├── samplers.py               #   Data samplers
│   ├── curriculum.py             #   Dimension & context length curriculum
│   ├── eval.py                   #   General evaluation utilities
│   ├── eval_icl_lr2x_speedcp.py  #   SpeedCP conformal evaluation
│   ├── eval_icl_lr2x_ci.py       #   RMSE with confidence intervals
│   ├── eval_icl_curve.py         #   ICL learning curves
│   ├── uq/                       #   Uncertainty quantification
│   │   ├── speedcp_conformal.py  #     SpeedCP implementation
│   │   └── conditional_conformal.py  # CondConf implementation
│   └── conf/gpt/                 #   All experiment configs (S13-S84)
│
├── pretrain_icl/                 # Part II: Pretrained LLM Evaluation
│   ├── data_utils.py             #   Data generation + prompt building
│   │                             #     (words2numbers + 10 other formats)
│   ├── tasks.py                  #   Task definitions (self-contained copy)
│   ├── samplers.py               #   Data samplers (self-contained copy)
│   ├── run_icl_sweep.py          #   Full sweep: tasks x dims x L
│   ├── run_snr_sweep.py          #   Noise sensitivity experiment
│   ├── plot_appendix_figures.py  #   ICL curves, dim/scale effect figures
│   ├── plot_snr_figure.py        #   SNR sensitivity figure
│   ├── plot_mechanism_comparison.py  # Mechanism comparison (6-panel)
│   └── README.md                 #   Detailed usage instructions
│
├── README.md                     # This file
└── environment.yml               # Conda environment
```

---

## `train_icl/`

Code for training GPT-2 style transformers from scratch on synthetic regression, then evaluating ICL performance and SpeedCP conformal prediction intervals.

### Experiments

| Thesis Section | Experiments | What Varies | Architecture |
|----------------|-------------|-------------|--------------|
| Results I: Architecture Scaling | S13-S24 | Width, depth, GPT-2 presets | Variable |
| Results II: Task & Noise | S52-S67 | 4 tasks x 4 noise levels | w256 d12 |
| Results III: Dimensionality | S69-S84 | d in {10, 20, 40, 100} | w512 d12 |

### Quick Start

```bash
# Train
python train_icl/train.py --config train_icl/conf/gpt/S16_gpt2_w256_d6_nlr80x40.yaml

# Evaluate RMSE
python train_icl/eval_icl_lr2x_ci.py --results-dir results/ --family s_nlr80_series

# Evaluate SpeedCP conformal intervals
python train_icl/eval_icl_lr2x_speedcp.py --results-dir results/ --family s_nlr80_series
```

### Tasks

All defined in [`train_icl/tasks.py`](train_icl/tasks.py):

| Task | Function | Noise |
|------|----------|-------|
| NLR | $y = w^\top x + \varepsilon$ | $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ |
| NQR | $y = (w^\top x)^2 + \varepsilon$ | Quadratic in projected direction |
| 2NN | $y = W_2 \cdot \text{ReLU}(W_1 x) + \varepsilon$ | 2-layer neural network |
| DT | $y = \text{DecisionTree}(x) + \varepsilon$ | Random decision tree |

---

## `pretrain_icl/`

Self-contained code for evaluating pretrained LLMs on the same synthetic regression tasks via vLLM. No dependency on `train_icl/`.

### Models Evaluated

| Model | Parameters | Type |
|-------|-----------|------|
| Nemotron-120B | 120B (MoE, 12B active) | Completions API |
| Mistral-Large-2411 | 123B | Completions API |
| Qwen3-32B / 14B / 8B / 0.6B | 0.6B - 32B | Completions API |

### Quick Start

```bash
# 1. Start vLLM
python -m vllm.entrypoints.openai.api_server \
    --model <path> --served-model-name <name> \
    --tensor-parallel-size 4 --port 8100 --dtype bfloat16

# 2. Run full sweep
python pretrain_icl/run_icl_sweep.py \
    --model <name> --endpoint http://127.0.0.1:8100/v1/completions

# 3. Run SNR experiment
python pretrain_icl/run_snr_sweep.py --model <name>
```

### Prompt Format

Primary format (`words2numbers`):
```
The task is to provide your best estimate for "Output".
Output only one number and nothing else.

Feature 0: 0.34
Feature 1: 1.49
Output: 0.38

Feature 0: 0.13
Feature 1: -0.72
Output:
```

10+ additional formats available in [`pretrain_icl/data_utils.py`](pretrain_icl/data_utils.py).

### Key Findings

1. **ICL Curves**: Pretrained LLMs show meaningful ICL at $d \leq 5$, degrading to near-random at $d \geq 10$
2. **Mechanism**: The LLM implements a rank-1 projection mechanism --- effective at $d_\text{eff} = 1$ (matching Ridge regression), but failing catastrophically at $d_\text{eff} \geq 2$ with irrelevant features
3. **Scale Effect**: Larger models (32B vs 0.6B) consistently improve ICL, but the gap vanishes at high dimensions
4. **SNR Sensitivity**: Higher noise degrades both the LLM and baselines, with the LLM's advantage over 1-NN shrinking at high noise

---

## Citation

```bibtex
@mastersthesis{lin2026icl,
  title={Conditional Conformal Prediction for In-Context Learning:
         Architecture Scaling, Task Complexity, and Signal-to-Noise Effects},
  author={Lin, Jinrui},
  school={University of California, Los Angeles},
  department={Department of Statistics},
  year={2026}
}
```

## License

This project is for academic purposes. Please contact the author for any usage beyond personal research.
