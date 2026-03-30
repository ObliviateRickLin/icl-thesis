# pretrain_icl — Pretrained LLM In-Context Learning Evaluation

Self-contained code for evaluating pretrained LLMs on synthetic regression tasks via ICL.
All experiments use **vLLM** OpenAI-compatible `/v1/completions` endpoint.

## Directory Structure

```
pretrain_icl/
├── data_utils.py            # Core: data generation, prompt building, prediction parsing
│                            #   - words2numbers prompt format (and 10+ other styles)
│                            #   - collect_xy(): generate (x, y) data from task definitions
│                            #   - _build_prompt(): format data into text prompts
│                            #   - _parse_prediction(): extract numbers from LLM output
├── tasks.py                 # Task definitions: NLR, NQR, 2NN, Decision Tree
├── samplers.py              # Data samplers: Gaussian
├── run_icl_sweep.py         # Full sweep: 4 tasks × 10 dims × L values
├── run_snr_sweep.py         # SNR experiment: 4 noise levels × 4 tasks
├── run_mechanism_weights.py # Causal weight recovery via y-perturbation
├── run_irrelevant_features.py # Irrelevant feature filtering experiment
├── extract_attention.py     # Attention weight analysis (requires HuggingFace)
├── plot_appendix_figures.py # ICL curves, dimension/scale effect figures
├── plot_snr_figure.py       # SNR sensitivity figure
├── plot_mechanism_comparison.py # Mechanism comparison figure (6-panel)
└── README.md
```

## Quick Start

```bash
# 1. Start vLLM with any model
python -m vllm.entrypoints.openai.api_server \
    --model <path> --served-model-name <name> \
    --tensor-parallel-size 4 --port 8100 --dtype bfloat16

# 2. Run full sweep
python run_icl_sweep.py --model <name> --endpoint http://127.0.0.1:8100/v1/completions

# 3. Run mechanism analysis
python run_mechanism_weights.py --model <name>

# 4. Run irrelevant feature experiment
python run_irrelevant_features.py --model <name>
```

## Prompt Styles

All styles are in `data_utils.py::_build_prompt()`:

| Style | Description | Used in paper |
|-------|-------------|---------------|
| `words2numbers` | `Feature 0: 0.34 \n Output: 0.38` | Primary (all experiments) |
| `words2numbers_linear_informed` | Adds "noisy linear regression" hint | Ablation |
| `words2numbers_calibrated` | Calibrated instruction | Ablation |
| `words2numbers_fixed` / `_scientific` | Fixed-point / scientific notation | Ablation |
| `compact` | `x=[0.34, 1.49] y=0.38` | Ablation |
| `compact_io` | `Input: ... Output: ...` | Ablation |
| `csv_real` | CSV format | Ablation |
| `mapping_arrow` | `[0.34, 1.49] -> 0.38` | Ablation |
| `scaled_integer_csv` | Integer-scaled CSV | Ablation |
| `jsonl_integer` | JSON lines | Ablation |

## Models Evaluated

| Model | Parameters | Endpoint |
|-------|-----------|----------|
| Nemotron-120B | 120B (MoE) | vLLM |
| Mistral-Large-2411 | 123B | vLLM |
| Qwen3-32B / 14B / 8B / 0.6B | 0.6B–32B | vLLM |

## Relationship to `train_icl/` (train_icl)

`train_icl/` contains code for **training** transformers from scratch on ICL tasks (the main text experiments). `pretrain_icl/` is **self-contained** and evaluates pretrained LLMs without training. Shared files (`tasks.py`, `samplers.py`) are copied here for independence.
