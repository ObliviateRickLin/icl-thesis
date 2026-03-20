# LLM ICL Regression: Appendix B Code

Code for evaluating pretrained LLM in-context learning on synthetic regression tasks.
All scripts use **vLLM** OpenAI-compatible `/v1/completions` endpoint.

## Setup

1. **Start vLLM** with any supported model:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model <model_path> \
    --served-model-name <name> \
    --tensor-parallel-size 4 \
    --port 8100 \
    --dtype bfloat16 \
    --max-model-len 16384
```

2. **Run experiments** pointing to the vLLM endpoint.

## Scripts

| Script | Paper Section | Description |
|--------|--------------|-------------|
| `run_icl_sweep.py` | B.2 ICL Curves, B.3 Dim Effect, B.4 Scale Effect | Full sweep: 4 tasks × 10 dims × multiple L, n=200 |
| `run_snr_sweep.py` | B.5 Noise Sensitivity | SNR sweep: 4 noise levels × 4 tasks × 6 L values, fixed d=5 |
| `run_mechanism_weights.py` | B.6 Mechanism Analysis | Causal weight recovery via y-perturbation, 4 tasks × 4 dims |
| `run_irrelevant_features.py` | B.6 Mechanism Analysis | Irrelevant feature filtering experiment |
| `plot_appendix_figures.py` | All figures | Generate publication-quality PDF figures |

## Example: Full sweep with Nemotron-120B

```bash
python run_icl_sweep.py \
    --model nemotron-120b \
    --endpoint http://127.0.0.1:8100/v1/completions \
    --output-dir results_nemotron-120b \
    --num-eval 200
```

## Example: Mechanism weight recovery

```bash
python run_mechanism_weights.py \
    --model nemotron-120b \
    --endpoint http://127.0.0.1:8100/v1/completions \
    --n-episodes 50
```

## Models evaluated in the paper

| Model | Parameters | Type |
|-------|-----------|------|
| Nemotron-120B | 120B (MoE) | vLLM completions |
| Mistral-Large-2411 | 123B | vLLM completions |
| Qwen3-32B | 32B | vLLM completions |
| Qwen3-14B | 14B | vLLM completions |
| Qwen3-8B | 8B | vLLM completions |
| Qwen3-0.6B | 0.6B | vLLM completions |

## Core dependencies

- `vllm` (serving)
- `torch`, `numpy`, `scipy` (computation)
- `matplotlib`, `pandas` (plotting)
- `../src/eval_icl_llm_rmse.py` (data generation and prompt building)
- `../src/tasks.py`, `../src/samplers.py` (task definitions)
