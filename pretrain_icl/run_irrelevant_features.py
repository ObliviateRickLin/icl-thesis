#!/usr/bin/env python3
"""
Irrelevant feature experiment: true function uses d_relevant dimensions,
but prompt presents d_present dimensions (d_present >= d_relevant).
Tests whether LLM can filter out irrelevant features.
Uses vLLM OpenAI-compatible /v1/completions endpoint.

Usage:
    python run_irrelevant_features.py --model <name> --endpoint http://127.0.0.1:8100/v1/completions
"""
import sys, json, time, re, argparse
import numpy as np
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import _build_prompt, _parse_prediction

SEEDS = [42, 123, 456, 789, 1024]


def query_vllm(api_url, model, prompt, timeout=120):
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": 16, "temperature": 0,
        "stop": ["\n\n", "\nFeature"],
    }).encode()
    req = urllib.request.Request(api_url, data=body,
                                headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    result = json.loads(resp.read())
    raw = result["choices"][0]["text"].strip()
    pred = _parse_prediction(raw, "first_number")
    if pred is None:
        numbers = re.findall(r'[-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?', raw)
        if numbers:
            pred = float(numbers[0])
    return pred


def run(api_url, model, d_present, d_relevant, w, L, N, seed, concurrency, timeout):
    rng = np.random.RandomState(seed)
    xs = rng.randn(N, L + 1, d_present)
    ys = (xs[:, :, :d_relevant] * w[None, None, :]).sum(axis=2) + 0.1 * rng.randn(N, L + 1)

    prompts = []
    for ep in range(N):
        prompts.append(_build_prompt(
            xs_ctx=xs[ep, :L], ys_ctx=ys[ep, :L], x_query=xs[ep, L],
            prompt_style="words2numbers", answer_format="first_number",
            x_decimals=2, y_decimals=2,
        ))

    preds = [None] * N

    def _call(idx):
        try:
            return idx, query_vllm(api_url, model, prompts[idx], timeout)
        except:
            return idx, None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(_call, i) for i in range(N)]
        for fut in as_completed(futs):
            idx, pred = fut.result()
            preds[idx] = pred

    mse = [((ys[i, L] - preds[i]) ** 2) for i in range(N) if preds[i] is not None]
    return np.sqrt(np.mean(mse)) if mse else float("inf")


def main():
    parser = argparse.ArgumentParser(description="Irrelevant features experiment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8100/v1/completions")
    parser.add_argument("--output-dir", type=str, default="results_irrelevant_features")
    parser.add_argument("--L", type=int, default=20)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    results = {}

    # Experiment 1: d_relevant=1, w=[1], vary d_present
    for d_rel, w_vals in [(1, [1.0]), (2, [1.0, 1.0]), (3, [1.0, 1.0, 1.0])]:
        w = np.array(w_vals)
        label = f"d_rel={d_rel}_w=unit"
        print(f"\n=== {label} ===")
        for d_pres in [d_rel, 5, 10, 20]:
            rmses = []
            for seed in SEEDS:
                r = run(args.endpoint, args.model, d_pres, d_rel, w,
                        args.L, args.N, seed, args.concurrency, args.timeout)
                rmses.append(r)
            mean_r, std_r = np.mean(rmses), np.std(rmses)
            key = f"{label}_d_pres={d_pres}"
            results[key] = {"mean_rmse": mean_r, "std_rmse": std_r, "rmses": rmses}
            print(f"  d_pres={d_pres:2d}: {mean_r:.3f} +/- {std_r:.3f}")

    out_file = out_dir / f"irrelevant_features_{args.model}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
