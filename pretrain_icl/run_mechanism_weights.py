#!/usr/bin/env python3
"""
Mechanism analysis: recover implicit weights via causal perturbation.
For each context point, perturb y_i by +/- delta, measure prediction change.
Uses vLLM OpenAI-compatible /v1/completions endpoint.

Usage:
    python run_mechanism_weights.py --model <name> --endpoint http://127.0.0.1:8100/v1/completions
"""
import sys, json, time, re, argparse
import numpy as np
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import (
    _seed_everything, collect_xy, _build_prompt, _parse_prediction,
)

TASKS = [
    ("noisy_linear_regression",    {"noise_std": 0.1, "normalize_w": True}, "NLR"),
    ("noisy_quadratic_regression", {"noise_std": 0.1, "normalize_w": True}, "NQR"),
    ("noisy_relu_2nn_regression",  {"noise_std": 0.1},                      "2NN"),
    ("noisy_decision_tree",        {"noise_std": 0.1},                      "DT"),
]

CONDITIONS = [(1, 10), (2, 12), (3, 12), (5, 20)]


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


def recover_weights(api_url, model_name, task_name, task_kwargs, d, L,
                    n_episodes=50, delta=0.3, concurrency=16, timeout=120):
    """Perturb each y_i by +/- delta, measure prediction change to recover weights."""
    _seed_everything(42)
    xs, ys = collect_xy(
        task_name=task_name, task_kwargs=task_kwargs,
        data_name="gaussian", n_dims=d, n_points=L + 1,
        num_eval_examples=n_episodes, batch_size=n_episodes, device="cpu",
    )
    xs_np, ys_np = xs.numpy(), ys.numpy()

    rank_weights = [[] for _ in range(L)]

    for ep in range(n_episodes):
        xs_ctx = xs_np[ep, :L]
        ys_ctx = ys_np[ep, :L]
        xq = xs_np[ep, L]
        dists = np.sqrt(np.sum((xs_ctx - xq) ** 2, axis=1))
        ranks = np.argsort(dists)

        # Build all prompts: base + 2*L perturbations
        prompts = {}
        prompts["base"] = _build_prompt(
            xs_ctx=xs_ctx, ys_ctx=ys_ctx, x_query=xq,
            prompt_style="words2numbers", answer_format="first_number",
            x_decimals=2, y_decimals=2,
        )
        for i in range(L):
            ys_p = ys_ctx.copy(); ys_p[i] += delta
            prompts[f"p_{i}"] = _build_prompt(
                xs_ctx=xs_ctx, ys_ctx=ys_p, x_query=xq,
                prompt_style="words2numbers", answer_format="first_number",
                x_decimals=2, y_decimals=2,
            )
            ys_m = ys_ctx.copy(); ys_m[i] -= delta
            prompts[f"m_{i}"] = _build_prompt(
                xs_ctx=xs_ctx, ys_ctx=ys_m, x_query=xq,
                prompt_style="words2numbers", answer_format="first_number",
                x_decimals=2, y_decimals=2,
            )

        # Concurrent queries
        preds = {}

        def _call(tag, prompt):
            try:
                return tag, query_vllm(api_url, model_name, prompt, timeout)
            except:
                return tag, None

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(_call, t, p) for t, p in prompts.items()]
            for fut in as_completed(futs):
                tag, pred = fut.result()
                preds[tag] = pred

        if preds.get("base") is None:
            continue

        for i in range(L):
            pp = preds.get(f"p_{i}")
            pm = preds.get(f"m_{i}")
            if pp is not None and pm is not None:
                w_i = (pp - pm) / (2 * delta)
                rank = int(np.where(ranks == i)[0][0])
                rank_weights[rank].append(w_i)

        if ep % 10 == 0:
            print(f"    ep {ep}/{n_episodes}")

    return rank_weights


def main():
    parser = argparse.ArgumentParser(description="Mechanism weight recovery")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8100/v1/completions")
    parser.add_argument("--output-dir", type=str, default="results_mechanism")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    all_results = {}
    for task_name, task_kwargs, task_short in TASKS:
        for d, L in CONDITIONS:
            key = f"{task_short}_d{d}_L{L}"
            print(f"{key}:", end=" ", flush=True)
            t0 = time.time()
            rw = recover_weights(
                args.endpoint, args.model, task_name, task_kwargs,
                d, L, args.n_episodes, args.delta, args.concurrency, args.timeout,
            )

            profile = [float(np.mean(rw[r])) if rw[r] else 0.0 for r in range(L)]
            top3 = profile[:3]
            rest = float(np.mean(profile[3:])) if len(profile) > 3 else 0
            print(f"rank0={top3[0]:.3f} rank1={top3[1]:.3f} rank2={top3[2]:.3f} "
                  f"rest={rest:.3f} ({time.time()-t0:.0f}s)")

            all_results[key] = {
                "profile": profile,
                "profile_std": [float(np.std(rw[r])) if rw[r] else 0.0 for r in range(L)],
                "n_per_rank": [len(rw[r]) for r in range(L)],
            }

    out_file = out_dir / f"weight_profile_{args.model}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
