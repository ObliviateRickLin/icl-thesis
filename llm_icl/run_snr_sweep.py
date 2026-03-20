#!/usr/bin/env python3
"""
SNR sensitivity sweep: evaluate ICL regression under different noise levels.
Uses vLLM OpenAI-compatible /v1/completions endpoint.

Usage:
    python run_snr_sweep.py --model <name> --endpoint http://127.0.0.1:8100/v1/completions
"""
import sys, json, time, re, argparse
import numpy as np
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval_icl_llm_rmse import (
    _seed_everything, collect_xy, _build_prompt, _parse_prediction,
)

TASKS_BASE = [
    ("noisy_linear_regression",    {"normalize_w": True}, "NLR"),
    ("noisy_quadratic_regression", {"normalize_w": True}, "NQR"),
    ("noisy_relu_2nn_regression",  {},                    "2NN"),
    ("noisy_decision_tree",        {},                    "DT"),
]

NOISE_STDS = [0.1, 0.25, 0.5, 1.0]
D = 5
L_VALUES = [3, 5, 10, 20, 30, 50]


def query_vllm(api_url, model, prompt, max_tokens=16, temperature=0, timeout=120):
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": temperature,
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
    return pred, raw[:200]


def run_condition(api_url, model, task_name, task_kwargs, task_short,
                  d, L, noise_std, num_eval, seed, concurrency, out_dir, timeout):
    tag = f"{task_short}_d{d}_L{L}_noise{noise_std}_n{num_eval}"
    out_file = out_dir / f"{tag}.json"
    if out_file.exists():
        return None

    kw = dict(task_kwargs)
    kw["noise_std"] = noise_std

    _seed_everything(seed)
    xs, ys = collect_xy(
        task_name=task_name, task_kwargs=kw,
        data_name="gaussian", n_dims=d, n_points=L + 1,
        num_eval_examples=num_eval, batch_size=min(num_eval, 50), device="cpu",
    )
    xs_np, ys_np = xs.numpy(), ys.numpy()

    prompts = []
    for i in range(num_eval):
        p = _build_prompt(
            xs_ctx=xs_np[i, :L], ys_ctx=ys_np[i, :L], x_query=xs_np[i, L],
            prompt_style="words2numbers", answer_format="first_number",
            x_decimals=2, y_decimals=2,
        )
        prompts.append(p)

    results_list = [None] * num_eval

    def _call(idx):
        try:
            return idx, query_vllm(api_url, model, prompts[idx], timeout=timeout)
        except Exception as e:
            return idx, (None, f"ERROR:{e}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(_call, i) for i in range(num_eval)]
        for fut in as_completed(futs):
            idx, (pred, raw) = fut.result()
            results_list[idx] = (pred, raw)

    details = []
    mse_list = []
    invalid = 0
    for i in range(num_eval):
        y_true = float(ys_np[i, L])
        pred, raw = results_list[i]
        xs_ctx, ys_ctx, x_query = xs_np[i, :L], ys_np[i, :L], xs_np[i, L]

        entry = {"episode": i, "target_y": y_true, "pred_y": pred, "raw_text": raw, "valid": pred is not None}
        if pred is not None:
            mse_list.append((y_true - pred) ** 2)
        else:
            invalid += 1

        dists = np.sum((xs_ctx - x_query) ** 2, axis=1)
        entry["knn1_pred"] = float(ys_ctx[np.argmin(dists)])
        entry["mean_y_pred"] = float(np.mean(ys_ctx))
        details.append(entry)

    rmse = np.sqrt(np.mean(mse_list)) if mse_list else float("inf")
    inv_rate = invalid / num_eval

    result = {
        "task": task_short, "task_name": task_name, "d": d, "L": L,
        "noise_std": noise_std, "num_eval": num_eval, "seed": seed,
        "model": model, "rmse": rmse, "invalid_rate": inv_rate,
        "details": details,
    }
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    return rmse, inv_rate


def main():
    parser = argparse.ArgumentParser(description="SNR sensitivity sweep")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8100/v1/completions")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-eval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    out_dir = Path(args.output_dir or f"results_{args.model}_snr")
    out_dir.mkdir(exist_ok=True)

    total = len(TASKS_BASE) * len(NOISE_STDS) * len(L_VALUES)
    print(f"SNR sweep: {args.model}, d={D}, {total} conditions")

    cell = 0
    for task_name, task_kwargs, task_short in TASKS_BASE:
        for noise_std in NOISE_STDS:
            for L in L_VALUES:
                cell += 1
                tag = f"{task_short}_d{D}_L{L}_noise{noise_std}_n{args.num_eval}"
                out_file = out_dir / f"{tag}.json"
                if out_file.exists():
                    print(f"  [{cell}/{total}] {tag} -- SKIP")
                    continue
                t0 = time.time()
                print(f"  [{cell}/{total}] {tag}...", end=" ", flush=True)
                try:
                    result = run_condition(
                        args.endpoint, args.model, task_name, task_kwargs, task_short,
                        D, L, noise_std, args.num_eval, args.seed, args.concurrency, out_dir, args.timeout)
                    if result is None:
                        print("SKIP")
                    else:
                        rmse, inv_rate = result
                        print(f"RMSE={rmse:.4f} inv={inv_rate:.2f} ({time.time()-t0:.0f}s)")
                except Exception as e:
                    print(f"ERROR: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
