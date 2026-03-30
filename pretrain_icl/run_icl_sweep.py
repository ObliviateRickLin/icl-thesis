#!/usr/bin/env python3
"""
Pretrained LLM ICL sweep: evaluate regression performance across tasks, dimensions, and context lengths.
Uses vLLM OpenAI-compatible /v1/completions endpoint. Works with any model served by vLLM.

Usage:
    # Start vLLM first:
    python -m vllm.entrypoints.openai.api_server \
        --model <model_path> --served-model-name <name> \
        --tensor-parallel-size 4 --port 8100 --dtype bfloat16

    # Then run sweep:
    python run_icl_sweep.py --model <name> --endpoint http://127.0.0.1:8100/v1/completions \
        --output-dir results_<name>
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

# ── Default configs ──
TASKS = [
    ("noisy_linear_regression",    {"noise_std": 0.1, "normalize_w": True}, "NLR"),
    ("noisy_quadratic_regression", {"noise_std": 0.1, "normalize_w": True}, "NQR"),
    ("noisy_relu_2nn_regression",  {"noise_std": 0.1},                      "2NN"),
    ("noisy_decision_tree",        {"noise_std": 0.1},                      "DT"),
]

DIM_L_CONFIGS = {
    1:   [1, 2, 4, 6, 10, 20, 40],
    2:   [1, 2, 4, 8, 12, 20, 40],
    3:   [2, 3, 6, 12, 18, 30, 60],
    4:   [2, 4, 8, 16, 24, 40, 80],
    5:   [3, 5, 10, 20, 30, 50, 100],
    8:   [4, 8, 16, 32, 48, 80],
    10:  [5, 10, 20, 40, 60, 100],
    20:  [10, 20, 40, 80, 120, 200],
    40:  [20, 40, 60, 80, 120, 160, 240],
    100: [10, 20, 30, 40, 50, 75, 100],
}


def query_vllm(api_url, model, prompt, max_tokens=16, temperature=0, timeout=120):
    """Query vLLM completions endpoint."""
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


def run_condition(api_url, model, task_name, task_kwargs, task_short, d, L,
                  num_eval, seed, concurrency, out_dir, timeout):
    """Run one (task, d, L) condition."""
    out_file = out_dir / f"{task_short}_d{d}_L{L}_n{num_eval}.json"
    if out_file.exists():
        return None

    _seed_everything(seed)
    xs, ys = collect_xy(
        task_name=task_name, task_kwargs=task_kwargs,
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
        xs_ctx = xs_np[i, :L]
        ys_ctx = ys_np[i, :L]
        x_query = xs_np[i, L]

        entry = {
            "episode": i, "target_y": y_true,
            "pred_y": pred, "raw_text": raw,
            "valid": pred is not None,
        }

        if pred is not None:
            mse_list.append((y_true - pred) ** 2)
        else:
            invalid += 1

        # Baselines
        dists = np.sum((xs_ctx - x_query) ** 2, axis=1)
        entry["knn1_pred"] = float(ys_ctx[np.argmin(dists)])
        k = min(5, L)
        entry["knn5_pred"] = float(np.mean(ys_ctx[np.argsort(dists)[:k]]))
        entry["mean_y_pred"] = float(np.mean(ys_ctx))
        if L > d + 1:
            try:
                X_aug = np.column_stack([xs_ctx, np.ones(L)])
                beta = np.linalg.lstsq(X_aug, ys_ctx, rcond=None)[0]
                entry["ols_pred"] = float(np.append(x_query, 1.0) @ beta)
            except:
                entry["ols_pred"] = None
        else:
            entry["ols_pred"] = None
        details.append(entry)

    rmse = np.sqrt(np.mean(mse_list)) if mse_list else float("inf")
    inv_rate = invalid / num_eval

    result = {
        "task": task_short, "task_name": task_name, "d": d, "L": L,
        "num_eval": num_eval, "seed": seed, "model": model,
        "rmse": rmse, "invalid_rate": inv_rate,
        "details": details,
    }
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    return rmse, inv_rate


def main():
    parser = argparse.ArgumentParser(description="Pretrained LLM ICL sweep")
    parser.add_argument("--model", type=str, required=True, help="Model name (as served by vLLM)")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8100/v1/completions")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-eval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    out_dir = Path(args.output_dir or f"results_{args.model}")
    out_dir.mkdir(exist_ok=True)

    total = sum(len(ls) for ls in DIM_L_CONFIGS.values()) * len(TASKS)
    print(f"LLM ICL sweep: {args.model}")
    print(f"  {len(TASKS)} tasks x {len(DIM_L_CONFIGS)} dims = {total} conditions")
    print(f"  n={args.num_eval}, endpoint={args.endpoint}")

    # Test API
    try:
        body = json.dumps({"model": args.model, "prompt": "1+1=", "max_tokens": 8, "temperature": 0}).encode()
        req = urllib.request.Request(args.endpoint, data=body, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=10)
        r = json.loads(resp.read())
        print(f"  API test OK: '{r['choices'][0]['text'].strip()}'")
    except Exception as e:
        print(f"  API test FAILED: {e}")
        return

    cell = 0
    for task_name, task_kwargs, task_short in TASKS:
        print(f"\n{'='*70}")
        print(f"Task: {task_short} ({task_name})")
        print(f"{'='*70}")

        for d in sorted(DIM_L_CONFIGS.keys()):
            for L in DIM_L_CONFIGS[d]:
                cell += 1
                out_file = out_dir / f"{task_short}_d{d}_L{L}_n{args.num_eval}.json"
                if out_file.exists():
                    print(f"  [{cell}/{total}] {task_short} d={d} L={L} -- SKIP")
                    continue

                t0 = time.time()
                print(f"  [{cell}/{total}] {task_short} d={d} L={L}...", end=" ", flush=True)
                try:
                    result = run_condition(
                        args.endpoint, args.model, task_name, task_kwargs, task_short,
                        d, L, args.num_eval, args.seed, args.concurrency, out_dir, args.timeout)
                    if result is None:
                        print("SKIP")
                    else:
                        rmse, inv_rate = result
                        elapsed = time.time() - t0
                        print(f"RMSE={rmse:.4f} inv={inv_rate:.2f} ({elapsed:.0f}s)")
                except Exception as e:
                    print(f"ERROR: {e}")

    # Summary CSV
    import csv
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "task", "d", "L", "rmse", "invalid_rate"])
        for jf in sorted(out_dir.glob("*.json")):
            try:
                with open(jf) as fh:
                    r = json.load(fh)
                writer.writerow([r["model"], r["task"], r["d"], r["L"],
                                f"{r['rmse']:.6f}", f"{r['invalid_rate']:.4f}"])
            except:
                pass
    print(f"\nSummary: {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()
