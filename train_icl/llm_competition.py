from __future__ import annotations

import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from eval_icl_llm_rmse import (
    _build_prompt,
    _messages_for_prompt,
    _parse_prediction,
    _postprocess_prediction,
    collect_xy,
)
from llm_cost import count_message_tokens, usage_cost_breakdown
from llm_oai_client import OpenAICompatClient


BASELINE_MEAN = "baseline/mean_y"
BASELINE_KNN = "baseline/knn"

TASK_PRESETS = {
    "Noisy Linear Regression": {
        "task_name": "noisy_linear_regression",
        "default_task_kwargs": {"noise_std": 0.1, "normalize_w": True},
    },
    "Noisy Quadratic Regression": {
        "task_name": "noisy_quadratic_regression",
        "default_task_kwargs": {"noise_std": 0.1, "normalize_w": True},
    },
    "Noisy ReLU 2NN Regression": {
        "task_name": "noisy_relu_2nn_regression",
        "default_task_kwargs": {"noise_std": 0.1},
    },
    "Noisy Decision Tree": {
        "task_name": "noisy_decision_tree",
        "default_task_kwargs": {"noise_std": 0.1, "depth": 4},
    },
}


def default_example_count(n_dims: int) -> int:
    return max(1, int(2 * n_dims))


def baseline_specs(knn_k: int = 5) -> list[dict]:
    return [
        {
            "id": BASELINE_MEAN,
            "name": "Baseline: Mean of context y",
            "kind": "baseline",
            "description": "Predict the average of all in-context y values.",
        },
        {
            "id": BASELINE_KNN,
            "name": f"Baseline: KNN (k<= {int(knn_k)})",
            "kind": "baseline",
            "description": "Fit KNN on the context examples and predict the query y.",
        },
    ]


def display_name_for_model(model_id: str, catalog_map: dict[str, dict], knn_k: int = 5) -> str:
    if model_id == BASELINE_MEAN:
        return "Baseline: Mean of context y"
    if model_id == BASELINE_KNN:
        return f"Baseline: KNN (k<= {int(knn_k)})"
    return catalog_map.get(model_id, {}).get("name", model_id)


@lru_cache(maxsize=1)
def fetch_openrouter_catalog() -> list[dict]:
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    out = []
    for item in payload.get("data", []):
        pricing = item.get("pricing") or {}
        try:
            prompt_rate = float(pricing.get("prompt", "nan"))
        except ValueError:
            prompt_rate = float("nan")
        try:
            completion_rate = float(pricing.get("completion", "nan"))
        except ValueError:
            completion_rate = float("nan")
        out.append(
            {
                "id": item.get("id", ""),
                "name": item.get("name", item.get("id", "")),
                "context_length": int(item.get("context_length") or 0),
                "prompt_rate": prompt_rate,
                "completion_rate": completion_rate,
                "description": item.get("description", ""),
            }
        )
    out.sort(key=lambda x: (x["id"] != BASELINE_MEAN, x["id"]))
    return out


def catalog_by_id() -> dict[str, dict]:
    return {item["id"]: item for item in fetch_openrouter_catalog()}


def _batch_size_for(num_eval_examples: int) -> int:
    num_eval_examples = int(num_eval_examples)
    for candidate in [20, 16, 10, 8, 5, 4, 2, 1]:
        if candidate <= num_eval_examples and num_eval_examples % candidate == 0:
            return candidate
    return num_eval_examples


def _cost_breakdown_for_model(
    *,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    catalog_map: dict[str, dict],
) -> dict[str, float]:
    if model_id in (BASELINE_MEAN, BASELINE_KNN):
        return {
            "prompt_cost_usd": 0.0,
            "completion_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        }
    meta = catalog_map.get(model_id)
    if meta is not None and not math.isnan(float(meta.get("prompt_rate", float("nan")))):
        prompt_cost = float(prompt_tokens) * float(meta["prompt_rate"])
        completion_cost = float(completion_tokens) * float(meta["completion_rate"])
        return {
            "prompt_cost_usd": prompt_cost,
            "completion_cost_usd": completion_cost,
            "total_cost_usd": prompt_cost + completion_cost,
        }
    return usage_cost_breakdown(
        model=model_id,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
    )


def estimate_competition_budget(
    *,
    model_ids: list[str],
    task_name: str,
    task_kwargs: dict,
    n_dims: int,
    example_count: int,
    num_eval_examples: int,
    prompt_style: str,
    answer_format: str,
    expected_completion_tokens: int,
    x_decimals: int = 2,
    y_decimals: int = 2,
    estimate_samples: int = 25,
    seed: int = 0,
) -> dict:
    batch_size = _batch_size_for(num_eval_examples)
    xs, ys = collect_xy(
        task_name=task_name,
        task_kwargs=task_kwargs,
        data_name="gaussian",
        n_dims=int(n_dims),
        n_points=int(example_count) + 1,
        num_eval_examples=int(num_eval_examples),
        batch_size=batch_size,
        device="cpu",
    )
    xs_np = xs.numpy()
    ys_np = ys.numpy()
    total_episodes = int(xs_np.shape[0])
    sample_count = min(int(estimate_samples), total_episodes)
    if sample_count <= 0:
        sample_count = total_episodes
    sample_indices = list(range(sample_count))
    catalog_map = catalog_by_id()

    rows = []
    total_cost = 0.0
    for model_id in model_ids:
        if model_id in (BASELINE_MEAN, BASELINE_KNN):
            rows.append(
                {
                    "model": model_id,
                    "display_name": display_name_for_model(model_id, catalog_map),
                    "prompt_tokens_total": 0,
                    "completion_tokens_total": 0,
                    "prompt_cost_usd": 0.0,
                    "completion_cost_usd": 0.0,
                    "total_cost_usd": 0.0,
                    "estimate_samples": sample_count,
                }
            )
            continue

        prompt_token_sum = 0
        for episode_idx in sample_indices:
            prompt = _build_prompt(
                xs_ctx=xs_np[episode_idx, :example_count, :],
                ys_ctx=ys_np[episode_idx, :example_count],
                x_query=xs_np[episode_idx, example_count, :],
                x_decimals=x_decimals,
                y_decimals=y_decimals,
                prompt_style=prompt_style,
                answer_format=answer_format,
            )
            prompt_token_sum += count_message_tokens(
                model=model_id,
                messages=_messages_for_prompt(prompt),
            )
        avg_prompt_tokens = float(prompt_token_sum) / float(sample_count)
        prompt_tokens_total = int(round(avg_prompt_tokens * float(num_eval_examples)))
        completion_tokens_total = int(expected_completion_tokens) * int(num_eval_examples)
        costs = _cost_breakdown_for_model(
            model_id=model_id,
            prompt_tokens=prompt_tokens_total,
            completion_tokens=completion_tokens_total,
            catalog_map=catalog_map,
        )
        total_cost += float(costs["total_cost_usd"])
        rows.append(
            {
                "model": model_id,
                "display_name": display_name_for_model(model_id, catalog_map),
                "prompt_tokens_total": prompt_tokens_total,
                "completion_tokens_total": completion_tokens_total,
                "prompt_cost_usd": costs["prompt_cost_usd"],
                "completion_cost_usd": costs["completion_cost_usd"],
                "total_cost_usd": costs["total_cost_usd"],
                "estimate_samples": sample_count,
            }
        )
    rows.sort(key=lambda x: x["total_cost_usd"], reverse=True)
    return {
        "rows": rows,
        "total_cost_usd": total_cost,
        "task_name": task_name,
        "n_dims": int(n_dims),
        "example_count": int(example_count),
        "num_eval_examples": int(num_eval_examples),
    }


def _baseline_mean_y(ctx_y: np.ndarray) -> float:
    return float(np.mean(ctx_y))


def _baseline_knn(ctx_x: np.ndarray, ctx_y: np.ndarray, x_query: np.ndarray, knn_k: int) -> float:
    k = max(1, min(int(knn_k), int(len(ctx_y))))
    reg = KNeighborsRegressor(n_neighbors=k, weights="distance")
    reg.fit(ctx_x, ctx_y)
    return float(reg.predict(x_query[None, :])[0])


def _summary_rows(detail_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in detail_rows:
        grouped.setdefault(row["model"], []).append(row)

    summary = []
    for model_id, rows in grouped.items():
        valid_rows = [r for r in rows if int(r["valid"]) == 1]
        if valid_rows:
            targets = np.array([float(r["target_y"]) for r in valid_rows], dtype=np.float64)
            preds = np.array([float(r["pred_y"]) for r in valid_rows], dtype=np.float64)
            mse = float(np.mean((targets - preds) ** 2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(targets - preds)))
        else:
            mse = float("nan")
            rmse = float("inf")
            mae = float("nan")

        summary.append(
            {
                "model": model_id,
                "display_name": rows[0]["display_name"],
                "rmse": rmse,
                "mse": mse,
                "mae": mae,
                "num_eval_examples": len(rows),
                "num_valid": len(valid_rows),
                "invalid_rate": float(len(rows) - len(valid_rows)) / float(len(rows) or 1),
                "avg_latency_s": float(np.mean([float(r["latency_s"]) for r in rows])) if rows else float("nan"),
                "prompt_tokens_total": int(sum(int(r["prompt_tokens"]) for r in rows)),
                "completion_tokens_total": int(sum(int(r["completion_tokens"]) for r in rows)),
                "total_cost_usd": float(sum(float(r["total_cost_usd"]) for r in rows)),
            }
        )
    summary.sort(key=lambda x: (float(x["rmse"]), float(x["invalid_rate"]), float(x["avg_latency_s"])))
    return summary


def episode_views(detail_rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in detail_rows:
        grouped.setdefault(int(row["episode_idx"]), []).append(row)
    out = []
    for episode_idx in sorted(grouped.keys()):
        rows = grouped[episode_idx]
        outputs = []
        for row in rows:
            outputs.append(
                {
                    "model": row["display_name"],
                    "raw_text": row["raw_text"],
                    "pred_y": row["pred_y"],
                    "target_y": row["target_y"],
                    "mse": row["mse"],
                    "valid": row["valid"],
                }
            )
        out.append(
            {
                "episode_idx": episode_idx,
                "prompt": rows[0]["prompt"],
                "target_y": rows[0]["target_y"],
                "outputs": outputs,
            }
        )
    return out


def run_competition(
    *,
    model_ids: list[str],
    api_key: str,
    task_name: str,
    task_kwargs: dict,
    n_dims: int,
    example_count: int,
    num_eval_examples: int,
    prompt_style: str,
    answer_format: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    x_decimals: int = 2,
    y_decimals: int = 2,
    knn_k: int = 5,
    timeout_s: float = 120.0,
    max_retries: int = 3,
    model_extra_body_map: dict[str, dict] | None = None,
    max_concurrency: int = 8,
    progress_callback=None,
) -> dict:
    batch_size = _batch_size_for(num_eval_examples)
    xs, ys = collect_xy(
        task_name=task_name,
        task_kwargs=task_kwargs,
        data_name="gaussian",
        n_dims=int(n_dims),
        n_points=int(example_count) + 1,
        num_eval_examples=int(num_eval_examples),
        batch_size=batch_size,
        device="cpu",
    )
    xs_np = xs.numpy()
    ys_np = ys.numpy()
    catalog_map = catalog_by_id()
    detail_rows = []
    total_steps = max(1, int(len(model_ids) * int(num_eval_examples)))
    step_idx = 0
    model_extra_body_map = model_extra_body_map or {}
    model_index = {model_id: idx for idx, model_id in enumerate(model_ids)}

    def _complete_remote(
        *,
        model_id: str,
        prompt: str,
        messages: list[dict],
        target_y: float,
        episode_idx: int,
        ys_ctx: np.ndarray,
    ) -> dict:
        client = OpenAICompatClient(
            base_url="https://openrouter.ai/api/v1",
            model=model_id,
            api_key=api_key,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )
        t0 = time.perf_counter()
        raw_text, response_json = client.complete_text(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            extra_body=model_extra_body_map.get(model_id),
            stop=None,
        )
        latency_s = time.perf_counter() - t0
        raw_pred = _parse_prediction(raw_text, answer_format)
        pred = _postprocess_prediction(raw_pred, ys_ctx, prompt_style)
        usage = response_json.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        costs = _cost_breakdown_for_model(
            model_id=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            catalog_map=catalog_map,
        )
        valid = int(pred is not None)
        mse = "" if pred is None else float((target_y - float(pred)) ** 2)
        return {
            "episode_idx": int(episode_idx),
            "model": model_id,
            "display_name": display_name_for_model(model_id, catalog_map, knn_k=knn_k),
            "prompt": prompt,
            "raw_text": raw_text,
            "target_y": float(target_y),
            "pred_y": "" if pred is None else float(pred),
            "valid": valid,
            "mse": mse,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "latency_s": float(latency_s),
            **costs,
        }

    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as executor:
        for episode_idx in range(int(num_eval_examples)):
            prompt = _build_prompt(
                xs_ctx=xs_np[episode_idx, :example_count, :],
                ys_ctx=ys_np[episode_idx, :example_count],
                x_query=xs_np[episode_idx, example_count, :],
                x_decimals=x_decimals,
                y_decimals=y_decimals,
                prompt_style=prompt_style,
                answer_format=answer_format,
            )
            messages = _messages_for_prompt(prompt)
            ctx_x = xs_np[episode_idx, :example_count, :]
            ctx_y = ys_np[episode_idx, :example_count]
            x_query = xs_np[episode_idx, example_count, :]
            target_y = float(ys_np[episode_idx, example_count])

            for model_id in model_ids:
                if model_id == BASELINE_MEAN:
                    pred = _baseline_mean_y(ctx_y)
                    raw_text = f"{pred:.{y_decimals}f}"
                    row = {
                        "episode_idx": int(episode_idx),
                        "model": model_id,
                        "display_name": display_name_for_model(model_id, catalog_map, knn_k=knn_k),
                        "prompt": prompt,
                        "raw_text": raw_text,
                        "target_y": float(target_y),
                        "pred_y": float(pred),
                        "valid": 1,
                        "mse": float((target_y - float(pred)) ** 2),
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_s": 0.0,
                        **_cost_breakdown_for_model(
                            model_id=model_id,
                            prompt_tokens=0,
                            completion_tokens=0,
                            catalog_map=catalog_map,
                        ),
                    }
                    detail_rows.append(row)
                    step_idx += 1
                    if progress_callback is not None:
                        progress_callback(step_idx, total_steps, model_id, episode_idx)
                elif model_id == BASELINE_KNN:
                    pred = _baseline_knn(ctx_x, ctx_y, x_query, knn_k=knn_k)
                    raw_text = f"{pred:.{y_decimals}f}"
                    row = {
                        "episode_idx": int(episode_idx),
                        "model": model_id,
                        "display_name": display_name_for_model(model_id, catalog_map, knn_k=knn_k),
                        "prompt": prompt,
                        "raw_text": raw_text,
                        "target_y": float(target_y),
                        "pred_y": float(pred),
                        "valid": 1,
                        "mse": float((target_y - float(pred)) ** 2),
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "latency_s": 0.0,
                        **_cost_breakdown_for_model(
                            model_id=model_id,
                            prompt_tokens=0,
                            completion_tokens=0,
                            catalog_map=catalog_map,
                        ),
                    }
                    detail_rows.append(row)
                    step_idx += 1
                    if progress_callback is not None:
                        progress_callback(step_idx, total_steps, model_id, episode_idx)
                else:
                    future = executor.submit(
                        _complete_remote,
                        model_id=model_id,
                        prompt=prompt,
                        messages=messages,
                        target_y=target_y,
                        episode_idx=episode_idx,
                        ys_ctx=ctx_y,
                    )
                    futures[future] = (model_id, episode_idx)

        for future in as_completed(futures):
            model_id, episode_idx = futures[future]
            row = future.result()
            detail_rows.append(row)
            step_idx += 1
            if progress_callback is not None:
                progress_callback(step_idx, total_steps, model_id, episode_idx)

    detail_rows.sort(key=lambda row: (int(row["episode_idx"]), model_index.get(row["model"], 9999)))

    return {
        "detail_rows": detail_rows,
        "summary_rows": _summary_rows(detail_rows),
        "episode_views": episode_views(detail_rows),
    }
