from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from eval_icl_llm_rmse import (
    _build_prompt,
    _messages_for_prompt,
    _parse_prediction,
    _postprocess_prediction,
    _seed_everything,
    collect_xy,
)
from llm_oai_client import OpenAICompatClient


MODEL_ID = "nvidia/nemotron-3-super-120b-a12b:free"
EXTRA_BODY = {"reasoning": {"effort": "none", "exclude": True}}
DIMS = [1, 2, 3, 4, 5]
EXAMPLE_COUNTS = list(range(5, 21)) + [24, 28, 32, 36, 40]
NUM_EVAL_EXAMPLES = 20
SEED = 23
TASK_NAME = "noisy_linear_regression"
TASK_KWARGS = {"noise_std": 0.1, "normalize_w": True}
PROMPT_STYLE = "words2numbers"
ANSWER_FORMAT = "strict_number"
SLEEP_S = 0.7

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results_prompt_ablation"
JSON_PATH = OUT_DIR / "nemotron_dim1to5_dense_scan_nlr_n20.json"
PNG_CURVE = OUT_DIR / "nemotron_dim1to5_dense_scan_nlr_n20.png"
PNG_HEAT = OUT_DIR / "nemotron_dim1to5_dense_scan_heatmap_n20.png"


def _load_results() -> dict:
    if JSON_PATH.exists():
        return json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return {
        "config": {
            "task": TASK_NAME,
            "task_kwargs": TASK_KWARGS,
            "model": MODEL_ID,
            "prompt_style": PROMPT_STYLE,
            "answer_format": ANSWER_FORMAT,
            "dims": DIMS,
            "example_counts": EXAMPLE_COUNTS,
            "num_eval_examples": NUM_EVAL_EXAMPLES,
            "seed": SEED,
        },
        "rows": [],
    }


def _save_results(results: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _row_exists(results: dict, d: int, examples: int) -> bool:
    for row in results["rows"]:
        if int(row["d"]) == int(d) and int(row["examples"]) == int(examples):
            return True
    return False


def _plot_results(results: dict) -> None:
    rows = results["rows"]
    if not rows:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, len(DIMS), figsize=(17, 4.2), dpi=180, sharey=True)
    if len(DIMS) == 1:
        axes = [axes]
    for ax, d in zip(axes, DIMS):
        sub = [r for r in rows if int(r["d"]) == int(d)]
        if not sub:
            continue
        xs_plot = [int(r["examples"]) for r in sub]
        ax.plot(xs_plot, [float(r["nemotron_rmse"]) for r in sub], marker="o", linewidth=2.2, color="#0f766e", label="Nemotron")
        ax.plot(xs_plot, [float(r["knn_rmse"]) for r in sub], marker="s", linewidth=1.9, color="#1d4ed8", label="KNN")
        ax.plot(xs_plot, [float(r["mean_y_rmse"]) for r in sub], marker="^", linewidth=1.9, color="#9a3412", label="Mean-y")
        ax.set_title(f"d={d}", fontsize=11)
        ax.set_xticks(xs_plot)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Nemotron NLR ICL Scan: Dimensions 1-5", fontsize=16, y=1.05)
    fig.supxlabel("Number of in-context examples", fontsize=11)
    fig.supylabel("RMSE", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(PNG_CURVE, bbox_inches="tight")
    plt.close(fig)

    heat = np.full((len(DIMS), len(EXAMPLE_COUNTS)), np.nan)
    delta = np.full((len(DIMS), len(EXAMPLE_COUNTS)), np.nan)
    for row in rows:
        i = DIMS.index(int(row["d"]))
        j = EXAMPLE_COUNTS.index(int(row["examples"]))
        heat[i, j] = float(row["nemotron_rmse"])
        delta[i, j] = float(row["nemotron_rmse"]) - float(row["knn_rmse"])

    fig2, axes2 = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=180)
    im1 = axes2[0].imshow(heat, aspect="auto", cmap="viridis")
    axes2[0].set_title("Nemotron RMSE")
    axes2[0].set_xticks(range(len(EXAMPLE_COUNTS)), EXAMPLE_COUNTS)
    axes2[0].set_yticks(range(len(DIMS)), DIMS)
    axes2[0].set_xlabel("Examples")
    axes2[0].set_ylabel("Dimension")
    fig2.colorbar(im1, ax=axes2[0], fraction=0.046, pad=0.04)

    im2 = axes2[1].imshow(delta, aspect="auto", cmap="coolwarm", vmin=-0.5, vmax=0.5)
    axes2[1].set_title("Nemotron - KNN RMSE")
    axes2[1].set_xticks(range(len(EXAMPLE_COUNTS)), EXAMPLE_COUNTS)
    axes2[1].set_yticks(range(len(DIMS)), DIMS)
    axes2[1].set_xlabel("Examples")
    axes2[1].set_ylabel("Dimension")
    fig2.colorbar(im2, ax=axes2[1], fraction=0.046, pad=0.04)
    fig2.tight_layout()
    fig2.savefig(PNG_HEAT, bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set")

    _seed_everything(SEED)
    results = _load_results()

    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        model=MODEL_ID,
        timeout_s=240.0,
        max_retries=3,
        retry_sleep_s=2.0,
    )

    max_examples = max(EXAMPLE_COUNTS)
    for d in DIMS:
        xs, ys = collect_xy(
            task_name=TASK_NAME,
            task_kwargs=TASK_KWARGS,
            data_name="gaussian",
            n_dims=d,
            n_points=max_examples + 1,
            num_eval_examples=NUM_EVAL_EXAMPLES,
            batch_size=NUM_EVAL_EXAMPLES,
            device="cpu",
        )
        xs_np = xs.numpy()
        ys_np = ys.numpy()
        y_true = ys_np[:, max_examples]

        for L in EXAMPLE_COUNTS:
            if _row_exists(results, d, L):
                print(json.dumps({"d": d, "examples": L, "status": "skip"}), flush=True)
                continue

            mean_preds = np.mean(ys_np[:, :L], axis=1)
            knn_preds = []
            for i in range(NUM_EVAL_EXAMPLES):
                reg = KNeighborsRegressor(n_neighbors=min(5, L), weights="distance")
                reg.fit(xs_np[i, :L, :], ys_np[i, :L])
                knn_preds.append(float(reg.predict(xs_np[i, max_examples, :][None, :])[0]))
            knn_preds = np.array(knn_preds)

            preds = []
            invalid = 0
            for i in range(NUM_EVAL_EXAMPLES):
                prompt = _build_prompt(
                    xs_ctx=xs_np[i, :L, :],
                    ys_ctx=ys_np[i, :L],
                    x_query=xs_np[i, max_examples, :],
                    x_decimals=2,
                    y_decimals=2,
                    prompt_style=PROMPT_STYLE,
                    answer_format=ANSWER_FORMAT,
                )
                try:
                    text, _ = client.complete_text(
                        messages=_messages_for_prompt(prompt),
                        max_tokens=24,
                        temperature=0.0,
                        top_p=1.0,
                        presence_penalty=0.0,
                        extra_body=EXTRA_BODY,
                        stop=None,
                    )
                    raw_pred = _parse_prediction(text, ANSWER_FORMAT)
                    pred = _postprocess_prediction(raw_pred, ys_np[i, :L], PROMPT_STYLE)
                except Exception:
                    pred = None

                if pred is None or not math.isfinite(float(pred)):
                    invalid += 1
                    preds.append(float("nan"))
                else:
                    preds.append(float(pred))
                time.sleep(SLEEP_S)

            arr = np.array(preds, dtype=float)
            mask = np.isfinite(arr)
            row = {
                "d": d,
                "examples": L,
                "nemotron_rmse": float(np.sqrt(np.mean((arr[mask] - y_true[mask]) ** 2))) if np.any(mask) else float("inf"),
                "nemotron_invalid_rate": invalid / float(NUM_EVAL_EXAMPLES),
                "knn_rmse": float(np.sqrt(np.mean((knn_preds - y_true) ** 2))),
                "mean_y_rmse": float(np.sqrt(np.mean((mean_preds - y_true) ** 2))),
            }
            results["rows"].append(row)
            results["rows"].sort(key=lambda x: (int(x["d"]), int(x["examples"])))
            _save_results(results)
            print(json.dumps(row), flush=True)

    _plot_results(results)
    print(str(JSON_PATH), flush=True)
    print(str(PNG_CURVE), flush=True)
    print(str(PNG_HEAT), flush=True)


if __name__ == "__main__":
    main()
