import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAID_DIR = ROOT / "results_openrouter_paid"
FREE_DIR = ROOT / "results_openrouter_free"
OUT_DIR = ROOT / "results_openrouter_report"


MAIN_N20_FILES = [
    PAID_DIR / "pilot_dense_linear_d10_L20_n20__openai__gpt-4o.json",
    PAID_DIR / "pilot_dense_linear_d10_L20_n20__openai__gpt-4o-mini.json",
    PAID_DIR / "pilot_dense_linear_d10_L20_n20__anthropic__claude-3-haiku.json",
    PAID_DIR / "pilot_dense_linear_d10_L20_n20__anthropic__claude-3.5-haiku.json",
    PAID_DIR / "pilot_dense_linear_d10_L20_n20__anthropic__claude-haiku-4.5.json",
    PAID_DIR / "pilot_dense_linear_d10_L20_n20__deepseek__deepseek-chat-v3.1.json",
    PAID_DIR / "pilot_dense_linear_d10_L20_n20__google__gemini-2.5-flash-lite.json",
    PAID_DIR / "pilot_dense_linear_d10_L20_n20__google__gemini-2.5-flash.json",
    FREE_DIR / "pilot_dense_linear_d10_L20_n20__arcee__trinity-large-preview__free.json",
]


STABILITY_SPECS = {
    "Anthropic: Claude 3 Haiku": [
        ("seed0 / n=20", PAID_DIR / "pilot_dense_linear_d10_L20_n20__anthropic__claude-3-haiku.json"),
        ("seed1 / n=20", PAID_DIR / "pilot_dense_linear_d10_L20_n20_seed1__anthropic__claude-3-haiku.json"),
        ("merged / n=40", PAID_DIR / "pilot_dense_linear_d10_L20_n40__anthropic__claude-3-haiku.json"),
    ],
    "Arcee AI: Trinity Large Preview (free)": [
        ("seed0 / n=20", FREE_DIR / "pilot_dense_linear_d10_L20_n20__arcee__trinity-large-preview__free.json"),
        ("seed1 / n=20", FREE_DIR / "pilot_dense_linear_d10_L20_n20_seed1__arcee__trinity-large-preview__free.json"),
        ("merged / n=40", FREE_DIR / "pilot_dense_linear_d10_L20_n40__arcee__trinity-large-preview__free.json"),
    ],
}


LONG_PROMPT_FILE = FREE_DIR / "probe_nlr_noise01_d40_L80_n3__combined_summary.json"
FREE_SWEEP_FILE = FREE_DIR / "big_free_sweep_nlr_d10_L20_n3.json"
EPISODE0_FILES = MAIN_N20_FILES


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _model_display_name(model_id: str) -> str:
    named = {
        "arcee-ai/trinity-large-preview:free": "Arcee AI: Trinity Large Preview (free)",
        "google/gemini-2.5-flash-lite": "Google: Gemini 2.5 Flash Lite",
        "google/gemini-2.5-flash": "Google: Gemini 2.5 Flash",
        "anthropic/claude-3-haiku": "Anthropic: Claude 3 Haiku",
        "anthropic/claude-3.5-haiku": "Anthropic: Claude 3.5 Haiku",
        "anthropic/claude-haiku-4.5": "Anthropic: Claude Haiku 4.5",
        "deepseek/deepseek-chat-v3.1": "DeepSeek: DeepSeek V3.1",
        "openai/gpt-4o": "OpenAI: GPT-4o",
        "openai/gpt-4o-mini": "OpenAI: GPT-4o-mini",
    }
    if model_id in named:
        return named[model_id]
    if model_id == "baseline/knn":
        return "Baseline: KNN (k<= 5)"
    if model_id == "baseline/mean_y":
        return "Baseline: Mean of context y"
    parts = model_id.split("/")
    if len(parts) == 2:
        vendor, model = parts
        return f"{vendor.title()}: {model}"
    return model_id


def _normalize_summary_rows(path: Path) -> list[dict]:
    obj = _load_json(path)
    rows: list[dict] = []

    if isinstance(obj, list):
        for row in obj:
            rows.append(dict(row))
        return rows

    if "summary_rows" in obj:
        for row in obj["summary_rows"]:
            rows.append(dict(row))
        return rows

    if "model_summary" in obj:
        model_summary = dict(obj["model_summary"])
        rows.append(
            {
                "model": model_summary["model"],
                "display_name": _model_display_name(model_summary["model"]),
                "rmse": model_summary["rmse"],
                "mse": np.nan,
                "mae": model_summary["mae"],
                "num_eval_examples": obj["task"]["num_eval_examples"],
                "num_valid": model_summary["num_valid"],
                "invalid_rate": model_summary["invalid_rate"],
                "avg_latency_s": model_summary["avg_latency_s"],
                "prompt_tokens_total": model_summary["prompt_tokens_total"],
                "completion_tokens_total": model_summary["completion_tokens_total"],
                "total_cost_usd": 0.0,
            }
        )
        for baseline_id, key in [
            ("baseline/mean_y", "baseline_mean_y"),
            ("baseline/knn", "baseline_knn"),
        ]:
            baseline = dict(obj[key])
            rows.append(
                {
                    "model": baseline_id,
                    "display_name": _model_display_name(baseline_id),
                    "rmse": baseline["rmse"],
                    "mse": np.nan,
                    "mae": baseline["mae"],
                    "num_eval_examples": obj["task"]["num_eval_examples"],
                    "num_valid": baseline.get("num_valid", obj["task"]["num_eval_examples"]),
                    "invalid_rate": baseline.get("invalid_rate", 0.0),
                    "avg_latency_s": baseline.get("avg_latency_s", 0.0) or 0.0,
                    "prompt_tokens_total": baseline.get("prompt_tokens_total", 0),
                    "completion_tokens_total": baseline.get("completion_tokens_total", 0),
                    "total_cost_usd": 0.0,
                }
            )
        return rows

    if "results" in obj:
        for row in obj["results"]:
            rows.append(dict(row))
        return rows

    raise ValueError(f"Unsupported summary format: {path}")


def _extract_task_meta(path: Path) -> dict:
    obj = _load_json(path)
    if "task" in obj:
        return dict(obj["task"])
    if "detail_rows" in obj and obj["detail_rows"]:
        return {
            "num_eval_examples": len(obj["detail_rows"]),
            "seed_runs": obj.get("seed_runs"),
        }
    return {}


def _normalize_detail_row(path: Path, episode_idx: int = 0) -> dict | None:
    obj = _load_json(path)

    if "detail_rows" in obj:
        for row in obj["detail_rows"]:
            if row.get("episode_idx") == episode_idx:
                return {
                    "model": row["model"],
                    "display_name": row.get("display_name", _model_display_name(row["model"])),
                    "prompt": row.get("prompt"),
                    "raw_text": row.get("raw_text"),
                    "pred_y": row.get("pred_y"),
                    "target_y": row.get("target_y"),
                    "mse": row.get("mse"),
                    "valid": row.get("valid"),
                }
        return None

    if "raw_rows" in obj:
        for row in obj["raw_rows"]:
            if row.get("episode_idx") == episode_idx:
                model = obj["model_summary"]["model"]
                return {
                    "model": model,
                    "display_name": _model_display_name(model),
                    "prompt": None,
                    "raw_text": row.get("raw_text"),
                    "pred_y": row.get("pred_y"),
                    "target_y": row.get("target_y"),
                    "mse": (row.get("pred_y") - row.get("target_y")) ** 2
                    if row.get("pred_y") is not None and row.get("target_y") is not None
                    else np.nan,
                    "valid": 1 if row.get("status") == "ok" else 0,
                }
        return None

    return None


def _type_from_model(model_id: str) -> str:
    if model_id.startswith("baseline/"):
        return "baseline"
    if model_id.endswith(":free"):
        return "free"
    return "paid"


def _short_name(display_name: str) -> str:
    replacements = {
        "OpenAI: GPT-4o": "GPT-4o",
        "OpenAI: GPT-4o-mini": "GPT-4o-mini",
        "Anthropic: claude-3-haiku": "Claude 3 Haiku",
        "Anthropic: claude-3.5-haiku": "Claude 3.5 Haiku",
        "Anthropic: claude-haiku-4.5": "Claude Haiku 4.5",
        "Deepseek: deepseek-chat-v3.1": "DeepSeek V3.1",
        "Google: gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
        "Google: gemini-2.5-flash": "Gemini 2.5 Flash",
        "Arcee Ai: trinity-large-preview:free": "Trinity Large Preview",
        "Arcee AI: Trinity Large Preview (free)": "Trinity Large Preview",
        "Baseline: Mean of context y": "Mean y",
        "Baseline: KNN (k<= 5)": "KNN",
    }
    return replacements.get(display_name, display_name.replace("(free)", "").strip())


def build_main_leaderboard() -> pd.DataFrame:
    rows = []
    for path in MAIN_N20_FILES:
        for row in _normalize_summary_rows(path):
            if row["model"] in {"baseline/knn", "baseline/mean_y"} and any(
                existing["model"] == row["model"] for existing in rows
            ):
                continue
            row = dict(row)
            row["source_file"] = path.name
            row["model_type"] = _type_from_model(row["model"])
            row["short_name"] = _short_name(row["display_name"])
            rows.append(row)
    df = pd.DataFrame(rows)
    df["sort_key"] = np.where(df["model_type"] == "baseline", -1, 0)
    df = df.sort_values(["sort_key", "rmse", "short_name"], ascending=[True, True, True]).reset_index(drop=True)
    return df.drop(columns=["sort_key"])


def build_stability_df() -> pd.DataFrame:
    rows = []
    for family, specs in STABILITY_SPECS.items():
        for run_label, path in specs:
            summaries = _normalize_summary_rows(path)
            by_model = {row["model"]: row for row in summaries}
            for series_name, model_id in [
                ("model", next(model for model in by_model if not model.startswith("baseline/"))),
                ("knn", "baseline/knn"),
                ("mean_y", "baseline/mean_y"),
            ]:
                row = by_model[model_id]
                rows.append(
                    {
                        "family": family,
                        "run_label": run_label,
                        "series": series_name,
                        "rmse": row["rmse"],
                    }
                )
    return pd.DataFrame(rows)


def build_long_prompt_df() -> pd.DataFrame:
    df = pd.DataFrame(_normalize_summary_rows(LONG_PROMPT_FILE))
    df["model_type"] = df["model"].map(_type_from_model)
    df["short_name"] = df["display_name"].map(_short_name)
    return df


def build_free_availability_df() -> pd.DataFrame:
    obj = _load_json(FREE_SWEEP_FILE)
    rows = []
    for row in obj["availability"]:
        status = row["status"]
        if status == "ok" and row.get("strict_number"):
            status_label = "OK: strict number"
        elif status == "ok" and row.get("has_reasoning"):
            status_label = "OK: reasoning only"
        elif status.startswith("http_429"):
            status_label = "429 rate limited"
        elif status.startswith("http_404"):
            status_label = "404 unavailable"
        else:
            status_label = status
        rows.append(
            {
                "model": row["model"],
                "provider": row.get("provider", ""),
                "status": status,
                "status_label": status_label,
                "strict_number": bool(row.get("strict_number")),
                "has_reasoning": bool(row.get("has_reasoning")),
            }
        )
    df = pd.DataFrame(rows)
    order = {
        "OK: strict number": 0,
        "OK: reasoning only": 1,
        "429 rate limited": 2,
        "404 unavailable": 3,
    }
    df["status_order"] = df["status_label"].map(lambda x: order.get(x, 9))
    df = df.sort_values(["status_order", "model"]).reset_index(drop=True)
    return df.drop(columns=["status_order"])


def build_episode0_outputs() -> tuple[str, pd.DataFrame]:
    prompt = None
    rows = []
    for path in EPISODE0_FILES:
        row = _normalize_detail_row(path, episode_idx=0)
        if not row:
            continue
        if row["prompt"] and prompt is None:
            prompt = row["prompt"]
        rows.append(
            {
                "model": row["model"],
                "display_name": row["display_name"],
                "short_name": _short_name(row["display_name"]),
                "raw_text": row["raw_text"],
                "pred_y": row["pred_y"],
                "target_y": row["target_y"],
                "mse": row["mse"],
                "valid": row["valid"],
                "model_type": _type_from_model(row["model"]),
            }
        )
    df = pd.DataFrame(rows).sort_values(["model_type", "mse", "short_name"]).reset_index(drop=True)
    return prompt or "", df


def save_tables(main_df: pd.DataFrame, stability_df: pd.DataFrame, long_df: pd.DataFrame, availability_df: pd.DataFrame, episode0_df: pd.DataFrame) -> None:
    main_df.to_csv(OUT_DIR / "pilot_d10_n20_leaderboard.csv", index=False)
    stability_df.to_csv(OUT_DIR / "stability_checks.csv", index=False)
    long_df.to_csv(OUT_DIR / "long_prompt_probe.csv", index=False)
    availability_df.to_csv(OUT_DIR / "free_model_availability.csv", index=False)
    episode0_df.to_csv(OUT_DIR / "episode0_outputs.csv", index=False)


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def plot_main_leaderboard(df: pd.DataFrame) -> None:
    plot_df = df.copy().sort_values("rmse", ascending=True)
    colors = plot_df["model_type"].map(
        {"baseline": "#6c757d", "paid": "#2b6cb0", "free": "#2f855a"}
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["short_name"], plot_df["rmse"], color=colors)
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        note = []
        if row["model_type"] == "paid":
            note.append(f"${row['total_cost_usd']:.3f}")
        if row["invalid_rate"] > 0:
            note.append(f"invalid {row['invalid_rate']:.0%}")
        if note:
            ax.text(row["rmse"] + 0.015, idx, " | ".join(note), va="center", fontsize=9)
    ax.set_title("Dense Linear Pilot: RMSE by Model\nnoise=0.1, d=10, examples=20, n=20", fontsize=13, pad=12)
    ax.set_xlabel("RMSE (lower is better)")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pilot_d10_n20_leaderboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stability(stability_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    colors = {"model": "#b83280", "knn": "#2d3748", "mean_y": "#718096"}
    markers = {"model": "o", "knn": "s", "mean_y": "^"}
    for ax, (family, fam_df) in zip(axes, stability_df.groupby("family", sort=False)):
        run_order = ["seed0 / n=20", "seed1 / n=20", "merged / n=40"]
        x = np.arange(len(run_order))
        for series in ["model", "knn", "mean_y"]:
            series_df = fam_df[fam_df["series"] == series].set_index("run_label").reindex(run_order)
            ax.plot(
                x,
                series_df["rmse"].values,
                marker=markers[series],
                linewidth=2,
                color=colors[series],
                label=series,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["seed0", "seed1", "merged"], rotation=0)
        ax.set_title(family, fontsize=11)
        ax.set_xlabel("evaluation slice")
        _style_axes(ax)
    axes[0].set_ylabel("RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    label_map = {"model": "LLM", "knn": "baseline KNN", "mean_y": "baseline mean y"}
    fig.legend(
        handles,
        [label_map[l] for l in labels],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Apparent Wins Collapse After Repeat Evaluation", fontsize=13, y=1.09)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT_DIR / "stability_rechecks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_long_prompt(long_df: pd.DataFrame) -> None:
    plot_df = long_df.copy()
    plot_df["plot_rmse"] = plot_df["rmse"].replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.sort_values(["plot_rmse", "short_name"], na_position="last")
    colors = plot_df["model_type"].map(
        {"baseline": "#6c757d", "paid": "#2b6cb0", "free": "#2f855a"}
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    valid_df = plot_df[plot_df["plot_rmse"].notna()]
    ax.barh(valid_df["short_name"], valid_df["plot_rmse"], color=colors[valid_df.index])
    invalid_df = plot_df[plot_df["plot_rmse"].isna()]
    note = ""
    if not invalid_df.empty:
        pieces = [f"{row['short_name']} invalid {row['invalid_rate']:.0%}" for _, row in invalid_df.iterrows()]
        note = "Invalid outputs: " + "; ".join(pieces)
    ax.set_title("More moe-icl-like Stress Test\nnoise=0.1, d=40, examples=80, n=3", fontsize=13, pad=12)
    ax.set_xlabel("RMSE (lower is better)")
    _style_axes(ax)
    if note:
        fig.text(0.12, 0.02, note, fontsize=9, ha="left")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(OUT_DIR / "long_prompt_probe.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_free_availability(availability_df: pd.DataFrame) -> None:
    color_map = {
        "OK: strict number": "#2f855a",
        "OK: reasoning only": "#d69e2e",
        "429 rate limited": "#c53030",
        "404 unavailable": "#718096",
    }
    fig, ax = plt.subplots(figsize=(11, 5.4))
    y = np.arange(len(availability_df))[::-1]
    ax.scatter(
        np.zeros(len(availability_df)),
        y,
        s=220,
        c=[color_map.get(v, "#4a5568") for v in availability_df["status_label"]],
        marker="s",
    )
    for idx, (_, row) in enumerate(availability_df.iterrows()):
        ypos = y[idx]
        ax.text(0.12, ypos, row["model"], va="center", fontsize=10)
        ax.text(2.05, ypos, row["status_label"], va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.2, 3.6)
    ax.set_ylim(-1, len(availability_df))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Large Free-Model Sweep: Availability Under Strict Numeric Protocol", fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "free_model_availability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_prompt_and_brief(main_df: pd.DataFrame, stability_df: pd.DataFrame, long_df: pd.DataFrame, availability_df: pd.DataFrame, prompt: str, episode0_df: pd.DataFrame) -> None:
    (OUT_DIR / "episode0_prompt.txt").write_text(prompt, encoding="utf-8")

    knn_row = main_df[main_df["model"] == "baseline/knn"].iloc[0]
    mean_row = main_df[main_df["model"] == "baseline/mean_y"].iloc[0]
    paid_df = main_df[main_df["model_type"] == "paid"].sort_values("rmse")
    free_df = main_df[main_df["model_type"] == "free"].sort_values("rmse")
    best_paid = paid_df.iloc[0]
    best_free = free_df.iloc[0]

    claude_n40 = _normalize_summary_rows(PAID_DIR / "pilot_dense_linear_d10_L20_n40__anthropic__claude-3-haiku.json")
    claude_n40_model = next(row for row in claude_n40 if row["model"] == "anthropic/claude-3-haiku")
    trinity_n40 = _normalize_summary_rows(FREE_DIR / "pilot_dense_linear_d10_L20_n40__arcee__trinity-large-preview__free.json")
    trinity_n40_model = next(row for row in trinity_n40 if row["model"] == "arcee-ai/trinity-large-preview:free")
    long_trinity = long_df[long_df["model"] == "arcee-ai/trinity-large-preview:free"].iloc[0]
    long_knn = long_df[long_df["model"] == "baseline/knn"].iloc[0]

    brief = f"""# LLM Regression Negative-Results Brief

## Main Takeaways

- We tested real LLMs on the same dense synthetic regression style used in `moe-icl`, starting from a controlled pilot: `noisy_linear_regression`, `noise_std=0.1`, `d=10`, `examples=20`.
- On the comparable `n=20` pilot slice, the strongest paid model we tested was **{best_paid['short_name']}** with `RMSE={best_paid['rmse']:.3f}`. The simple `KNN` baseline was still better at `RMSE={knn_row['rmse']:.3f}`.
- On the same pilot slice, the strongest free model was **{best_free['short_name']}** with `RMSE={best_free['rmse']:.3f}`.
- Early "good" results did not survive repeat evaluation. After re-running to `n=40`:
  - `Claude 3 Haiku` moved to `RMSE={claude_n40_model['rmse']:.3f}`
  - `Trinity Large Preview` moved to `RMSE={trinity_n40_model['rmse']:.3f}`
  - Both ended up worse than `KNN` (`RMSE={knn_row['rmse']:.3f}` on the seed0 pilot, `RMSE=0.639` on the merged recheck).
- In a more moe-icl-like long-context stress test (`d=40`, `examples=80`), `Trinity Large Preview` degraded to `RMSE={long_trinity['rmse']:.3f}` while `KNN` stayed at `RMSE={long_knn['rmse']:.3f}`.

## What This Means

- The current result is not "we did not try enough models".
- The stronger claim we can defend is: **under the current prompt format and task protocol, general-purpose LLM APIs do not show robust in-context regression gains on these synthetic dense tasks.**
- The negative result is strongest where we repeated apparently promising models and the gains disappeared.

## What To Show Tomorrow

1. `pilot_d10_n20_leaderboard.png`
   This is the clean same-task leaderboard with baselines included.
2. `stability_rechecks.png`
   This is the key evidence that small-sample optimism is real and misleading.
3. `free_model_availability.png`
   This shows we also pushed on the free/large-model frontier, but availability and output-format issues blocked many options.
4. `long_prompt_probe.png`
   This shows performance gets worse when we move closer to the longer prompts used in `moe-icl`.

## Suggested Spoken Framing

- "I did not stop at a single bad run. I widened the model pool, added paid and free providers, repeated the apparently strongest models, and kept the classical baselines in the loop."
- "The negative result is now reproducible enough to be informative: once the evaluation is widened from tiny pilots to repeated runs, the LLM advantage disappears."
- "The next decision is not whether we tried hard enough. The next decision is whether we want to keep investing in prompt-only real-LLM regression, or treat this as evidence that the synthetic `moe-icl` setup is outside the comfort zone of current general LLM APIs."
"""
    (OUT_DIR / "boss_tomorrow_brief.md").write_text(brief, encoding="utf-8")


def main() -> None:
    _ensure_dir(OUT_DIR)
    main_df = build_main_leaderboard()
    stability_df = build_stability_df()
    long_df = build_long_prompt_df()
    availability_df = build_free_availability_df()
    prompt, episode0_df = build_episode0_outputs()

    save_tables(main_df, stability_df, long_df, availability_df, episode0_df)
    plot_main_leaderboard(main_df)
    plot_stability(stability_df)
    plot_long_prompt(long_df)
    plot_free_availability(availability_df)
    write_prompt_and_brief(main_df, stability_df, long_df, availability_df, prompt, episode0_df)

    print(f"Wrote report artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
