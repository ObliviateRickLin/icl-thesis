from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAID_DIR = ROOT / "results_openrouter_paid_dimension"
FREE_DIR = ROOT / "results_openrouter_free_dimension"
FREE_D40_DIR = ROOT / "results_openrouter_free_dimension_d40"
OUT_DIR = ROOT / "results_dimension_report"

DIMS = [1, 2, 4, 8, 10, 40, 100]
TASK_ORDER = [
    "Noisy Linear Regression",
    "Noisy Quadratic Regression",
    "Noisy ReLU 2NN Regression",
    "Noisy Decision Tree",
]
MODEL_ORDER = [
    "baseline/knn",
    "baseline/mean_y",
    "arcee-ai/trinity-large-preview:free",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-chat-v3.1",
    "google/gemini-2.5-flash-lite",
]

DISPLAY_NAME = {
    "baseline/knn": "KNN",
    "baseline/mean_y": "Mean y",
    "arcee-ai/trinity-large-preview:free": "Trinity free",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "anthropic/claude-3-haiku": "Claude 3 Haiku",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "deepseek/deepseek-chat-v3.1": "DeepSeek V3.1",
    "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
}

COLOR_MAP = {
    "baseline/knn": "#0f172a",
    "baseline/mean_y": "#64748b",
    "arcee-ai/trinity-large-preview:free": "#0f766e",
    "openai/gpt-4o": "#c2410c",
    "openai/gpt-4o-mini": "#fb923c",
    "anthropic/claude-3-haiku": "#7c3aed",
    "anthropic/claude-haiku-4.5": "#a855f7",
    "deepseek/deepseek-chat-v3.1": "#1d4ed8",
    "google/gemini-2.5-flash-lite": "#0891b2",
}

BG = "#f7f4ee"
PANEL = "#fffdf8"
GRID = "#d6d3d1"
TEXT = "#1f2937"
SUBTEXT = "#6b7280"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _apply_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#d6d3d1",
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d6d3d1")
    ax.spines["bottom"].set_color("#d6d3d1")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summaries_from_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    out = []
    for model, model_rows in grouped.items():
        valid_rows = [r for r in model_rows if int(r.get("valid", 1)) == 1 and r.get("mse") is not None]
        invalid_count = sum(1 for r in model_rows if int(r.get("valid", 1)) != 1)
        mse = float("nan")
        rmse = float("nan")
        if valid_rows:
            mse = sum(float(r["mse"]) for r in valid_rows) / len(valid_rows)
            rmse = math.sqrt(mse)
        out.append(
            {
                "model": model,
                "rmse": rmse,
                "mse": mse,
                "num_eval_examples": len(model_rows),
                "num_valid": len(valid_rows),
                "invalid_rate": invalid_count / len(model_rows) if model_rows else float("nan"),
            }
        )
    return out


def _extract_rows(path: Path, source_tag: str) -> list[dict]:
    obj = _load_json(path)
    meta = obj["meta"]
    if "summary_rows" in obj:
        summaries = obj["summary_rows"]
    else:
        summaries = _summaries_from_rows(obj["detail_rows"])
    rows = []
    for row in summaries:
        rows.append(
            {
                "task": meta["task_label"],
                "dim": int(meta["n_dims"]),
                "example_count": int(meta["example_count"]),
                "model": row["model"],
                "rmse": float(row["rmse"]),
                "invalid_rate": float(row.get("invalid_rate", 0.0)),
                "source": source_tag,
                "file": path.name,
            }
        )
    return rows


def _collect_all_rows() -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(FREE_DIR.glob("*.json")):
        if path.name in {"manifest.json", "free_probe_candidates.json"}:
            continue
        rows.extend(_extract_rows(path, "free_main"))
    for path in sorted(FREE_D40_DIR.glob("*.json")):
        if path.name == "manifest.json":
            continue
        rows.extend(_extract_rows(path, "free_d40"))
    for path in sorted(PAID_DIR.glob("*.json")):
        if path.name == "manifest.json":
            continue
        rows.extend(_extract_rows(path, "paid"))

    df = pd.DataFrame(rows)
    source_priority = {"paid": 0, "free_d40": 1, "free_main": 2}
    df["source_priority"] = df["source"].map(source_priority)
    df["model_order"] = df["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    df["task_order"] = df["task"].map({t: i for i, t in enumerate(TASK_ORDER)})
    df = (
        df.sort_values(["task_order", "dim", "model_order", "source_priority"])
        .drop_duplicates(subset=["task", "dim", "model"], keep="first")
        .reset_index(drop=True)
    )
    return df.drop(columns=["source_priority", "model_order", "task_order"])


def _plot_task_curves(df: pd.DataFrame) -> None:
    _apply_theme()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    axes = axes.flatten()
    x_positions = list(range(len(DIMS)))

    for ax, task in zip(axes, TASK_ORDER):
        task_df = df[df["task"] == task]
        for model in MODEL_ORDER:
            sub = task_df[task_df["model"] == model].sort_values("dim")
            if sub.empty:
                continue
            xs = [x_positions[DIMS.index(d)] for d in sub["dim"] if d in DIMS]
            ys = [sub[sub["dim"] == d]["rmse"].iloc[0] for d in sub["dim"] if d in DIMS]
            linewidth = 3.0 if model.startswith("baseline/") else 2.1
            alpha = 0.98 if model.startswith("baseline/") else 0.9
            marker = "o" if model.startswith("baseline/") else "D" if model.endswith(":free") else "s"
            ax.plot(
                xs,
                ys,
                label=DISPLAY_NAME[model],
                color=COLOR_MAP[model],
                linewidth=linewidth,
                marker=marker,
                markersize=5,
                alpha=alpha,
                markeredgewidth=0,
            )
        ax.axvspan(-0.4, 4.4, color="#f5efe6", alpha=0.55, zorder=0)
        ax.axvspan(4.6, 6.4, color="#eef6f5", alpha=0.55, zorder=0)
        ax.set_title(task, fontsize=13, fontweight="bold")
        ax.set_xticks(x_positions, [str(d) for d in DIMS])
        ax.set_xlabel("Dimension")
        ax.set_ylabel("RMSE")
        _style_axis(ax)
        ax.text(0.02, 0.96, "Low to medium d", transform=ax.transAxes, va="top", fontsize=9, color=SUBTEXT)
        ax.text(0.72, 0.96, "High d", transform=ax.transAxes, va="top", fontsize=9, color=SUBTEXT)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Dimension Scan: RMSE Across Tasks and Models", fontsize=16, fontweight="bold")
    fig.savefig(OUT_DIR / "dimension_scan_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    for task in TASK_ORDER:
        fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
        task_df = df[df["task"] == task]
        for model in MODEL_ORDER:
            sub = task_df[task_df["model"] == model].sort_values("dim")
            if sub.empty:
                continue
            xs = [x_positions[DIMS.index(d)] for d in sub["dim"] if d in DIMS]
            ys = [sub[sub["dim"] == d]["rmse"].iloc[0] for d in sub["dim"] if d in DIMS]
            ax.plot(
                xs,
                ys,
                label=DISPLAY_NAME[model],
                color=COLOR_MAP[model],
                linewidth=3.1 if model.startswith("baseline/") else 2.3,
                marker="o" if model.startswith("baseline/") else "D" if model.endswith(":free") else "s",
                markersize=6,
                alpha=0.97 if model.startswith("baseline/") else 0.92,
                markeredgewidth=0,
            )
        ax.axvspan(-0.4, 4.4, color="#f5efe6", alpha=0.6, zorder=0)
        ax.axvspan(4.6, 6.4, color="#eef6f5", alpha=0.6, zorder=0)
        ax.set_xticks(x_positions, [str(d) for d in DIMS])
        ax.set_xlabel("Dimension")
        ax.set_ylabel("RMSE")
        ax.set_title(task, fontsize=16, fontweight="bold", loc="left")
        ax.text(0.0, 1.03, "Examples = 2 × dimension, n = 40 per point", transform=ax.transAxes, fontsize=10, color=SUBTEXT)
        _style_axis(ax)
        ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        slug = (
            task.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )
        fig.savefig(OUT_DIR / f"{slug}_curve.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def _plot_delta_heatmap(df: pd.DataFrame) -> None:
    _apply_theme()
    knn = (
        df[df["model"] == "baseline/knn"][["task", "dim", "rmse"]]
        .rename(columns={"rmse": "knn_rmse"})
        .reset_index(drop=True)
    )
    merged = df.merge(knn, on=["task", "dim"], how="left")
    merged["delta_vs_knn"] = merged["rmse"] - merged["knn_rmse"]
    plot_df = merged[~merged["model"].isin(["baseline/knn", "baseline/mean_y"])].copy()
    plot_df["column"] = plot_df["task"].map(
        {
            "Noisy Linear Regression": "NLR",
            "Noisy Quadratic Regression": "NQR",
            "Noisy ReLU 2NN Regression": "2NN",
            "Noisy Decision Tree": "NDT",
        }
    ) + " d=" + plot_df["dim"].astype(str)
    pivot = (
        plot_df.pivot(index="model", columns="column", values="delta_vs_knn")
        .reindex(index=[m for m in MODEL_ORDER if not m.startswith("baseline/")])
    )
    ordered_columns = []
    for task_short in ["NLR", "NQR", "2NN", "NDT"]:
        for d in DIMS:
            col = f"{task_short} d={d}"
            if col in pivot.columns:
                ordered_columns.append(col)
    pivot = pivot[ordered_columns]

    fig, ax = plt.subplots(figsize=(18, 6.7), constrained_layout=True)
    image = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), [DISPLAY_NAME[m] for m in pivot.index])
    ax.set_title("RMSE Delta vs KNN (negative is better)", fontsize=15, fontweight="bold")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.isna(value):
                text = "NA"
            else:
                text = f"{value:+.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=TEXT)
    fig.colorbar(image, ax=ax, shrink=0.9, label="RMSE(model) - RMSE(KNN)")
    fig.savefig(OUT_DIR / "dimension_scan_delta_vs_knn.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_leaderboard(df: pd.DataFrame) -> None:
    _apply_theme()
    agg = (
        df.groupby("model", as_index=False)["rmse"]
        .mean()
        .sort_values("rmse", ascending=True)
        .reset_index(drop=True)
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    colors = [COLOR_MAP.get(model, "#888888") for model in agg["model"]]
    labels = [DISPLAY_NAME[m] for m in agg["model"]]
    ax.barh(labels, agg["rmse"], color=colors, height=0.62)
    ax.invert_yaxis()
    ax.set_xlabel("Average RMSE Across Available Points")
    ax.set_title("Overall Leaderboard", fontsize=15, fontweight="bold", loc="left")
    ax.text(0.0, 1.03, "Computed over currently available points in the dimension scan", transform=ax.transAxes, fontsize=10, color=SUBTEXT)
    _style_axis(ax)
    for idx, value in enumerate(agg["rmse"]):
        ax.text(value + 0.01, idx, f"{value:.3f}", va="center", fontsize=9)
    fig.savefig(OUT_DIR / "dimension_scan_leaderboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _ensure_dir(OUT_DIR)
    df = _collect_all_rows()
    df["display_name"] = df["model"].map(DISPLAY_NAME)
    df = df.sort_values(
        by=["task", "dim", "model"],
        key=lambda s: s.map({**{t: i for i, t in enumerate(TASK_ORDER)}, **{m: i for i, m in enumerate(MODEL_ORDER)}}).fillna(999)
        if s.name in {"task", "model"}
        else s,
    ).reset_index(drop=True)
    df.to_csv(OUT_DIR / "dimension_scan_summary.csv", index=False, encoding="utf-8-sig")
    _plot_task_curves(df)
    _plot_delta_heatmap(df)
    _plot_leaderboard(df)
    print(f"Wrote report to {OUT_DIR}")


if __name__ == "__main__":
    main()
