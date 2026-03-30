from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from llm_competition import (
    BASELINE_KNN,
    BASELINE_MEAN,
    TASK_PRESETS,
    default_example_count,
    run_competition,
)


DEFAULT_DIMS = [1, 2, 4, 8, 10, 100]
DEFAULT_FREE_MODELS = [
    "arcee-ai/trinity-large-preview:free",
]

MODEL_EXTRA_BODY_MAP = {
    "nvidia/nemotron-3-nano-30b-a3b:free": {"reasoning": {"effort": "none", "exclude": True}},
    "nvidia/nemotron-3-super-120b-a12b:free": {"reasoning": {"effort": "none", "exclude": True}},
}


def _slugify_task(task_label: str) -> str:
    return (
        task_label.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_int_csv(text: str) -> list[int]:
    out = []
    for tok in (text or "").split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _parse_str_csv(text: str) -> list[str]:
    out = []
    for tok in (text or "").split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


def _normalize_task_labels(task_labels: list[str]) -> list[str]:
    wanted = {label.strip().lower() for label in task_labels if label.strip()}
    out = []
    for task_label in TASK_PRESETS:
        if task_label.lower() in wanted:
            out.append(task_label)
    return out


def _print_progress(step_idx: int, total_steps: int, model_id: str, episode_idx: int) -> None:
    if model_id.startswith("baseline/"):
        return
    print(f"[{step_idx}/{total_steps}] episode={episode_idx:02d} model={model_id}", flush=True)


def _run_one(
    *,
    api_key: str,
    task_label: str,
    task_name: str,
    task_kwargs: dict,
    n_dims: int,
    num_eval_examples: int,
    model_ids: list[str],
    out_dir: Path,
    max_tokens: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    timeout_s: float,
    max_retries: int,
    max_concurrency: int,
    prompt_style: str,
    answer_format: str,
    x_decimals: int,
    y_decimals: int,
) -> Path:
    example_count = default_example_count(n_dims)
    slug = f"{_slugify_task(task_label)}__d{n_dims}__e{example_count}__n{num_eval_examples}"
    out_path = out_dir / f"{slug}.json"

    if out_path.exists():
        print(f"[skip] {out_path.name}", flush=True)
        return out_path

    print(
        f"[run] task={task_label} dims={n_dims} examples={example_count} evals={num_eval_examples}",
        flush=True,
    )

    result = run_competition(
        model_ids=model_ids,
        api_key=api_key,
        task_name=task_name,
        task_kwargs=task_kwargs,
        n_dims=n_dims,
        example_count=example_count,
        num_eval_examples=num_eval_examples,
        prompt_style=prompt_style,
        answer_format=answer_format,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        x_decimals=x_decimals,
        y_decimals=y_decimals,
        timeout_s=timeout_s,
        max_retries=max_retries,
        max_concurrency=max_concurrency,
        model_extra_body_map=MODEL_EXTRA_BODY_MAP,
        progress_callback=_print_progress,
    )
    payload = {
        "meta": {
            "task_label": task_label,
            "task_name": task_name,
            "task_kwargs": task_kwargs,
            "n_dims": n_dims,
            "example_count": example_count,
            "num_eval_examples": num_eval_examples,
            "model_ids": model_ids,
            "prompt_style": prompt_style,
            "answer_format": answer_format,
        },
        **result,
    }
    _save_json(out_path, payload)
    print(f"[done] {out_path.name}", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the free-model dimension scan across 4 regression tasks.")
    parser.add_argument("--api-key", type=str, default="", help="OpenRouter API key. Defaults to OPENROUTER_API_KEY.")
    parser.add_argument("--dims", type=str, default="1,2,4,8,10,100")
    parser.add_argument("--num-eval-examples", type=int, default=40)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_FREE_MODELS))
    parser.add_argument(
        "--task-labels",
        type=str,
        default="",
        help="Optional comma-separated subset of task labels to run. Must match keys in TASK_PRESETS.",
    )
    parser.add_argument("--include-baselines", action="store_true", default=True)
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "results_openrouter_free_dimension"))
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--prompt-style", type=str, default="words2numbers")
    parser.add_argument("--answer-format", type=str, default="strict_number")
    parser.add_argument("--x-decimals", type=int, default=2)
    parser.add_argument("--y-decimals", type=int, default=2)
    parser.add_argument("--sleep-s", type=float, default=3.5, help="Sleep between configs to be gentle with free endpoints.")
    args = parser.parse_args()

    api_key = args.api_key or __import__("os").environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("Missing OpenRouter API key. Set OPENROUTER_API_KEY or pass --api-key.")

    dims = _parse_int_csv(args.dims)
    model_ids = _parse_str_csv(args.models)
    task_labels = _normalize_task_labels(_parse_str_csv(args.task_labels)) if args.task_labels else list(TASK_PRESETS.keys())
    if args.include_baselines:
        model_ids = [BASELINE_KNN, BASELINE_MEAN] + model_ids

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_rows = []
    for task_label in task_labels:
        preset = TASK_PRESETS[task_label]
        for d in dims:
            config_rows.append(
                {
                    "task_label": task_label,
                    "task_name": preset["task_name"],
                    "task_kwargs": preset["default_task_kwargs"],
                    "n_dims": int(d),
                }
            )

    manifest = {
        "dims": dims,
        "models": model_ids,
        "num_eval_examples": int(args.num_eval_examples),
        "configs": config_rows,
        "prompt_style": args.prompt_style,
        "answer_format": args.answer_format,
    }
    _save_json(out_dir / "manifest.json", manifest)

    total = len(config_rows)
    for idx, cfg in enumerate(config_rows, start=1):
        print(f"\n=== [{idx}/{total}] {cfg['task_label']} d={cfg['n_dims']} ===", flush=True)
        try:
            _run_one(
                api_key=api_key,
                task_label=cfg["task_label"],
                task_name=cfg["task_name"],
                task_kwargs=cfg["task_kwargs"],
                n_dims=cfg["n_dims"],
                num_eval_examples=args.num_eval_examples,
                model_ids=model_ids,
                out_dir=out_dir,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                timeout_s=args.timeout_s,
                max_retries=args.max_retries,
                max_concurrency=args.max_concurrency,
                prompt_style=args.prompt_style,
                answer_format=args.answer_format,
                x_decimals=args.x_decimals,
                y_decimals=args.y_decimals,
            )
        except Exception as exc:
            err_path = out_dir / f"{_slugify_task(cfg['task_label'])}__d{cfg['n_dims']}__ERROR.txt"
            err_path.write_text(str(exc), encoding="utf-8")
            print(f"[error] {cfg['task_label']} d={cfg['n_dims']}: {exc}", flush=True)
        if idx < total and args.sleep_s > 0:
            time.sleep(float(args.sleep_s))

    print(f"\nCompleted free dimension scan into {out_dir}", flush=True)


if __name__ == "__main__":
    main()
