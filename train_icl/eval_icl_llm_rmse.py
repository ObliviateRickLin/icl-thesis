from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from llm_cost import aggregate_costs, count_message_tokens, usage_cost_breakdown
from llm_oai_client import LiteLLMClient, OpenAICompatClient
from samplers import get_data_sampler
from tasks import get_task_sampler


FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?")
FULL_FLOAT_RE = re.compile(r"\s*[-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?\s*\Z")


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _parse_int_csv(s: str) -> list[int]:
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def _parse_json_dict(s: str) -> dict:
    s = (s or "").strip()
    if not s:
        return {}
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("--task-kwargs and --extra-body must be JSON objects")
    return obj


def _format_num(x: float, decimals: int) -> str:
    if math.isnan(float(x)) or math.isinf(float(x)):
        raise ValueError("NaN/Inf in prompt values is not supported")
    return f"{float(x):.{decimals}f}"


def _format_vec(xs: np.ndarray, decimals: int) -> str:
    return "[" + ", ".join(_format_num(v, decimals) for v in xs.tolist()) + "]"


def _format_feature_lines(xs: np.ndarray, decimals: int) -> list[str]:
    return [f"Feature {i}: {_format_num(v, decimals)}" for i, v in enumerate(xs.tolist())]


def _format_feature_lines_token(xs: np.ndarray, decimals: int, number_style: str) -> list[str]:
    return [f"Feature {i}: {_format_num_token(v, decimals, number_style)}" for i, v in enumerate(xs.tolist())]


def _format_num_token(x: float, decimals: int, number_style: str) -> str:
    if math.isnan(float(x)) or math.isinf(float(x)):
        raise ValueError("NaN/Inf in prompt values is not supported")
    value = float(x)
    if number_style == "signed_fixed":
        return f"{value:+.{decimals}f}"
    if number_style == "signed_scientific":
        return f"{value:+.{decimals}e}"
    return _format_num(value, decimals)


def _format_vec_token(xs: np.ndarray, decimals: int, number_style: str) -> str:
    return "[" + ",".join(_format_num_token(v, decimals, number_style) for v in xs.tolist()) + "]"


def _order_context(xs_ctx: np.ndarray, ys_ctx: np.ndarray, x_query: np.ndarray, prompt_style: str) -> tuple[np.ndarray, np.ndarray]:
    if prompt_style not in {"compact_ordered_fixed", "compact_ordered_scientific"}:
        return xs_ctx, ys_ctx
    dists = np.linalg.norm(xs_ctx - x_query[None, :], axis=1)
    order = np.argsort(dists, kind="stable")
    return xs_ctx[order], ys_ctx[order]


def _ctx_y_stats(ys_ctx: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(ys_ctx))
    std = float(np.std(ys_ctx))
    if not math.isfinite(std) or std < 1e-6:
        std = 1.0
    return mean, std


def _postprocess_prediction(pred: float | None, ys_ctx: np.ndarray, prompt_style: str) -> float | None:
    if pred is None:
        return None
    if prompt_style == "compact_yz":
        y_mean, y_std = _ctx_y_stats(ys_ctx)
        return float(y_mean + y_std * float(pred))
    if prompt_style in {"scaled_integer_csv", "jsonl_integer"}:
        return float(pred) / 100.0
    return float(pred)


def _build_prompt(
    *,
    xs_ctx: np.ndarray,
    ys_ctx: np.ndarray,
    x_query: np.ndarray,
    x_decimals: int,
    y_decimals: int,
    prompt_style: str,
    answer_format: str,
) -> str:
    xs_ctx, ys_ctx = _order_context(xs_ctx, ys_ctx, x_query, prompt_style)
    if answer_format == "final_tag":
        answer_instruction = (
            "You may reason step by step, but the final line must be exactly `Final: <number>`."
        )
        final_stub = "Final:"
    else:
        answer_instruction = "Output only one number and nothing else."
        final_stub = "Output:"

    if prompt_style == "compact":
        lines = [
            "Predict the scalar target for the query vector from the examples of the same function.",
            answer_instruction,
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            lines.append(f"x={_format_vec(xs_ctx[i], x_decimals)} y={_format_num(ys_ctx[i], y_decimals)}")
        if answer_format == "final_tag":
            lines.append(f"x={_format_vec(x_query, x_decimals)}")
            lines.append("Final:")
        else:
            lines.append(f"x={_format_vec(x_query, x_decimals)} y=")
        return "\n".join(lines)

    if prompt_style == "compact_ordered_fixed":
        lines = [
            "Examples below come from the same unknown real-valued function.",
            "The examples are ordered from most similar to the query to least similar.",
            answer_instruction,
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            lines.append(
                f"input={_format_vec_token(xs_ctx[i], x_decimals, 'signed_fixed')} "
                f"output={_format_num_token(ys_ctx[i], y_decimals, 'signed_fixed')}"
            )
        if answer_format == "final_tag":
            lines.append(f"input={_format_vec_token(x_query, x_decimals, 'signed_fixed')}")
            lines.append("Final:")
        else:
            lines.append(f"input={_format_vec_token(x_query, x_decimals, 'signed_fixed')} output=")
        return "\n".join(lines)

    if prompt_style == "compact_ordered_scientific":
        lines = [
            "Examples below come from the same unknown real-valued function.",
            "The examples are ordered from most similar to the query to least similar.",
            answer_instruction,
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            lines.append(
                f"input={_format_vec_token(xs_ctx[i], x_decimals, 'signed_scientific')} "
                f"output={_format_num_token(ys_ctx[i], y_decimals, 'signed_scientific')}"
            )
        if answer_format == "final_tag":
            lines.append(f"input={_format_vec_token(x_query, x_decimals, 'signed_scientific')}")
            lines.append("Final:")
        else:
            lines.append(f"input={_format_vec_token(x_query, x_decimals, 'signed_scientific')} output=")
        return "\n".join(lines)

    if prompt_style == "compact_io":
        lines = [
            "Predict the output for the final input from the examples of the same function.",
            answer_instruction,
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            lines.append(f"Input: {_format_vec(xs_ctx[i], x_decimals)}")
            lines.append(f"Output: {_format_num(ys_ctx[i], y_decimals)}")
            lines.append("")
        lines.append(f"Input: {_format_vec(x_query, x_decimals)}")
        lines.append(final_stub)
        return "\n".join(lines)

    if prompt_style == "compact_yz":
        y_mean, y_std = _ctx_y_stats(ys_ctx)
        stat_decimals = max(4, y_decimals)
        lines = [
            "Predict the standardized target z for the query vector from the examples of the same function.",
            "The targets in the examples use z = (y - mean) / std for this context.",
            f"context_mean={_format_num(y_mean, stat_decimals)} context_std={_format_num(y_std, stat_decimals)}",
            "Return only the standardized z value for the query.",
            answer_instruction,
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            z = (float(ys_ctx[i]) - y_mean) / y_std
            lines.append(f"x={_format_vec(xs_ctx[i], x_decimals)} z={_format_num(z, y_decimals)}")
        if answer_format == "final_tag":
            lines.append(f"x={_format_vec(x_query, x_decimals)}")
            lines.append("Final:")
        else:
            lines.append(f"x={_format_vec(x_query, x_decimals)} z=")
        return "\n".join(lines)

    if prompt_style == "words2numbers":
        lines = [
            'The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.',
            answer_instruction,
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            lines.extend(_format_feature_lines(xs_ctx[i], x_decimals))
            lines.append(f"Output: {_format_num(ys_ctx[i], y_decimals)}")
            lines.append("")
        lines.extend(_format_feature_lines(x_query, x_decimals))
        lines.append(final_stub)
        return "\n".join(lines)

    if prompt_style == "words2numbers_linear_informed":
        lines = [
            "The examples below are generated by the same noisy linear regression model.",
            "There is one fixed linear relationship between the input vector and the output scalar, and the examples differ only because of small noise.",
            "Use the examples to infer the linear relationship and estimate the Output for the final example.",
            answer_instruction,
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            lines.extend(_format_feature_lines(xs_ctx[i], x_decimals))
            lines.append(f"Output: {_format_num(ys_ctx[i], y_decimals)}")
            lines.append("")
        lines.extend(_format_feature_lines(x_query, x_decimals))
        lines.append(final_stub)
        return "\n".join(lines)

    if prompt_style == "words2numbers_calibrated":
        lines = [
            "You are given input-output examples from the same noisy real-valued function.",
            "Use the numeric relationship between the Features and the Output in the examples to estimate the most likely Output for the final example.",
            "The final answer must be a single real number.",
            "Do not explain your reasoning. Do not repeat the prompt.",
            answer_instruction,
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            lines.extend(_format_feature_lines(xs_ctx[i], x_decimals))
            lines.append(f"Output: {_format_num(ys_ctx[i], y_decimals)}")
            lines.append("")
        lines.extend(_format_feature_lines(x_query, x_decimals))
        lines.append(final_stub)
        return "\n".join(lines)

    if prompt_style == "scaled_integer_csv":
        lines = [
            "All values below are integers obtained by multiplying the original real values by 100 and rounding.",
            "Each row is one example from the same unknown function.",
            "Predict the missing integer output for the final row.",
            "Return only the integer output.",
            "",
        ]
        header = ",".join([f"f{i}" for i in range(xs_ctx.shape[1])] + ["y"])
        lines.append(header)
        for i in range(xs_ctx.shape[0]):
            x_vals = [str(int(round(float(v) * 100.0))) for v in xs_ctx[i].tolist()]
            y_val = str(int(round(float(ys_ctx[i]) * 100.0)))
            lines.append(",".join(x_vals + [y_val]))
        x_query_vals = [str(int(round(float(v) * 100.0))) for v in x_query.tolist()]
        lines.append(",".join(x_query_vals + ["?"]))
        return "\n".join(lines)

    if prompt_style == "csv_real":
        lines = [
            "Each row below is one example from the same unknown real-valued function.",
            "The last column is the output value.",
            "Predict the missing output in the final row.",
            answer_instruction,
            "",
        ]
        header = ",".join([f"f{i}" for i in range(xs_ctx.shape[1])] + ["y"])
        lines.append(header)
        for i in range(xs_ctx.shape[0]):
            x_vals = [_format_num(v, x_decimals) for v in xs_ctx[i].tolist()]
            y_val = _format_num(float(ys_ctx[i]), y_decimals)
            lines.append(",".join(x_vals + [y_val]))
        x_query_vals = [_format_num(v, x_decimals) for v in x_query.tolist()]
        lines.append(",".join(x_query_vals + ["?"]))
        return "\n".join(lines)

    if prompt_style == "jsonl_integer":
        lines = [
            "All values below are integers obtained by multiplying the original real values by 100 and rounding.",
            "Each JSON line is one example from the same unknown function.",
            "Predict the missing integer output for the final JSON line.",
            "Return only the integer output.",
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            x_vals = [int(round(float(v) * 100.0)) for v in xs_ctx[i].tolist()]
            y_val = int(round(float(ys_ctx[i]) * 100.0))
            lines.append(json.dumps({"input": x_vals, "output": y_val}, separators=(",", ":")))
        x_query_vals = [int(round(float(v) * 100.0)) for v in x_query.tolist()]
        lines.append(json.dumps({"input": x_query_vals, "output": "?"}, separators=(",", ":")))
        return "\n".join(lines)

    if prompt_style == "mapping_arrow":
        lines = [
            "Each line below is one input-output example from the same unknown real-valued function.",
            "Predict the missing output for the final input.",
            answer_instruction,
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            lines.append(f"{_format_vec(xs_ctx[i], x_decimals)} -> {_format_num(float(ys_ctx[i]), y_decimals)}")
        if answer_format == "final_tag":
            lines.append(f"{_format_vec(x_query, x_decimals)}")
            lines.append("Final:")
        else:
            lines.append(f"{_format_vec(x_query, x_decimals)} ->")
        return "\n".join(lines)

    if prompt_style == "words2numbers_fixed":
        lines = [
            'The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.',
            answer_instruction,
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            lines.extend(_format_feature_lines_token(xs_ctx[i], x_decimals, "signed_fixed"))
            lines.append(f"Output: {_format_num_token(ys_ctx[i], y_decimals, 'signed_fixed')}")
            lines.append("")
        lines.extend(_format_feature_lines_token(x_query, x_decimals, "signed_fixed"))
        lines.append(final_stub)
        return "\n".join(lines)

    if prompt_style == "words2numbers_scientific":
        lines = [
            'The task is to provide your best estimate for "Output". Please provide that and only that, without any additional text.',
            answer_instruction,
            "",
        ]
        for i in range(xs_ctx.shape[0]):
            lines.extend(_format_feature_lines_token(xs_ctx[i], x_decimals, "signed_scientific"))
            lines.append(f"Output: {_format_num_token(ys_ctx[i], y_decimals, 'signed_scientific')}")
            lines.append("")
        lines.extend(_format_feature_lines_token(x_query, x_decimals, "signed_scientific"))
        lines.append(final_stub)
        return "\n".join(lines)

    lines = [
        "You are given examples of a function from a real-valued vector x to a real-valued scalar y.",
        "Predict y for the final x.",
        answer_instruction,
    ]
    for i in range(xs_ctx.shape[0]):
        lines.append(
            f"Example {i + 1}: x={_format_vec(xs_ctx[i], x_decimals)} y={_format_num(ys_ctx[i], y_decimals)}"
        )
    if answer_format == "final_tag":
        lines.append(f"Query: x={_format_vec(x_query, x_decimals)}")
        lines.append("Final:")
    else:
        lines.append(f"Query: x={_format_vec(x_query, x_decimals)} y=")
    return "\n".join(lines)


def _parse_prediction(text: str, answer_format: str) -> float | None:
    text = (text or "").strip()
    if answer_format == "final_tag":
        matches = re.findall(r"(?im)^\s*Final\s*:\s*([-+]?(?:\d+\.\d*|\d+|\.\d+)(?:[eE][-+]?\d+)?)\s*$", text)
        if matches:
            try:
                return float(matches[-1])
            except ValueError:
                return None
    if not FULL_FLOAT_RE.fullmatch(text):
        return None
    match = FLOAT_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _task_to_label(task_name: str, task_kwargs: dict) -> str:
    if not task_kwargs:
        return task_name
    items = ",".join([f"{k}={task_kwargs[k]}" for k in sorted(task_kwargs.keys())])
    return f"{task_name}({items})"


@torch.no_grad()
def collect_xy(
    *,
    task_name: str,
    task_kwargs: dict,
    data_name: str,
    n_dims: int,
    n_points: int,
    num_eval_examples: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert num_eval_examples % batch_size == 0
    data_sampler = get_data_sampler(data_name, n_dims=n_dims)
    task_sampler = get_task_sampler(task_name, n_dims, batch_size, **task_kwargs)
    xs_all, ys_all = [], []
    for _ in range(num_eval_examples // batch_size):
        xs = data_sampler.sample_xs(n_points, batch_size, n_dims, device=device)
        task = task_sampler(device=device)
        ys = task.evaluate(xs)
        xs_all.append(xs.detach().cpu())
        ys_all.append(ys.detach().cpu())
    return torch.cat(xs_all, dim=0), torch.cat(ys_all, dim=0)


def _messages_for_prompt(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def _build_client(args):
    if args.client_type == "litellm":
        endpoint = "" if (args.endpoint or "").strip() == "http://127.0.0.1:8000/v1" else args.endpoint
        return LiteLLMClient(
            model=args.model,
            api_key=args.api_key,
            base_url=endpoint,
            timeout_s=args.timeout_s,
            max_retries=args.max_retries,
        )
    return OpenAICompatClient(
        base_url=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
    )


def _cost_model_name(args) -> str:
    return (args.cost_model or "").strip() or str(args.model)


def _tokenizer_model_name(args) -> str:
    return (args.tokenizer_model or "").strip() or _cost_model_name(args)


def _estimate_cost_rows(
    *,
    xs_np: np.ndarray,
    ys_np: np.ndarray,
    icl_lens: list[int],
    num_eval_examples: int,
    x_decimals: int,
    y_decimals: int,
    prompt_style: str,
    answer_format: str,
    estimate_samples: int,
    expected_completion_tokens: int,
    cost_model: str,
    tokenizer_model: str,
    seed: int,
) -> list[dict]:
    total_episodes = int(xs_np.shape[0])
    if estimate_samples and estimate_samples > 0 and estimate_samples < total_episodes:
        rng = random.Random(int(seed))
        sample_indices = sorted(rng.sample(range(total_episodes), int(estimate_samples)))
        exact = False
    else:
        sample_indices = list(range(total_episodes))
        exact = True

    rows = []
    for L in icl_lens:
        prompt_token_sum = 0
        for episode_idx in sample_indices:
            prompt = _build_prompt(
                xs_ctx=xs_np[episode_idx, :L, :],
                ys_ctx=ys_np[episode_idx, :L],
                x_query=xs_np[episode_idx, L, :],
                x_decimals=x_decimals,
                y_decimals=y_decimals,
                prompt_style=prompt_style,
                answer_format=answer_format,
            )
            prompt_token_sum += count_message_tokens(
                model=tokenizer_model,
                messages=_messages_for_prompt(prompt),
            )

        sampled_requests = len(sample_indices)
        avg_prompt_tokens = float(prompt_token_sum) / float(sampled_requests)
        if exact:
            prompt_tokens_total = int(prompt_token_sum)
        else:
            prompt_tokens_total = int(round(avg_prompt_tokens * float(num_eval_examples)))
        completion_tokens_total = int(expected_completion_tokens) * int(num_eval_examples)
        costs = usage_cost_breakdown(
            model=cost_model,
            prompt_tokens=prompt_tokens_total,
            completion_tokens=completion_tokens_total,
        )
        rows.append(
            {
                "icl_len": int(L),
                "num_eval_examples": int(num_eval_examples),
                "estimate_samples": int(sampled_requests),
                "estimate_exact": int(exact),
                "avg_prompt_tokens": float(avg_prompt_tokens),
                "avg_completion_tokens": float(expected_completion_tokens),
                "prompt_tokens_total": int(prompt_tokens_total),
                "completion_tokens_total": int(completion_tokens_total),
                **costs,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--client-type", default="openai_compat", choices=["openai_compat", "litellm"])
    p.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", required=True)
    p.add_argument("--cost-model", default="", help="LiteLLM pricing model name. Defaults to --model.")
    p.add_argument("--tokenizer-model", default="", help="LiteLLM tokenizer model name. Defaults to --cost-model/--model.")
    p.add_argument("--task-name", required=True)
    p.add_argument("--task-kwargs", default="{}")
    p.add_argument("--data-name", default="gaussian")
    p.add_argument("--n-dims", type=int, required=True)
    p.add_argument("--n-points", type=int, required=True, help="Total points including the query point.")
    p.add_argument("--icl-lens", default="", help="Comma-separated ICL lengths. Default: final point only.")
    p.add_argument("--num-eval-examples", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="Sampling device for synthetic data.")
    p.add_argument("--x-decimals", type=int, default=2)
    p.add_argument("--y-decimals", type=int, default=2)
    p.add_argument(
        "--prompt-style",
        default="default",
        choices=[
            "default",
            "words2numbers",
            "words2numbers_linear_informed",
            "words2numbers_calibrated",
            "words2numbers_fixed",
            "words2numbers_scientific",
            "scaled_integer_csv",
            "csv_real",
            "jsonl_integer",
            "mapping_arrow",
            "compact",
            "compact_io",
            "compact_yz",
            "compact_ordered_fixed",
            "compact_ordered_scientific",
        ],
    )
    p.add_argument("--answer-format", default="strict_number", choices=["strict_number", "final_tag"])
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--presence-penalty", type=float, default=0.0)
    p.add_argument("--stop-newline", action="store_true")
    p.add_argument("--timeout-s", type=float, default=300.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--disable-thinking", action="store_true")
    p.add_argument("--extra-body", default="{}")
    p.add_argument("--estimate-only", action="store_true")
    p.add_argument(
        "--estimate-samples",
        type=int,
        default=0,
        help="Number of sampled prompts per ICL length for budget estimation. 0 means exact over all prompts.",
    )
    p.add_argument(
        "--estimate-completion-tokens",
        type=int,
        default=-1,
        help="Expected completion tokens per request for budget estimation. Defaults to --max-tokens.",
    )
    p.add_argument("--save-prompts", action="store_true")
    p.add_argument("--out-dir", default="../results_llm")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    _seed_everything(args.seed)

    task_kwargs = _parse_json_dict(args.task_kwargs)
    extra_body = _parse_json_dict(args.extra_body)
    if args.disable_thinking:
        chat_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_kwargs

    icl_lens = _parse_int_csv(args.icl_lens)
    if not icl_lens:
        icl_lens = [int(args.n_points) - 1]
    icl_lens = sorted(set(int(x) for x in icl_lens))
    for L in icl_lens:
        if L <= 0 or L >= int(args.n_points):
            raise SystemExit(f"ICL length must be in [1, n_points-1], got {L}")

    task_label = _task_to_label(args.task_name, task_kwargs)
    tag = args.tag.strip() or (
        f"{args.model.split('/')[-1]}__{args.task_name}__d{args.n_dims}__p{args.n_points}__n{args.num_eval_examples}__s{args.seed}"
    )
    out_dir = Path(args.out_dir).resolve() / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = out_dir / "raw_predictions.csv"
    summary_csv = out_dir / "summary.csv"
    estimate_csv = out_dir / "cost_estimate.csv"

    expected_completion_tokens = (
        int(args.max_tokens) if int(args.estimate_completion_tokens) < 0 else int(args.estimate_completion_tokens)
    )
    cost_model = _cost_model_name(args)
    tokenizer_model = _tokenizer_model_name(args)

    xs, ys = collect_xy(
        task_name=args.task_name,
        task_kwargs=task_kwargs,
        data_name=args.data_name,
        n_dims=args.n_dims,
        n_points=args.n_points,
        num_eval_examples=args.num_eval_examples,
        batch_size=args.batch_size,
        device=args.device,
    )
    xs_np = xs.numpy()
    ys_np = ys.numpy()

    if args.estimate_only:
        estimate_rows = _estimate_cost_rows(
            xs_np=xs_np,
            ys_np=ys_np,
            icl_lens=icl_lens,
            num_eval_examples=args.num_eval_examples,
            x_decimals=args.x_decimals,
            y_decimals=args.y_decimals,
            prompt_style=args.prompt_style,
            answer_format=args.answer_format,
            estimate_samples=args.estimate_samples,
            expected_completion_tokens=expected_completion_tokens,
            cost_model=cost_model,
            tokenizer_model=tokenizer_model,
            seed=args.seed,
        )
        estimate_fields = [
            "created_at_utc",
            "client_type",
            "model",
            "cost_model",
            "tokenizer_model",
            "task_name",
            "task_kwargs_json",
            "data_name",
            "n_dims",
            "n_points",
            "icl_len",
            "num_eval_examples",
            "estimate_samples",
            "estimate_exact",
            "avg_prompt_tokens",
            "avg_completion_tokens",
            "prompt_tokens_total",
            "completion_tokens_total",
            "prompt_cost_usd",
            "completion_cost_usd",
            "total_cost_usd",
        ]
        with estimate_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=estimate_fields)
            writer.writeheader()
            for row in estimate_rows:
                writer.writerow(
                    {
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "client_type": args.client_type,
                        "model": args.model,
                        "cost_model": cost_model,
                        "tokenizer_model": tokenizer_model,
                        "task_name": args.task_name,
                        "task_kwargs_json": json.dumps(task_kwargs, sort_keys=True),
                        "data_name": args.data_name,
                        "n_dims": int(args.n_dims),
                        "n_points": int(args.n_points),
                        **row,
                    }
                )
        overall = aggregate_costs(
            model=cost_model,
            prompt_tokens=sum(int(r["prompt_tokens_total"]) for r in estimate_rows),
            completion_tokens=sum(int(r["completion_tokens_total"]) for r in estimate_rows),
        )
        print(f"[estimate] wrote budget estimate to {estimate_csv}")
        print(
            "[estimate] total prompt_tokens={prompt_tokens_total} completion_tokens={completion_tokens_total} "
            "cost_usd={total_cost_usd:.6f}".format(**overall)
        )
        return

    client = _build_client(args)

    done = set()
    if raw_csv.exists():
        with raw_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((int(row["episode_idx"]), int(row["icl_len"])))

    raw_fields = [
        "episode_idx",
        "icl_len",
        "target_y",
        "pred_y",
        "raw_text",
        "valid",
        "prompt_tokens",
        "completion_tokens",
        "prompt_cost_usd",
        "completion_cost_usd",
        "total_cost_usd",
        "latency_s",
    ]
    if args.save_prompts:
        raw_fields.append("prompt")

    needs_header = not raw_csv.exists()
    with raw_csv.open("a", newline="", encoding="utf-8") as raw_f:
        raw_writer = csv.DictWriter(raw_f, fieldnames=raw_fields)
        if needs_header:
            raw_writer.writeheader()

        for episode_idx in range(xs_np.shape[0]):
            for L in icl_lens:
                key = (episode_idx, L)
                if key in done:
                    continue

                prompt = _build_prompt(
                    xs_ctx=xs_np[episode_idx, :L, :],
                    ys_ctx=ys_np[episode_idx, :L],
                    x_query=xs_np[episode_idx, L, :],
                    x_decimals=args.x_decimals,
                    y_decimals=args.y_decimals,
                    prompt_style=args.prompt_style,
                    answer_format=args.answer_format,
                )
                messages = _messages_for_prompt(prompt)

                t0 = time.perf_counter()
                text, response_json = client.complete_text(
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    presence_penalty=args.presence_penalty,
                    extra_body=extra_body,
                    stop=["\n"] if args.stop_newline else None,
                )
                latency_s = time.perf_counter() - t0
                raw_pred = _parse_prediction(text, args.answer_format)
                pred = _postprocess_prediction(raw_pred, ys_np[episode_idx, :L], args.prompt_style)
                usage = response_json.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                cost_breakdown = usage_cost_breakdown(
                    model=cost_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                row = {
                    "episode_idx": int(episode_idx),
                    "icl_len": int(L),
                    "target_y": float(ys_np[episode_idx, L]),
                    "pred_y": "" if pred is None else float(pred),
                    "raw_text": text,
                    "valid": int(pred is not None),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    **cost_breakdown,
                    "latency_s": float(latency_s),
                }
                if args.save_prompts:
                    row["prompt"] = prompt

                raw_writer.writerow(row)
                raw_f.flush()

    grouped = defaultdict(list)
    with raw_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[int(row["icl_len"])].append(row)

    summary_fields = [
        "created_at_utc",
        "model",
        "endpoint",
        "task",
        "task_name",
        "task_kwargs_json",
        "data_name",
        "n_dims",
        "n_points",
        "icl_len",
        "num_eval_examples",
        "num_valid",
        "num_invalid",
        "mse",
        "rmse",
        "mae",
        "prompt_tokens_total",
        "completion_tokens_total",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "prompt_cost_usd",
        "completion_cost_usd",
        "total_cost_usd",
        "avg_latency_s",
        "seed",
        "x_decimals",
        "y_decimals",
        "disable_thinking",
        "client_type",
        "cost_model",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for L in icl_lens:
            rows = grouped.get(L, [])
            valid_rows = [r for r in rows if int(r["valid"]) == 1]
            targets = np.array([float(r["target_y"]) for r in valid_rows], dtype=np.float64)
            preds = np.array([float(r["pred_y"]) for r in valid_rows], dtype=np.float64)
            if len(valid_rows):
                se = (targets - preds) ** 2
                ae = np.abs(targets - preds)
                mse = float(np.mean(se))
                rmse = float(np.sqrt(mse))
                mae = float(np.mean(ae))
            else:
                mse = float("nan")
                rmse = float("nan")
                mae = float("nan")
            prompt_tokens = np.array([int(r["prompt_tokens"]) for r in rows], dtype=np.int64)
            completion_tokens = np.array([int(r["completion_tokens"]) for r in rows], dtype=np.int64)
            latencies = np.array([float(r["latency_s"]) for r in rows], dtype=np.float64)
            cost_totals = aggregate_costs(
                model=cost_model,
                prompt_tokens=int(prompt_tokens.sum()) if len(prompt_tokens) else 0,
                completion_tokens=int(completion_tokens.sum()) if len(completion_tokens) else 0,
            )
            writer.writerow(
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "model": args.model,
                    "endpoint": args.endpoint,
                    "task": task_label,
                    "task_name": args.task_name,
                    "task_kwargs_json": json.dumps(task_kwargs, sort_keys=True),
                    "data_name": args.data_name,
                    "n_dims": int(args.n_dims),
                    "n_points": int(args.n_points),
                    "icl_len": int(L),
                    "num_eval_examples": int(len(rows)),
                    "num_valid": int(len(valid_rows)),
                    "num_invalid": int(len(rows) - len(valid_rows)),
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "prompt_tokens_total": int(prompt_tokens.sum()) if len(prompt_tokens) else 0,
                    "completion_tokens_total": int(completion_tokens.sum()) if len(completion_tokens) else 0,
                    "avg_prompt_tokens": float(prompt_tokens.mean()) if len(prompt_tokens) else float("nan"),
                    "avg_completion_tokens": float(completion_tokens.mean()) if len(completion_tokens) else float("nan"),
                    "prompt_cost_usd": cost_totals["prompt_cost_usd"],
                    "completion_cost_usd": cost_totals["completion_cost_usd"],
                    "total_cost_usd": cost_totals["total_cost_usd"],
                    "avg_latency_s": float(np.nanmean(latencies)) if len(latencies) else float("nan"),
                    "seed": int(args.seed),
                    "x_decimals": int(args.x_decimals),
                    "y_decimals": int(args.y_decimals),
                    "disable_thinking": int(bool(args.disable_thinking)),
                    "client_type": args.client_type,
                    "cost_model": cost_model,
                }
            )

    print(f"[done] wrote raw predictions to {raw_csv}")
    print(f"[done] wrote summary to {summary_csv}")


if __name__ == "__main__":
    main()
