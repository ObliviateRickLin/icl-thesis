"""
Core data utilities for pretrained LLM ICL evaluation.
Provides: data generation, prompt building (all styles including words2numbers),
prediction parsing, and seeding.
"""
from __future__ import annotations

import json
import math
import random
import re

import numpy as np
import torch

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

