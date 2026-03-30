from __future__ import annotations

from typing import Iterable


def _require_litellm():
    try:
        from litellm import token_counter
        from litellm.cost_calculator import cost_per_token
        return token_counter, cost_per_token
    except ImportError:
        return None, None


def count_message_tokens(*, model: str, messages: list[dict]) -> int:
    token_counter, _ = _require_litellm()
    if token_counter is None:
        return 0
    return int(token_counter(model=model, messages=messages))


def usage_cost_breakdown(*, model: str, prompt_tokens: int, completion_tokens: int) -> dict[str, float]:
    _, cost_per_token = _require_litellm()
    if cost_per_token is None:
        return {"prompt_cost_usd": 0.0, "completion_cost_usd": 0.0, "total_cost_usd": 0.0}
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
    )
    prompt_cost = float(prompt_cost or 0.0)
    completion_cost = float(completion_cost or 0.0)
    return {
        "prompt_cost_usd": prompt_cost,
        "completion_cost_usd": completion_cost,
        "total_cost_usd": prompt_cost + completion_cost,
    }


def estimate_messages_cost(
    *,
    model: str,
    messages: list[dict],
    expected_completion_tokens: int,
    tokenizer_model: str | None = None,
) -> dict[str, float]:
    prompt_tokens = count_message_tokens(model=tokenizer_model or model, messages=messages)
    costs = usage_cost_breakdown(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=int(expected_completion_tokens),
    )
    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(expected_completion_tokens),
        **costs,
    }


def aggregate_costs(*, model: str, prompt_tokens: int, completion_tokens: int) -> dict[str, float]:
    costs = usage_cost_breakdown(
        model=model,
        prompt_tokens=int(prompt_tokens),
        completion_tokens=int(completion_tokens),
    )
    return {
        "prompt_tokens_total": int(prompt_tokens),
        "completion_tokens_total": int(completion_tokens),
        **costs,
    }


def sum_cost_rows(rows: Iterable[dict]) -> dict[str, float]:
    prompt_tokens = 0
    completion_tokens = 0
    prompt_cost = 0.0
    completion_cost = 0.0
    total_cost = 0.0
    for row in rows:
        prompt_tokens += int(row.get("prompt_tokens", 0) or 0)
        completion_tokens += int(row.get("completion_tokens", 0) or 0)
        prompt_cost += float(row.get("prompt_cost_usd", 0.0) or 0.0)
        completion_cost += float(row.get("completion_cost_usd", 0.0) or 0.0)
        total_cost += float(row.get("total_cost_usd", 0.0) or 0.0)
    return {
        "prompt_tokens_total": int(prompt_tokens),
        "completion_tokens_total": int(completion_tokens),
        "prompt_cost_usd": float(prompt_cost),
        "completion_cost_usd": float(completion_cost),
        "total_cost_usd": float(total_cost),
    }
