from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_competition import (
    BASELINE_KNN,
    BASELINE_MEAN,
    TASK_PRESETS,
    baseline_specs,
    catalog_by_id,
    default_example_count,
    estimate_competition_budget,
    fetch_openrouter_catalog,
    run_competition,
)


class BudgetRequest(BaseModel):
    model_ids: list[str]
    task_name: str
    task_kwargs: dict[str, Any] = Field(default_factory=dict)
    n_dims: int
    example_count: int
    num_eval_examples: int
    prompt_style: str = "words2numbers"
    answer_format: str = "strict_number"
    expected_completion_tokens: int = 16
    estimate_samples: int = 25
    x_decimals: int = 2
    y_decimals: int = 2


class MatchRequest(BaseModel):
    model_ids: list[str]
    task_name: str
    task_kwargs: dict[str, Any] = Field(default_factory=dict)
    n_dims: int
    example_count: int
    num_eval_examples: int
    prompt_style: str = "words2numbers"
    answer_format: str = "strict_number"
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    x_decimals: int = 2
    y_decimals: int = 2
    knn_k: int = 5
    api_key: str = ""


app = FastAPI(title="moe-icl arena api", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/tasks")
def tasks() -> dict[str, Any]:
    rows = []
    for label, spec in TASK_PRESETS.items():
        rows.append(
            {
                "label": label,
                "task_name": spec["task_name"],
                "default_task_kwargs": spec["default_task_kwargs"],
            }
        )
    return {"tasks": rows}


@app.get("/api/models")
def models() -> dict[str, Any]:
    catalog = fetch_openrouter_catalog()
    rows = baseline_specs() + catalog
    return {"models": rows}


@app.get("/api/defaults")
def defaults() -> dict[str, Any]:
    return {
        "recommended_examples_rule": "2x_dimension",
        "default_example_count_for_dim_10": default_example_count(10),
        "baseline_ids": [BASELINE_MEAN, BASELINE_KNN],
    }


@app.post("/api/budget")
def budget(req: BudgetRequest) -> dict[str, Any]:
    if not req.model_ids:
        raise HTTPException(status_code=400, detail="model_ids must not be empty")
    return estimate_competition_budget(
        model_ids=req.model_ids,
        task_name=req.task_name,
        task_kwargs=req.task_kwargs,
        n_dims=req.n_dims,
        example_count=req.example_count,
        num_eval_examples=req.num_eval_examples,
        prompt_style=req.prompt_style,
        answer_format=req.answer_format,
        expected_completion_tokens=req.expected_completion_tokens,
        x_decimals=req.x_decimals,
        y_decimals=req.y_decimals,
        estimate_samples=req.estimate_samples,
    )


@app.post("/api/run")
def run(req: MatchRequest) -> dict[str, Any]:
    if not req.model_ids:
        raise HTTPException(status_code=400, detail="model_ids must not be empty")
    api_key = (req.api_key or "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
    needs_remote_models = any(not model_id.startswith("baseline/") for model_id in req.model_ids)
    if needs_remote_models and not api_key:
        raise HTTPException(
            status_code=400,
            detail="No API key provided. Set OPENROUTER_API_KEY or send api_key in the request.",
        )
    return run_competition(
        model_ids=req.model_ids,
        api_key=api_key,
        task_name=req.task_name,
        task_kwargs=req.task_kwargs,
        n_dims=req.n_dims,
        example_count=req.example_count,
        num_eval_examples=req.num_eval_examples,
        prompt_style=req.prompt_style,
        answer_format=req.answer_format,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        presence_penalty=req.presence_penalty,
        x_decimals=req.x_decimals,
        y_decimals=req.y_decimals,
        knn_k=req.knn_k,
    )


@app.get("/api/model/{model_id:path}")
def model_detail(model_id: str) -> dict[str, Any]:
    meta = catalog_by_id().get(model_id)
    if meta is None and model_id not in {BASELINE_MEAN, BASELINE_KNN}:
        raise HTTPException(status_code=404, detail="model not found")
    if model_id in {BASELINE_MEAN, BASELINE_KNN}:
        return {"id": model_id, "kind": "baseline"}
    return meta
