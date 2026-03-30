from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def _prepend_transformers_src(transformers_src: str | None) -> None:
    if not transformers_src:
        return
    src = str(Path(transformers_src).resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = 0.0
    top_p: float = 1.0
    presence_penalty: float = 0.0
    stop: list[str] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


def _message_content_to_text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "".join(parts)


def _messages_to_template(messages: list[ChatMessage]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        out.append({"role": msg.role, "content": _message_content_to_text(msg.content)})
    return out


def _first_device(model) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


def _trim_stop(text: str, stops: list[str] | None) -> str:
    if not stops:
        return text
    cut = len(text)
    for stop in stops:
        if not stop:
            continue
        idx = text.find(stop)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut]


def _count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(len(ids))


def _strip_think_block(text: str) -> str:
    if "</think>" in text:
        return re.sub(r"(?s)^.*?</think>\s*", "", text, count=1)
    return text


def build_app(args: argparse.Namespace) -> FastAPI:
    _prepend_transformers_src(args.transformers_src)

    from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    model = Qwen3_5MoeForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    input_device = _first_device(model)

    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model_path": args.model_path,
            "device": str(input_device),
            "dtype": str(getattr(model, "dtype", "unknown")),
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest) -> dict[str, Any]:
        try:
            with torch.inference_mode():
                templated_messages = _messages_to_template(req.messages)
                prompt_text = tokenizer.apply_chat_template(
                    templated_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(req.chat_template_kwargs or {}),
                )
                inputs = tokenizer(prompt_text, return_tensors="pt")
                inputs = {k: v.to(input_device) for k, v in inputs.items()}

                do_sample = float(req.temperature) > 0.0
                gen_kwargs = {
                    "max_new_tokens": int(req.max_tokens),
                    "do_sample": do_sample,
                    "top_p": float(req.top_p),
                    "use_cache": True,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    gen_kwargs["temperature"] = float(req.temperature)

                t0 = time.time()
                outputs = model.generate(**inputs, **gen_kwargs)
                latency_s = time.time() - t0

                prompt_tokens = int(inputs["input_ids"].shape[-1])
                completion_ids = outputs[0, prompt_tokens:]
                completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                completion_text = _strip_think_block(completion_text)
                completion_text = _trim_stop(completion_text, req.stop)
                print(f"[serve_hf_qwen_oai] completion_text={completion_text!r}", flush=True)
                completion_tokens = _count_tokens(tokenizer, completion_text)

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model or args.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": completion_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "x_latency_s": latency_s,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", default="")
    p.add_argument("--transformers-src", default="")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device-map", default="auto")
    args = p.parse_args()
    if not args.model_name:
        args.model_name = Path(args.model_path).name

    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
