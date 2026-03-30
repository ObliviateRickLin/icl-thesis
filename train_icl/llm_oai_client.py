from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def _extract_text(response_json: dict) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _response_to_json(response_obj) -> dict:
    if isinstance(response_obj, dict):
        return response_obj
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(response_obj, attr, None)
        if callable(fn):
            return fn()
    raise TypeError(f"Unsupported response type: {type(response_obj)!r}")


class OpenAICompatClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 300.0,
        max_retries: int = 5,
        retry_sleep_s: float = 2.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.retry_sleep_s = float(retry_sleep_s)

    def chat_completion(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        presence_penalty: float,
        extra_body: dict | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "presence_penalty": float(presence_penalty),
        }
        if stop:
            payload["stop"] = stop
        if extra_body:
            payload.update(extra_body)

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as err:
                last_err = err
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(self.retry_sleep_s * (attempt + 1))
        raise RuntimeError(f"OpenAI-compatible request failed after {self.max_retries} attempts: {last_err}")

    def complete_text(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        presence_penalty: float,
        extra_body: dict | None = None,
        stop: list[str] | None = None,
    ) -> tuple[str, dict]:
        response_json = self.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            extra_body=extra_body,
            stop=stop,
        )
        return _extract_text(response_json), response_json


class LiteLLMClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "",
        timeout_s: float = 300.0,
        max_retries: int = 5,
        retry_sleep_s: float = 2.0,
    ) -> None:
        try:
            from litellm import completion
        except ImportError as err:
            raise RuntimeError(
                "LiteLLM is not installed. Install it with `pip install litellm` in a Python >= 3.9 environment."
            ) from err

        self._completion = completion
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.retry_sleep_s = float(retry_sleep_s)

    def chat_completion(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        presence_penalty: float,
        extra_body: dict | None = None,
        stop: list[str] | None = None,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "presence_penalty": float(presence_penalty),
            "timeout": self.timeout_s,
        }
        if self.api_key:
            payload["api_key"] = self.api_key
        if self.base_url:
            payload["base_url"] = self.base_url
        if stop:
            payload["stop"] = stop
        if extra_body:
            payload.update(extra_body)

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._completion(**payload)
                return _response_to_json(response)
            except Exception as err:
                last_err = err
                if attempt + 1 >= self.max_retries:
                    break
                time.sleep(self.retry_sleep_s * (attempt + 1))
        raise RuntimeError(f"LiteLLM request failed after {self.max_retries} attempts: {last_err}")

    def complete_text(
        self,
        *,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        presence_penalty: float,
        extra_body: dict | None = None,
        stop: list[str] | None = None,
    ) -> tuple[str, dict]:
        response_json = self.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            extra_body=extra_body,
            stop=stop,
        )
        return _extract_text(response_json), response_json
