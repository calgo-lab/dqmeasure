from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError, URLError

_TIMEOUT_SECONDS = 60.0
_MAX_ATTEMPTS = 5
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

# A dedicated random instance so the backoff jitter doesn't advance the global `random` stream that `tab_err` seeds.
_jitter = random.Random(0)

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"accurate": {"type": "boolean"}},
            "required": ["accurate"],
            "additionalProperties": False,
        },
    },
}


def _post_with_retry(url: str, data: bytes, headers: dict[str, str]) -> bytes:
    """POST with up to 5 attempts on failures, exponential backoff, taking into account a server's `Retry-After`.

    Retries on 408/409/425/429/500/502/503/504 and on connection-level failures (`URLError`,
    `TimeoutError`). Anything else is raised on the first attempt.
    """
    for attempt in range(_MAX_ATTEMPTS):
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                return bytes(response.read())
        except HTTPError as error:
            if error.code not in _RETRYABLE_STATUS_CODES or attempt == _MAX_ATTEMPTS - 1:
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            delay = float(retry_after) if retry_after is not None else _jitter.uniform(0, 2**attempt)
        except (URLError, TimeoutError):
            if attempt == _MAX_ATTEMPTS - 1:
                raise
            delay = _jitter.uniform(0, 2**attempt)
        time.sleep(delay)
    raise AssertionError("unreachable: the loop above always returns or raises")


def openai_completion(
    model: str,
    url: str,
    api_key: str | None,
    provider: str | None = None,
) -> Callable[[str], str]:
    """Build a completion callable against an OpenAI-compatible endpoint.

    `provider`, if given, pins the request to one upstream provider (relevant for routers such as OpenRouter,
    whose providers for a model differ in quantization and price).
    """

    def complete(prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 32,
            "reasoning": {"enabled": False},
            "response_format": _VERDICT_SCHEMA,
        }
        if provider is not None:
            payload["provider"] = {"order": [provider], "allow_fallbacks": False}
        headers = {"Content-Type": "application/json"}
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = _post_with_retry(f"{url.rstrip('/')}/chat/completions", json.dumps(payload).encode(), headers)
        parsed = json.loads(body)
        try:
            return str(parsed["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:  # an error body, or a server that is not compatible
            raise RuntimeError(f"Unexpected chat-completions response from {url}: {parsed!r}") from error

    return complete


def complete_many(complete: Callable[[str], str], prompts: Iterable[str], n_jobs: int = 1) -> list[str]:
    """Support concurrently running jobs, where `n_jobs=1` is sequential."""
    prompts = list(prompts)
    if n_jobs == 1:
        return [complete(prompt) for prompt in prompts]
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        return list(pool.map(complete, prompts))


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except TypeError:
        return True


def render_record(row: dict[str, Any]) -> str:
    """Render a record's values, pipe-separated in column order, with missing values as `<missing>`."""
    return " | ".join("<missing>" if is_missing(value) else str(value) for value in row.values())
