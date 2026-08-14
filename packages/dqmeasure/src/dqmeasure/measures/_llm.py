from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any

_TIMEOUT_SECONDS = 60.0


def openai_completion(model: str, url: str, api_key: str | None) -> Callable[[str], str]:
    """Build a completion callable against an OpenAI-compatible chat-completions endpoint."""

    def complete(prompt: str) -> str:
        payload: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        headers = {"Content-Type": "application/json"}
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            f"{url.rstrip('/')}/chat/completions", data=json.dumps(payload).encode(), headers=headers
        )
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read())
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:  # an error body, or a server that is not compatible
            raise RuntimeError(f"Unexpected chat-completions response from {url}: {body!r}") from error

    return complete


def render_record(row: dict[str, Any]) -> str:
    return ", ".join(f"{name}: {'' if value is None else value}" for name, value in row.items())
