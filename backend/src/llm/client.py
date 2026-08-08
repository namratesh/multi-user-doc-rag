"""Chat-completion client for the RAG pipeline.

OpenRouter exposes an OpenAI-compatible API, so the same `OPENROUTER_API_KEY`
and base URL already used for embeddings (`ingest/embed_and_store.py`) drive
chat completions too, via the official `openai` SDK pointed at OpenRouter.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from openai import OpenAI

from src.config.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    if not settings.openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to .env to use chat completions."
        )
    return OpenAI(api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url)


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    model: str | None = None,
) -> str:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model or settings.chat_model_name,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def parse_json_response(text: str, default: dict[str, Any]) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM response.

    Models occasionally wrap JSON in prose or code fences despite
    instructions; this pulls out the first `{...}` block rather than
    requiring an exact match. Falls back to `default` on any failure so a
    single malformed response degrades gracefully instead of crashing the
    pipeline.
    """
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        logger.warning("No JSON object found in LLM response: %r", text[:200])
        return default
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON from LLM response: %r", text[:200])
        return default
