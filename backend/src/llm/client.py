"""Chat-completion client for the RAG pipeline.

Groq and OpenRouter both expose an OpenAI-compatible API, so chat completions
go through the official `openai` SDK pointed at whichever one `CHAT_PROVIDER`
(.env / settings) selects. Embeddings are unaffected and always go through
OpenRouter (`ingest/embed_and_store.py`).
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


_PROVIDERS = {
    "groq": ("groq_api_key", "groq_base_url", "GROQ_API_KEY"),
    "openrouter": ("openrouter_api_key", "openrouter_base_url", "OPENROUTER_API_KEY"),
}


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    provider = settings.chat_provider.lower()
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown CHAT_PROVIDER {settings.chat_provider!r}. "
            f"Expected one of {sorted(_PROVIDERS)}."
        )
    api_key_field, base_url_field, env_var = _PROVIDERS[provider]
    api_key = getattr(settings, api_key_field)
    if not api_key:
        raise ValueError(f"{env_var} is not set. Add it to .env to use chat completions.")
    return OpenAI(api_key=api_key, base_url=getattr(settings, base_url_field))


def chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    model: str | None = None,
) -> str:
    client = get_openai_client()
    model_name = model or settings.chat_model_name
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    logger.info(
        "[chat_completion] input model=%s messages=%d last_user=%r",
        model_name,
        len(messages),
        last_user[:200],
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    content = response.choices[0].message.content or ""
    logger.info("[chat_completion] output content_len=%d", len(content))
    return content


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
