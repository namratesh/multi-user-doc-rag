"""LangSmith tracing client shared by the LLM client and LangGraph nodes.

Constructed explicitly from `settings` (rather than relying on LangSmith's
own `LANGSMITH_*` os.environ lookup) because `.env` is read directly by
pydantic-settings and never exported to the process environment -- see
`settings.py`. `langsmith_client` is passed explicitly to every `@traceable`
call and `trace()` context manager in `llm/client.py`, `graph/graph.py` and
`graph/nodes.py`. `LANGSMITH_TRACING` still has to land in the process
environment because that's the only switch the langsmith SDK checks to
decide whether tracing is on at all; when the API key isn't set, tracing
stays off and every `@traceable`/`trace()` call becomes a no-op rather than
raising, so local dev without LangSmith configured is unaffected.
"""

from __future__ import annotations

import os

from langsmith import Client

from .logger import get_logger
from .settings import settings

logger = get_logger(__name__)

langsmith_enabled = bool(settings.langsmith_api_key)

langsmith_client = Client(
    api_key=settings.langsmith_api_key,
    api_url=settings.langsmith_endpoint,
)

os.environ["LANGSMITH_TRACING"] = "true" if langsmith_enabled else "false"
os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

if langsmith_enabled:
    logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
else:
    logger.info("LangSmith tracing disabled (LANGSMITH_API_KEY not set)")
