"""Builds and runs the conversational RAG LangGraph.

    classify --continue--> rephrase -> fetch -> build_answer -> guardrail -> END
             --greet/deny-> canned_response ------------------------------> END

`classify` is UX routing only; access control is enforced deterministically
in `fetch_node` via `allowed_companies`, which callers must derive from a
fresh permissions lookup (see `api/routes/chat.py`), not from the graph.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator, Literal

from langgraph.graph import END, START, StateGraph

from src.config.logger import get_logger
from src.config.settings import settings
from src.llm.client import stream_chat_completion

from .nodes import (
    GUARDRAIL_FALLBACK_MESSAGE,
    NO_CONTEXT_MESSAGE,
    build_answer_messages,
    build_answer_node,
    canned_response_node,
    classify_node,
    fetch_node,
    guardrail_node,
    guardrail_verdict,
    rephrase_node,
    select_citations,
)
from .state import ChatState, ConversationTurn

logger = get_logger(__name__)


def _route_after_classify(state: ChatState) -> Literal["continue", "greet", "deny"]:
    return state.get("route", "continue")


@lru_cache(maxsize=1)
def get_chat_graph():
    builder = StateGraph(ChatState)
    builder.add_node("classify", classify_node)
    builder.add_node("canned_response", canned_response_node)
    builder.add_node("rephrase", rephrase_node)
    builder.add_node("fetch", fetch_node)
    builder.add_node("build_answer", build_answer_node)
    builder.add_node("guardrail", guardrail_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"continue": "rephrase", "greet": "canned_response", "deny": "canned_response"},
    )
    builder.add_edge("canned_response", END)
    builder.add_edge("rephrase", "fetch")
    builder.add_edge("fetch", "build_answer")
    builder.add_edge("build_answer", "guardrail")
    builder.add_edge("guardrail", END)
    return builder.compile()


def run_chat_graph(
    *,
    email: str,
    allowed_companies: list[str],
    conv_id: str,
    question: str,
    history: list[ConversationTurn],
) -> ChatState:
    graph = get_chat_graph()
    initial_state: ChatState = {
        "email": email,
        "allowed_companies": allowed_companies,
        "conv_id": conv_id,
        "question": question,
        "history": history,
    }
    return graph.invoke(initial_state)


def stream_chat_graph_answer(
    *,
    email: str,
    allowed_companies: list[str],
    conv_id: str,
    question: str,
    history: list[ConversationTurn],
) -> Iterator[dict]:
    """Runs the same classify/rephrase/fetch steps as `run_chat_graph`, but
    streams the answer-generation LLM call token-by-token as `delta` events.

    The guardrail still needs the *complete* answer to judge groundedness,
    so it only runs after streaming finishes; the trailing `done` event
    carries the authoritative answer/citations. If the guardrail rejects the
    answer, `done.answer` will differ from the concatenated `delta` text --
    callers must treat `done` as the source of truth and replace whatever
    was rendered from deltas, not just append to it.
    """
    state: ChatState = {
        "email": email,
        "allowed_companies": allowed_companies,
        "conv_id": conv_id,
        "question": question,
        "history": history,
    }
    state.update(classify_node(state))
    route = state.get("route", "continue")

    if route in ("greet", "deny"):
        state.update(canned_response_node(state))
        yield {
            "type": "done",
            "route": route,
            "answer": state["final_answer"],
            "citations": [],
            "guardrail_passed": True,
        }
        return

    state.update(rephrase_node(state))
    state.update(fetch_node(state))
    chunks = state.get("chunks") or []
    answer_question = state.get("standalone_question") or question

    if not chunks:
        yield {
            "type": "done",
            "route": route,
            "answer": NO_CONTEXT_MESSAGE,
            "citations": [],
            "guardrail_passed": True,
        }
        return

    messages = build_answer_messages(answer_question, chunks)
    parts: list[str] = []
    try:
        for delta in stream_chat_completion(messages, temperature=settings.chat_temperature):
            parts.append(delta)
            yield {"type": "delta", "text": delta}
    except Exception:
        logger.exception("[stream_chat_graph_answer] answer stream failed")
        yield {
            "type": "done",
            "route": route,
            "answer": "I ran into an error generating an answer. Please try again.",
            "citations": [],
            "guardrail_passed": True,
        }
        return

    answer = "".join(parts).strip()
    citations = select_citations(answer, chunks)
    passed, reason = guardrail_verdict(answer, chunks)

    if passed:
        yield {
            "type": "done",
            "route": route,
            "answer": answer,
            "citations": citations,
            "guardrail_passed": True,
        }
    else:
        logger.warning("[stream_chat_graph_answer] guardrail rejected answer reason=%s", reason)
        yield {
            "type": "done",
            "route": route,
            "answer": GUARDRAIL_FALLBACK_MESSAGE,
            "citations": [],
            "guardrail_passed": False,
        }
