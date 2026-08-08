"""Builds and runs the conversational RAG LangGraph.

    classify --continue--> rephrase -> fetch -> build_answer -> guardrail -> END
             --greet/deny-> canned_response ------------------------------> END

`classify` is UX routing only; access control is enforced deterministically
in `fetch_node` via `allowed_companies`, which callers must derive from a
fresh permissions lookup (see `api/routes/chat.py`), not from the graph.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langgraph.graph import END, START, StateGraph

from .nodes import (
    build_answer_node,
    canned_response_node,
    classify_node,
    fetch_node,
    guardrail_node,
    rephrase_node,
)
from .state import ChatState, ConversationTurn


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
