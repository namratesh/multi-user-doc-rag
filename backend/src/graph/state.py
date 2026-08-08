"""Shared state threaded through the LangGraph conversational RAG pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict


class ConversationTurn(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class ChatState(TypedDict, total=False):
    # Inputs, set once before graph.invoke()
    email: str
    allowed_companies: list[str]
    conv_id: str
    question: str
    history: list[ConversationTurn]

    # Populated by classify_node
    route: Literal["greet", "deny", "continue"]

    # Populated by rephrase_node
    standalone_question: str

    # Populated by decompose_node -- one entry per company the question asks
    # about (or a single entry with company_id=None for a general/
    # single-company question). `route_to_fetch` fans each one out to its
    # own `fetch_one` run via `Send`, so they execute concurrently.
    sub_queries: list[dict]

    # Populated by `fetch_one` -- each parallel branch contributes a
    # single-item list (its own sub_query + "chunks"), concatenated here by
    # the `operator.add` reducer as branches complete. `route_to_answer`
    # then fans these out to `answer_one`, again via `Send`.
    sub_results: Annotated[list[dict], operator.add]

    # Populated by `answer_one` -- same fan-out/reduce shape as sub_results,
    # one `{status, company_id, ...}` record per sub_query.
    answered: Annotated[list[dict], operator.add]

    # Populated by combine_answer_node -- "chunks" is the union across all
    # sub_results, kept for the guardrail step's groundedness check
    answer: str
    citations: list[dict]
    chunks: list[dict]

    # Populated by guardrail_node / canned_response_node
    guardrail_passed: bool
    final_answer: str


class FetchOneInput(TypedDict):
    """Input for a single `fetch_one` branch, dispatched via `Send` --
    intentionally narrower than `ChatState`: each branch only needs its own
    sub_query plus the caller's ACL, not the whole conversation state."""

    sub_query: dict
    allowed_companies: list[str]


class AnswerOneInput(TypedDict):
    """Input for a single `answer_one` branch, dispatched via `Send`."""

    sub_result: dict
