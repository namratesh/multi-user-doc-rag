"""Shared state threaded through the LangGraph conversational RAG pipeline."""

from __future__ import annotations

from typing import Literal, TypedDict


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

    # Populated by fetch_node
    chunks: list[dict]

    # Populated by build_answer_node
    answer: str
    citations: list[dict]

    # Populated by guardrail_node / canned_response_node
    guardrail_passed: bool
    final_answer: str
