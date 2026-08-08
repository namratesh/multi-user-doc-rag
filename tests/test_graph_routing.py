"""Graph-level tests for the conversational RAG pipeline.

All LLM calls and retrieval are stubbed so these run offline (no
OPENROUTER_API_KEY / MongoDB required) while still exercising the real
LangGraph wiring in `src/graph/graph.py` and node logic in
`src/graph/nodes.py`.
"""

from __future__ import annotations

import json

from src.graph import nodes
from src.graph.graph import get_chat_graph

_SAMPLE_CHUNK = {
    "chunk_id": "TCS_Q1_2026_qa_001",
    "company_id": "TCS",
    "doc_id": "TCS_Q1_2026",
    "fiscal_quarter": "Q1",
    "fiscal_year": "2026",
    "speaker_name": "CEO",
    "text": "Revenue grew 10% year over year.",
    "score": 0.91,
}
_ANSWER_TEXT = f"Revenue grew 10% [{_SAMPLE_CHUNK['chunk_id']}]."


def _fake_chat_completion(route: str, guardrail_grounded: bool, guardrail_safe: bool):
    def _fake(messages, *, temperature=0.0, model=None):
        system = messages[0]["content"]
        if '"route"' in system:
            return json.dumps({"route": route})
        if "standalone question" in system:
            return "What was TCS revenue growth?"
        if "grounded" in system:
            return json.dumps(
                {"grounded": guardrail_grounded, "safe": guardrail_safe, "reason": "test"}
            )
        return _ANSWER_TEXT

    return _fake


def _run(monkeypatch, *, route: str, guardrail_grounded: bool = True, guardrail_safe: bool = True):
    monkeypatch.setattr(
        nodes, "chat_completion", _fake_chat_completion(route, guardrail_grounded, guardrail_safe)
    )
    monkeypatch.setattr(nodes, "retrieve", lambda *args, **kwargs: [_SAMPLE_CHUNK])
    monkeypatch.setattr(nodes, "_get_embedder", lambda: None)

    graph = get_chat_graph()
    return graph.invoke(
        {
            "email": "alice@example.com",
            "allowed_companies": ["TCS"],
            "conv_id": "conv-1",
            "question": "How did revenue do?",
            "history": [],
        }
    )


def test_greet_route_short_circuits_before_retrieval(monkeypatch) -> None:
    result = _run(monkeypatch, route="greet")
    assert result["route"] == "greet"
    assert "chunks" not in result
    assert result["final_answer"]


def test_deny_route_short_circuits_before_retrieval(monkeypatch) -> None:
    result = _run(monkeypatch, route="deny")
    assert result["route"] == "deny"
    assert "chunks" not in result
    assert result["final_answer"]


def test_continue_route_produces_grounded_answer_with_citations(monkeypatch) -> None:
    result = _run(monkeypatch, route="continue")
    assert result["route"] == "continue"
    assert result["guardrail_passed"] is True
    assert result["final_answer"] == _ANSWER_TEXT
    assert result["citations"] == [_SAMPLE_CHUNK]


def test_guardrail_blocks_ungrounded_answer(monkeypatch) -> None:
    result = _run(monkeypatch, route="continue", guardrail_grounded=False)
    assert result["guardrail_passed"] is False
    assert result["citations"] == []
    assert result["final_answer"] != _ANSWER_TEXT
