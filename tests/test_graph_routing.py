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


def _fake_chat_completion(
    route: str, guardrail_grounded: bool, guardrail_safe: bool, answer_text: str = _ANSWER_TEXT
):
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
        return answer_text

    return _fake


def _run(
    monkeypatch,
    *,
    route: str,
    guardrail_grounded: bool = True,
    guardrail_safe: bool = True,
    answer_text: str = _ANSWER_TEXT,
):
    monkeypatch.setattr(
        nodes,
        "chat_completion",
        _fake_chat_completion(route, guardrail_grounded, guardrail_safe, answer_text),
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
    assert result["final_answer"] == "Revenue grew 10%."
    assert _SAMPLE_CHUNK["chunk_id"] not in result["final_answer"]
    assert result["citations"] == [{**_SAMPLE_CHUNK, "cited": True}]


def test_guardrail_blocks_ungrounded_answer(monkeypatch) -> None:
    result = _run(monkeypatch, route="continue", guardrail_grounded=False)
    assert result["guardrail_passed"] is False
    assert result["citations"] == []
    assert result["final_answer"] != _ANSWER_TEXT


def test_model_denial_suppresses_citations_despite_retrieved_chunks(monkeypatch) -> None:
    """Retrieval can return chunks that are similar but don't actually
    answer the question -- when the model follows the prompt and replies
    with the exact "not found" sentence, none of those chunks should be
    shown as sources."""
    result = _run(monkeypatch, route="continue", answer_text=nodes.NO_CONTEXT_MESSAGE)
    assert result["final_answer"] == nodes.NO_CONTEXT_MESSAGE
    assert result["citations"] == []


def test_multi_company_question_answers_only_authorized_companies(monkeypatch) -> None:
    """A question naming two companies where the caller is authorized for
    only one should be decomposed, answered for the authorized company, and
    never name the other -- not withhold the whole answer just because part
    of it can't be answered. It should still surface a generic
    PARTIAL_COVERAGE_NOTE so the user knows the question wasn't silently
    dropped, without confirming/denying which company was skipped or why."""
    decompose_json = json.dumps(
        {
            "sub_queries": [
                {"company_id": "TCS", "question": "What was TCS revenue growth?"},
                {"company_id": "Hdfc", "question": "What was Hdfc revenue growth?"},
            ]
        }
    )

    def _fake(messages, *, temperature=0.0, model=None):
        system = messages[0]["content"]
        if '"route"' in system:
            return json.dumps({"route": "continue"})
        if "standalone question" in system:
            return "What was TCS and Hdfc revenue growth?"
        if "sub_queries" in system:
            return decompose_json
        if "grounded" in system:
            return json.dumps({"grounded": True, "safe": True, "reason": ""})
        return _ANSWER_TEXT  # the only company that ever reaches this call is TCS

    def _fake_retrieve(query, companies, top_k, embedder):
        assert companies != ["Hdfc"], "unauthorized company must never reach retrieval"
        return [_SAMPLE_CHUNK] if companies == ["TCS"] else []

    monkeypatch.setattr(nodes, "chat_completion", _fake)
    monkeypatch.setattr(nodes, "retrieve", _fake_retrieve)
    monkeypatch.setattr(nodes, "_get_embedder", lambda: None)

    graph = get_chat_graph()
    result = graph.invoke(
        {
            "email": "alice@example.com",
            "allowed_companies": ["TCS"],  # not authorized for Hdfc
            "conv_id": "conv-1",
            "question": "What was TCS and HDFC revenue growth?",
            "history": [],
        }
    )

    assert result["route"] == "continue"
    assert "Revenue grew 10%" in result["final_answer"]
    assert "hdfc" not in result["final_answer"].lower()
    assert nodes.PARTIAL_COVERAGE_NOTE in result["final_answer"]
    assert result["citations"] == [{**_SAMPLE_CHUNK, "cited": True}]
