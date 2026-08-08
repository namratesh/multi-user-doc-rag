"""API tests for `src/api/routes/chat.py` (`POST /api/conversations/{conv_id}/messages`).

The LangGraph pipeline is stubbed via `run_chat_graph` (graph wiring itself
is covered by `test_graph_routing.py`) and the history store is stubbed with
`FakeHistoryCollection`, so these focus on the route's own responsibilities:
404-ing on unknown/foreign conversations, 401-ing on revoked users, shaping
the response, and persisting both turns.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import chat as chat_route
from src.api.schemas import UserInfo
from src.api.security import get_current_user
from src.store import history_store
from tests._fakes import FakeHistoryCollection

_FAKE_CITATION = {
    "chunk_id": "TCS_Q1_2026_qa_001",
    "company_id": "TCS",
    "doc_id": "TCS_Q1_2026",
    "fiscal_quarter": "Q1",
    "fiscal_year": "2026",
    "speaker_name": "CEO",
    "score": 0.91,
    "text": "Revenue grew 10% year-over-year driven by strong deal wins.",
    "cited": False,
}

_FAKE_RESULT = {
    "final_answer": "Revenue grew 10%.",
    "route": "continue",
    "citations": [_FAKE_CITATION],
}


@pytest.fixture
def fake_collection(monkeypatch):
    collection = FakeHistoryCollection()
    monkeypatch.setattr(history_store, "get_history_collection", lambda: collection)
    return collection


@pytest.fixture
def fake_graph(monkeypatch):
    calls: list[dict] = []

    def _fake_run_chat_graph(**kwargs):
        calls.append(kwargs)
        return dict(_FAKE_RESULT)

    monkeypatch.setattr(chat_route, "run_chat_graph", _fake_run_chat_graph)
    return calls


@pytest.fixture
def client(fake_collection, fake_graph):
    app.dependency_overrides[get_current_user] = lambda: UserInfo(
        email="alice@example.com", companies=["TCS", "Infosys"]
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_send_message_returns_404_for_unknown_conversation(client) -> None:
    response = client.post("/api/conversations/does-not-exist/messages", json={"message": "hi"})

    assert response.status_code == 404


def test_send_message_returns_404_for_another_users_conversation(client) -> None:
    history_store.create_conversation("bob@example.com", "conv-owned-by-bob")

    response = client.post(
        "/api/conversations/conv-owned-by-bob/messages", json={"message": "hi"}
    )

    assert response.status_code == 404


def test_send_message_returns_answer_and_citations(client) -> None:
    history_store.create_conversation("alice@example.com", "conv-1")

    response = client.post(
        "/api/conversations/conv-1/messages", json={"message": "How did revenue do?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["conv_id"] == "conv-1"
    assert body["answer"] == "Revenue grew 10%."
    assert body["route"] == "continue"
    assert body["citations"] == [_FAKE_CITATION]


def test_send_message_persists_user_and_assistant_turns(client) -> None:
    history_store.create_conversation("alice@example.com", "conv-1")

    client.post("/api/conversations/conv-1/messages", json={"message": "How did revenue do?"})

    thread = history_store.get_full_thread("alice@example.com", "conv-1")
    assert [t["role"] for t in thread] == ["user", "assistant"]
    assert thread[0]["content"] == "How did revenue do?"
    assert thread[1]["content"] == "Revenue grew 10%."
    assert thread[1]["citations"] == [_FAKE_CITATION]


def test_send_message_passes_fresh_permissions_and_recent_history_to_graph(
    client, fake_graph
) -> None:
    history_store.create_conversation("alice@example.com", "conv-1")
    history_store.append_turn("alice@example.com", "conv-1", "user", "earlier question")

    client.post("/api/conversations/conv-1/messages", json={"message": "follow up"})

    assert len(fake_graph) == 1
    call = fake_graph[0]
    assert call["email"] == "alice@example.com"
    assert call["allowed_companies"] == ["TCS", "Infosys"]
    assert call["conv_id"] == "conv-1"
    assert call["question"] == "follow up"
    assert call["history"] == [{"role": "user", "content": "earlier question"}]


def test_send_message_returns_401_for_user_no_longer_in_access_control_map(
    fake_collection, fake_graph
) -> None:
    app.dependency_overrides[get_current_user] = lambda: UserInfo(
        email="ghost@example.com", companies=[]
    )
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/conversations/conv-1/messages", json={"message": "hi"}
        )
    app.dependency_overrides.clear()

    assert response.status_code == 401
    assert fake_graph == []


def test_send_message_requires_authentication() -> None:
    with TestClient(app) as unauthenticated_client:
        response = unauthenticated_client.post(
            "/api/conversations/conv-1/messages", json={"message": "hi"}
        )

    assert response.status_code == 401
