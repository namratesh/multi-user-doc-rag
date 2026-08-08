"""API tests for `src/api/routes/conversations.py`: list/create/get-thread.

The Mongo-backed history store is stubbed with `FakeHistoryCollection` and
auth is bypassed via a `get_current_user` dependency override, so these run
offline against the real FastAPI app/routing (no JWT, no MongoDB needed).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import UserInfo
from src.api.security import get_current_user
from src.store import history_store
from tests._fakes import FakeHistoryCollection


@pytest.fixture
def fake_collection(monkeypatch):
    collection = FakeHistoryCollection()
    monkeypatch.setattr(history_store, "get_history_collection", lambda: collection)
    return collection


@pytest.fixture
def client(fake_collection):
    app.dependency_overrides[get_current_user] = lambda: UserInfo(
        email="alice@example.com", companies=["TCS", "Infosys"]
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_create_conversation_returns_new_empty_conv_id(client) -> None:
    response = client.post("/api/conversations")

    assert response.status_code == 200
    conv_id = response.json()["conv_id"]
    assert conv_id

    thread = client.get(f"/api/conversations/{conv_id}")
    assert thread.status_code == 200
    assert thread.json() == {"conv_id": conv_id, "messages": []}


def test_list_conversations_returns_only_callers_conversations(client) -> None:
    history_store.create_conversation("alice@example.com", "conv-1")
    history_store.append_turn("alice@example.com", "conv-1", "user", "hi")
    history_store.create_conversation("bob@example.com", "conv-2")

    response = client.get("/api/conversations")

    assert response.status_code == 200
    conv_ids = [c["conv_id"] for c in response.json()]
    assert conv_ids == ["conv-1"]


def test_get_conversation_returns_404_for_unknown_conv_id(client) -> None:
    response = client.get("/api/conversations/does-not-exist")

    assert response.status_code == 404


def test_get_conversation_returns_404_for_another_users_conversation(client) -> None:
    history_store.create_conversation("bob@example.com", "conv-owned-by-bob")

    response = client.get("/api/conversations/conv-owned-by-bob")

    assert response.status_code == 404


def test_get_conversation_returns_full_turn_history(client) -> None:
    history_store.create_conversation("alice@example.com", "conv-1")
    history_store.append_turn("alice@example.com", "conv-1", "user", "How did revenue do?")
    history_store.append_turn(
        "alice@example.com", "conv-1", "assistant", "It grew 10%.", citations=["chunk-1"]
    )

    response = client.get("/api/conversations/conv-1")

    assert response.status_code == 200
    assert response.json()["messages"] == [
        {"role": "user", "content": "How did revenue do?", "citations": []},
        {"role": "assistant", "content": "It grew 10%.", "citations": ["chunk-1"]},
    ]


def test_conversation_routes_require_authentication() -> None:
    with TestClient(app) as unauthenticated_client:
        response = unauthenticated_client.get("/api/conversations")

    assert response.status_code == 401
