"""Unit tests for `src/store/history_store.py`.

The Mongo collection is stubbed with `FakeHistoryCollection` (see
`tests/_fakes.py`) so these run offline, while still exercising the real
`_doc_id` keying scheme and the per-function Mongo query/update shapes.
"""

from __future__ import annotations

import pytest

from src.store import history_store
from tests._fakes import FakeHistoryCollection


@pytest.fixture(autouse=True)
def fake_collection(monkeypatch):
    collection = FakeHistoryCollection()
    monkeypatch.setattr(history_store, "get_history_collection", lambda: collection)
    return collection


def test_create_conversation_is_idempotent(fake_collection) -> None:
    history_store.create_conversation("alice@example.com", "conv-1")
    history_store.create_conversation("alice@example.com", "conv-1")

    thread = history_store.get_full_thread("alice@example.com", "conv-1")
    assert thread == []


def test_append_turn_creates_conversation_if_missing() -> None:
    history_store.append_turn("alice@example.com", "conv-1", "user", "Hello")

    thread = history_store.get_full_thread("alice@example.com", "conv-1")
    assert thread == [{"role": "user", "content": "Hello", "citations": []}]


def test_append_turn_stores_citations_when_given() -> None:
    history_store.append_turn(
        "alice@example.com", "conv-1", "assistant", "Answer", citations=["chunk-1", "chunk-2"]
    )

    thread = history_store.get_full_thread("alice@example.com", "conv-1")
    assert thread[0]["citations"] == ["chunk-1", "chunk-2"]


def test_append_turn_omits_citations_key_when_not_given() -> None:
    history_store.append_turn("alice@example.com", "conv-1", "user", "Hello")

    thread = history_store.get_full_thread("alice@example.com", "conv-1")
    assert thread[0]["citations"] == []


def test_get_full_thread_returns_none_for_unknown_conversation() -> None:
    assert history_store.get_full_thread("alice@example.com", "does-not-exist") is None


def test_get_full_thread_isolates_users_sharing_a_conv_id(fake_collection) -> None:
    # Same conv_id, two different users -- must never resolve to each other's
    # thread, since `_doc_id` namespaces the document by email.
    history_store.append_turn("alice@example.com", "conv-1", "user", "Alice's message")
    history_store.append_turn("bob@example.com", "conv-1", "user", "Bob's message")

    alice_thread = history_store.get_full_thread("alice@example.com", "conv-1")
    bob_thread = history_store.get_full_thread("bob@example.com", "conv-1")

    assert alice_thread[0]["content"] == "Alice's message"
    assert bob_thread[0]["content"] == "Bob's message"


def test_get_full_thread_guards_against_user_email_mismatch(fake_collection) -> None:
    # Simulates a document whose _id happens to match the lookup key but
    # whose stored user_email disagrees -- the explicit equality check in
    # `get_full_thread` must still refuse to return it.
    fake_collection.seed(
        "alice@example.com::conv-1", user_email="mallory@example.com", turns=[]
    )
    assert history_store.get_full_thread("alice@example.com", "conv-1") is None


def test_load_recent_turns_returns_empty_list_for_unknown_conversation() -> None:
    assert history_store.load_recent_turns("alice@example.com", "does-not-exist", limit=6) == []


def test_load_recent_turns_respects_limit_and_drops_citations() -> None:
    for i in range(4):
        history_store.append_turn("alice@example.com", "conv-1", "user", f"msg-{i}")

    turns = history_store.load_recent_turns("alice@example.com", "conv-1", limit=2)

    assert turns == [{"role": "user", "content": "msg-2"}, {"role": "user", "content": "msg-3"}]


def test_list_conversations_orders_by_updated_at_descending() -> None:
    history_store.create_conversation("alice@example.com", "conv-old")
    history_store.append_turn("alice@example.com", "conv-old", "user", "first ever")
    history_store.create_conversation("alice@example.com", "conv-new")
    history_store.append_turn("alice@example.com", "conv-new", "user", "most recent")

    conversations = history_store.list_conversations("alice@example.com")

    assert [c["conv_id"] for c in conversations] == ["conv-new", "conv-old"]


def test_list_conversations_titles_from_first_turn_and_truncates() -> None:
    history_store.create_conversation("alice@example.com", "conv-1")
    history_store.append_turn("alice@example.com", "conv-1", "user", "x" * 80)
    history_store.append_turn("alice@example.com", "conv-1", "assistant", "irrelevant second turn")

    [conversation] = history_store.list_conversations("alice@example.com")

    assert conversation["title"] == "x" * 60 + "…"


def test_list_conversations_titles_empty_thread_as_new_conversation() -> None:
    history_store.create_conversation("alice@example.com", "conv-1")

    [conversation] = history_store.list_conversations("alice@example.com")

    assert conversation["title"] == "New conversation"


def test_list_conversations_only_returns_requesting_users_conversations() -> None:
    history_store.create_conversation("alice@example.com", "conv-1")
    history_store.create_conversation("bob@example.com", "conv-2")

    conversations = history_store.list_conversations("alice@example.com")

    assert [c["conv_id"] for c in conversations] == ["conv-1"]
