"""Conversation history storage: MongoDB, keyed by `user_email` + `conv_id`.

Every document's `_id` is `f"{user_email}::{conv_id}"`, so two different
users can never collide on the same conversation id -- even a guessed or
reused `conv_id` from another session addresses a different document,
without needing a query-time ownership check to enforce it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

from pymongo.collection import Collection

from src.config.settings import settings
from src.store.mongo_store import get_client


def _doc_id(email: str, conv_id: str) -> str:
    return f"{email}::{conv_id}"


@lru_cache(maxsize=1)
def get_history_collection() -> Collection:
    client = get_client()
    collection = client[settings.mongodb_db_name][settings.mongodb_history_collection]
    collection.create_index("user_email")
    return collection


def load_recent_turns(email: str, conv_id: str, limit: int) -> list[dict]:
    collection = get_history_collection()
    doc = collection.find_one(
        {"_id": _doc_id(email, conv_id)},
        {"turns": {"$slice": -limit}},
    )
    if not doc:
        return []
    return [{"role": t["role"], "content": t["content"]} for t in doc.get("turns", [])]


def create_conversation(email: str, conv_id: str) -> None:
    collection = get_history_collection()
    now = datetime.now(timezone.utc)
    collection.update_one(
        {"_id": _doc_id(email, conv_id)},
        {
            "$setOnInsert": {
                "user_email": email,
                "conv_id": conv_id,
                "turns": [],
                "created_at": now,
                "updated_at": now,
            }
        },
        upsert=True,
    )


def list_conversations(email: str) -> list[dict]:
    collection = get_history_collection()
    cursor = collection.find(
        {"user_email": email},
        {"conv_id": 1, "turns": {"$slice": 1}, "created_at": 1, "updated_at": 1},
    ).sort("updated_at", -1)

    conversations = []
    for doc in cursor:
        turns = doc.get("turns", [])
        first_content = turns[0]["content"] if turns else ""
        title = first_content[:60] + "…" if len(first_content) > 60 else first_content
        conversations.append(
            {
                "conv_id": doc["conv_id"],
                "title": title or "New conversation",
                "updated_at": doc.get("updated_at", doc.get("created_at", datetime.now(timezone.utc))),
            }
        )
    return conversations


def get_full_thread(email: str, conv_id: str) -> list[dict] | None:
    """Returns the complete turn list for `conv_id`, or None if it doesn't
    exist or doesn't belong to `email`. The explicit `user_email` check
    guards against IDOR even though `_doc_id` already namespaces the lookup
    by email -- this is the one line that actually enforces isolation if
    that scheme ever changes."""
    collection = get_history_collection()
    doc = collection.find_one({"_id": _doc_id(email, conv_id)})
    if not doc or doc.get("user_email") != email:
        return None
    return [
        {"role": t["role"], "content": t["content"], "citations": t.get("citations", [])}
        for t in doc.get("turns", [])
    ]


def append_turn(
    email: str,
    conv_id: str,
    role: str,
    content: str,
    citations: list[str] | None = None,
) -> None:
    collection = get_history_collection()
    now = datetime.now(timezone.utc)
    turn: dict = {"role": role, "content": content, "created_at": now}
    if citations:
        turn["citations"] = citations

    collection.update_one(
        {"_id": _doc_id(email, conv_id)},
        {
            "$setOnInsert": {
                "user_email": email,
                "conv_id": conv_id,
                "created_at": now,
            },
            "$push": {"turns": turn},
            "$set": {"updated_at": now},
        },
        upsert=True,
    )
