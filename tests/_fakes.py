"""In-memory stand-in for the pymongo `Collection` surface that
`src/store/history_store.py` actually uses, so history-store and route tests
can run offline (no real MongoDB) the same way `test_graph_routing.py` stubs
the LLM/retrieval boundary.

Only the operations `history_store.py` calls are implemented: `find_one`,
`find(...).sort(...)`, `update_one` (with `$setOnInsert`/`$set`/`$push`), and
`create_index`. It is not a general Mongo emulator.
"""

from __future__ import annotations

from typing import Any


def _apply_projection(doc: dict, projection: dict | None) -> dict:
    if projection is None:
        return dict(doc)
    result: dict[str, Any] = {}
    for key, spec in projection.items():
        if key not in doc:
            continue
        value = doc[key]
        if isinstance(spec, dict) and "$slice" in spec:
            n = spec["$slice"]
            value = value[n:] if n < 0 else value[:n]
        result[key] = value
    return result


class FakeCursor(list):
    def sort(self, field: str, direction: int = 1) -> "FakeCursor":
        super().sort(key=lambda d: d.get(field), reverse=(direction == -1))
        return self


class FakeHistoryCollection:
    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}

    def create_index(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def find_one(self, filter: dict, projection: dict | None = None) -> dict | None:
        doc = self._docs.get(filter["_id"])
        if doc is None:
            return None
        return _apply_projection(doc, projection)

    def find(self, filter: dict, projection: dict | None = None) -> FakeCursor:
        matched = [
            doc for doc in self._docs.values() if all(doc.get(k) == v for k, v in filter.items())
        ]
        return FakeCursor(_apply_projection(doc, projection) for doc in matched)

    def update_one(self, filter: dict, update: dict, upsert: bool = False) -> None:
        doc_id = filter["_id"]
        doc = self._docs.get(doc_id)
        if doc is None:
            if not upsert:
                return
            doc = {"_id": doc_id}
            self._docs[doc_id] = doc
            for key, value in update.get("$setOnInsert", {}).items():
                doc[key] = value

        for key, value in update.get("$set", {}).items():
            doc[key] = value
        for key, value in update.get("$push", {}).items():
            doc.setdefault(key, []).append(value)

    # Test-only helper, not part of the pymongo surface.
    def seed(self, doc_id: str, **fields: Any) -> None:
        self._docs[doc_id] = {"_id": doc_id, **fields}
