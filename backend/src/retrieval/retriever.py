"""Company-scoped chunk retrieval: embed the query, then search MongoDB
restricted to the caller's authorized companies.

`allowed_companies` must come from `UserInfo.companies` (decoded from the
caller's JWT in `api/security.py`), never from client-supplied input --
otherwise a user could request another company's `company_id` directly.
"""

from __future__ import annotations

from src.ingest.embed_and_store import Embedder
from src.store import mongo_store


def retrieve(
    query: str,
    allowed_companies: list[str],
    top_k: int = 5,
    embedder: Embedder | None = None,
) -> list[dict]:
    if not allowed_companies:
        return []

    embedder = embedder or Embedder()
    query_vector = embedder.embed_query(query)
    collection = mongo_store.get_collection()
    return mongo_store.vector_search(
        collection,
        query_vector=query_vector,
        allowed_companies=allowed_companies,
        top_k=top_k,
    )
