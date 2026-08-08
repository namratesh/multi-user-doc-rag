"""Company-scoped chunk retrieval: embed the query, then search MongoDB
restricted to the caller's authorized companies.

`allowed_companies` must come from `UserInfo.companies` (decoded from the
caller's JWT in `api/security.py`), never from client-supplied input --
otherwise a user could request another company's `company_id` directly.
"""

from __future__ import annotations

from src.config.logger import get_logger
from src.ingest.embed_and_store import Embedder
from src.store import mongo_store

logger = get_logger(__name__)


def retrieve(
    query: str,
    allowed_companies: list[str],
    top_k: int = 5,
    embedder: Embedder | None = None,
) -> list[dict]:
    logger.info(
        "[retrieve] input query=%r allowed_companies=%s top_k=%d",
        query,
        allowed_companies,
        top_k,
    )
    if not allowed_companies:
        logger.info("[retrieve] no allowed companies; output chunks=0 (deny-by-default)")
        return []

    embedder = embedder or Embedder()
    query_vector = embedder.embed_query(query)
    logger.info("[retrieve] step: embedded query into vector of dim=%d", len(query_vector))

    collection = mongo_store.get_collection()
    results = mongo_store.vector_search(
        collection,
        query_vector=query_vector,
        allowed_companies=allowed_companies,
        top_k=top_k,
    )
    logger.info("[retrieve] output chunks=%d", len(results))
    return results
