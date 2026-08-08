"""MongoDB Atlas vector store: connection, vector index management, and
company-scoped `$vectorSearch` retrieval.

Access control is enforced *inside* the vector search itself: the Atlas Search
index declares `company_id` as a `filter` field, and every query passes
`filter={"company_id": {"$in": allowed_companies}}` as part of the
`$vectorSearch` stage. Unauthorized chunks are excluded from the ANN search
itself -- never fetched, never ranked, never returned -- not filtered out of
results after the fact. Callers must source `allowed_companies` from the
caller's JWT (`UserInfo.companies`, see `api/security.py`), never from
user-supplied input.
"""

from __future__ import annotations

import time
from functools import lru_cache

from pymongo import MongoClient, ReplaceOne
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from src.config.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)

_EMBEDDING_FIELD = "embedding"


@lru_cache(maxsize=1)
def get_client() -> MongoClient:
    if not settings.mongodb_uri:
        raise ValueError(
            "MONGODB_URI is not set. Add your Atlas connection string to .env."
        )
    return MongoClient(settings.mongodb_uri)


def get_collection() -> Collection:
    client = get_client()
    return client[settings.mongodb_db_name][settings.mongodb_collection]


def ensure_vector_index(
    collection: Collection,
    dimensions: int,
    index_name: str | None = None,
    wait: bool = True,
    timeout_s: int = 120,
) -> None:
    """Create the Atlas vector search index if it doesn't already exist.

    `company_id` is declared as a `filter` field so `$vectorSearch` can
    pre-filter candidates by it -- the mechanism the access-control model
    depends on.
    """
    index_name = index_name or settings.mongodb_vector_index
    existing = {idx["name"] for idx in collection.list_search_indexes()}
    if index_name in existing:
        return

    model = SearchIndexModel(
        name=index_name,
        type="vectorSearch",
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": _EMBEDDING_FIELD,
                    "numDimensions": dimensions,
                    "similarity": "cosine",
                },
                {"type": "filter", "path": "company_id"},
            ]
        },
    )
    collection.create_search_index(model)
    logger.info("Created Atlas vector index %r (dimensions=%d)", index_name, dimensions)

    if not wait:
        return

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        indexes = list(collection.list_search_indexes(index_name))
        if indexes and indexes[0].get("queryable"):
            logger.info("Vector index %r is queryable", index_name)
            return
        time.sleep(2)
    raise TimeoutError(f"Vector index {index_name!r} did not become queryable within {timeout_s}s")


def upsert_chunks(
    collection: Collection,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """Upsert chunks keyed by `chunk_id` (as `_id`), storing metadata + embedding."""
    logger.info("[upsert_chunks] input chunks=%d", len(chunks))
    if not chunks:
        return 0

    operations = []
    for chunk, embedding in zip(chunks, embeddings):
        doc = {k: v for k, v in chunk.items() if k != "chunk_id"}
        doc[_EMBEDDING_FIELD] = embedding
        operations.append(ReplaceOne({"_id": chunk["chunk_id"]}, doc, upsert=True))

    result = collection.bulk_write(operations, ordered=False)
    written = result.upserted_count + result.modified_count
    logger.info(
        "[upsert_chunks] output written=%d (upserted=%d modified=%d)",
        written,
        result.upserted_count,
        result.modified_count,
    )
    return written


def vector_search(
    collection: Collection,
    query_vector: list[float],
    allowed_companies: list[str],
    top_k: int = 5,
    num_candidates: int | None = None,
    index_name: str | None = None,
) -> list[dict]:
    """Company-scoped ANN search. Returns [] if `allowed_companies` is empty
    (deny-by-default) rather than searching unfiltered."""
    logger.info(
        "[vector_search] input allowed_companies=%s top_k=%d", allowed_companies, top_k
    )
    if not allowed_companies:
        logger.info("[vector_search] no allowed companies; output results=0")
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name or settings.mongodb_vector_index,
                "path": _EMBEDDING_FIELD,
                "queryVector": query_vector,
                "numCandidates": num_candidates or settings.vector_search_num_candidates,
                "limit": top_k,
                "filter": {"company_id": {"$in": allowed_companies}},
            }
        },
        {
            "$project": {
                _EMBEDDING_FIELD: 0,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    results = list(collection.aggregate(pipeline))
    for doc in results:
        doc["chunk_id"] = doc.pop("_id")
    logger.info("[vector_search] output results=%d", len(results))
    return results
