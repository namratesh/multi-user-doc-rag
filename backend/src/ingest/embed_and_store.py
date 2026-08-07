"""Embed chunked transcript text and persist it into MongoDB Atlas.

Each chunk keeps its `company_id`/`doc_id` (and other chunker.py fields) as document
fields, so retrieval can filter to a user's authorized companies as part of the
`$vectorSearch` itself (`filter={"company_id": {"$in": [...]}}`) rather than after
the fact -- see `src/store/mongo_store.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from pymongo.collection import Collection

from src.config.logger import get_logger
from src.config.settings import settings
from src.store import mongo_store

logger = get_logger(__name__)


class Embedder:
    """Embeds text via the OpenRouter embeddings API (default model: nvidia/nemotron-3-embed-1b:free)."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self.api_key = api_key or settings.openrouter_api_key
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to .env to use the OpenRouter embedding model."
            )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )
        self._url = f"{settings.openrouter_base_url.rstrip('/')}/embeddings"
        logger.info("Using OpenRouter embedding model %s", self.model_name)

    def _request(self, texts: list[str]) -> list[list[float]]:
        response = self._session.post(
            self._url,
            json={"model": self.model_name, "input": texts, "encoding_format": "float"},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        data.sort(key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings.extend(self._request(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._request([text])[0]


def embed_chunk_file(
    chunk_path: Path,
    embedder: Embedder,
    collection: Collection,
    batch_size: int,
) -> int:
    chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
    if not chunks:
        return 0

    embeddings = embedder.embed_documents([c["text"] for c in chunks], batch_size=batch_size)
    # Upsert first: Atlas Search index creation requires the collection to already
    # exist, and bulk_write(upsert=True) implicitly creates it on first write.
    mongo_store.upsert_chunks(collection, chunks, embeddings)
    mongo_store.ensure_vector_index(collection, dimensions=len(embeddings[0]))
    return len(chunks)


def embed_directory(
    chunks_dir: Path,
    batch_size: int = 32,
    embedder: Embedder | None = None,
) -> int:
    chunk_paths = sorted(chunks_dir.glob("*.json"))
    if not chunk_paths:
        logger.warning("No chunk files found in %s", chunks_dir)
        return 0

    embedder = embedder or Embedder()
    collection = mongo_store.get_collection()

    total = 0
    for chunk_path in chunk_paths:
        count = embed_chunk_file(chunk_path, embedder, collection, batch_size)
        total += count
        print(f"{chunk_path.name}: embedded {count} chunks -> {settings.mongodb_collection}")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"), help="Directory of chunker.py JSON output")
    parser.add_argument("--batch-size", type=int, default=settings.embedding_batch_size)
    args = parser.parse_args()

    total = embed_directory(args.chunks_dir, args.batch_size)
    print(f"Done: embedded and stored {total} chunks -> MongoDB[{settings.mongodb_db_name}.{settings.mongodb_collection}]")


if __name__ == "__main__":
    main()
