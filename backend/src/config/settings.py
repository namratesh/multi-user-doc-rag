"""Centralized application settings, loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root .env, resolved from this file's location so it loads regardless of cwd
# (the ingest scripts are run from backend/, not the repo root).
_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_prefix="", extra="ignore")

    app_name: str = "multi-user-doc-rag"
    env: str = "development"

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_to_file: bool = True

    # Auth / JWT
    secret_key: str = "dev-secret-key-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # Embeddings
    embedding_model_name: str = "nvidia/nemotron-3-embed-1b:free"
    embedding_batch_size: int = 32
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Vector store (MongoDB Atlas + $vectorSearch)
    mongodb_uri: str | None = Field(
        default=None, validation_alias=AliasChoices("MONGODB_URI", "MONGO_DB_STRING")
    )
    mongodb_db_name: str = "multi_user_rag"
    mongodb_collection: str = "transcript_chunks"
    mongodb_vector_index: str = "chunk_vector_index"
    vector_search_num_candidates: int = 100

    # Conversational RAG (LangGraph pipeline). chat_provider picks which
    # OpenAI-compatible backend serves chat completions ("groq" or
    # "openrouter"); the matching <provider>_api_key / <provider>_base_url
    # pair above/below is used.
    chat_provider: str = "groq"
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    chat_model_name: str = "llama-3.3-70b-versatile"
    chat_temperature: float = 0.1
    chat_top_k: int = 6
    guardrail_enabled: bool = True
    history_max_turns: int = 6
    mongodb_history_collection: str = "conversations"

    # Observability (LangSmith tracing, backend/src/config/observability.py).
    # Tracing is skipped (no-op) when the API key isn't set.
    langsmith_api_key: str | None = None
    langsmith_project: str = "multi-user-rag"
    langsmith_endpoint: str = "https://api.smith.langchain.com"


settings = Settings()
