"""FastAPI application entrypoint.

Run from the repo root with:
    uvicorn backend.src.api.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config.logger import get_logger
from ..config.settings import settings
from .routes.auth import router as auth_router
from .routes.query import router as query_router

logger = get_logger(__name__)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(query_router)


@app.on_event("startup")
def on_startup() -> None:
    logger.info("%s starting up (env=%s)", settings.app_name, settings.env)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
