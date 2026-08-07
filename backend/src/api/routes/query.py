"""Retrieval endpoint: vector-searches transcript chunks restricted to the
authenticated caller's authorized companies (from the JWT, never from the
request body)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ...config.logger import get_logger
from ...retrieval.retriever import retrieve
from ..schemas import QueryRequest, QueryResponse, UserInfo
from ..security import get_current_user

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    current_user: UserInfo = Depends(get_current_user),
) -> QueryResponse:
    logger.info(
        "Query from %s (companies=%s): %r", current_user.email, current_user.companies, payload.query
    )
    results = retrieve(payload.query, current_user.companies, top_k=payload.top_k)
    return QueryResponse(query=payload.query, results=results)
