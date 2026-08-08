"""Conversational RAG endpoint: routes the user's message through the
LangGraph pipeline (intent routing -> rephrase -> ACL-scoped retrieval ->
answer generation -> grounding guardrail), persisting turns per user +
conversation.

Authorization is always re-derived via a fresh `get_user_companies` lookup
keyed on the JWT's `email` claim -- the JWT's own `companies` claim is
treated as identity context only, never trusted for access control, so a
still-valid token issued before a permission change can't leak access to a
company that's since been revoked.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ...config.logger import get_logger
from ...config.settings import settings
from ...config.users import get_user_companies
from ...graph.graph import run_chat_graph
from ...store import history_store
from ..schemas import MessageRequest, MessageResponse, Citation, UserInfo
from ..security import get_current_user

logger = get_logger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["chat"])


@router.post("/{conv_id}/messages", response_model=MessageResponse)
def send_message(
    conv_id: str,
    payload: MessageRequest,
    current_user: UserInfo = Depends(get_current_user),
) -> MessageResponse:
    allowed_companies = get_user_companies(current_user.email)
    if allowed_companies is None:
        logger.warning("Chat rejected: %s no longer a recognized user", current_user.email)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown user")

    if history_store.get_full_thread(current_user.email, conv_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    history = history_store.load_recent_turns(
        current_user.email, conv_id, limit=settings.history_max_turns
    )

    logger.info(
        "Chat from %s (companies=%s, conv_id=%s): %r",
        current_user.email,
        allowed_companies,
        conv_id,
        payload.message,
    )

    result = run_chat_graph(
        email=current_user.email,
        allowed_companies=allowed_companies,
        conv_id=conv_id,
        question=payload.message,
        history=history,
    )

    final_answer = result.get("final_answer", "")
    citations = [
        Citation(
            chunk_id=c["chunk_id"],
            company_id=c.get("company_id", ""),
            doc_id=c.get("doc_id", ""),
            fiscal_quarter=c.get("fiscal_quarter"),
            fiscal_year=c.get("fiscal_year"),
            speaker_name=c.get("speaker_name"),
            score=c.get("score", 0.0),
        )
        for c in result.get("citations", [])
    ]

    history_store.append_turn(current_user.email, conv_id, "user", payload.message)
    history_store.append_turn(
        current_user.email,
        conv_id,
        "assistant",
        final_answer,
        citations=[c.chunk_id for c in citations],
    )

    return MessageResponse(
        conv_id=conv_id,
        answer=final_answer,
        route=result.get("route", "continue"),
        citations=citations,
    )
