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

import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ...config.logger import get_logger
from ...config.settings import settings
from ...config.users import get_user_companies
from ...graph.graph import run_chat_graph, stream_chat_graph_answer
from ...store import history_store
from ..schemas import MessageRequest, MessageResponse, Citation, UserInfo
from ..security import get_current_user

logger = get_logger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["chat"])


def _to_citations(chunks: list[dict]) -> list[Citation]:
    return [
        Citation(
            chunk_id=c["chunk_id"],
            company_id=c.get("company_id", ""),
            doc_id=c.get("doc_id", ""),
            fiscal_quarter=c.get("fiscal_quarter"),
            fiscal_year=c.get("fiscal_year"),
            speaker_name=c.get("speaker_name"),
            score=c.get("score", 0.0),
        )
        for c in chunks
    ]


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
        "[send_message] input user=%s companies=%s conv_id=%s history_turns=%d message=%r",
        current_user.email,
        allowed_companies,
        conv_id,
        len(history),
        payload.message,
    )

    logger.info("[send_message] step: running chat graph")
    result = run_chat_graph(
        email=current_user.email,
        allowed_companies=allowed_companies,
        conv_id=conv_id,
        question=payload.message,
        history=history,
    )

    final_answer = result.get("final_answer", "")
    citations = _to_citations(result.get("citations", []))

    logger.info("[send_message] step: persisting turns")
    history_store.append_turn(current_user.email, conv_id, "user", payload.message)
    history_store.append_turn(
        current_user.email,
        conv_id,
        "assistant",
        final_answer,
        citations=[c.chunk_id for c in citations],
    )

    logger.info(
        "[send_message] output route=%s answer_len=%d citations=%d",
        result.get("route", "continue"),
        len(final_answer),
        len(citations),
    )
    return MessageResponse(
        conv_id=conv_id,
        answer=final_answer,
        route=result.get("route", "continue"),
        citations=citations,
    )


@router.post("/{conv_id}/messages/stream")
def send_message_stream(
    conv_id: str,
    payload: MessageRequest,
    current_user: UserInfo = Depends(get_current_user),
) -> StreamingResponse:
    """Server-Sent Events variant of `send_message`: streams the answer as
    `delta` events while it's generated, then a `done` event with the
    authoritative final answer/citations (see `stream_chat_graph_answer` for
    why `done` can override streamed text -- the guardrail runs after).
    """
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
        "[send_message_stream] input user=%s companies=%s conv_id=%s history_turns=%d message=%r",
        current_user.email,
        allowed_companies,
        conv_id,
        len(history),
        payload.message,
    )

    def event_stream():
        final_answer = ""
        citations: list[Citation] = []
        for event in stream_chat_graph_answer(
            email=current_user.email,
            allowed_companies=allowed_companies,
            conv_id=conv_id,
            question=payload.message,
            history=history,
        ):
            if event["type"] == "done":
                final_answer = event["answer"]
                citations = _to_citations(event["citations"])
                event = {**event, "citations": [c.model_dump() for c in citations]}
            yield f"data: {json.dumps(event)}\n\n"

        history_store.append_turn(current_user.email, conv_id, "user", payload.message)
        history_store.append_turn(
            current_user.email,
            conv_id,
            "assistant",
            final_answer,
            citations=[c.chunk_id for c in citations],
        )
        logger.info(
            "[send_message_stream] output answer_len=%d citations=%d",
            len(final_answer),
            len(citations),
        )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
