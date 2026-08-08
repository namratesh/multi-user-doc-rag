"""Conversation lifecycle endpoints: list, create, and fetch a full thread.

Every route resolves the owning user from the JWT (`get_current_user`), never
from a request body or path param -- so isolation between users holds even
if a caller passes another user's conv_id.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from ...config.logger import get_logger
from ...store import history_store
from ..schemas import (
    ConversationSummary,
    ConversationThreadResponse,
    CreateConversationResponse,
    UserInfo,
)
from ..security import get_current_user

logger = get_logger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    current_user: UserInfo = Depends(get_current_user),
) -> list[ConversationSummary]:
    conversations = history_store.list_conversations(current_user.email)
    return [ConversationSummary(**c) for c in conversations]


@router.post("", response_model=CreateConversationResponse)
def create_conversation(
    current_user: UserInfo = Depends(get_current_user),
) -> CreateConversationResponse:
    conv_id = str(uuid4())
    history_store.create_conversation(current_user.email, conv_id)
    logger.info("Created conversation %s for %s", conv_id, current_user.email)
    return CreateConversationResponse(conv_id=conv_id)


@router.get("/{conv_id}", response_model=ConversationThreadResponse)
def get_conversation(
    conv_id: str,
    current_user: UserInfo = Depends(get_current_user),
) -> ConversationThreadResponse:
    thread = history_store.get_full_thread(current_user.email, conv_id)
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return ConversationThreadResponse(conv_id=conv_id, messages=thread)
