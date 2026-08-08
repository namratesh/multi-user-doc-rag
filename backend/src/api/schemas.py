"""Pydantic request/response models for the API layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):
    email: EmailStr


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str
    companies: list[str]


class UserInfo(BaseModel):
    email: str
    companies: list[str]


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrievedChunk(BaseModel):
    chunk_id: str
    company_id: str
    doc_id: str
    section_type: str | None = None
    speaker_name: str | None = None
    fiscal_quarter: str | None = None
    fiscal_year: str | None = None
    text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    results: list[RetrievedChunk]


class MessageRequest(BaseModel):
    message: str


class Citation(BaseModel):
    chunk_id: str
    company_id: str
    doc_id: str
    fiscal_quarter: str | None = None
    fiscal_year: str | None = None
    speaker_name: str | None = None
    score: float = 0.0
    text: str = ""
    cited: bool = False


class MessageResponse(BaseModel):
    conv_id: str
    answer: str
    route: Literal["greet", "deny", "continue"]
    citations: list[Citation] = []


class ConversationSummary(BaseModel):
    conv_id: str
    title: str
    updated_at: datetime


class CreateConversationResponse(BaseModel):
    conv_id: str


class ThreadMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation | str] = []


class ConversationThreadResponse(BaseModel):
    conv_id: str
    messages: list[ThreadMessage]
