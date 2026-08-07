"""Pydantic request/response models for the API layer."""

from __future__ import annotations

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
