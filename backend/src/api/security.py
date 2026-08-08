"""JWT issuance/verification and the `get_current_user` auth dependency.

Every authenticated endpoint depends on `get_current_user`, which decodes the
bearer token into an email + authorized company_id list. Downstream retrieval
code must filter by `user.companies` so a session never sees another user's
(or another session's) documents.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config.logger import get_logger
from ..config.settings import settings
from .schemas import UserInfo

logger = get_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


def create_access_token(email: str, companies: list[str]) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {"sub": email, "companies": companies, "exp": expires_at}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> UserInfo:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired, please log in again"
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token"
        ) from exc

    return UserInfo(email=payload["sub"], companies=payload.get("companies", []))


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> UserInfo:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = decode_access_token(credentials.credentials)
    logger.debug("Authenticated request for %s", user.email)
    return user
