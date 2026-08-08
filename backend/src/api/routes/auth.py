"""Login endpoints: dummy email-based auth per the assignment's UI requirement."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ...config.logger import get_logger
from ...config.users import get_user_companies
from ..schemas import LoginRequest, LoginResponse, UserInfo
from ..security import create_access_token, get_current_user

logger = get_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    companies = get_user_companies(payload.email)
    if companies is None:
        logger.warning("Login rejected for unknown email: %s", payload.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unknown user email",
        )

    token = create_access_token(payload.email, companies)
    logger.info("Login succeeded for %s (companies=%s)", payload.email, companies)
    return LoginResponse(access_token=token, email=payload.email, companies=companies)


@router.get("/me", response_model=UserInfo)
def me(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
    return current_user
