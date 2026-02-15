from fastapi import Depends, Header, HTTPException, status

from app.core.security import decode_admin_token


def get_bearer_token(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    return authorization.split(" ", 1)[1]


def require_admin(token: str = Depends(get_bearer_token)) -> dict:
    return decode_admin_token(token)

