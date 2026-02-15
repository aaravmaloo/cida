from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_admin
from app.core.config import get_settings
from app.core.security import create_admin_token, verify_passkey
from app.db.session import get_db
from app.models.entities import AdminAuditLog
from app.schemas.admin import AdminLoginRequest, AdminLoginResponse
from app.utils.http import client_ip

router = APIRouter()


@router.post("/admin/login", response_model=AdminLoginResponse)
async def admin_login(body: AdminLoginRequest, request: Request, db: AsyncSession = Depends(get_db)):
    settings = get_settings()
    if body.username != settings.admin_user or body.passkey != settings.admin_pass:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials")

    token, expires = create_admin_token()

    db.add(
        AdminAuditLog(
            action="admin_login",
            ip_address=client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
        )
    )
    await db.commit()

    return AdminLoginResponse(access_token=token, expires_at=expires)


@router.get("/admin/ping")
async def admin_ping(_=Depends(require_admin)):
    return {"status": "ok"}

