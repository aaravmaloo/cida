from __future__ import annotations

import httpx

from app.core.config import get_settings


async def verify_turnstile(token: str | None, remote_ip: str) -> bool:
    settings = get_settings()
    if not settings.turnstile_secret:
        return True
    if not token:
        return False

    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data={
                "secret": settings.turnstile_secret,
                "response": token,
                "remoteip": remote_ip,
            },
        )
    payload = response.json()
    return bool(payload.get("success"))
