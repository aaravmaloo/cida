from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException, Request, status
from starlette.requests import ClientDisconnect


async def read_json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except ClientDisconnect as exc:
        raise HTTPException(status_code=499, detail="Client disconnected") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="JSON body must be an object")

    return payload
