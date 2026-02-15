from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextvars import ContextVar

from fastapi import Request

trace_id_ctx: ContextVar[str] = ContextVar("trace_id", default="")


def make_trace_id() -> str:
    return uuid.uuid4().hex


async def trace_context_middleware(request: Request, call_next):
    trace_id = request.headers.get("x-trace-id") or make_trace_id()
    token = trace_id_ctx.set(trace_id)
    try:
        response = await call_next(request)
        response.headers["x-trace-id"] = trace_id
        return response
    finally:
        trace_id_ctx.reset(token)


def get_trace_id() -> str:
    return trace_id_ctx.get() or make_trace_id()

