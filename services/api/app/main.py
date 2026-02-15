from __future__ import annotations

from datetime import datetime, timezone

import sentry_sdk
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import text

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.redis import close_redis
from app.db.base import Base
from app.db.session import engine
from app.schemas.common import ErrorResponse, HealthResponse
from app.utils.trace import get_trace_id, trace_context_middleware

settings = get_settings()
configure_logging()
logger = get_logger(__name__)

if settings.sentry_dsn:
    sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.environment)


app = FastAPI(title=settings.app_name, default_response_class=ORJSONResponse)
app.middleware("http")(trace_context_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline';"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return ORJSONResponse(
        status_code=422,
        content=ErrorResponse(detail=str(exc), trace_id=get_trace_id()).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("unhandled_exception", error=str(exc), trace_id=get_trace_id())
    return ORJSONResponse(
        status_code=500,
        content=ErrorResponse(detail="Internal server error", trace_id=get_trace_id()).model_dump(),
    )


@app.on_event("startup")
async def startup_event() -> None:
    if settings.environment == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    Instrumentator().instrument(app).expose(app)
    logger.info("startup_complete", environment=settings.environment)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await close_redis()
    await engine.dispose()


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/readyz")
async def readyz() -> dict:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return {"status": "ready", "time": datetime.now(timezone.utc).isoformat()}


app.include_router(v1_router, prefix=settings.api_prefix)

