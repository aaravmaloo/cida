from fastapi import APIRouter

from app.api.v1 import admin, analytics, analyze, humanize, reports

router = APIRouter()
router.include_router(analyze.router, tags=["analyze"])
router.include_router(humanize.router, tags=["humanize"])
router.include_router(reports.router, tags=["reports"])
router.include_router(analytics.router, tags=["analytics"])
router.include_router(admin.router, tags=["admin"])

