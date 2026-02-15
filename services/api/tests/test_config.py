import pytest

from app.core.config import Settings


@pytest.mark.parametrize(
    "database_url",
    [
        "postgres://postgres:postgres@localhost:5432/cida",
        "postgresql://postgres:postgres@localhost:5432/cida",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/cida",
    ],
)
def test_settings_normalizes_database_url(monkeypatch, database_url):
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    settings = Settings()

    assert settings.database_url.startswith("postgresql+asyncpg://")


@pytest.mark.parametrize(
    ("raw_origins", "expected"),
    [
        ("https://cida-web.vercel.app/", ["https://cida-web.vercel.app"]),
        ("cida-web.vercel.app", ["https://cida-web.vercel.app"]),
        (
            "https://cida-web.vercel.app, http://localhost:3000/",
            ["https://cida-web.vercel.app", "http://localhost:3000"],
        ),
        (
            '["https://cida-web.vercel.app/","http://localhost:3000"]',
            ["https://cida-web.vercel.app", "http://localhost:3000"],
        ),
    ],
)
def test_settings_normalizes_cors_origins(monkeypatch, raw_origins, expected):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/cida")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", raw_origins)

    settings = Settings()

    assert settings.cors_origins == expected


def test_settings_reads_cors_origin_regex(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/cida")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("CORS_ALLOW_ORIGIN_REGEX", r"^https://cida-web\.vercel\.app$")

    settings = Settings()

    assert settings.cors_origin_regex == r"^https://cida-web\.vercel\.app$"
