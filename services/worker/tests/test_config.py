import pytest

from app.config import WorkerSettings


@pytest.mark.parametrize(
    "database_url",
    [
        "postgres://postgres:postgres@localhost:5432/cida",
        "postgresql://postgres:postgres@localhost:5432/cida",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/cida",
    ],
)
def test_worker_settings_normalizes_database_url(monkeypatch, database_url):
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    settings = WorkerSettings()
    assert settings.database_url.startswith("postgresql+asyncpg://")
    assert settings.redis_url.startswith("redis")

