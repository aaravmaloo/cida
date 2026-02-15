from app.config import WorkerSettings


def test_worker_settings_has_urls(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/cida")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    settings = WorkerSettings()
    assert settings.database_url.startswith("postgresql")
    assert settings.redis_url.startswith("redis")

