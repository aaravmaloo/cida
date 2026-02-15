from arq.connections import RedisSettings

from app.config import WorkerSettings
from app.jobs import generate_report

settings = WorkerSettings()


class WorkerSettingsConfig:
    functions = [generate_report]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 16
    queue_name = "arq:queue"

