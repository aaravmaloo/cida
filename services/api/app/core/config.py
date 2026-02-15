from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _normalize_async_database_url(url: str) -> str:
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgres://"):
        return f"postgresql+asyncpg://{url[len('postgres://'):]}"
    if url.startswith("postgresql://"):
        return f"postgresql+asyncpg://{url[len('postgresql://'):]}"
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="CIDA API", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    api_prefix: str = Field(default="/v1", alias="API_PREFIX")

    database_url: str = Field(alias="DATABASE_URL")
    redis_url: str = Field(alias="REDIS_URL")

    jwt_secret: str = Field(default="dev-jwt-secret-keep-it-secret", alias="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=30, alias="JWT_EXPIRE_MINUTES")
    admin_user: str = Field(default="admin", alias="ADMIN_USER")
    admin_pass: str = Field(default="admin", alias="ADMIN_PASS")

    cors_allowed_origins: str = Field(default="http://localhost:3000", alias="CORS_ALLOWED_ORIGINS")

    model_version: str = Field(default="deberta-v3-base-v1", alias="MODEL_VERSION")
    detector_model_name: str = Field(default="microsoft/deberta-v3-base", alias="DETECTOR_MODEL_NAME")
    detector_onnx_path: str = Field(default="../trainer/artifacts/latest/model.onnx", alias="DETECTOR_ONNX_PATH")
    calibration_path: str = Field(default="../trainer/artifacts/latest/calibration.json", alias="CALIBRATION_PATH")

    cache_ttl_seconds: int = Field(default=600, alias="CACHE_TTL_SECONDS")
    max_upload_bytes: int = Field(default=3_145_728, alias="MAX_UPLOAD_BYTES")

    r2_endpoint: str = Field(default="", alias="R2_ENDPOINT")
    r2_bucket: str = Field(default="", alias="R2_BUCKET")
    r2_region: str = Field(default="auto", alias="R2_REGION")
    r2_access_key: str = Field(default="", alias="R2_ACCESS_KEY")
    r2_secret_key: str = Field(default="", alias="R2_SECRET_KEY")
    report_local_dir: str = Field(default="./generated_reports", alias="REPORT_LOCAL_DIR")
    public_base_url: str = Field(default="http://localhost:8000", alias="PUBLIC_BASE_URL")

    sentry_dsn: str = Field(default="", alias="SENTRY_DSN")
    turnstile_secret: str = Field(default="", alias="TURNSTILE_SECRET")

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: object) -> object:
        if isinstance(value, str):
            return _normalize_async_database_url(value)
        return value

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


@lru_cache

def get_settings() -> Settings:
    return Settings()
