from functools import lru_cache
import json
from urllib.parse import urlsplit

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

    cors_allowed_origins: str = Field(
        default="http://localhost:3000,https://cida-web.vercel.app",
        alias="CORS_ALLOWED_ORIGINS",
    )
    cors_allow_origin_regex: str = Field(default="", alias="CORS_ALLOW_ORIGIN_REGEX")

    model_version: str = Field(default="hf-shahxeebhassan-bert-base-ai-content-detector", alias="MODEL_VERSION")
    detector_model_name: str = Field(
        default="shahxeebhassan/bert_base_ai_content_detector",
        alias="DETECTOR_MODEL_NAME",
    )
    detector_allow_remote_download: bool = Field(default=True, alias="DETECTOR_ALLOW_REMOTE_DOWNLOAD")
    detector_ai_label: int = Field(default=1, alias="DETECTOR_AI_LABEL")
    detector_max_length: int = Field(default=512, alias="DETECTOR_MAX_LENGTH")
    detector_eager_load: bool = Field(default=False, alias="DETECTOR_EAGER_LOAD")
    release_detector_for_humanizer: bool = Field(default=True, alias="RELEASE_DETECTOR_FOR_HUMANIZER")

    humanizer_model_name: str = Field(default="google/flan-t5-small", alias="HUMANIZER_MODEL_NAME")
    humanizer_use_remote_api: bool = Field(default=True, alias="HUMANIZER_USE_REMOTE_API")
    humanizer_allow_remote_download: bool = Field(default=True, alias="HUMANIZER_ALLOW_REMOTE_DOWNLOAD")
    humanizer_require_model: bool = Field(default=True, alias="HUMANIZER_REQUIRE_MODEL")
    humanizer_api_timeout_seconds: float = Field(default=60.0, alias="HUMANIZER_API_TIMEOUT_SECONDS")
    humanizer_api_url: str = Field(default="", alias="HUMANIZER_API_URL")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    hf_router_base_url: str = Field(default="https://router.huggingface.co", alias="HF_ROUTER_BASE_URL")
    humanizer_max_input_tokens: int = Field(default=512, alias="HUMANIZER_MAX_INPUT_TOKENS")
    humanizer_max_new_tokens: int = Field(default=256, alias="HUMANIZER_MAX_NEW_TOKENS")

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

    @staticmethod
    def _normalize_origin(origin: str) -> str:
        candidate = origin.strip().strip("'\"")
        if not candidate:
            return ""

        if "://" not in candidate:
            candidate = f"https://{candidate}"

        parsed = urlsplit(candidate)
        if not parsed.scheme or not parsed.netloc:
            return ""

        # CORS matching is exact on scheme+host+port; paths must be removed.
        normalized = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return normalized

    @property
    def cors_origins(self) -> list[str]:
        raw = self.cors_allowed_origins.strip()
        if not raw:
            return []

        values: list[str]
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    values = [str(item) for item in parsed]
                else:
                    values = [raw]
            except json.JSONDecodeError:
                values = [raw]
        else:
            values = raw.split(",")

        normalized = [self._normalize_origin(value) for value in values]
        return [origin for origin in normalized if origin]

    @property
    def cors_origin_regex(self) -> str | None:
        value = self.cors_allow_origin_regex.strip()
        return value or None


@lru_cache

def get_settings() -> Settings:
    return Settings()
