"""Настройки окружения и общие правила проверки конфигураций пайплайнов"""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationInfo,
    field_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseModel):
    """Запрещает неизвестные параметры и задаёт версию данных для логов"""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    schema_version: Literal[1]
    data_version: str | None = Field(default=None, min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        """Не принимает логическое значение вместо целочисленной версии"""
        if type(value) is not int:
            raise ValueError("Версия схемы должна быть целым числом")

        return value

    @field_validator("*", mode="before")
    @classmethod
    def validate_path(cls, value: object, info: ValidationInfo) -> object:
        """Отклоняет пустую строку вместо пути входного или выходного файла"""
        name = info.field_name or ""
        if (
            name.endswith(("_path", "_dir"))
            and isinstance(value, str)
            and not value.strip()
        ):
            raise ValueError("Путь не должен быть пустым")

        return value


class Settings(BaseSettings):
    """Читает адреса сервисов, учётные данные и ограничения пула из окружения"""

    model_config = SettingsConfigDict(
        env_prefix="NLP_RANKER_",
        env_file=".env",
        extra="ignore",
        hide_input_in_errors=True,
    )

    environment: Literal["local", "test", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["console", "json"] = "console"

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = Field(default="nlp-ranker", min_length=1)

    postgres_dsn: str = Field(
        default="postgresql+asyncpg://nlp_ranker:local-dev-only@localhost:5432/nlp_ranker",
        repr=False,
    )
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=5, ge=0)
    db_pool_timeout: float = Field(default=5.0, gt=0)

    redis_url: str = Field(default="redis://localhost:6379/0", repr=False)
    redis_max_connections: int = Field(default=10, ge=1)
    cache_ttl_seconds: int = Field(default=300, ge=1)
    cache_enabled: bool = True
    cache_timeout_seconds: float = Field(default=0.1, gt=0)
    cache_max_bytes: int = Field(default=65536, ge=1024)

    search_enabled: bool = True
    search_config_path: Path = Path("configs/search.yaml")
    search_active_limit: int = Field(default=1, ge=1, le=8)
    search_waiting_limit: int = Field(default=8, ge=0, le=64)
    search_queue_timeout_seconds: float = Field(default=1.0, gt=0)
    search_timeout_seconds: float = Field(default=10.0, gt=0)

    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket: str = Field(default="nlp-ranker", min_length=3, max_length=63)
    s3_access_key: SecretStr = Field(
        default=SecretStr("nlp_ranker"),
        validation_alias="AWS_ACCESS_KEY_ID",
        repr=False,
    )
    s3_secret_key: SecretStr = Field(
        default=SecretStr("local-dev-only"),
        validation_alias="AWS_SECRET_ACCESS_KEY",
        repr=False,
    )
    s3_concurrency: int = Field(default=2, ge=1, le=16)

    @field_validator(
        "postgres_dsn", "redis_url", "s3_endpoint_url", "mlflow_tracking_uri"
    )
    @classmethod
    def validate_service_url(cls, value: str, info: ValidationInfo) -> str:
        """Проверяет протокол, адрес и порт до создания клиентов сервисов"""
        try:
            url = urlsplit(value)
            port = url.port
        except ValueError:
            raise ValueError("Неверный адрес или порт сервиса") from None

        protocols = {
            "postgres_dsn": {"postgresql+asyncpg"},
            "redis_url": {"redis", "rediss"},
            "s3_endpoint_url": {"http", "https"},
            "mlflow_tracking_uri": {"http", "https"},
        }

        if url.scheme not in protocols[info.field_name or ""] or not url.hostname:
            raise ValueError("Неверный протокол или адрес сервиса")

        if port is not None and port < 1:
            raise ValueError("Порт должен быть положительным")

        if info.field_name == "postgres_dsn" and (
            not url.username or not url.path.strip("/")
        ):
            raise ValueError("Для PostgreSQL нужны пользователь и имя базы")

        return value
