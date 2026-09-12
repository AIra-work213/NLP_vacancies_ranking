"""Проверенные данные импорта документов и конфликты записи"""

import hashlib
import json
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentInput(BaseModel):
    """Проверяет данные документа перед записью в БД"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: UUID
    source: str = Field(min_length=1, max_length=100)
    source_id: str = Field(min_length=1, max_length=255)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    employer: str | None = Field(default=None, max_length=255)
    city: str | None = Field(default=None, max_length=255)
    work_format: str | None = Field(default=None, max_length=50)
    experience: str | None = Field(default=None, max_length=100)
    region: str | None = Field(default=None, max_length=255)
    schedule: str | None = Field(default=None, max_length=100)

    def database_values(self) -> dict[str, object]:
        """Добавляет отпечаток содержимого для пропуска повторного импорта"""
        values = self.model_dump()
        content = self.model_dump(mode="json", exclude={"id"})
        serialized = json.dumps(content, ensure_ascii=False, sort_keys=True)
        values["content_hash"] = hashlib.sha256(serialized.encode()).hexdigest()
        return values


class DocumentConflictError(Exception):
    """Обозначает конфликт изменения документа с устаревшей версией"""
