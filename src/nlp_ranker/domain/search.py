"""Фильтры и результаты поиска без зависимости от транспорта"""

from dataclasses import dataclass
from typing import Literal

WorkFormat = Literal["remote", "office", "hybrid"]
Experience = Literal["no_experience", "1_to_3_years", "3_to_6_years", "over_6_years"]


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """Описывает фильтры до отбора кандидатов без зависимости от HTTP"""

    city: str | None = None
    work_format: str | None = None
    experience: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentDetails:
    """Представляет документ вне сессии ORM"""

    document_id: str
    title: str
    text: str
    employer: str | None
    city: str | None
    work_format: str | None
    experience: str | None
    region: str | None
    schedule: str | None
    version: int


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Описывает неизменяемый документ с оценкой релевантности"""

    document_id: str
    score: float
    text: str
    document: DocumentDetails | None = None


class SearchUnavailableError(RuntimeError):
    """Сообщает, что поисковые ресурсы не настроены или недоступны"""
