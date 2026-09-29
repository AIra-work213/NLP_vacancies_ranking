"""HTTP-контракт поиска и открытия вакансий"""

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from nlp_ranker.domain.search import Experience, SearchHit, WorkFormat

FilterText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
]


class SearchFiltersRequest(BaseModel):
    """Не принимает неизвестные фильтры и пустые значения"""

    model_config = ConfigDict(extra="forbid")
    city: FilterText | None = None
    work_format: WorkFormat | None = None
    experience: Experience | None = None

    @field_validator("experience", mode="before")
    @classmethod
    def normalize_experience(cls, value: object) -> object:
        """Принимает прежнее имя категории и приводит его к значению корпуса"""
        return "over_6_years" if value == "more_than_6_years" else value


class FilterOptions(BaseModel):
    """Передаёт поддерживаемые значения фильтров из текущих документов"""

    city: list[str]
    work_format: list[WorkFormat]
    experience: list[Experience]


class SearchRequest(BaseModel):
    """Ограничивает запрос и число результатов до входа в очередь"""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "query": "Java-разработчик backend со Spring",
                    "limit": 10,
                    "filters": {"work_format": "remote"},
                }
            ]
        },
    )
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    limit: int = Field(default=10, ge=1, le=100, strict=True)
    filters: SearchFiltersRequest = Field(default_factory=SearchFiltersRequest)


class DocumentResponse(BaseModel):
    """Возвращает исходный документ и метаданные из PostgreSQL"""

    model_config = ConfigDict(from_attributes=True)
    document_id: UUID
    title: str
    text: str
    employer: str | None
    city: str | None
    work_format: str | None
    experience: str | None
    region: str | None
    schedule: str | None
    version: int


class SearchHitResponse(DocumentResponse):
    """Добавляет ранг и score, который не является вероятностью"""

    rank: int = Field(ge=1)
    score: float = Field(
        allow_inf_nan=False, description="Оценка ранжирования, не вероятность"
    )


class SearchResponse(BaseModel):
    """Сохраняет порядок результатов и версию поискового выпуска"""

    release_version: str
    hits: list[SearchHitResponse]

    @classmethod
    def from_hits(cls, release_version: str, hits: list[SearchHit]) -> "SearchResponse":
        """Представляет внутренние результаты в публичной схеме ответа"""
        return cls(
            release_version=release_version,
            hits=[
                SearchHitResponse(
                    **DocumentResponse.model_validate(hit.document).model_dump(),
                    rank=rank,
                    score=hit.score,
                )
                for rank, hit in enumerate(hits, start=1)
            ],
        )
