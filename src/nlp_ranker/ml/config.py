"""Конфигурация энкодера, BM25 и объединения поиска"""

import re
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nlp_ranker.core.config import PipelineConfig


class RetrievalConfig(BaseModel):
    """Задаёт число кандидатов и ещё не выбранное объединение поиска"""

    model_config = ConfigDict(extra="forbid", strict=True)

    candidate_limit: int = Field(ge=1)
    fusion: Literal["rrf", "weighted_sum"] | None
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)


class DenseConfig(BaseModel):
    """Описывает модель эмбеддингов без автоматического выбора или загрузки"""

    model_config = ConfigDict(extra="forbid", strict=True)

    model_name: str | None = Field(min_length=1)
    revision: str | None = Field(min_length=1)
    device: Literal["cpu", "cuda", "mps"]
    dimensions: int = Field(default=384, ge=384, le=384)
    batch_size: int = Field(default=8, ge=1, le=128)
    threads: int = Field(default=4, ge=1, le=32)
    max_length: int = Field(default=512, ge=32, le=512)
    chunk_tokens: int = Field(default=384, ge=16, le=448)
    overlap_tokens: int = Field(default=64, ge=0)
    artifact_path: Path = Field(
        default=Path("artifacts/indexes/embeddings.parquet"), strict=False
    )

    @model_validator(mode="after")
    def validate_chunking(self) -> Self:
        """Проверяет запас для заголовка и шаг перекрывающихся фрагментов"""
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("Перекрытие должно быть меньше размера фрагмента")
        if self.chunk_tokens + 64 > self.max_length:
            raise ValueError("Оставьте 64 токена для заголовка, префикса и границ")
        if self.model_name and (
            not self.revision or not re.fullmatch(r"[0-9a-f]{40}", self.revision)
        ):
            raise ValueError("Для модели нужна фиксированная revision из 40 символов")
        return self


class RerankingConfig(BaseModel):
    """Определяет использование ранжирования и количество его результатов"""

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool
    top_k: int = Field(ge=1)
    model_name: str | None = None
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    batch_size: int = Field(default=8, ge=1, le=128)
    max_length: int = Field(default=512, ge=32, le=512)
    threads: int = Field(default=4, ge=1, le=32)


class BM25Config(BaseModel):
    """Задаёт файлы и параметры первого лексического индекса"""

    model_config = ConfigDict(extra="forbid", strict=True)

    corpus_path: Path = Field(
        default=Path("data/processed/documents.parquet"), strict=False
    )
    index_path: Path = Field(default=Path("artifacts/indexes/bm25.zip"), strict=False)
    title_repeat: int = Field(default=3, ge=1, le=10)
    k1: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    b: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)


class SearchConfig(PipelineConfig):
    """Проверяет параметры поиска и согласованность числа кандидатов"""

    bm25: BM25Config = Field(default_factory=BM25Config)
    retrieval: RetrievalConfig
    dense: DenseConfig
    reranking: RerankingConfig

    @model_validator(mode="after")
    def validate_candidate_limit(self) -> Self:
        """Не разрешает запрашивать больше результатов, чем отобрано кандидатов"""
        if self.reranking.top_k > self.retrieval.candidate_limit:
            raise ValueError("top_k не должен превышать candidate_limit")

        return self
