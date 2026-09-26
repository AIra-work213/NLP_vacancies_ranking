"""Конфигурация оценки поиска и порога качества"""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nlp_ranker.core.config import PipelineConfig


class QualityGateConfig(BaseModel):
    """Проверяет совместный выбор базового отчёта и допустимого снижения NDCG"""

    model_config = ConfigDict(extra="forbid", strict=True)

    baseline_metrics_path: Path | None = Field(strict=False)
    max_ndcg_regression: float | None = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_baseline(self) -> Self:
        """Требует оба параметра проверки качества либо отключение обоих"""
        if (self.baseline_metrics_path is None) != (self.max_ndcg_regression is None):
            raise ValueError("Для quality_gate нужны базовый отчёт и порог вместе")

        return self


class EvaluationConfig(PipelineConfig):
    """Проверяет пути оценки и поддерживаемые метрики без их вычисления"""

    dataset_path: Path = Field(strict=False)
    index_path: Path = Field(strict=False)
    metrics_path: Path = Field(strict=False)

    metrics: list[Literal["ndcg_at_10", "mrr_at_10", "recall_at_100"]] = Field(
        min_length=1
    )
    quality_gate: QualityGateConfig
    qrels_path: Path = Field(default=Path("data/processed/qrels.parquet"), strict=False)
    corpus_path: Path = Field(
        default=Path("data/processed/documents.parquet"), strict=False
    )
    split: Literal["train", "validation", "test", "all"] = "validation"
    seed: int = Field(default=42, ge=0)
    latency_repeats: int = Field(default=10, ge=1, le=1000)
    method: Literal["bm25", "dense", "hybrid", "rerank"] = "bm25"
    search_config_path: Path = Field(default=Path("configs/search.yaml"), strict=False)
    rerank_candidates: int = Field(default=20, ge=10, le=100)

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        """Отклоняет повторение одной метрики в отчёте"""
        if len(self.metrics) != len(set(self.metrics)):
            raise ValueError("Метрики не должны повторяться")

        if (
            self.quality_gate.baseline_metrics_path is not None
            and "ndcg_at_10" not in self.metrics
        ):
            raise ValueError("Для quality_gate нужна метрика ndcg_at_10")
        return self
