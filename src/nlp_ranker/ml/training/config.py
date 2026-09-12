"""Конфигурация отложенного дообучения моделей"""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nlp_ranker.core.config import PipelineConfig


class ModelConfig(BaseModel):
    """Хранит ещё не выбранную модель и её фиксированную ревизию"""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str | None = Field(min_length=1)
    revision: str | None = Field(min_length=1)


class TrainingParameters(BaseModel):
    """Проверяет известные параметры, допуская отложенный выбор настроек"""

    model_config = ConfigDict(extra="forbid", strict=True)

    device: Literal["cpu", "cuda", "mps"]
    batch_size: int | None = Field(gt=0)
    epochs: int | None = Field(gt=0)
    learning_rate: float | None = Field(gt=0, allow_inf_nan=False)


class TrainingConfig(PipelineConfig):
    """Разделяет обучающие и валидационные файлы и параметры обучения"""

    seed: int = Field(ge=0, le=4294967295)
    dataset_path: Path = Field(strict=False)
    validation_path: Path = Field(strict=False)
    output_dir: Path = Field(strict=False)

    model: ModelConfig
    training: TrainingParameters

    @model_validator(mode="after")
    def validate_dataset_paths(self) -> Self:
        """Отклоняет один и тот же файл для обучения и валидации"""
        if self.dataset_path.resolve() == self.validation_path.resolve():
            raise ValueError("Обучающий и валидационный файлы должны различаться")

        return self
