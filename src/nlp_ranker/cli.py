"""Команды подготовки данных, индексации, обучения и оценки"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import structlog
import yaml
from pydantic import ValidationError

from nlp_ranker.core.config import PipelineConfig, Settings
from nlp_ranker.core.logging import configure_logging
from nlp_ranker.ml.config import SearchConfig
from nlp_ranker.ml.evaluation.config import EvaluationConfig
from nlp_ranker.ml.training.config import TrainingConfig
from nlp_ranker.ml.training.trainer import mine_negatives, train
from nlp_ranker.mlops.tracking import (
    log_inputs,
    log_results,
    pipeline_names,
    tracked_run,
)
from nlp_ranker.pipelines.data import DataConfig, prepare
from nlp_ranker.pipelines.embeddings import build_embeddings, load_dense
from nlp_ranker.pipelines.evaluate import evaluate
from nlp_ranker.pipelines.index import build_index

CONFIG_SCHEMAS: dict[str, type[PipelineConfig]] = {
    "prepare": DataConfig,
    "index": SearchConfig,
    "embed": SearchConfig,
    "load-dense": SearchConfig,
    "mine-negatives": TrainingConfig,
    "train": TrainingConfig,
    "evaluate": EvaluationConfig,
}


def load_config(path: Path, stage: str) -> PipelineConfig:
    """Проверяет все параметры YAML по схеме выбранного этапа"""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    return CONFIG_SCHEMAS[stage].model_validate(config)


def main() -> None:
    """Проверяет конфигурацию и логирует исход этапа без содержимого данных"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=list(CONFIG_SCHEMAS))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--experiment")
    parser.add_argument("--run-name")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"])
    args = parser.parse_args()

    try:
        settings = Settings()
    except ValidationError as exc:
        fields = [str(error["loc"][0]) for error in exc.errors(include_input=False)]
        parser.exit(
            status=2,
            message="Неверные настройки окружения: " + ", ".join(fields) + "\n",
        )

    configure_logging(settings)
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(stage=args.stage, data_version=None)

    logger = structlog.stdlib.get_logger("nlp_ranker.pipeline")
    started = perf_counter()
    logger.info("Этап запущен", dry_run=args.dry_run)

    try:
        config = load_config(args.config, args.stage)
        if args.split:
            if not isinstance(config, EvaluationConfig):
                parser.error("Параметр --split доступен только для evaluate")
            config.split = args.split
            config.metrics_path = config.metrics_path.with_name(
                f"metrics-{config.method}-{args.split}.json"
            )
        structlog.contextvars.bind_contextvars(data_version=config.data_version)

        if args.dry_run:
            logger.info("Конфигурация проверена; артефакты не записаны")
        else:
            stages = {
                "prepare": prepare,
                "index": build_index,
                "embed": build_embeddings,
                "load-dense": load_dense,
                "mine-negatives": mine_negatives,
                "train": train,
                "evaluate": evaluate,
            }
            values = config.model_dump(mode="json")
            params = {
                "stage": args.stage,
                "config": json.dumps(values, ensure_ascii=False),
                "data_version": config.data_version or "unversioned",
            }
            experiment, run_name = pipeline_names(args.stage, values)
            if args.experiment:
                settings.mlflow_experiment = args.experiment
            elif settings.mlflow_experiment == "nlp-ranker":
                settings.mlflow_experiment = experiment

            context = (
                tracked_run(settings, args.run_name or run_name, params)
                if args.track
                else nullcontext()
            )
            with context:
                if args.track:
                    log_inputs(values, args.config)
                stages[args.stage](values)
                if args.track:
                    log_results(args.stage, values, settings)

    except (
        ValidationError,
        yaml.YAMLError,
        OSError,
        NotImplementedError,
        ValueError,
    ) as exc:
        logger.error(
            "Этап завершился ошибкой",
            error_type=type(exc).__name__,
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )

        if isinstance(exc, ValidationError):
            messages = {
                "missing": "обязательный параметр отсутствует",
                "extra_forbidden": "неизвестный параметр",
                "int_type": "ожидалось целое число",
                "float_type": "ожидалось число",
                "bool_type": "ожидалось логическое значение",
                "literal_error": "неподдерживаемое значение",
                "greater_than": "значение должно быть больше нижней границы",
                "greater_than_equal": "значение ниже минимального",
                "less_than_equal": "значение выше максимального",
                "finite_number": "ожидалось конечное число",
            }
            details = []

            for error in exc.errors(include_input=False, include_url=False):
                field = ".".join(str(part) for part in error["loc"]) or "конфигурация"
                reason = messages.get(error["type"], "недопустимое значение")
                details.append(f"{field}: {reason}")

            message = "Неверная конфигурация: " + "; ".join(details)
        elif isinstance(exc, (NotImplementedError, ValueError)):
            message = str(exc)
        else:
            message = "Не удалось прочитать YAML-конфигурацию"

        parser.exit(status=2, message=message + "\n")

    except Exception as exc:
        logger.error(
            "Этап завершился ошибкой",
            error_type=type(exc).__name__,
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )
        parser.exit(status=1, message="Ошибка выполнения этапа\n")

    else:
        logger.info(
            "Этап завершён", duration_ms=round((perf_counter() - started) * 1000, 3)
        )

    finally:
        structlog.contextvars.clear_contextvars()


if __name__ == "__main__":
    main()
