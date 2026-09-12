"""Общий формат логов приложения и пайплайнов"""

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog
from structlog.types import Processor

from nlp_ranker.core.config import Settings


def remove_exception_details(
    logger: object, method: str, event: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Не выводит traceback, который может содержать тексты запросов и секреты"""
    event.pop("exc_info", None)
    event.pop("stack_info", None)

    return event


def configure_logging(settings: Settings) -> None:
    """Объединяет structlog и logging в один читаемый или JSON-формат"""
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        remove_exception_details,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(settings.log_level)

    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(settings.log_level)

    logging.getLogger("uvicorn.access").disabled = True

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )
