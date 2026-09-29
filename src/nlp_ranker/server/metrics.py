"""Метрики одного процесса API без пользовательских значений в метках"""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    ProcessCollector,
    generate_latest,
)
from prometheus_fastapi_instrumentator import Instrumentator


class SearchMetrics:
    """Измеряет этапы поиска и фактическое занятие ограниченной очереди"""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Регистрирует показатели в реестре приложения или отдельном тестовом"""
        registry = registry if registry is not None else CollectorRegistry()
        self.stage = Histogram(
            "nlp_ranker_search_stage_seconds",
            "Время этапа поиска, включая неуспешное завершение",
            ["stage"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
            registry=registry,
        )
        self.active = Gauge(
            "nlp_ranker_search_active",
            "Вычисляемые поисковые задачи",
            registry=registry,
        )
        self.waiting = Gauge(
            "nlp_ranker_search_waiting", "Задачи в ожидании слота", registry=registry
        )
        self.rejected = Counter(
            "nlp_ranker_search_rejected_total",
            "Отказы очереди",
            ["reason"],
            registry=registry,
        )
        self.ready = Gauge(
            "nlp_ranker_search_ready",
            "Готовность выпуска и PostgreSQL",
            registry=registry,
        )
        self.release = Info(
            "nlp_ranker_release", "Загруженный поисковый выпуск", registry=registry
        )

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Учитывает время до завершения этапа, ошибки или отмены"""
        started = perf_counter()
        try:
            yield
        finally:
            self.stage.labels(stage).observe(perf_counter() - started)


def install_metrics(app: FastAPI) -> None:
    """Подключает HTTP-метрики и проверяет готовность при каждом сборе"""
    registry = CollectorRegistry()
    ProcessCollector(registry=registry)
    app.state.metrics_registry = registry
    metrics = SearchMetrics(registry)
    app.state.search_metrics = metrics
    Instrumentator(
        registry=registry,
        should_group_status_codes=False,
        should_ignore_untemplated=False,
        excluded_handlers=["^/metrics$"],
    ).instrument(app)

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        """Отдаёт показатели процесса с актуальной проверкой обязательной БД"""
        metrics.ready.set(await app.state.search_backend.check_ready())
        return Response(
            generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )
