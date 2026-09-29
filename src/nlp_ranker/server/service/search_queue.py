"""Ограниченная очередь поиска с удержанием слота до завершения вычислений"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from nlp_ranker.core.config import Settings
from nlp_ranker.domain.errors import SearchOverloadedError, SearchTimeoutError
from nlp_ranker.domain.search import SearchHit
from nlp_ranker.server.metrics import SearchMetrics


class SearchQueue:
    """Учитывает активные задачи и ожидание в одном процессе API"""

    def __init__(self, settings: Settings, metrics: SearchMetrics) -> None:
        """Создаёт семафор и счётчик без запуска фоновых задач"""
        self.settings = settings
        self.metrics = metrics
        self.semaphore = asyncio.Semaphore(settings.search_active_limit)
        self.pending = 0
        self.tasks: set[asyncio.Task[list[SearchHit]]] = set()

    async def run(
        self, operation: Callable[[], Coroutine[Any, Any, list[SearchHit]]]
    ) -> list[SearchHit]:
        """Ограничивает очередь и время ответа, отмена ждёт реальную работу"""
        if (
            self.pending
            >= self.settings.search_active_limit + self.settings.search_waiting_limit
        ):
            self.metrics.rejected.labels("full").inc()
            raise SearchOverloadedError("Очередь поиска заполнена")
        self.pending += 1
        self.metrics.waiting.inc()
        try:
            with self.metrics.measure("queue"):
                await asyncio.wait_for(
                    self.semaphore.acquire(), self.settings.search_queue_timeout_seconds
                )
        except BaseException as exc:
            self.pending -= 1
            if isinstance(exc, TimeoutError):
                self.metrics.rejected.labels("wait_timeout").inc()
                raise SearchOverloadedError("Истекло время ожидания поиска") from exc
            raise
        finally:
            self.metrics.waiting.dec()
        self.metrics.active.inc()
        task = asyncio.create_task(operation())
        self.tasks.add(task)
        task.add_done_callback(self._finished)
        try:
            return await asyncio.wait_for(
                asyncio.shield(task), self.settings.search_timeout_seconds
            )
        except TimeoutError as exc:
            task.cancel()
            self.metrics.rejected.labels("execution_timeout").inc()
            raise SearchTimeoutError("Истекло время выполнения поиска") from exc
        except asyncio.CancelledError:
            task.cancel()
            raise

    def _finished(self, task: asyncio.Task[list[SearchHit]]) -> None:
        """Убирает завершённую задачу и получает её исключение после отмены HTTP"""
        self.metrics.active.dec()
        self.tasks.discard(task)
        self.pending -= 1
        self.semaphore.release()
        if not task.cancelled():
            task.exception()

    async def close(self) -> None:
        """Дожидается всех уже принятых вычислений до закрытия ресурсов"""
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
