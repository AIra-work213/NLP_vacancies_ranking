"""Контракт поискового сервиса для HTTP-слоя"""

from typing import Protocol

from nlp_ranker.domain.search import (
    SearchFilters,
    SearchHit,
    SearchUnavailableError,
)


class SearchBackend(Protocol):
    """Определяет операции, необходимые поисковому сервису"""

    release_version: str

    async def check_ready(self) -> bool:
        """Проверяет обязательные ресурсы поиска"""
        ...

    @property
    def ready(self) -> bool:
        """Возвращает признак готовности поисковых ресурсов"""
        ...

    async def search(
        self, query: str, limit: int, filters: SearchFilters | None = None
    ) -> list[SearchHit]:
        """Возвращает ранжированные документы по запросу с заданным лимитом"""
        ...


class UnconfiguredBackend:
    """Отклоняет запросы, пока не подключены модель и индекс"""

    release_version = "unconfigured"

    async def check_ready(self) -> bool:
        """Не объявляет не загруженный поиск готовым"""
        return False

    @property
    def ready(self) -> bool:
        """Возвращает признак готовности поисковых ресурсов"""
        return False

    async def search(
        self, query: str, limit: int, filters: SearchFilters | None = None
    ) -> list[SearchHit]:
        """Отклоняет запрос из-за отсутствия модели и индекса"""
        raise SearchUnavailableError("Поисковый индекс и модель не настроены")
