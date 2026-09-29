"""Короткий кеш ID и оценок поиска с ограниченным временем обращения к Redis"""

import asyncio
import hashlib
import json
from time import perf_counter
from typing import Literal
from uuid import UUID

import structlog
from prometheus_client import CollectorRegistry, Counter, Histogram
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from nlp_ranker.core.config import Settings
from nlp_ranker.domain.search import SearchFilters, SearchHit


class CachedHit(BaseModel):
    """Проверяет ID и конечную оценку перед использованием записи"""

    model_config = ConfigDict(extra="forbid")
    document_id: UUID
    score: float = Field(allow_inf_nan=False)


class CacheEntry(BaseModel):
    """Различает пустую успешную выдачу и отсутствующую запись"""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    hits: list[CachedHit] = Field(max_length=100)


class SearchCache:
    """Хранит версию выпуска в ключе и продолжает поиск при отказе Redis"""

    def __init__(
        self,
        client: Redis,
        settings: Settings,
        registry: CollectorRegistry | None = None,
    ) -> None:
        """Сохраняет клиент, лимиты и отдельные метрики процесса"""
        self.client = client
        self.settings = settings
        self.requests = Counter(
            "nlp_ranker_search_cache_requests_total",
            "Обращения к кешу поиска",
            ["result"],
            registry=registry,
        )
        self.duration = Histogram(
            "nlp_ranker_search_cache_duration_seconds",
            "Время обращения к кешу",
            ["operation"],
            registry=registry,
        )

    def key(self, release: str, query: str, limit: int, filters: SearchFilters) -> str:
        """Учитывает выпуск, фактический запрос, число результатов и все фильтры"""
        value = {
            "release": release,
            "query": query,
            "limit": limit,
            "city": filters.city,
            "work_format": filters.work_format,
            "experience": filters.experience,
        }
        digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        return f"nlp_ranker:search:v1:{digest}"

    async def get(self, key: str) -> list[SearchHit] | None:
        """Ограничивает ожидание и не принимает повреждённые записи"""
        if not self.settings.cache_enabled:
            return None
        started = perf_counter()
        status = "miss"
        try:
            async with asyncio.timeout(self.settings.cache_timeout_seconds):
                raw = await self.client.get(key)
            if raw is None:
                return None
            if len(raw) > self.settings.cache_max_bytes:
                status = "invalid"
                return None
            entry = CacheEntry.model_validate_json(raw)
            ids = [hit.document_id for hit in entry.hits]
            if len(ids) != len(set(ids)):
                status = "invalid"
                return None
            status = "hit"
            return [
                SearchHit(str(hit.document_id), hit.score, "") for hit in entry.hits
            ]
        except (RedisError, TimeoutError):
            status = "error"
            return None
        except ValidationError:
            status = "invalid"
            return None
        finally:
            self.requests.labels(status).inc()
            self.duration.labels("read").observe(perf_counter() - started)
            structlog.get_logger().info("Кеш поиска прочитан", cache_status=status)

    async def put(self, key: str, hits: list[SearchHit]) -> None:
        """Записывает только успешную выдачу с TTL и ограничением размера"""
        if not self.settings.cache_enabled:
            return
        entry = CacheEntry(
            hits=[
                CachedHit(document_id=UUID(hit.document_id), score=hit.score)
                for hit in hits
            ]
        )
        raw = entry.model_dump_json().encode()
        if len(raw) > self.settings.cache_max_bytes:
            self.requests.labels("oversize").inc()
            return
        started = perf_counter()
        try:
            async with asyncio.timeout(self.settings.cache_timeout_seconds):
                await self.client.set(key, raw, ex=self.settings.cache_ttl_seconds)
        except (RedisError, TimeoutError):
            self.requests.labels("error").inc()
            structlog.get_logger().warning("Кеш поиска недоступен при записи")
        finally:
            self.duration.labels("write").observe(perf_counter() - started)
