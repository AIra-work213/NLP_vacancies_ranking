"""Проверки ограниченной очереди, отмены и кеша без загрузки моделей"""

import asyncio
import threading
from pathlib import Path
from uuid import uuid4

import pytest
from prometheus_client import CollectorRegistry
from redis.asyncio import Redis

from nlp_ranker.core.concurrency import run_blocking
from nlp_ranker.core.config import Settings
from nlp_ranker.domain.errors import SearchOverloadedError, SearchTimeoutError
from nlp_ranker.domain.search import SearchFilters, SearchHit
from nlp_ranker.server.metrics import SearchMetrics
from nlp_ranker.server.repository.cache.search import SearchCache
from nlp_ranker.server.service.search import SearchService, load_search_config


async def test_timeout_keeps_slot_until_cpu_finishes():
    """После 504 вычисление удерживает слот, перегрузка не запускает новую работу"""
    started = threading.Event()
    finish = threading.Event()
    settings = Settings(
        _env_file=None, search_waiting_limit=0, search_timeout_seconds=0.02
    )
    redis = Redis.from_url(settings.redis_url)
    registry = CollectorRegistry()
    backend = SearchService(
        load_search_config(Path("configs/search.yaml")),
        settings,
        None,
        SearchCache(redis, settings),
        SearchMetrics(registry),
    )
    backend.ready = True

    async def search(query, limit, filters):
        """Ожидает разрешения в реальном потоке"""

        def compute():
            """Имитирует незавершённый CPU-инференс"""
            started.set()
            assert finish.wait(5)
            return []

        return await run_blocking(compute)

    backend._search = search
    try:
        with pytest.raises(SearchTimeoutError):
            await backend.search("первый", 1)
        assert started.is_set()
        assert backend.pending == 1
        assert registry.get_sample_value("nlp_ranker_search_active") == 1
        assert registry.get_sample_value("nlp_ranker_search_waiting") == 0
        with pytest.raises(SearchOverloadedError):
            await backend.search("второй", 1)
        closing = asyncio.create_task(backend.close())
        await asyncio.sleep(0)
        assert not closing.done()
        finish.set()
        await closing
        assert backend.pending == 0
        assert registry.get_sample_value("nlp_ranker_search_active") == 0
        assert (
            registry.get_sample_value(
                "nlp_ranker_search_rejected_total", {"reason": "execution_timeout"}
            )
            == 1
        )
        assert (
            registry.get_sample_value(
                "nlp_ranker_search_rejected_total", {"reason": "full"}
            )
            == 1
        )
    finally:
        finish.set()
        await backend.close()
        await redis.aclose()


async def test_repeated_cancellation_does_not_free_slot_early():
    """Повторная отмена ждёт поток, закрытие не оставляет фоновых вычислений"""
    started = threading.Event()
    finish = threading.Event()

    def compute():
        """Отмечает начало работы и ждёт завершения"""
        started.set()
        assert finish.wait(5)
        return 1

    task = asyncio.create_task(run_blocking(compute))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_queue_wait_is_bounded():
    """Ожидающий запрос получает отказ по таймауту, счётчик возвращается к нулю"""
    settings = Settings(
        _env_file=None, search_waiting_limit=1, search_queue_timeout_seconds=0.02
    )
    redis = Redis.from_url(settings.redis_url)
    backend = SearchService(
        load_search_config(Path("configs/search.yaml")),
        settings,
        None,
        SearchCache(redis, settings),
    )
    backend.ready = True
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def search(query, limit, filters):
        """Удерживает единственный активный слот"""
        entered.set()
        await finish.wait()
        return []

    backend._search = search
    first = asyncio.create_task(backend.search("первый", 1))
    try:
        await entered.wait()
        with pytest.raises(SearchOverloadedError):
            await backend.search("второй", 1)
        assert backend.pending == 1
        finish.set()
        assert await first == []
        assert backend.pending == 0
    finally:
        finish.set()
        await asyncio.gather(first, return_exceptions=True)
        await backend.close()
        await redis.aclose()


def test_cache_keys_separate_versions_filters_and_limits():
    """Разные выпуски и параметры не разделяют запись кеша"""
    settings = Settings(_env_file=None)
    cache = SearchCache(Redis.from_url(settings.redis_url), settings)
    original = cache.key("v1", "Java", 10, SearchFilters())
    assert original != cache.key("v2", "Java", 10, SearchFilters())
    assert original != cache.key("v1", "Java", 1, SearchFilters())
    assert original != cache.key("v1", "Java", 10, SearchFilters(city="Москва"))
    assert original != cache.key("v1", "Java", 10, SearchFilters(work_format="remote"))
    assert original != cache.key(
        "v1", "Java", 10, SearchFilters(experience="no_experience")
    )


async def test_unavailable_redis_is_a_cache_miss():
    """Отказ соединения не превращается в ошибку поиска"""
    settings = Settings(
        _env_file=None, redis_url="redis://127.0.0.1:1", cache_timeout_seconds=0.02
    )
    redis = Redis.from_url(settings.redis_url, socket_connect_timeout=0.02)
    cache = SearchCache(redis, settings)
    try:
        assert await cache.get("key") is None
        await cache.put("key", [SearchHit(str(uuid4()), 0.5, "")])
    finally:
        await redis.aclose()


async def test_startup_mismatch_closes_resources(tmp_path, monkeypatch):
    """Сбой до готовности закрывает поиск, Redis и БД в обратном порядке"""
    import yaml
    from sqlalchemy.ext.asyncio import AsyncEngine

    from nlp_ranker.server.app import create_app

    config = yaml.safe_load(Path("configs/search.yaml").read_text())
    config["bm25"]["title_repeat"] += 1
    path = tmp_path / "incompatible.yaml"
    path.write_text(yaml.safe_dump(config))
    closed = []
    close_search = SearchService.close
    close_redis = Redis.aclose
    close_engine = AsyncEngine.dispose

    async def search_close(self):
        """Отмечает закрытие настоящего поискового механизма"""
        closed.append("search")
        await close_search(self)

    async def redis_close(self, *args, **kwargs):
        """Отмечает закрытие настоящего Redis-клиента"""
        closed.append("redis")
        await close_redis(self, *args, **kwargs)

    async def database_close(self, *args, **kwargs):
        """Отмечает освобождение настоящего пула БД"""
        closed.append("database")
        await close_engine(self, *args, **kwargs)

    monkeypatch.setattr(SearchService, "close", search_close)
    monkeypatch.setattr(Redis, "aclose", redis_close)
    monkeypatch.setattr(AsyncEngine, "dispose", database_close)
    app = create_app(Settings(_env_file=None, search_config_path=path))
    with pytest.raises(ValueError, match="несовместимы"):
        async with app.router.lifespan_context(app):
            pytest.fail("Несовместимый выпуск не должен запускаться")
    assert closed == ["search", "redis", "database"]
