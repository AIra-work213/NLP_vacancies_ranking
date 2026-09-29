"""Сборка API и владение моделями, БД и Redis в одном процессе"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import structlog
from fastapi import FastAPI
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from nlp_ranker.core.config import Settings
from nlp_ranker.core.logging import configure_logging
from nlp_ranker.domain.errors import SearchOverloadedError, SearchTimeoutError
from nlp_ranker.domain.search import SearchUnavailableError
from nlp_ranker.server.controller import documents, health, search
from nlp_ranker.server.errors import handle_search_error, handle_unexpected_error
from nlp_ranker.server.metrics import install_metrics
from nlp_ranker.server.middleware import RequestLoggingMiddleware
from nlp_ranker.server.repository.cache.search import SearchCache
from nlp_ranker.server.repository.postgres.connection import (
    create_engine,
    create_session_factory,
)
from nlp_ranker.server.service.contracts import SearchBackend, UnconfiguredBackend
from nlp_ranker.server.service.documents import DocumentService
from nlp_ranker.server.service.search import SearchService, load_search_config


def create_app(
    settings: Settings | None = None, backend: SearchBackend | None = None
) -> FastAPI:
    """Передаёт настройки и зависимости, не загружая модели при импорте"""
    settings = settings or Settings()
    configure_logging(settings)
    app = FastAPI(
        title="Поиск и ранжирование вакансий", version="0.1.0", lifespan=lifespan
    )
    app.state.settings = settings
    app.state.search_backend = backend if backend is not None else UnconfiguredBackend()
    app.state.backend_injected = backend is not None
    for controller in (health, search, documents):
        app.include_router(controller.router)
    for error in (SearchOverloadedError, SearchTimeoutError, SearchUnavailableError):
        app.add_exception_handler(error, handle_search_error)
    install_metrics(app)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_exception_handler(Exception, handle_unexpected_error)
    return app


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Закрывает поиск перед клиентами, в том числе при ошибке загрузки"""
    settings: Settings = app.state.settings
    logger = structlog.stdlib.get_logger("nlp_ranker.lifecycle")
    async with AsyncExitStack() as resources:
        engine = create_engine(settings)
        resources.push_async_callback(engine.dispose)
        cache: Redis = Redis.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            socket_connect_timeout=settings.cache_timeout_seconds,
            socket_timeout=settings.cache_timeout_seconds,
            retry=Retry(NoBackoff(), 0),
        )
        resources.push_async_callback(cache.aclose)
        sessions = create_session_factory(engine)
        app.state.session_factory = sessions
        app.state.document_service = DocumentService(sessions, settings)
        app.state.cache = cache
        if settings.search_enabled and not app.state.backend_injected:
            search_cache = SearchCache(cache, settings, app.state.metrics_registry)
            backend = SearchService(
                load_search_config(settings.search_config_path),
                settings,
                sessions,
                search_cache,
                app.state.search_metrics,
            )
            resources.push_async_callback(backend.close)
            await backend.start()
            app.state.search_backend = backend
        logger.info(
            "API запущен",
            environment=settings.environment,
            search_ready=app.state.search_backend.ready,
        )
        try:
            yield
        finally:
            logger.info("API останавливается")
    logger.info("Ресурсы API закрыты")
