"""Фабрики асинхронного пула и независимых сессий"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from nlp_ranker.core.config import Settings


def create_engine(settings: Settings) -> AsyncEngine:
    """Создаёт ограниченный асинхронный пул без открытия соединения"""
    return create_async_engine(
        settings.postgres_dsn,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_pre_ping=True,
        connect_args={"timeout": 5, "command_timeout": 10},
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Создаёт фабрику независимых сессий БД для отдельных задач"""
    return async_sessionmaker(engine, expire_on_commit=False)
