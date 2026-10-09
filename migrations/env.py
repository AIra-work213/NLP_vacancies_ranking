"""Асинхронные миграции Alembic с общими для API настройками подключения"""

import asyncio

from alembic import context
from sqlalchemy import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from nlp_ranker.core.config import Settings
from nlp_ranker.server.model.documents import Base

target_metadata = Base.metadata


def run_migrations(connection: Connection) -> None:
    """Применяет миграции через открытое соединение"""
    context.configure(
        connection=connection, target_metadata=target_metadata, compare_type=True
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_online() -> None:
    """Открывает соединение для миграций и закрывает его после выполнения"""
    engine = create_async_engine(Settings().postgres_dsn, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=Settings().postgres_dsn,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(run_online())
