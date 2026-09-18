"""Проверки записи и конкурентного доступа в изолированном PostgreSQL"""

import asyncio
import os
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError, TimeoutError

from nlp_ranker.core.config import Settings
from nlp_ranker.domain.documents import DocumentConflictError, DocumentInput
from nlp_ranker.server.model.documents import Document
from nlp_ranker.server.repository.postgres.connection import (
    create_engine,
    create_session_factory,
)
from nlp_ranker.server.repository.postgres.documents import DocumentRepository

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def database():
    """Очищает только специально указанную тестовую БД и закрывает её пул"""
    dsn = os.environ.get("NLP_RANKER_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Запустите make test-database")
    if not dsn.endswith("/nlp_ranker_test"):
        pytest.fail("Для проверки нужна отдельная БД nlp_ranker_test")
    engine = create_engine(
        Settings(
            _env_file=None,
            postgres_dsn=dsn,
            db_pool_size=2,
            db_max_overflow=0,
            db_pool_timeout=1,
        )
    )
    sessions = create_session_factory(engine)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("TRUNCATE documents CASCADE"))
        yield engine, sessions
    finally:
        await engine.dispose()


def vacancy(**changes):
    """Создаёт небольшой документ для проверки поведения репозитория"""
    values = dict(
        id=uuid4(),
        source="test",
        source_id=str(uuid4()),
        title="Разработчик",
        text="Python и SQL",
        city=None,
        work_format="remote",
        experience="no_experience",
    )
    return DocumentInput.model_validate(values | changes)


async def test_repeat_import_and_changed_metadata(database):
    """Повторный импорт не обновляет строку, изменение метаданных повышает версию"""
    _, sessions = database
    doc = vacancy()
    async with sessions() as session, session.begin():
        repository = DocumentRepository(session)
        assert await repository.save_many([doc]) == 1
        assert await repository.save_many([doc]) == 0
        original = (await repository.get_many([doc.id]))[0]
        timestamp = original.updated_at
        assert original.version == 1
    async with sessions() as session, session.begin():
        repository = DocumentRepository(session)
        assert await repository.save_many([doc]) == 0
        stored = (await repository.get_many([doc.id]))[0]
        assert stored.updated_at == timestamp
        assert (
            await repository.save_many([doc.model_copy(update={"city": "Москва"})]) == 1
        )
    async with sessions() as session:
        stored = (await DocumentRepository(session).get_many([doc.id]))[0]
        assert stored.version == 2
        assert stored.city == "Москва"


async def test_order_filters_and_missing_documents(database):
    """Восстанавливает порядок выдачи, проверяет фильтры и отсутствие документа"""
    _, sessions = database
    first = vacancy(city="Москва")
    second = vacancy(work_format=None, experience=None)
    async with sessions() as session, session.begin():
        repository = DocumentRepository(session)
        await repository.save_many([first, second])
        assert [doc.id for doc in await repository.get_many([second.id, first.id])] == [
            second.id,
            first.id,
        ]
        assert await repository.list_ids(
            city="Москва", work_format="remote", experience="no_experience"
        ) == [first.id]
        assert set(await repository.list_ids()) == {first.id, second.id}
        assert await repository.list_ids(city="Казань") == []
        with pytest.raises(LookupError):
            await repository.get_many([uuid4()])


async def test_concurrent_updates_and_versioned_delete(database):
    """Один из двух писателей получает конфликт вместо потери чужого изменения"""
    _, sessions = database
    doc = vacancy()
    async with sessions() as session, session.begin():
        await DocumentRepository(session).save_many([doc])

    async def change(title):
        """Изменяет документ с прочитанной версией через собственную сессию"""
        async with sessions() as session, session.begin():
            return await DocumentRepository(session).replace(
                doc.model_copy(update={"title": title}), expected_version=1
            )

    results = await asyncio.gather(
        change("Первый"), change("Второй"), return_exceptions=True
    )
    assert sum(isinstance(result, DocumentConflictError) for result in results) == 1
    assert 2 in results
    async with sessions() as session, session.begin():
        repository = DocumentRepository(session)
        with pytest.raises(DocumentConflictError):
            await repository.delete(doc.id, expected_version=1)
        await repository.delete(doc.id, expected_version=2)
        assert await repository.list_ids() == []


async def test_concurrent_insert_and_rollback(database):
    """Проверяет параллельный импорт и откат записи при ошибке"""
    _, sessions = database
    doc = vacancy()

    async def insert():
        """Импортирует документ в отдельной транзакции"""
        async with sessions() as session, session.begin():
            return await DocumentRepository(session).save_many([doc])

    assert sorted(await asyncio.gather(insert(), insert())) == [0, 1]
    async with sessions() as session:
        assert (await DocumentRepository(session).get_many([doc.id]))[0].version == 1
    with pytest.raises(RuntimeError):
        async with sessions() as session, session.begin():
            await DocumentRepository(session).save_many([vacancy()])
            raise RuntimeError("Прервана загрузка")
    async with sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Document)) == 1


async def test_pool_and_statement_timeout(database):
    """Проверяет лимит пула и восстановление после таймаута запроса"""
    engine, sessions = database
    async with engine.connect(), engine.connect():
        with pytest.raises(TimeoutError):
            async with engine.connect():
                pass
    with pytest.raises(DBAPIError):
        async with sessions() as session, session.begin():
            await session.execute(text("SET LOCAL statement_timeout = '50ms'"))
            await session.execute(text("SELECT pg_sleep(0.2)"))
    async with sessions() as session:
        assert await session.scalar(text("SELECT 1")) == 1


async def test_document_identity_cannot_change(database):
    """Отклоняет новый ID для существующей пары источник/ID и сохраняет строку"""
    _, sessions = database
    doc = vacancy()
    async with sessions() as session, session.begin():
        await DocumentRepository(session).save_many([doc])
    with pytest.raises(DocumentConflictError):
        async with sessions() as session, session.begin():
            changed = doc.model_copy(update={"id": uuid4(), "title": "Новое название"})
            await DocumentRepository(session).save_many([changed])
    async with sessions() as session:
        stored = (await DocumentRepository(session).get_many([doc.id]))[0]
        assert stored.version == 1
        assert stored.title == doc.title
