"""Чтение документов и представление результатов вне ORM"""

import asyncio
from typing import get_args
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nlp_ranker.core.config import Settings
from nlp_ranker.domain.search import (
    DocumentDetails,
    Experience,
    SearchUnavailableError,
    WorkFormat,
)
from nlp_ranker.server.model.documents import Document
from nlp_ranker.server.repository.postgres.documents import DocumentRepository


def document_details(document: Document) -> DocumentDetails:
    """Отделяет результат чтения БД от ORM и HTTP-схем"""
    return DocumentDetails(
        document_id=str(document.id),
        title=document.title,
        text=document.text,
        employer=document.employer,
        city=document.city,
        work_format=document.work_format,
        experience=document.experience,
        region=document.region,
        schedule=document.schedule,
        version=document.version,
    )


class DocumentService:
    """Читает актуальные документы независимо от готовности индекса поиска"""

    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], settings: Settings
    ) -> None:
        """Получает фабрику сессий и ограничение времени чтения"""
        self.sessions = sessions
        self.settings = settings

    async def get(self, document_id: UUID) -> DocumentDetails | None:
        """Получает актуальный документ напрямую из PostgreSQL без кеширования"""
        try:
            async with asyncio.timeout(self.settings.search_timeout_seconds):
                async with self.sessions() as session:
                    document = await DocumentRepository(session).get(document_id)
                    return document_details(document) if document is not None else None
        except (SQLAlchemyError, OSError, TimeoutError) as exc:
            raise SearchUnavailableError("Хранилище документов недоступно") from exc

    async def filter_options(self) -> dict[str, list[str]]:
        """Оставляет значения корпуса, поддерживаемые поисковым контрактом"""
        try:
            async with asyncio.timeout(self.settings.search_timeout_seconds):
                async with self.sessions() as session, session.begin():
                    await session.connection(
                        execution_options={"isolation_level": "REPEATABLE READ"}
                    )
                    options = await DocumentRepository(session).filter_values()
            options["work_format"] = [
                value
                for value in options["work_format"]
                if value in get_args(WorkFormat)
            ]
            options["experience"] = [
                value
                for value in options["experience"]
                if value in get_args(Experience)
            ]
            return options
        except (SQLAlchemyError, OSError, TimeoutError) as exc:
            raise SearchUnavailableError("Фильтры временно недоступны") from exc
