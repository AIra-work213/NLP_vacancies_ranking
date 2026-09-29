"""Проверка соответствия документов PostgreSQL поисковому выпуску"""

from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from nlp_ranker.domain.search import SearchUnavailableError
from nlp_ranker.server.model.documents import Document, DocumentEmbedding


class ReleaseRepository:
    """Проверяет документы и векторы в транзакции вызывающего сервиса"""

    def __init__(self, session: AsyncSession) -> None:
        """Использует тот же снимок БД, что и поисковая выдача"""
        self.session = session

    async def validate(self, hashes: dict[UUID, str], metadata: dict[str, Any]) -> None:
        """Отклоняет изменённые документы и неполный набор векторов"""
        rows = await self.session.execute(
            select(Document.id, Document.content_hash).where(Document.id.in_(hashes))
        )
        if dict(rows.tuples().all()) != hashes:
            raise SearchUnavailableError(
                "Документы изменились, пересоберите поисковый выпуск"
            )
        count = await self.session.scalar(
            select(func.count())
            .select_from(DocumentEmbedding)
            .join(Document)
            .where(
                DocumentEmbedding.index_key == metadata["index_key"],
                DocumentEmbedding.content_hash == Document.content_hash,
            )
        )
        if count != metadata["build"]["chunks"]:
            raise SearchUnavailableError("Dense-выпуск неполон, загрузите эмбеддинги")
