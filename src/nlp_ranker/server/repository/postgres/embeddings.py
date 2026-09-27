"""Загрузка эмбеддингов и векторный поиск через pgvector"""

import asyncio
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from nlp_ranker.domain.search import SearchHit
from nlp_ranker.ml.artifacts import read_embeddings
from nlp_ranker.server.model.documents import Document, DocumentEmbedding


async def import_embeddings(session: AsyncSession, path: Path) -> dict[str, Any]:
    """Проверяет документы под блокировкой и атомарно загружает выпуск без дублей"""
    rows, metadata = await asyncio.to_thread(read_embeddings, path)
    ids = sorted({UUID(row["document_id"]) for row in rows})
    stored = await session.execute(
        select(Document.id, Document.content_hash)
        .where(Document.id.in_(ids))
        .order_by(Document.id)
        .with_for_update(read=True)
    )
    hashes = {doc_id: content_hash for doc_id, content_hash in stored}
    if any(hashes.get(UUID(row["document_id"])) != row["content_hash"] for row in rows):
        raise ValueError("Документы PostgreSQL не совпадают с корпусом эмбеддингов")
    for start in range(0, len(rows), 100):
        values = [
            dict(
                row,
                document_id=UUID(row["document_id"]),
                index_key=metadata["index_key"],
            )
            for row in rows[start : start + 100]
        ]
        await session.execute(
            insert(DocumentEmbedding).values(values).on_conflict_do_nothing()
        )
    return metadata


class EmbeddingRepository:
    """Ищет документы через pgvector в переданной транзакции"""

    def __init__(self, session: AsyncSession) -> None:
        """Сохраняет сессию вызывающего сценария"""
        self.session = session

    async def search(
        self,
        vector: list[float],
        metadata: dict[str, Any],
        limit: int = 100,
        city: str | None = None,
        work_format: str | None = None,
        experience: str | None = None,
    ) -> list[SearchHit]:
        """Проверяет выпуск и применяет фильтры перед отбором top-k"""
        import math

        if limit < 1 or len(vector) != 384 or not all(math.isfinite(x) for x in vector):
            raise ValueError("Нужны положительный лимит и конечный вектор из 384 чисел")
        if not any(vector):
            raise ValueError("Нулевой вектор не имеет cosine расстояния")
        valid = [
            DocumentEmbedding.index_key == metadata["index_key"],
            DocumentEmbedding.content_hash == Document.content_hash,
        ]
        count = await self.session.scalar(
            select(func.count())
            .select_from(DocumentEmbedding)
            .join(Document)
            .where(*valid)
        )
        if count != metadata["build"]["chunks"]:
            raise ValueError("Dense-выпуск неполон или устарел, пересоберите векторы")
        distance = func.min(DocumentEmbedding.embedding.cosine_distance(vector))
        statement = (
            select(Document.id, distance.label("distance"))
            .join(DocumentEmbedding)
            .where(*valid)
            .group_by(Document.id)
            .order_by(distance, Document.id)
            .limit(limit)
        )
        for column, value in [
            (Document.city, city),
            (Document.work_format, work_format),
            (Document.experience, experience),
        ]:
            if value is not None:
                statement = statement.where(column == value)
        rows = await self.session.execute(statement)
        return [SearchHit(str(doc_id), 1 - float(score), "") for doc_id, score in rows]
