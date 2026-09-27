"""Инференс запроса и вызов векторного репозитория"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nlp_ranker.domain.search import SearchHit
from nlp_ranker.ml.inference.encoder import Encoder
from nlp_ranker.server.repository.postgres.embeddings import EmbeddingRepository


class DenseSearch:
    """Ищет точным cosine расстоянием и объединяет фрагменты до top-k документов"""

    def __init__(
        self,
        encoder: Encoder,
        sessions: async_sessionmaker[AsyncSession],
        metadata: dict[str, Any],
    ) -> None:
        """Сохраняет модель, фабрику отдельных сессий и версию выпуска"""
        if encoder.config.model_dump(mode="json") != metadata["dense"]:
            raise ValueError("Параметры запросов не совпадают с моделью корпуса")
        self.encoder = encoder
        self.sessions = sessions
        self.metadata = metadata

    async def search(
        self,
        query: str,
        limit: int = 100,
        city: str | None = None,
        work_format: str | None = None,
        experience: str | None = None,
    ) -> list[SearchHit]:
        """Фильтрует в SQL и отклоняет неполный или устаревший выпуск"""
        if limit < 1:
            raise ValueError("Лимит должен быть положительным")
        if not query.strip():
            return []
        vector = await self.encoder.query(query)
        return await self.search_vector(vector, limit, city, work_format, experience)

    async def search_vector(
        self,
        vector: list[float],
        limit: int = 100,
        city: str | None = None,
        work_format: str | None = None,
        experience: str | None = None,
    ) -> list[SearchHit]:
        """Читает один согласованный снимок базы без потери кандидатов на фильтрах"""
        import math

        if limit < 1 or len(vector) != 384 or not all(math.isfinite(x) for x in vector):
            raise ValueError("Нужны положительный лимит и конечный вектор из 384 чисел")
        if not any(vector):
            raise ValueError("Нулевой вектор не имеет cosine расстояния")
        async with self.sessions() as session, session.begin():
            await session.connection(
                execution_options={"isolation_level": "REPEATABLE READ"}
            )
            return await self.search_vector_in_session(
                session, vector, limit, city, work_format, experience
            )

    async def search_vector_in_session(
        self,
        session: AsyncSession,
        vector: list[float],
        limit: int = 100,
        city: str | None = None,
        work_format: str | None = None,
        experience: str | None = None,
    ) -> list[SearchHit]:
        """Передаёт вектор репозиторию в снимке вызывающего сервиса"""
        return await EmbeddingRepository(session).search(
            vector, self.metadata, limit, city, work_format, experience
        )
