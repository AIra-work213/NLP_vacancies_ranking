"""Поиск в согласованном снимке с кешем и ограничением нагрузки"""

import asyncio
from pathlib import Path
from uuid import UUID

import yaml
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from nlp_ranker.core.concurrency import run_blocking
from nlp_ranker.core.config import Settings
from nlp_ranker.domain.search import (
    SearchFilters,
    SearchHit,
    SearchUnavailableError,
)
from nlp_ranker.ml.config import SearchConfig
from nlp_ranker.ml.inference.release import load_release
from nlp_ranker.ml.inference.reranker import Reranker
from nlp_ranker.ml.retrieval.bm25 import BM25Index
from nlp_ranker.ml.retrieval.hybrid import fuse_results
from nlp_ranker.server.metrics import SearchMetrics
from nlp_ranker.server.repository.cache.search import SearchCache
from nlp_ranker.server.repository.postgres.documents import DocumentRepository
from nlp_ranker.server.repository.postgres.releases import ReleaseRepository
from nlp_ranker.server.service.dense import DenseSearch
from nlp_ranker.server.service.documents import document_details
from nlp_ranker.server.service.search_queue import SearchQueue


class SearchService:
    """Владеет индексом и моделями, каждая поисковая задача владеет своей сессией"""

    def __init__(
        self,
        config: SearchConfig,
        settings: Settings,
        sessions: async_sessionmaker[AsyncSession],
        cache: SearchCache,
        metrics: SearchMetrics | None = None,
    ) -> None:
        """Создаёт ограниченную очередь без загрузки тяжёлых ресурсов"""
        self.config = config
        self.settings = settings
        self.sessions = sessions
        self.cache = cache
        self.metrics = metrics if metrics is not None else SearchMetrics()
        self.ready = False
        self.closing = False
        self.release_version = "unconfigured"
        self.index: BM25Index | None = None
        self.dense: DenseSearch | None = None
        self.reranker: Reranker | None = None
        self.hashes: dict[UUID, str] = {}
        self.chunks: dict[str, list[str]] = {}
        self.document_filters: dict[str, dict[str, object]] = {}
        self.queue = SearchQueue(settings, self.metrics)

    async def start(self) -> None:
        """Собирает поиск и проверяет PostgreSQL до объявления готовности"""
        release = await run_blocking(lambda: load_release(self.config))
        self.index = release.index
        self.dense = DenseSearch(release.encoder, self.sessions, release.metadata)
        self.reranker = release.reranker
        self.hashes = release.hashes
        self.chunks = release.chunks
        self.document_filters = release.document_filters
        self.release_version = release.version
        async with self.sessions() as session, session.begin():
            await session.connection(
                execution_options={"isolation_level": "REPEATABLE READ"}
            )
            await self.validate_release(session)
        self.metrics.release.info({"version": self.release_version})
        self.ready = True

    async def validate_release(self, session: AsyncSession) -> None:
        """Делегирует репозиторию проверку версии в снимке запроса"""
        if self.dense is None:
            raise SearchUnavailableError("Поисковый выпуск не загружен")
        await ReleaseRepository(session).validate(self.hashes, self.dense.metadata)

    async def check_ready(self) -> bool:
        """Проверяет обязательную БД и выпуск, Redis остаётся необязательным"""
        if not self.ready or self.closing:
            return False
        try:
            async with asyncio.timeout(self.settings.search_queue_timeout_seconds):
                async with self.sessions() as session, session.begin():
                    await session.connection(
                        execution_options={"isolation_level": "REPEATABLE READ"}
                    )
                    await self.validate_release(session)
            return True
        except (SQLAlchemyError, OSError, TimeoutError, SearchUnavailableError):
            return False

    async def search(
        self, query: str, limit: int, filters: SearchFilters | None = None
    ) -> list[SearchHit]:
        """Проверяет запрос и передаёт поиск ограниченной очереди"""
        if not self.ready or self.closing:
            raise SearchUnavailableError("Поиск не готов")
        if not query.strip() or not 1 <= limit <= 100:
            raise ValueError("Нужны непустой запрос и лимит от 1 до 100")
        return await self.queue.run(
            lambda: self._run(
                " ".join(query.split()), limit, filters or SearchFilters()
            )
        )

    @property
    def pending(self) -> int:
        """Возвращает число принятых запросов, включая ожидание"""
        return self.queue.pending

    async def _run(
        self, query: str, limit: int, filters: SearchFilters
    ) -> list[SearchHit]:
        """Освобождает слот только после выхода всех вычислений и транзакции"""
        try:
            if self.closing:
                raise SearchUnavailableError("API останавливается")
            return await self._search(query, limit, filters)
        except (SQLAlchemyError, OSError, LookupError, ValueError) as exc:
            raise SearchUnavailableError(
                "Поисковый выпуск недоступен или устарел"
            ) from exc

    async def _search(
        self, query: str, limit: int, filters: SearchFilters
    ) -> list[SearchHit]:
        """Читает версии, ищет и получает документы в одной транзакции"""
        if self.index is None or self.dense is None:
            raise SearchUnavailableError("Поиск не загружен")
        index, dense = self.index, self.dense
        async with self.sessions() as session, session.begin():
            await session.connection(
                execution_options={"isolation_level": "REPEATABLE READ"}
            )
            with self.metrics.measure("validate_release"):
                await self.validate_release(session)
            key = self.cache.key(self.release_version, query, limit, filters)
            hits = await self.cache.get(key)
            if hits is not None and (
                len(hits) > limit
                or any(
                    UUID(hit.document_id) not in self.hashes
                    or any(
                        value is not None
                        and self.document_filters.get(hit.document_id, {}).get(name)
                        != value
                        for name, value in [
                            ("city", filters.city),
                            ("work_format", filters.work_format),
                            ("experience", filters.experience),
                        ]
                    )
                    for hit in hits
                )
            ):
                hits = None
            cached = hits is not None
            if hits is None:
                count = max(limit, self.config.retrieval.candidate_limit)
                with self.metrics.measure("bm25"):
                    lexical = await run_blocking(
                        lambda: index.search(
                            query,
                            count,
                            city=filters.city,
                            work_format=filters.work_format,
                            experience=filters.experience,
                        )
                    )
                with self.metrics.measure("encoder"):
                    vector = await dense.encoder.query(query)
                with self.metrics.measure("dense"):
                    semantic = await dense.search_vector_in_session(
                        session,
                        vector,
                        count,
                        filters.city,
                        filters.work_format,
                        filters.experience,
                    )
                hits = fuse_results(lexical, semantic, self.config.retrieval, limit)
                if self.reranker is not None:
                    rerank_count = max(limit, self.config.reranking.top_k)
                    candidates = fuse_results(
                        lexical, semantic, self.config.retrieval, rerank_count
                    )
                    with self.metrics.measure("rerank"):
                        hits = await self.reranker.search(
                            query, candidates, self.chunks, limit
                        )
            with self.metrics.measure("documents"):
                documents = await DocumentRepository(session).get_many(
                    [UUID(hit.document_id) for hit in hits]
                )
            if any(
                (filters.city is not None and doc.city != filters.city)
                or (
                    filters.work_format is not None
                    and doc.work_format != filters.work_format
                )
                or (
                    filters.experience is not None
                    and doc.experience != filters.experience
                )
                for doc in documents
            ):
                raise SearchUnavailableError("Фильтры кешированной выдачи несовместимы")
            result = [
                SearchHit(hit.document_id, hit.score, doc.text, document_details(doc))
                for hit, doc in zip(hits, documents, strict=True)
            ]
            if not cached:
                await self.cache.put(key, hits)
            return result

    async def close(self) -> None:
        """Не принимает новую работу и ждёт завершения вычислений до закрытия БД"""
        self.closing = True
        self.ready = False
        self.metrics.ready.set(0)
        await self.queue.close()
        self.index = None
        self.dense = None
        self.reranker = None
        self.chunks.clear()


def load_search_config(path: Path) -> SearchConfig:
    """Читает проверенный YAML выпуска перед сборкой поиска"""
    return SearchConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
