"""Проверки фрагментов, возобновления, фильтров и объединения поиска"""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import yaml
from sqlalchemy import text

from nlp_ranker.domain.documents import DocumentInput
from nlp_ranker.domain.search import SearchHit
from nlp_ranker.ml.artifacts import read_embeddings, write_embeddings
from nlp_ranker.ml.config import DenseConfig, RetrievalConfig
from nlp_ranker.ml.inference.encoder import Encoder
from nlp_ranker.ml.retrieval.hybrid import fuse_results
from nlp_ranker.pipelines.embeddings import build_embeddings
from nlp_ranker.server.repository.postgres.connection import (
    create_engine,
    create_session_factory,
)
from nlp_ranker.server.repository.postgres.documents import DocumentRepository
from nlp_ranker.server.repository.postgres.embeddings import import_embeddings
from nlp_ranker.server.service.dense import DenseSearch

pytest.importorskip("pyarrow")


def hit(doc_id, score):
    """Создаёт кандидата без загрузки моделей"""
    return SearchHit(doc_id, score, "")


def test_fusion_deduplicates_and_handles_empty_sources():
    """Проверяет ручной RRF и нормированную сумму при равных оценках"""
    config = RetrievalConfig(candidate_limit=100, fusion="rrf", rrf_k=60)
    hits = fuse_results(
        [hit("b", 8), hit("a", 2), hit("b", 1)], [hit("a", 0.9)], config
    )
    assert [row.document_id for row in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fuse_results([], [], config) == []
    assert [row.document_id for row in fuse_results([], [hit("x", 1)], config)] == ["x"]
    config.fusion = "weighted_sum"
    config.dense_weight = 0.0
    assert fuse_results([], [hit("x", 1)], config) == []
    assert [
        row.document_id for row in fuse_results([hit("b", 1), hit("a", 1)], [], config)
    ] == ["a", "b"]


def test_embedding_resume_after_interrupted_batch(tmp_path, monkeypatch):
    """После сбоя повторно считает только незавершённый пакет"""
    import pyarrow.parquet as pq

    import nlp_ranker.pipelines.embeddings as embeddings

    doc = DocumentInput(
        id=uuid4(), source="test", source_id="1", title="Название", text="Текст"
    )
    corpus = tmp_path / "corpus.parquet"
    import pyarrow as pa

    pq.write_table(
        pa.Table.from_pylist(
            [dict(doc.model_dump(mode="json"), document_id=str(doc.id))]
        ),
        corpus,
    )
    config = yaml.safe_load(Path("configs/search.yaml").read_text())
    config["bm25"]["corpus_path"] = str(corpus)
    config["dense"]["artifact_path"] = str(tmp_path / "vectors.parquet")
    config["dense"]["batch_size"] = 1
    calls = []

    class FakeEncoder:
        """Останавливает первый запуск между двумя пакетами"""

        def __init__(self, config):
            """Принимает параметры выбранного этапа"""
            pass

        def chunks(self, title, text):
            """Создаёт два различимых фрагмента"""
            return ["первый", "второй"]

        def encode(self, texts, prefix):
            """Воспроизводит сбой на втором пакете один раз"""
            calls.append(texts)
            if len(calls) == 2:
                raise RuntimeError("Прервано")
            return [[1.0] + [0.0] * 383]

    monkeypatch.setattr(embeddings, "Encoder", FakeEncoder)
    with pytest.raises(RuntimeError):
        build_embeddings(config)
    build_embeddings(config)
    rows, metadata = read_embeddings(Path(config["dense"]["artifact_path"]))
    assert calls == [["первый"], ["второй"], ["второй"]]
    assert len(rows) == 2
    assert metadata["build"]["resumed_chunks"] == 1
    assert not (tmp_path / "vectors.partial.parquet").exists()


async def test_cancelled_encoding_keeps_inference_slot():
    """Отмена ожидающего запроса не запускает второй инференс раньше первого"""
    import threading

    started = threading.Event()
    finish = threading.Event()
    calls = []
    encoder = object.__new__(Encoder)
    encoder.lock = asyncio.Lock()

    def encode(texts, prefix):
        """Блокирует первый инференс до разрешения теста"""
        calls.append(texts[0])
        started.set()
        assert finish.wait(5)
        return [[1.0]]

    encoder.encode = encode
    first = asyncio.create_task(encoder.query("первый"))
    await asyncio.to_thread(started.wait, 5)
    first.cancel()
    second = asyncio.create_task(encoder.query("второй"))
    await asyncio.sleep(0.02)
    assert calls == ["первый"]
    first.cancel()
    await asyncio.sleep(0.02)
    assert calls == ["первый"]
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second == [1.0]


@pytest.mark.integration
async def test_pgvector_filters_chunks_repeat_import_and_stale_document(tmp_path):
    """Проверяет точный поиск, фильтр до top-k и отказ при изменении корпуса"""
    from nlp_ranker.core.config import Settings

    dsn = os.environ.get("NLP_RANKER_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Нужен тестовый PostgreSQL")
    assert dsn.endswith("/nlp_ranker_test")
    engine = create_engine(Settings(_env_file=None, postgres_dsn=dsn))
    sessions = create_session_factory(engine)
    first = DocumentInput(
        id=uuid4(),
        source="test",
        source_id="1",
        title="Python",
        text="SQL",
        city="Москва",
    )
    second = DocumentInput(
        id=uuid4(),
        source="test",
        source_id="2",
        title="Python",
        text="SQL",
        city="Казань",
    )
    rows = []
    for doc, vectors in [(first, [[1.0, 0.0], [1.0, 0.0]]), (second, [[0.8, 0.6]])]:
        for number, vector in enumerate(vectors):
            rows.append(
                {
                    "document_id": str(doc.id),
                    "chunk_number": number,
                    "content_hash": doc.database_values()["content_hash"],
                    "text": doc.text,
                    "embedding": vector + [0.0] * 382,
                }
            )
    config = DenseConfig(model_name=None, revision=None, device="cpu")
    metadata = {
        "index_key": "test",
        "dense": config.model_dump(mode="json"),
        "build": {"chunks": 3},
    }
    path = tmp_path / "vectors.parquet"
    write_embeddings(path, rows, metadata)
    try:
        async with sessions() as session, session.begin():
            await session.execute(text("TRUNCATE documents CASCADE"))
            await DocumentRepository(session).save_many([first, second])
        for _ in range(2):
            async with sessions() as session, session.begin():
                await import_embeddings(session, path)
        search = DenseSearch(SimpleNamespace(config=config), sessions, metadata)
        vector = [1.0] + [0.0] * 383
        assert [row.document_id for row in await search.search_vector(vector)] == [
            str(first.id),
            str(second.id),
        ]
        assert [
            row.document_id
            for row in await search.search_vector(vector, 1, city="Казань")
        ] == [str(second.id)]
        assert await search.search_vector(vector, city="Сочи") == []
        async with sessions() as session, session.begin():
            await DocumentRepository(session).replace(
                first.model_copy(update={"text": "Новый текст"}), 1
            )
        with pytest.raises(ValueError, match="неполон"):
            await search.search_vector(vector)
        with pytest.raises(ValueError, match="не совпадают"):
            async with sessions() as session, session.begin():
                await import_embeddings(session, path)
    finally:
        await engine.dispose()


def test_reranker_deduplicates_and_uses_best_fragment():
    """Проверяет max по фрагментам, равные оценки и ограничение длинного входа"""
    import numpy as np

    from nlp_ranker.ml.config import RerankingConfig
    from nlp_ranker.ml.inference.reranker import Reranker

    class Tokenizer:
        """Разбивает тестовый текст по словам"""

        def encode(self, text, **kwargs):
            """Возвращает слова вместо числовых токенов"""
            return text.split()

        def decode(self, tokens, **kwargs):
            """Соединяет тестовые токены"""
            return " ".join(tokens)

    class Model:
        """Сохраняет пары и возвращает заданные оценки"""

        tokenizer = Tokenizer()

        def predict(self, pairs, **kwargs):
            """Проверяет лимит пар и задаёт равные лучшие оценки двух документов"""
            assert len(pairs) == 3
            assert all(
                len(query.split()) + len(text.split()) <= 512 for query, text in pairs
            )
            return np.array([0.0, 2.0, 2.0])

    reranker = object.__new__(Reranker)
    reranker.config = RerankingConfig(enabled=False, top_k=10)
    reranker.model = Model()
    chunks = {
        "a": ["Заголовок\n" + "слово " * 1000, "Другой фрагмент"],
        "b": ["Название"],
    }
    assert [
        row.document_id
        for row in reranker.rank(
            "запрос " * 1000, [hit("a", 1), hit("b", 1), hit("a", 0)], chunks
        )
    ] == ["a", "b"]
    assert reranker.rank("запрос", [], {}) == []
    assert reranker.rank(" ", [hit("a", 1)], {}) == []
    with pytest.raises(LookupError):
        reranker.rank("запрос", [hit("c", 1)], chunks)
