"""Построение эмбеддингов с возобновлением и загрузка в PostgreSQL"""

import asyncio
import fcntl
import hashlib
import json
from time import perf_counter
from typing import Any

from nlp_ranker.domain.documents import DocumentInput
from nlp_ranker.ml.artifacts import read_embeddings, write_embeddings
from nlp_ranker.ml.config import SearchConfig
from nlp_ranker.ml.inference.encoder import Encoder
from nlp_ranker.server.repository.postgres.embeddings import import_embeddings


def build_embeddings(config: dict[str, object]) -> None:
    """Возобновляет завершённые пакеты и публикует полный артефакт корпуса"""
    import pyarrow.parquet as pq

    settings = SearchConfig.model_validate(config)
    path = settings.dense.artifact_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with settings.bm25.corpus_path.open("rb") as stream:
            corpus_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            documents = pq.read_table(stream).to_pylist()
        dense_config = settings.dense.model_dump(mode="json")
        identity = {"dense": dense_config, "corpus_sha256": corpus_hash}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        metadata = {
            **identity,
            "index_key": key,
            "data_version": settings.data_version,
            "distance": "cosine",
            "aggregation": "max_chunk_score",
        }
        if path.exists():
            _, previous = read_embeddings(path)
            if previous["index_key"] == key:
                return
        encoder = Encoder(settings.dense)
        chunks = []
        for doc in documents:
            content_hash = DocumentInput.model_validate(
                {
                    "id": doc["document_id"],
                    **{
                        name: doc.get(name)
                        for name in DocumentInput.model_fields
                        if name != "id"
                    },
                }
            ).database_values()["content_hash"]
            for number, text in enumerate(encoder.chunks(doc["title"], doc["text"])):
                chunks.append(
                    {
                        "document_id": doc["document_id"],
                        "chunk_number": number,
                        "content_hash": content_hash,
                        "text": text,
                    }
                )
        checkpoint = path.with_suffix(".partial.parquet")
        rows: list[dict[str, Any]] = []
        if checkpoint.exists():
            rows, previous = read_embeddings(checkpoint)
            if previous["index_key"] != key:
                raise ValueError("Checkpoint относится к другому корпусу или модели")
            if [dict(row, embedding=None) for row in rows] != [
                dict(row, embedding=None) for row in chunks[: len(rows)]
            ]:
                raise ValueError("Checkpoint не совпадает с порядком фрагментов")
        started = perf_counter()
        resumed = len(rows)
        for start in range(resumed, len(chunks), settings.dense.batch_size):
            batch = chunks[start : start + settings.dense.batch_size]
            vectors = encoder.encode([row["text"] for row in batch], "passage: ")
            rows.extend(
                dict(row, embedding=vector)
                for row, vector in zip(batch, vectors, strict=True)
            )
            write_embeddings(checkpoint, rows, metadata)
        metadata["build"] = {
            "seconds": perf_counter() - started,
            "resumed_chunks": resumed,
            "documents": len(documents),
            "chunks": len(rows),
        }
        write_embeddings(path, rows, metadata)
        checkpoint.unlink(missing_ok=True)


async def load_embeddings(config: dict[str, object]) -> None:
    """Загружает готовый артефакт в собственной транзакции и закрывает пул"""
    from nlp_ranker.core.config import Settings
    from nlp_ranker.server.repository.postgres.connection import (
        create_engine,
        create_session_factory,
    )

    settings = SearchConfig.model_validate(config)
    engine = create_engine(Settings())
    try:
        async with create_session_factory(engine)() as session, session.begin():
            await import_embeddings(session, settings.dense.artifact_path)
    finally:
        await engine.dispose()


def load_dense(config: dict[str, object]) -> None:
    """Восстанавливает PostgreSQL из проверенного артефакта"""
    asyncio.run(load_embeddings(config))
