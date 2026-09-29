"""Загрузка совместимых индексов и моделей без подключения к HTTP и PostgreSQL"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from nlp_ranker.ml.artifacts import read_embeddings
from nlp_ranker.ml.config import SearchConfig
from nlp_ranker.ml.inference.encoder import Encoder
from nlp_ranker.ml.inference.reranker import Reranker
from nlp_ranker.ml.retrieval.bm25 import BM25Index


@dataclass(slots=True)
class SearchRelease:
    """Объединяет согласованные ресурсы одного поискового выпуска"""

    index: BM25Index
    encoder: Encoder
    metadata: dict[str, Any]
    reranker: Reranker | None
    hashes: dict[UUID, str]
    chunks: dict[str, list[str]]
    document_filters: dict[str, dict[str, object]]
    version: str


def load_release(config: SearchConfig) -> SearchRelease:
    """Проверяет артефакты и загружает модели один раз на процесс"""
    hashes: dict[UUID, str] = {}
    chunks: dict[str, list[str]] = {}
    index = BM25Index.load(config.bm25.index_path)
    rows, metadata = read_embeddings(config.dense.artifact_path)
    if (
        index.metadata["corpus_sha256"] != metadata["corpus_sha256"]
        or index.metadata["bm25"] != config.bm25.model_dump(mode="json")
        or index.metadata["data_version"] != config.data_version
        or metadata["dense"] != config.dense.model_dump(mode="json")
    ):
        raise ValueError("Индекс, эмбеддинги и конфигурация несовместимы")
    for row in rows:
        doc_id = UUID(row["document_id"])
        if doc_id in hashes and hashes[doc_id] != row["content_hash"]:
            raise ValueError("Фрагменты документа имеют разные версии")
        hashes[doc_id] = row["content_hash"]
        chunks.setdefault(str(doc_id), []).append(row["text"])
    if set(hashes) != {UUID(str(doc["document_id"])) for doc in index.documents}:
        raise ValueError("BM25 и dense содержат разные документы")
    encoder = Encoder(config.dense)
    reranker = None
    if config.reranking.enabled:
        reranker = Reranker(config.reranking)
    document_filters = {str(doc["document_id"]): doc for doc in index.documents}
    identity = {
        "schema_version": 1,
        "bm25": index.archive_sha256,
        "embeddings": metadata["artifact_sha256"],
        "config": config.model_dump(mode="json"),
    }
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    return SearchRelease(
        index, encoder, metadata, reranker, hashes, chunks, document_filters, version
    )
