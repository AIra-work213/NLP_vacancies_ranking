"""Построение индекса BM25 из подготовленного корпуса"""

import hashlib

from nlp_ranker.ml.config import SearchConfig
from nlp_ranker.ml.retrieval.bm25 import BM25Index


def build_index(config: dict[str, object]) -> None:
    """Строит первый BM25-индекс из подготовленного корпуса"""
    import pyarrow.parquet as pq

    settings = SearchConfig.model_validate(config)
    with settings.bm25.corpus_path.open("rb") as stream:
        corpus_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        documents = pq.read_table(stream).to_pylist()
    index = BM25Index.build(
        documents, settings.bm25, settings.data_version, corpus_hash
    )
    index.save(settings.bm25.index_path)
