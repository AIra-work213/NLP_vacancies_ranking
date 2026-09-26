"""Запуск offline-оценки поиска с записью отчёта"""

import asyncio
import hashlib
import json
import os
import platform
from contextlib import AsyncExitStack
from math import ceil
from statistics import mean, median
from time import perf_counter
from typing import Any

from nlp_ranker.ml.evaluation.config import EvaluationConfig
from nlp_ranker.ml.evaluation.metrics import checksum, compare_reports, query_metrics


def evaluate(config: dict[str, object]) -> None:
    """Оценивает выбранный поиск и закрывает созданные ресурсы при любой ошибке"""

    async def run() -> None:
        """Владеет пулом PostgreSQL в пределах одного эксперимента"""
        async with AsyncExitStack() as resources:
            await evaluate_search(config, resources)

    asyncio.run(run())


async def evaluate_search(config: dict[str, object], resources: AsyncExitStack) -> None:
    """Сохраняет качество, полную выдачу и задержки выбранного offline-поиска"""
    import pyarrow.parquet as pq

    from nlp_ranker.ml.retrieval.bm25 import BM25Index

    settings = EvaluationConfig.model_validate(config)
    started = perf_counter()
    index = BM25Index.load(settings.index_path)
    corpus_hash = checksum(settings.corpus_path)
    if index.metadata["corpus_sha256"] != corpus_hash:
        raise ValueError("Индекс построен на другой версии корпуса")
    if (
        settings.data_version is not None
        and index.metadata["data_version"] != settings.data_version
    ):
        raise ValueError("Версия данных в конфиге не совпадает с индексом")
    from nlp_ranker.domain.search import SearchHit
    from nlp_ranker.ml.config import SearchConfig
    from nlp_ranker.ml.retrieval.hybrid import fuse_results

    dense = None
    reranker = None
    search_config = None
    chunks: dict[str, list[str]] = {}
    extra: dict[str, Any] = {"method": settings.method}
    if settings.method != "bm25":
        import yaml

        from nlp_ranker.core.config import Settings
        from nlp_ranker.ml.artifacts import read_embeddings
        from nlp_ranker.ml.inference.encoder import Encoder
        from nlp_ranker.server.repository.postgres.connection import (
            create_engine,
            create_session_factory,
        )
        from nlp_ranker.server.service.dense import DenseSearch

        search_config = SearchConfig.model_validate(
            yaml.safe_load(settings.search_config_path.read_text(encoding="utf-8"))
        )
        rows, metadata = read_embeddings(search_config.dense.artifact_path)
        if metadata["corpus_sha256"] != corpus_hash:
            raise ValueError("Эмбеддинги относятся к другому корпусу")
        encoder = await asyncio.to_thread(Encoder, search_config.dense)
        engine = create_engine(Settings())
        resources.push_async_callback(engine.dispose)
        dense = DenseSearch(encoder, create_session_factory(engine), metadata)
        for row in rows:
            chunks.setdefault(row["document_id"], []).append(row["text"])
        extra.update(
            search_config=search_config.model_dump(mode="json"),
            embeddings_sha256=metadata["artifact_sha256"],
            dense_index_key=metadata["index_key"],
        )
        if settings.method == "rerank":
            from nlp_ranker.ml.inference.reranker import Reranker

            reranker = await asyncio.to_thread(Reranker, search_config.reranking)

    async def search(query: str) -> list[SearchHit]:
        """Исполняет явный режим без скрытого перехода на другой алгоритм"""
        if settings.method == "bm25":
            return await asyncio.to_thread(index.search, query, 100)
        if dense is None or search_config is None:
            raise ValueError("Dense-поиск не настроен")
        candidate_limit = search_config.retrieval.candidate_limit
        if settings.method == "dense":
            return await dense.search(query, 100)
        lexical, semantic = await asyncio.gather(
            asyncio.to_thread(index.search, query, candidate_limit),
            dense.search(query, candidate_limit),
        )
        hits = fuse_results(lexical, semantic, search_config.retrieval, 100)
        if reranker is not None:
            candidates = hits[: settings.rerank_candidates]
            ranked = await reranker.search(
                query, candidates, chunks, settings.rerank_candidates
            )
            return ranked + hits[len(candidates) :]
        return hits

    load_ms = (perf_counter() - started) * 1000
    with settings.dataset_path.open("rb") as stream:
        queries_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        queries = pq.read_table(stream).to_pylist()
    queries = [
        row
        for row in queries
        if settings.split == "all" or row["split"] == settings.split
    ]
    if not queries or len({row["query_id"] for row in queries}) != len(queries):
        raise ValueError("Для оценки нужны непустые уникальные запросы")
    with settings.qrels_path.open("rb") as stream:
        qrels_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        labels = pq.read_table(stream).to_pylist()
    judgments: dict[str, dict[str, int]] = {row["query_id"]: {} for row in queries}
    for row in labels:
        if row["query_id"] in judgments:
            query_labels = judgments[row["query_id"]]
            if row["document_id"] in query_labels:
                raise ValueError("Оценка пары запрос/документ повторяется")
            query_labels[row["document_id"]] = row["relevance"]
    per_query: dict[str, dict[str, Any]] = {}
    timings = []
    for query in queries:
        hits = await search(query["text"])
        ids = [hit.document_id for hit in hits]
        metrics = query_metrics(judgments[query["query_id"]], ids)
        durations = []
        for _ in range(settings.latency_repeats):
            started = perf_counter()
            await search(query["text"])
            durations.append((perf_counter() - started) * 1000)
        timings.extend(durations)
        per_query[query["query_id"]] = {
            **{name: metrics[name] for name in settings.metrics},
            "text": query["text"],
            "category": query["category"],
            "split": query["split"],
            "top10_judged": len(ids[:10]),
            "retrieved": len(ids),
            "judged_at_100": sum(doc in judgments[query["query_id"]] for doc in ids),
            "latency_median_ms": median(durations),
            "hits": [
                {"document_id": hit.document_id, "score": hit.score} for hit in hits
            ],
        }
    report: dict[str, Any] = {
        **extra,
        "schema_version": 1,
        "config": settings.model_dump(mode="json"),
        "split": settings.split,
        "queries_sha256": queries_hash,
        "qrels_sha256": qrels_hash,
        "corpus_sha256": corpus_hash,
        "index_sha256": index.archive_sha256,
        "metric_definition": {
            "ndcg_gain": "linear",
            "relevance_threshold": 1,
            "recall_scope": "known_judgments",
        },
        "metrics": {
            name: mean(row[name] for row in per_query.values())
            for name in settings.metrics
        },
        "per_query": per_query,
        "performance": {
            "index_load_ms": load_ms,
            "search_p50_ms": median(timings),
            "search_p95_ms": sorted(timings)[ceil(len(timings) * 0.95) - 1],
            "percentile_method": "nearest_rank",
            "samples": len(timings),
            "warmup_per_query": 1,
            "corpus_documents": len(index.documents),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
        },
        "release_status": "baseline_measurement",
    }
    gate = settings.quality_gate
    if gate.baseline_metrics_path is not None:
        baseline = json.loads(gate.baseline_metrics_path.read_text(encoding="utf-8"))
        report["comparison"] = compare_reports(report, baseline, settings.seed)
        tolerance = gate.max_ndcg_regression or 0.0
        report["release_status"] = (
            "passed" if report["comparison"]["ndcg_delta"] >= -tolerance else "rejected"
        )
    settings.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if report["release_status"] == "rejected":
        raise ValueError("NDCG ниже допуска относительно базового отчёта")
