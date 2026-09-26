"""Метрики релевантности и сравнение одинаковых наборов запросов"""

import hashlib
import random
from pathlib import Path
from statistics import mean
from typing import Any


def checksum(path: Path) -> str:
    """Вычисляет версию входного файла для проверки совместимости результатов"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def query_metrics(judgments: dict[str, int], ranking: list[str]) -> dict[str, float]:
    """Вычисляет ranx-метрики с линейным NDCG и порогом релевантности 1"""
    from ranx import evaluate as ranx_evaluate

    require_judged_results({"query": ranking}, {"query": judgments})
    if any(grade not in {0, 1, 2} for grade in judgments.values()):
        raise ValueError("Оценки должны быть 0, 1 или 2")
    names = {
        "ndcg@10": "ndcg_at_10",
        "mrr@10": "mrr_at_10",
        "recall@100": "recall_at_100",
    }
    if not ranking or not any(judgments.values()):
        return dict.fromkeys(names.values(), 0.0)
    run = {doc_id: len(ranking) - position for position, doc_id in enumerate(ranking)}
    values = ranx_evaluate({"query": judgments}, {"query": run}, list(names), threads=1)
    return {name: float(values[metric]) for metric, name in names.items()}


def compare_reports(
    current: dict[str, Any], baseline: dict[str, Any], seed: int = 42
) -> dict[str, float]:
    """Сравнивает одинаковые запросы парным бутстрэпом с фиксированным seed"""
    for field in [
        "queries_sha256",
        "qrels_sha256",
        "corpus_sha256",
        "split",
        "metric_definition",
    ]:
        if current[field] != baseline[field]:
            raise ValueError("Базовый отчёт относится к другим данным или метрикам")
    query_ids = sorted(current["per_query"])
    if not query_ids or query_ids != sorted(baseline["per_query"]):
        raise ValueError("В отчётах разные запросы")
    differences = [
        current["per_query"][qid]["ndcg_at_10"]
        - baseline["per_query"][qid]["ndcg_at_10"]
        for qid in query_ids
    ]
    generator = random.Random(seed)
    samples = sorted(
        mean(generator.choices(differences, k=len(differences))) for _ in range(2000)
    )
    return {
        "ndcg_delta": mean(differences),
        "ci95_low": samples[49],
        "ci95_high": samples[1949],
    }


def require_judged_results(
    results: dict[str, list[str]], judgments: dict[str, dict[str, int]], limit: int = 10
) -> None:
    """Запрещает оценку выдачи с неизвестными документами вместо подстановки нулей"""
    if limit < 1 or not results or set(results) != set(judgments):
        raise ValueError(
            "Выдача и оценки должны содержать один непустой набор запросов"
        )

    for query_id, ranked in results.items():
        if len(ranked) != len(set(ranked)):
            raise ValueError("Выдача не должна содержать повторов документов")
        if any(doc not in judgments[query_id] for doc in ranked[:limit]):
            raise ValueError(f"Дополните разметку top-{limit} для запроса {query_id}")
