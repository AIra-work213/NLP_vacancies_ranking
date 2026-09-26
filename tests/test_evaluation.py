"""Проверки метрик на понятных примерах и сопоставимости отчётов"""

from math import log2

import pytest

pytest.importorskip("ranx")

from nlp_ranker.ml.evaluation.metrics import compare_reports, query_metrics


def test_metrics_on_known_example():
    """Сверяет линейный NDCG, MRR и Recall с ручным расчётом"""
    metrics = query_metrics(
        {"best": 2, "partial": 1, "bad": 0}, ["bad", "partial", "best"]
    )
    expected = (1 / log2(3) + 2 / log2(4)) / (2 + 1 / log2(3))
    assert metrics["ndcg_at_10"] == pytest.approx(expected)
    assert metrics["mrr_at_10"] == 0.5
    assert metrics["recall_at_100"] == 1


def test_unknown_results_and_short_ranking():
    """Отклоняет неизвестные top-10 и допускает настоящую короткую выдачу"""
    with pytest.raises(ValueError, match="Дополните"):
        query_metrics({"known": 2}, ["unknown"])
    assert query_metrics({"known": 2}, ["known"])["ndcg_at_10"] == 1
    assert query_metrics({"known": 2}, []) == {
        "ndcg_at_10": 0,
        "mrr_at_10": 0,
        "recall_at_100": 0,
    }
    assert query_metrics({"known": 0}, ["known"])["ndcg_at_10"] == 0


def test_recall_is_relative_to_known_judgments():
    """Считает известные положительные документы и показывает неполный отбор"""
    assert query_metrics({"first": 1, "second": 2}, ["first"])["recall_at_100"] == 0.5


def test_paired_comparison_checks_inputs():
    """Парное сравнение повторяемо и отклоняет другие версии разметки"""
    report = {
        "queries_sha256": "queries",
        "qrels_sha256": "labels",
        "corpus_sha256": "corpus",
        "split": "validation",
        "metric_definition": {"gain": "linear"},
        "per_query": {"q1": {"ndcg_at_10": 0.5}, "q2": {"ndcg_at_10": 0.8}},
    }
    assert compare_reports(report, report) == {
        "ndcg_delta": 0,
        "ci95_low": 0,
        "ci95_high": 0,
    }
    with pytest.raises(ValueError, match="другим данным"):
        compare_reports(report, dict(report, qrels_sha256="other"))


def test_corpus_compatibility_and_quality_gate(tmp_path):
    """Проверяет отказ при регрессии и несовместимом корпусе на настоящих файлах"""
    import json

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from nlp_ranker.ml.config import BM25Config
    from nlp_ranker.ml.evaluation.metrics import checksum
    from nlp_ranker.ml.retrieval.bm25 import BM25Index
    from nlp_ranker.pipelines.evaluate import evaluate

    corpus = tmp_path / "documents.parquet"
    queries = tmp_path / "queries.parquet"
    labels = tmp_path / "qrels.parquet"
    index_path = tmp_path / "index.zip"
    report_path = tmp_path / "metrics.json"
    rows = [
        {"document_id": "wrong", "title": "Python Python", "text": "Python SQL"},
        {"document_id": "good", "title": "Разработчик", "text": "Python SQL"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), corpus)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "query_id": "q",
                    "text": "Python",
                    "category": "role",
                    "split": "validation",
                }
            ]
        ),
        queries,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"query_id": "q", "document_id": "wrong", "relevance": 0},
                {"query_id": "q", "document_id": "good", "relevance": 2},
            ]
        ),
        labels,
    )
    BM25Index.build(rows, BM25Config(), "toy", checksum(corpus)).save(index_path)
    config = dict(
        schema_version=1,
        data_version="toy",
        dataset_path=str(queries),
        qrels_path=str(labels),
        corpus_path=str(corpus),
        index_path=str(index_path),
        metrics_path=str(report_path),
        metrics=["ndcg_at_10", "mrr_at_10", "recall_at_100"],
        latency_repeats=1,
        quality_gate=dict(baseline_metrics_path=None, max_ndcg_regression=None),
    )
    evaluate(config)
    report = json.loads(report_path.read_text())
    baseline = tmp_path / "baseline.json"
    ideal = query_metrics({"wrong": 0, "good": 2}, ["good", "wrong"])
    report["per_query"]["q"].update(ideal)
    report["metrics"] = ideal
    baseline.write_text(json.dumps(report))
    config["quality_gate"] = dict(
        baseline_metrics_path=str(baseline), max_ndcg_regression=0.0
    )
    with pytest.raises(ValueError, match="NDCG ниже"):
        evaluate(config)
    assert json.loads(report_path.read_text())["release_status"] == "rejected"
    rows[0]["text"] = "Другой корпус"
    pq.write_table(pa.Table.from_pylist(rows), corpus)
    with pytest.raises(ValueError, match="другой версии корпуса"):
        evaluate(config)
