"""Сравнение готовых энкодеров на CPU и общем пуле валидационной разметки"""

import argparse
import csv
import gc
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import mlflow
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

from nlp_ranker.domain.search import SearchHit
from nlp_ranker.ml.artifacts import read_embeddings
from nlp_ranker.ml.config import SearchConfig
from nlp_ranker.ml.evaluation.metrics import checksum, query_metrics
from nlp_ranker.ml.retrieval.bm25 import BM25Index
from nlp_ranker.ml.retrieval.hybrid import fuse_results

MODELS = [
    "intfloat/multilingual-e5-small",
    "cointegrated/rubert-tiny2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]
EXPERIMENT = "E5-small / MiniLM-L12-v2 / ruBERT-tiny2 — сравнение моделей на CPU"
OUTPUT = Path("artifacts/reports/encoder-comparison")


def chunks(model, documents):
    """Делит тексты с учётом окна модели, сохраняя заголовки в каждом фрагменте"""
    tokenizer = model.tokenizer
    size = min(384, model.max_seq_length - 64)
    overlap = min(64, size // 4)
    rows = []
    for document in documents:
        heading = tokenizer.decode(
            tokenizer.encode(document["title"], add_special_tokens=False)[:48]
        )
        tokens = tokenizer.encode(document["text"], add_special_tokens=False)
        for start in range(0, max(1, len(tokens)), size - overlap):
            rows.append(
                {
                    "document_id": document["document_id"],
                    "text": heading
                    + "\n"
                    + tokenizer.decode(tokens[start : start + size]),
                }
            )
            if start + size >= len(tokens):
                break
    return rows, size, overlap


def retrieve(rows, vectors, query):
    """Объединяет cosine оценки фрагментов максимумом и разрешает ничьи по ID"""
    scores = {}
    for row, score in zip(rows, vectors @ query, strict=True):
        document_id = row["document_id"]
        scores[document_id] = max(scores.get(document_id, -1), float(score))
    ordered = sorted(
        scores, key=lambda document_id: (-scores[document_id], document_id)
    )
    return [
        SearchHit(document_id, scores[document_id], "") for document_id in ordered[:100]
    ]


def collect(config, documents, queries):
    """Сохраняет реальные выдачи всех моделей, не вычисляя метрики с пробелами"""
    index = BM25Index.load(config.bm25.index_path)
    rankings = {
        "BM25": {
            query["query_id"]: [
                hit.document_id for hit in index.search(query["text"], 100)
            ]
            for query in queries
        }
    }
    details = {}
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    for name in MODELS:
        short = name.split("/")[-1]
        revision = (
            config.dense.revision if name == MODELS[0] else HfApi().model_info(name).sha
        )
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(
            run_name=f"{short} — индексация и поиск, validation"
        ) as run:
            print(f"Модель: {name}", flush=True)
            model = SentenceTransformer(
                name, revision=revision, device="cpu", trust_remote_code=False
            )
            model.max_seq_length = min(model.max_seq_length, 512)
            model.eval()
            started = perf_counter()
            if name == MODELS[0]:
                rows, _metadata = read_embeddings(config.dense.artifact_path)
                vectors = np.asarray(
                    [row["embedding"] for row in rows], dtype=np.float32
                )
                size, overlap = 384, 64
                prefix = "query: "
                reused = True
            else:
                rows, size, overlap = chunks(model, documents)
                with torch.inference_mode():
                    vectors = model.encode(
                        [row["text"] for row in rows],
                        batch_size=8,
                        normalize_embeddings=True,
                        show_progress_bar=True,
                    )
                prefix = ""
                reused = False
            build_seconds = perf_counter() - started
            samples = []
            dense_ranks, hybrid_ranks = {}, {}
            with torch.inference_mode():
                model.encode([prefix + queries[0]["text"]], normalize_embeddings=True)
                for query in queries:
                    query_id = query["query_id"]
                    for repeat in range(10):
                        started = perf_counter()
                        vector = model.encode(
                            [prefix + query["text"]], normalize_embeddings=True
                        )[0]
                        dense = retrieve(rows, vectors, vector)
                        elapsed = (perf_counter() - started) * 1000
                        samples.append(elapsed)
                        if repeat == 0:
                            dense_ranks[query_id] = [hit.document_id for hit in dense]
                            lexical = index.search(query["text"], 50)
                            hybrid_ranks[query_id] = [
                                hit.document_id
                                for hit in fuse_results(
                                    lexical, dense[:50], config.retrieval
                                )
                            ]
            params = {
                "model_name": name,
                "revision": revision,
                "split": "validation",
                "backend": "numpy_exact_cosine",
                "aggregation": "max_chunk_score",
                "dimensions": vectors.shape[1],
                "max_length": model.max_seq_length,
                "chunk_tokens": size,
                "overlap_tokens": overlap,
                "threads": 4,
                "batch_size": 8,
                "query_prefix": prefix,
                "corpus_vectors_reused": reused,
                "corpus_sha256": checksum(config.bm25.corpus_path),
                "queries_sha256": checksum(Path("data/processed/queries.parquet")),
                "rrf_k": 60,
                "candidate_limit": 50,
            }
            performance = {
                "dense_p50_ms": float(np.median(samples)),
                "dense_p95_ms": float(
                    np.quantile(samples, 0.95, method="inverted_cdf")
                ),
                "build_seconds": build_seconds,
            }
            mlflow.log_params(params)
            mlflow.log_metrics(performance)
            mlflow.log_dict(
                {"dense": dense_ranks, "hybrid": hybrid_ranks}, "rankings.json"
            )
            mlflow.log_artifact(__file__, "source")
            details[short] = {
                "params": params,
                "performance": performance,
                "retrieval_run_id": run.info.run_id,
            }
            rankings[f"{short} — dense"] = dense_ranks
            rankings[f"{short} — BM25 + RRF"] = hybrid_ranks
        del model, vectors, rows
        gc.collect()
    (OUTPUT / "rankings.json").write_text(
        json.dumps(
            {"rankings": rankings, "models": details}, ensure_ascii=False, indent=2
        )
    )


def report(documents, queries):
    """Требует общую полную разметку top-10, затем записывает сравнение в MLflow"""
    saved = json.loads((OUTPUT / "rankings.json").read_text())
    labels = pq.read_table("data/processed/qrels.parquet").to_pylist()
    additions = OUTPUT / "additional-qrels.csv"
    if additions.exists():
        with additions.open(newline="") as stream:
            labels.extend(
                dict(row, relevance=int(row["relevance"]))
                for row in csv.DictReader(stream)
            )
    judgments = {
        query["query_id"]: {
            row["document_id"]: row["relevance"]
            for row in labels
            if row["query_id"] == query["query_id"]
        }
        for query in queries
    }
    by_id = {document["document_id"]: document for document in documents}
    missing = {}
    for ranks in saved["rankings"].values():
        for query_id, ranked in ranks.items():
            for document_id in ranked[:10]:
                if document_id not in judgments[query_id]:
                    missing[(query_id, document_id)] = {
                        "query_id": query_id,
                        "document": by_id[document_id],
                    }
    (OUTPUT / "unjudged.json").write_text(
        json.dumps(list(missing.values()), ensure_ascii=False, indent=2)
    )
    if missing:
        print(f"До сравнения нужно разметить {len(missing)} новых пар", flush=True)
        return
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(
        run_name="E5-small / MiniLM-L12 / ruBERT-tiny2 — итоговое сравнение, validation"
    ) as parent:
        mlflow.log_dict(judgments, "evaluation/common-qrels.json")
        mlflow.log_artifact(__file__, "source")
        mlflow.log_artifact("configs/search.yaml", "config")
        mlflow.log_artifact("data/queries.csv", "evaluation")
        mlflow.log_artifact("data/qrels.csv", "evaluation")
        if additions.exists():
            mlflow.log_artifact(str(additions), "evaluation")
        mlflow.log_artifact(str(OUTPUT / "rankings.json"))
        summary = {}
        for name, ranks in saved["rankings"].items():
            per_query = {
                query_id: query_metrics(judgments[query_id], ranked)
                for query_id, ranked in ranks.items()
            }
            metrics = {
                metric: mean(values[metric] for values in per_query.values())
                for metric in ["ndcg_at_10", "mrr_at_10", "recall_at_100"]
            }
            with mlflow.start_run(
                run_name=f"{name} — оценка, validation", nested=True
            ) as run:
                mlflow.log_params(
                    {
                        "split": "validation",
                        "backend": "bm25" if name == "BM25" else "numpy_exact_cosine",
                        "annotator": "codex",
                        "queries": len(queries),
                        "judgments": sum(len(values) for values in judgments.values()),
                    }
                )
                if name != "BM25":
                    short = name.split(" — ")[0]
                    mlflow.log_params(saved["models"][short]["params"])
                    mlflow.set_tag(
                        "retrieval_run_id", saved["models"][short]["retrieval_run_id"]
                    )
                mlflow.log_metrics(metrics)
                mlflow.log_dict(per_query, "evaluation/per-query.json")
                mlflow.log_dict(judgments, "evaluation/common-qrels.json")
                summary[name] = {"metrics": metrics, "run_id": run.info.run_id}
        result = {
            "summary": summary,
            "models": saved["models"],
            "comparison_run_id": parent.info.run_id,
            "experiment_id": parent.info.experiment_id,
        }
        mlflow.log_dict(result, "evaluation/comparison.json")
        (OUTPUT / "comparison.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    """Запускает инференс или пересчитывает метрики по сохранённым выдачам"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("http://localhost:5000")
    config = SearchConfig.model_validate(
        yaml.safe_load(Path("configs/search.yaml").read_text())
    )
    documents = pq.read_table(config.bm25.corpus_path).to_pylist()
    queries = [
        row
        for row in pq.read_table("data/processed/queries.parquet").to_pylist()
        if row["split"] == "validation"
    ]
    if not args.report_only:
        collect(config, documents, queries)
    report(documents, queries)


if __name__ == "__main__":
    main()
