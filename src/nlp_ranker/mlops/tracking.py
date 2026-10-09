"""Запись настоящих этапов и выпусков в MLflow без импорта ML-библиотек в API"""

import asyncio
import hashlib
import json
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from nlp_ranker.core.config import Settings

if TYPE_CHECKING:
    from mlflow import ActiveRun


def pipeline_names(stage: str, config: dict[str, object]) -> tuple[str, str]:
    """Указывает модель, цель этапа и разбиение в названиях MLflow"""
    import yaml

    from nlp_ranker.ml.config import SearchConfig

    model = "BM25" if stage == "index" or stage == "evaluate" else "NLP ranking"
    method = str(config.get("method", "bm25"))
    search_values = config
    if stage == "evaluate" and method != "bm25":
        path = Path(str(config["search_config_path"]))
        search_values = yaml.safe_load(path.read_text(encoding="utf-8"))

    if stage in {"embed", "load-dense"} or (stage == "evaluate" and method != "bm25"):
        search = SearchConfig.model_validate(search_values)
        model = (search.dense.model_name or "dense").split("/")[-1]
        if method == "rerank":
            model += " + " + (search.reranking.model_name or "reranker").split("/")[-1]

    purposes = {
        "embed": "подготовка эмбеддингов",
        "index": "построение индекса",
        "evaluate": "оценка поиска",
        "prepare": "подготовка данных",
        "load-dense": "загрузка векторов",
        "mine-negatives": "подготовка отрицательных примеров",
        "train": "обучение",
    }
    purpose = purposes.get(stage, stage)
    experiment = f"{model} - {purpose}"
    name = experiment
    if stage == "evaluate":
        name += f" - {method}, {config.get('split', 'validation')}"
    return experiment, name


@contextmanager
def tracked_run(
    settings: Settings, name: str, params: Mapping[str, str]
) -> Iterator["ActiveRun"]:
    """Создаёт запуск и отмечает ошибку этапа средствами MLflow"""
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(dict(params))
        yield run


def log_inputs(config: dict[str, object], config_path: Path) -> None:
    """Сохраняет эффективный конфиг, исходники и версии без зависимости от Git"""
    import mlflow

    mlflow.log_dict(config, "config/effective.json")
    mlflow.log_artifact(str(config_path), "config")
    mlflow.log_artifacts("configs", "config/files")
    source_hash = hashlib.sha256()
    package = Path(__file__).parents[1]
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        source_hash.update(str(relative).encode())
        source_hash.update(path.read_bytes())
        mlflow.log_artifact(str(path), str(Path("source/nlp_ranker") / relative.parent))
    mlflow.log_param("source_sha256", source_hash.hexdigest())
    for name in [
        "uv.lock",
        "dvc.lock",
        "dvc.yaml",
        ".dvc/config",
        "data/raw/vacancies_sample.csv.dvc",
    ]:
        path = Path(name)
        if path.exists():
            mlflow.log_artifact(str(path), "versions")


def log_results(stage: str, config: dict[str, object], settings: Settings) -> None:
    """Связывает индекс MinIO или отчёт оценки с текущим запуском"""
    import mlflow

    from nlp_ranker.ml.config import SearchConfig
    from nlp_ranker.ml.evaluation.config import EvaluationConfig
    from nlp_ranker.ml.retrieval.bm25 import BM25Index
    from nlp_ranker.server.repository.s3.artifacts import ArtifactStore

    if stage == "embed":
        from nlp_ranker.ml.artifacts import read_embeddings
        from nlp_ranker.ml.evaluation.metrics import checksum

        search = SearchConfig.model_validate(config)
        with TemporaryDirectory() as directory:
            snapshot = Path(directory) / "embeddings.parquet"
            shutil.copyfile(search.dense.artifact_path, snapshot)
            rows, metadata = read_embeddings(snapshot)
            if metadata["dense"] != search.dense.model_dump(mode="json"):
                raise ValueError("Артефакт заменён другим конфигом эмбеддингов")
            store = ArtifactStore(settings)

            async def publish_vectors() -> str:
                """Сохраняет согласованный снимок эмбеддингов в MinIO"""
                await store.ensure_bucket()
                return await store.publish(
                    snapshot, "indexes", search.data_version or "unversioned"
                )

            try:
                key = asyncio.run(publish_vectors())
            finally:
                store.close()
            artifact_hash = checksum(snapshot)
        mlflow.log_dict(
            {
                "manifest_key": key,
                "metadata": metadata,
                "embeddings_sha256": artifact_hash,
            },
            "dense-release.json",
        )
        mlflow.log_params(
            {
                "dense_index_key": metadata["index_key"],
                "embeddings_sha256": artifact_hash,
                "model_name": search.dense.model_name,
                "revision": search.dense.revision,
            }
        )
        mlflow.log_metric("chunks", len(rows))
    elif stage == "index":
        search = SearchConfig.model_validate(config)
        path = search.bm25.index_path
        with TemporaryDirectory() as directory:
            snapshot = Path(directory) / "index.zip"
            shutil.copyfile(path, snapshot)
            index = BM25Index.load(snapshot)
            if index.metadata["bm25"] != search.bm25.model_dump(mode="json"):
                raise ValueError("Индекс был заменён выпуском с другим конфигом")
            store = ArtifactStore(settings)

            async def publish() -> str:
                """Публикует завершённый снимок индекса в рабочее хранилище"""
                await store.ensure_bucket()
                return await store.publish(
                    snapshot, "indexes", search.data_version or "unversioned"
                )

            try:
                key = asyncio.run(publish())
            finally:
                store.close()
        run = mlflow.active_run()
        release = {
            "run_id": run.info.run_id,
            "bucket": store.bucket,
            "manifest_key": key,
            "index_sha256": index.archive_sha256,
            "index_metadata": index.metadata,
        }
        mlflow.log_dict(release, "release.json")
        mlflow.log_params(
            {
                "manifest_key": key,
                "index_sha256": release["index_sha256"],
                "corpus_sha256": index.metadata["corpus_sha256"],
            }
        )
        mlflow.log_metric("corpus_documents", len(index.documents))
    elif stage == "evaluate":
        evaluation = EvaluationConfig.model_validate(config)
        report = json.loads(evaluation.metrics_path.read_text(encoding="utf-8"))
        mlflow.log_artifact(str(evaluation.metrics_path), "evaluation")
        mlflow.log_metrics(report["metrics"])
        mlflow.log_metrics(
            {
                name: value
                for name, value in report["performance"].items()
                if name.endswith("_ms")
            }
        )
        mlflow.log_params(
            {
                name: report[name]
                for name in [
                    "index_sha256",
                    "corpus_sha256",
                    "qrels_sha256",
                    "queries_sha256",
                    "split",
                ]
            }
        )
        mlflow.set_tag("release_status", report["release_status"])
        mlflow.log_param("method", report.get("method", "bm25"))
        for name in ["embeddings_sha256", "dense_index_key"]:
            if name in report:
                mlflow.log_param(name, report[name])
        if "search_config" in report:
            mlflow.log_dict(report["search_config"], "config/search-effective.json")
        run = mlflow.active_run()
        client = mlflow.MlflowClient()
        parents = client.search_runs(
            [experiment.experiment_id for experiment in client.search_experiments()],
            filter_string=(
                "params.stage = 'index' and "
                f"params.index_sha256 = '{report['index_sha256']}'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if parents:
            mlflow.set_tag("index_run_id", parents[0].info.run_id)

        if "embeddings_sha256" in report:
            dense_runs = client.search_runs(
                [
                    experiment.experiment_id
                    for experiment in client.search_experiments()
                ],
                filter_string=(
                    "params.stage = 'embed' and "
                    f"params.embeddings_sha256 = '{report['embeddings_sha256']}'"
                ),
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if dense_runs:
                mlflow.set_tag("dense_run_id", dense_runs[0].info.run_id)
