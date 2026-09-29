"""Проверки конфигурации ML-процессов"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nlp_ranker.cli import load_config
from nlp_ranker.core.config import Settings


def test_invalid_config_is_rejected(tmp_path):
    """Проверяет отказ для неподдерживаемой версии конфигурации"""
    path = tmp_path / "invalid.yaml"
    path.write_text("schema_version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_config(path, "prepare")


@pytest.mark.parametrize(
    ("stage", "filename"),
    [
        ("prepare", "data.yaml"),
        ("index", "search.yaml"),
        ("train", "training.yaml"),
        ("mine-negatives", "training.yaml"),
        ("evaluate", "evaluation.yaml"),
    ],
)
def test_current_configs_are_valid(stage, filename):
    """Проверяет существующие схемы, включая явно отложенные параметры"""
    assert load_config(Path("configs") / filename, stage).schema_version == 1


@pytest.mark.parametrize(
    ("stage", "filename", "section", "field", "value"),
    [
        ("prepare", "data.yaml", None, "seed", -1),
        ("prepare", "data.yaml", None, "schema_version", True),
        ("prepare", "data.yaml", None, "output_path", ""),
        ("prepare", "data.yaml", None, "seed", True),
        ("prepare", "data.yaml", None, "unexpected", "секрет"),
        ("index", "search.yaml", "retrieval", "candidate_limit", 0),
        ("index", "search.yaml", "retrieval", "candidate_limit", "100"),
        ("index", "search.yaml", "retrieval", "typo", 1),
        ("index", "search.yaml", "reranking", "top_k", 101),
        ("index", "search.yaml", "dense", "device", "invalid"),
        ("train", "training.yaml", "training", "batch_size", 0),
        ("train", "training.yaml", "training", "learning_rate", float("nan")),
        (
            "train",
            "training.yaml",
            None,
            "validation_path",
            "data/processed/train.parquet",
        ),
        ("evaluate", "evaluation.yaml", None, "metrics", []),
        ("evaluate", "evaluation.yaml", None, "metrics", ["unknown"]),
        ("evaluate", "evaluation.yaml", None, "metrics", ["mrr_at_10", "mrr_at_10"]),
        ("evaluate", "evaluation.yaml", "quality_gate", "max_ndcg_regression", 0.1),
    ],
)
def test_invalid_stage_parameters_are_rejected(
    tmp_path, stage, filename, section, field, value
):
    """Проверяет границы, опечатки, неправильные типы и связи параметров"""
    config = yaml.safe_load((Path("configs") / filename).read_text())
    target = config[section] if section else config
    target[field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(path, stage)


def test_dry_run_does_not_write_artifacts(tmp_path):
    """Запускает CLI в отдельном процессе и проверяет отсутствие выходного файла"""
    config = yaml.safe_load(Path("configs/data.yaml").read_text())
    output = tmp_path / "documents.parquet"
    config["output_path"] = str(output)
    config["data_version"] = "corpus-v1"
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nlp_ranker.cli",
            "prepare",
            "--config",
            str(path),
            "--dry-run",
        ],
        env=dict(os.environ, NLP_RANKER_LOG_FORMAT="json"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert not output.exists()
    assert '"data_version": "corpus-v1"' in result.stdout
    assert "Этап завершён" in result.stdout


def test_cli_validation_error_does_not_echo_input(tmp_path):
    """Не допускает вывода ошибочного значения или traceback при отказе CLI"""
    path = tmp_path / "invalid.yaml"
    path.write_text("schema_version: 1\nseed: private-token\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nlp_ranker.cli",
            "prepare",
            "--config",
            str(path),
            "--dry-run",
        ],
        env=dict(os.environ, NLP_RANKER_LOG_FORMAT="json"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "seed" in result.stderr
    assert "private-token" not in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("log_level", "DEBIG"),
        ("log_format", "invalid"),
        ("db_pool_size", 0),
        ("redis_max_connections", 0),
        ("postgres_dsn", "postgresql://user:password@localhost/db"),
        ("redis_url", "redis://localhost:99999"),
        ("s3_endpoint_url", "not-a-url"),
    ],
)
def test_invalid_settings_are_rejected(field, value):
    """Проверяет настройки до создания ресурсов приложения"""
    with pytest.raises(ValueError):
        Settings(_env_file=None, **{field: value})


def test_cli_settings_error_does_not_echo_secret():
    """Отклоняет неверный адрес до логирования без вывода значения окружения"""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nlp_ranker.cli",
            "prepare",
            "--config",
            "configs/data.yaml",
            "--dry-run",
        ],
        env=dict(os.environ, NLP_RANKER_REDIS_URL="redis://localhost:private-port"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "redis_url" in result.stderr
    assert "private-port" not in result.stdout + result.stderr
    assert "Traceback" not in result.stdout + result.stderr
