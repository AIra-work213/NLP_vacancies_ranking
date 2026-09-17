"""Анализ и подготовка корпуса вакансий"""

import csv
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from statistics import median
from uuid import NAMESPACE_URL, uuid5

import structlog
from pydantic import Field

from nlp_ranker.core.config import PipelineConfig


class DataConfig(PipelineConfig):
    """Проверяет настройки подготовки корпуса"""

    seed: int = Field(ge=0, le=4294967295)
    input_path: Path | None = Field(strict=False)
    output_path: Path = Field(strict=False)
    split_group_field: str = Field(min_length=1)


class VacancyTextParser(HTMLParser):
    """Извлекает текст HTML, сохраняя границы абзацев и пунктов списка"""

    def __init__(self) -> None:
        """Создаёт буфер текста одного документа"""
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Отмечает границы блоков и скрытые элементы"""
        if tag in {"script", "style"}:
            self.hidden = True
        elif tag in {"p", "div", "li", "br", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Завершает абзац или скрытый элемент"""
        if tag in {"script", "style"}:
            self.hidden = False
        elif tag in {"p", "div", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Сохраняет видимый текст"""
        if not self.hidden:
            self.parts.append(data)


def clean_text(value: str) -> str:
    """Удаляет HTML и лишние пробелы, сохраняя непустые строки"""
    parser = VacancyTextParser()
    parser.feed(value)
    parser.close()
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line)


def prepare(config: dict[str, object]) -> None:
    """Очищает вакансии и сохраняет корпус с устойчивыми ID в Parquet"""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    settings = DataConfig.model_validate(config)
    if settings.input_path is None:
        raise ValueError("Укажите input_path для подготовки данных")
    if settings.input_path.resolve() == settings.output_path.resolve():
        raise ValueError("Выходной файл не должен совпадать с исходником")

    source = pd.read_csv(settings.input_path, sep=";", dtype=str, keep_default_na=False)
    if not isinstance(source.index, pd.RangeIndex):
        raise ValueError("Количество полей CSV не соответствует заголовку")

    required = {"data_source", "id", "name", "description"}
    if not required.issubset(source.columns):
        raise ValueError("В CSV отсутствуют обязательные поля вакансии")

    experience_names = {
        "0": "no_experience",
        "1-3": "1_to_3_years",
        "3-6": "3_to_6_years",
        "6-": "over_6_years",
    }
    documents = []
    seen_ids: set[tuple[str, str]] = set()
    missing = 0
    duplicate_ids = 0
    groups: Counter[str] = Counter()

    for row in source.to_dict(orient="records"):
        origin = row["data_source"].strip()
        source_id = row["id"].strip()
        title = clean_text(row["name"])
        text = clean_text(row["description"])

        if not origin or not source_id or not title or not text:
            missing += 1
            continue

        key = (origin, source_id)
        if key in seen_ids:
            duplicate_ids += 1
            continue
        seen_ids.add(key)

        document_id = str(uuid5(NAMESPACE_URL, f"nlp-ranker:{origin}:{source_id}"))
        group_text = re.sub(r"\s+", " ", text).casefold()
        group_id = str(uuid5(NAMESPACE_URL, "nlp-ranker:text:" + group_text))
        groups[group_id] += 1
        schedule = row.get("schedule", "").strip()

        documents.append(
            {
                "document_id": document_id,
                "source": origin,
                "source_id": source_id,
                "title": title,
                "text": text,
                "employer": clean_text(row.get("employer_name", "")) or None,
                "city": None,
                "region": row.get("region_name", "").strip() or None,
                "schedule": schedule or None,
                "work_format": "remote" if schedule == "remote" else None,
                "experience": experience_names.get(row.get("experience", "").strip()),
                "group_id": group_id,
            }
        )

    required_columns = [
        "document_id",
        "source",
        "source_id",
        "title",
        "text",
        "group_id",
    ]
    optional_columns = [
        "employer",
        "city",
        "region",
        "schedule",
        "work_format",
        "experience",
    ]
    schema = pa.schema(
        [pa.field(name, pa.string(), nullable=False) for name in required_columns]
        + [pa.field(name, pa.string()) for name in optional_columns]
    )
    table = pa.Table.from_pylist(documents, schema=schema)
    settings.output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, settings.output_path)

    structlog.stdlib.get_logger("nlp_ranker.data").info(
        "Корпус подготовлен",
        input_rows=len(source),
        documents=len(documents),
        excluded_missing=missing,
        excluded_duplicate_ids=duplicate_ids,
        repeated_descriptions=sum(count - 1 for count in groups.values()),
        output_path=str(settings.output_path),
    )


def inspect_source(path: Path) -> dict[str, object]:
    """Подсчитывает пропуски, дубли и длины описаний исходных вакансий"""
    required = ("data_source", "id", "name", "description")
    missing = Counter({field: 0 for field in required})
    keys: Counter[tuple[str, str]] = Counter()
    descriptions: Counter[str] = Counter()
    lengths: list[int] = []
    total = 0
    usable = 0

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";", strict=True)

        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError("В CSV отсутствуют обязательные поля вакансии")

        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Количество полей CSV не соответствует заголовку")

            total += 1
            values = {field: row[field].strip() for field in required}

            for field, value in values.items():
                if not value:
                    missing[field] += 1

            if all(values.values()):
                usable += 1

            if values["data_source"] and values["id"]:
                keys[(values["data_source"], values["id"])] += 1

            if values["description"]:
                descriptions[values["description"]] += 1
                lengths.append(len(row["description"]))

    return {
        "rows": total,
        "usable_required_fields": usable,
        "missing_values": dict(missing),
        "duplicate_source_ids": sum(count - 1 for count in keys.values()),
        "duplicate_descriptions": sum(count - 1 for count in descriptions.values()),
        "description_characters": {
            "min": min(lengths) if lengths else None,
            "median": median(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
        },
    }


def prepare_labels(
    corpus_path: Path = Path("data/processed/documents.parquet"),
    queries_path: Path = Path("data/queries.csv"),
    qrels_path: Path = Path("data/qrels.csv"),
    output_dir: Path = Path("data/processed"),
) -> None:
    """Проверяет готовые оценки и сохраняет запросы и qrels в Parquet"""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    queries = pd.read_csv(queries_path, dtype=str, keep_default_na=False)
    qrels = pd.read_csv(qrels_path, dtype=str, keep_default_na=False)
    corpus = pq.read_table(corpus_path, columns=["document_id", "group_id"]).to_pandas()

    query_columns = ["query_id", "text", "group_id", "split", "category"]
    label_columns = ["query_id", "document_id", "relevance", "reason"]
    for frame, columns in [(queries, query_columns), (qrels, label_columns)]:
        if not set(columns).issubset(frame.columns):
            raise ValueError("В разметке отсутствуют обязательные колонки")
        if frame.empty or frame[columns].eq("").any().any():
            raise ValueError("Запросы и оценки не должны быть пустыми")

    if queries["query_id"].duplicated().any():
        raise ValueError("ID запросов должны быть уникальными")
    if qrels.duplicated(["query_id", "document_id"]).any():
        raise ValueError("Пара запрос/документ размечена повторно")
    if not queries["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Неизвестное разбиение запросов")
    if not qrels["relevance"].isin(["0", "1", "2"]).all():
        raise ValueError("Оценка релевантности должна быть 0, 1 или 2")
    if not qrels["query_id"].isin(queries["query_id"]).all():
        raise ValueError("В оценках указан неизвестный запрос")
    if not qrels["document_id"].isin(corpus["document_id"]).all():
        raise ValueError("В оценках указан отсутствующий документ")
    if not queries["query_id"].isin(qrels["query_id"]).all():
        raise ValueError("Не все запросы имеют оценки")
    if queries.groupby("group_id")["split"].nunique().gt(1).any():
        raise ValueError("Перефразировки запроса попали в разные разбиения")

    if "selection" not in qrels.columns:
        qrels["selection"] = "manual"

    qrels["relevance"] = qrels["relevance"].astype("int8")
    qrels = qrels.merge(
        queries[["query_id", "split"]], on="query_id", validate="many_to_one"
    )
    qrels = qrels.merge(
        corpus.rename(columns={"group_id": "document_group_id"}),
        on="document_id",
        validate="many_to_one",
    )
    if (
        qrels[qrels["relevance"] > 0]
        .groupby("document_group_id")["split"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("Связанные документы попали в разные разбиения")

    query_schema = pa.schema([pa.field(name, pa.string()) for name in query_columns])
    label_schema = pa.schema(
        [
            pa.field(name, pa.int8() if name == "relevance" else pa.string())
            for name in label_columns + ["split", "document_group_id", "selection"]
        ]
    )
    metadata = {b"annotator": b"codex", b"guideline_version": b"2"}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame, schema in [
        ("queries", queries, query_schema),
        ("qrels", qrels, label_schema),
    ]:
        table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, output_dir / f"{name}.parquet")

    structlog.stdlib.get_logger("nlp_ranker.data").info(
        "Разметка подготовлена",
        queries=len(queries),
        judgments=len(qrels),
        documents=qrels["document_id"].nunique(),
        splits=queries["split"].value_counts().to_dict(),
    )
