"""Чтение и атомарная запись эмбеддингов в Parquet"""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def read_embeddings(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Читает векторы и параметры из одного снимка Parquet"""
    import numpy as np
    import pyarrow.parquet as pq

    with path.open("rb") as stream:
        artifact_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        table = pq.read_table(stream)
    metadata = json.loads((table.schema.metadata or {})[b"nlp_ranker"])
    metadata["artifact_sha256"] = artifact_hash
    rows = table.to_pylist()
    vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
    if (
        not rows
        or vectors.shape != (len(rows), 384)
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
        or len({(row["document_id"], row["chunk_number"]) for row in rows}) != len(rows)
    ):
        raise ValueError("Нужны уникальные нормированные векторы размерности 384")
    return rows, metadata


def write_embeddings(
    path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    """Атомарно сохраняет завершённые пакеты без частично записанного файла"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows).replace_schema_metadata(
        {b"nlp_ranker": json.dumps(metadata, ensure_ascii=False).encode()}
    )
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / "embeddings.parquet"
        pq.write_table(table, temporary)
        temporary.replace(path)
