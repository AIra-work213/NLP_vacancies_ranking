"""Пакетный импорт подготовленных документов в PostgreSQL"""

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

import structlog

from nlp_ranker.core.config import Settings
from nlp_ranker.core.logging import configure_logging
from nlp_ranker.domain.documents import DocumentInput
from nlp_ranker.server.repository.postgres.connection import (
    create_engine,
    create_session_factory,
)
from nlp_ranker.server.repository.postgres.documents import DocumentRepository


async def import_documents(
    documents: Sequence[DocumentInput], settings: Settings, batch_size: int = 500
) -> int:
    """Загружает пакеты в одной транзакции и закрывает собственный пул соединений"""
    if not 1 <= batch_size <= 1000:
        raise ValueError("Размер пакета должен быть от 1 до 1000")
    identities = {(doc.source, doc.source_id) for doc in documents}
    if len(identities) != len(documents):
        raise ValueError("Загрузка содержит повторные пары источник/ID")
    engine = create_engine(settings)
    sessions = create_session_factory(engine)
    try:
        async with sessions() as session, session.begin():
            repository = DocumentRepository(session)
            changed = 0
            for start in range(0, len(documents), batch_size):
                changed += await repository.save_many(
                    documents[start : start + batch_size]
                )
            return changed
    finally:
        await engine.dispose()


def main() -> None:
    """Читает подготовленный Parquet и запускает асинхронный импорт в PostgreSQL"""
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, nargs="?", default=Path("data/processed/documents.parquet")
    )
    args = parser.parse_args()
    rows = pq.read_table(args.path).to_pylist()
    documents = []
    for row in rows:
        values = {
            name: row.get(name) for name in DocumentInput.model_fields if name != "id"
        }
        documents.append(
            DocumentInput.model_validate({"id": row["document_id"], **values})
        )
    settings = Settings()
    configure_logging(settings)
    changed = asyncio.run(import_documents(documents, settings))
    structlog.stdlib.get_logger("nlp_ranker.database").info(
        "Документы загружены", documents=len(documents), changed=changed
    )


if __name__ == "__main__":
    main()
