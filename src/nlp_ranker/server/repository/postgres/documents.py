"""Чтение документов и изменения с проверкой версии"""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import (
    delete,
    func,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import (
    AsyncSession,
)

from nlp_ranker.domain.documents import DocumentConflictError, DocumentInput
from nlp_ranker.server.model.documents import Document


class DocumentRepository:
    """Читает и изменяет документы в транзакции вызывающего кода"""

    def __init__(self, session: AsyncSession) -> None:
        """Сохраняет переданную зависимость"""
        self.session = session

    async def filter_values(self) -> dict[str, list[str]]:
        """Читает непустые значения фильтров из документов без выдуманных категорий"""
        options = {}
        for name in ("city", "work_format", "experience"):
            column = getattr(Document, name)
            statement = (
                select(column)
                .where(column.is_not(None), column != "")
                .distinct()
                .order_by(column)
            )
            options[name] = list(await self.session.scalars(statement))
        return options

    async def get(self, document_id: UUID) -> Document | None:
        """Читает документ по первичному ключу без открытия новой транзакции"""
        return await self.session.get(Document, document_id)

    async def get_many(self, document_ids: Sequence[UUID]) -> list[Document]:
        """Читает одним запросом, сохраняя порядок и проверяя наличие ID"""
        if not document_ids:
            return []
        rows = await self.session.scalars(
            select(Document)
            .where(Document.id.in_(document_ids))
            .execution_options(populate_existing=True)
        )
        found = {doc.id: doc for doc in rows}
        if any(doc_id not in found for doc_id in document_ids):
            raise LookupError(
                "Документ выдачи отсутствует в базе; проверьте версию индекса"
            )
        return [found[doc_id] for doc_id in document_ids]

    async def list_ids(
        self,
        city: str | None = None,
        work_format: str | None = None,
        experience: str | None = None,
        limit: int = 100,
    ) -> list[UUID]:
        """Отбирает ID по заданным фильтрам; None означает отсутствие фильтра"""
        if not 1 <= limit <= 10000:
            raise ValueError("Число документов должно быть от 1 до 10000")
        query = select(Document.id).order_by(Document.id).limit(limit)
        for column, value in [
            (Document.city, city),
            (Document.work_format, work_format),
            (Document.experience, experience),
        ]:
            if value is not None:
                query = query.where(column == value)
        return list(await self.session.scalars(query))

    async def save_many(self, documents: Sequence["DocumentInput"]) -> int:
        """Импортирует снимок: повышает версию изменений и пропускает повторы"""
        if not documents:
            return 0
        values = [doc.database_values() for doc in documents]
        values.sort(key=lambda row: (str(row["source"]), str(row["source_id"])))
        if len({(row["source"], row["source_id"]) for row in values}) != len(values):
            raise ValueError("Пакет содержит повторные пары источник/ID")

        statement = insert(Document).values(values)
        fields: dict[str, object] = {
            name: statement.excluded[name]
            for name in values[0]
            if name not in {"id", "source", "source_id"}
        }
        fields.update(version=Document.version + 1, updated_at=func.now())
        upsert = statement.on_conflict_do_update(
            constraint="uq_documents_source",
            set_=fields,
            where=Document.content_hash != statement.excluded.content_hash,
        ).returning(Document.id)
        changed = len(list(await self.session.scalars(upsert)))
        expected_ids = {(doc.source, doc.source_id): doc.id for doc in documents}
        stored = await self.session.execute(
            select(Document.source, Document.source_id, Document.id).where(
                tuple_(Document.source, Document.source_id).in_(list(expected_ids))
            )
        )
        for source, source_id, document_id in stored:
            if document_id != expected_ids[(source, source_id)]:
                raise DocumentConflictError("ID существующего документа нельзя менять")
        return changed

    async def replace(self, document: "DocumentInput", expected_version: int) -> int:
        """Заменяет документ только при совпадении прочитанной версии"""
        values = document.database_values()
        values.pop("id")
        values.pop("source")
        values.pop("source_id")
        new_version = await self.session.scalar(
            update(Document)
            .where(
                Document.id == document.id,
                Document.source == document.source,
                Document.source_id == document.source_id,
                Document.version == expected_version,
            )
            .values(**values, version=Document.version + 1, updated_at=func.now())
            .returning(Document.version)
        )
        if new_version is None:
            raise DocumentConflictError(
                "Документ изменён или удалён; прочитайте его заново"
            )
        return int(new_version)

    async def delete(self, document_id: UUID, expected_version: int) -> None:
        """Удаляет документ только при совпадении ожидаемой версии"""
        deleted = await self.session.scalar(
            delete(Document)
            .where(Document.id == document_id, Document.version == expected_version)
            .returning(Document.id)
        )
        if deleted is None:
            raise DocumentConflictError("Документ изменён или уже удалён")
