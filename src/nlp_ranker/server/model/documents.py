"""ORM-модели документов и фрагментов с векторами"""

from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Объединяет метаданные моделей SQLAlchemy"""

    pass


class Document(Base):
    """Хранит документ источника и метаданные для фильтрации и версионирования"""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_documents_source"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    source: Mapped[str] = mapped_column(String(100))
    source_id: Mapped[str] = mapped_column(String(255))
    title: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    employer: Mapped[str | None] = mapped_column(String(255))
    city: Mapped[str | None] = mapped_column(String(255), index=True)
    work_format: Mapped[str | None] = mapped_column(String(50), index=True)
    experience: Mapped[str | None] = mapped_column(String(100), index=True)
    region: Mapped[str | None] = mapped_column(String(255))
    schedule: Mapped[str | None] = mapped_column(String(100))
    content_hash: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DocumentEmbedding(Base):
    """Связывает фрагмент и вектор с неизменяемым выпуском и содержимым документа"""

    __tablename__ = "document_embeddings"

    index_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    chunk_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64))
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))
