"""Добавляет точный векторный поиск по фрагментам документов"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Создаёт таблицу векторов без приблизительного индекса"""
    op.create_table(
        "document_embeddings",
        sa.Column("index_key", sa.String(64), primary_key=True),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("chunk_number", sa.Integer(), primary_key=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
    )


def downgrade() -> None:
    """Удаляет только векторы, сохраняя документы"""
    op.drop_table("document_embeddings")
