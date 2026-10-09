"""Включает pgvector и добавляет поле уровня опыта для фильтрации"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Включает расширение и добавляет необязательный уровень опыта вакансии"""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("documents", sa.Column("experience", sa.String(100), nullable=True))
    op.create_index("ix_documents_experience", "documents", ["experience"])


def downgrade() -> None:
    """Удаляет поле опыта, сохраняя расширение для других таблиц базы"""
    op.drop_index("ix_documents_experience", table_name="documents")
    op.drop_column("documents", "experience")
