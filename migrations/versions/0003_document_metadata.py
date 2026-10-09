"""Сохраняет регион и исходный график работы документа"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Добавляет поля корпуса без изменения существующих документов"""
    op.add_column("documents", sa.Column("region", sa.String(255), nullable=True))
    op.add_column("documents", sa.Column("schedule", sa.String(100), nullable=True))


def downgrade() -> None:
    """Удаляет дополнительные метаданные документа"""
    op.drop_column("documents", "schedule")
    op.drop_column("documents", "region")
