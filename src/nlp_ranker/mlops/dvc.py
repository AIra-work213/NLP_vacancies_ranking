"""Запуск DVC с настройками MinIO из окружения"""

import os
import subprocess
import sys

from nlp_ranker.core.config import Settings


def run_dvc(arguments: list[str], settings: Settings) -> int:
    """Передаёт DVC учётные данные из .env без записи секретов в его конфиг"""
    environment = os.environ.copy()
    environment.update(
        AWS_ACCESS_KEY_ID=settings.s3_access_key.get_secret_value(),
        AWS_SECRET_ACCESS_KEY=settings.s3_secret_key.get_secret_value(),
        AWS_DEFAULT_REGION="us-east-1",
    )
    return subprocess.run(
        [sys.executable, "-m", "dvc", *arguments], env=environment, check=False
    ).returncode
