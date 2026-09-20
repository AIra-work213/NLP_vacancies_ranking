"""CLI для MinIO и DVC"""

import argparse
import asyncio
from pathlib import Path

import urllib3
from minio.error import S3Error

from nlp_ranker.core.config import Settings
from nlp_ranker.mlops.dvc import run_dvc
from nlp_ranker.server.repository.s3.artifacts import ArtifactStore


def main() -> None:
    """Запускает настройку MinIO, публикацию, восстановление или команду DVC"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    publish = commands.add_parser("publish")
    publish.add_argument("path", type=Path)
    publish.add_argument("--prefix", choices=["indexes", "models"], required=True)
    publish.add_argument("--version", required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("manifest_key")
    restore.add_argument("destination", type=Path)
    dvc = commands.add_parser("dvc")
    dvc.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    settings = Settings()
    if args.command == "dvc":
        raise SystemExit(run_dvc(args.arguments, settings))
    store = ArtifactStore(settings)

    async def execute() -> None:
        """Выполняет одну операцию CLI"""
        if args.command == "init":
            await store.ensure_bucket()
            print("Bucket MinIO готов")
        elif args.command == "publish":
            print(await store.publish(args.path, args.prefix, args.version))
        else:
            await store.restore(args.manifest_key, args.destination)
            print("Файл восстановлен")

    try:
        asyncio.run(execute())
    except (OSError, ValueError, S3Error, urllib3.exceptions.HTTPError) as exc:
        parser.exit(1, f"Ошибка хранилища: {type(exc).__name__}\n")
    finally:
        store.close()


if __name__ == "__main__":
    main()
