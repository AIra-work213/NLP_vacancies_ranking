"""Публикация и восстановление артефактов MinIO"""

import asyncio
import hashlib
import io
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeVar
from urllib.parse import urlsplit
from uuid import uuid4

import urllib3
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel, ConfigDict, Field

from nlp_ranker.core.concurrency import run_blocking
from nlp_ranker.core.config import Settings

T = TypeVar("T")


class ArtifactManifest(BaseModel):
    """Описывает файл завершённого выпуска для проверки при восстановлении"""

    model_config = ConfigDict(extra="forbid")

    object_key: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)


def file_checksum(path: Path) -> str:
    """Вычисляет SHA256 файла без загрузки всего содержимого в память"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ArtifactStore:
    """Хранит отдельные выпуски и ограничивает параллельные вызовы SDK в потоках"""

    def __init__(self, settings: Settings) -> None:
        """Создаёт клиент с таймаутами без соединения с хранилищем"""
        endpoint = urlsplit(settings.s3_endpoint_url)
        self.bucket = settings.s3_bucket
        self.http = urllib3.PoolManager(
            timeout=urllib3.Timeout(connect=3, read=10), retries=False
        )
        self.client = Minio(
            endpoint.netloc,
            access_key=settings.s3_access_key.get_secret_value(),
            secret_key=settings.s3_secret_key.get_secret_value(),
            secure=endpoint.scheme == "https",
            region="us-east-1",
            http_client=self.http,
        )
        self.semaphore = asyncio.Semaphore(settings.s3_concurrency)

    async def run(self, operation: Callable[[], T]) -> T:
        """Сохраняет слот до фактического завершения потока при повторной отмене"""
        async with self.semaphore:
            return await run_blocking(operation)

    async def ensure_bucket(self) -> None:
        """Создаёт bucket при первом запуске, сохраняя существующие объекты"""

        def create() -> None:
            """Обрабатывает одновременное создание одного bucket"""
            if not self.client.bucket_exists(self.bucket):
                try:
                    self.client.make_bucket(self.bucket)
                except S3Error as exc:
                    if exc.code != "BucketAlreadyOwnedByYou":
                        raise

        await self.run(create)

    async def publish(self, path: Path, prefix: str, version: str) -> str:
        """Проверяет загруженный файл и публикует manifest последним"""
        if prefix not in {"indexes", "models"}:
            raise ValueError("Префикс должен быть indexes или models")
        if (
            not version
            or any(part in version for part in ("/", "\\"))
            or version in {".", ".."}
        ):
            raise ValueError("Версия должна быть непустым именем без разделителей пути")
        return await self.run(lambda: self._publish(path, prefix, version))

    def _publish(self, path: Path, prefix: str, version: str) -> str:
        """Изолирует каждую попытку загрузки отдельным адресом выпуска"""
        folder = f"{prefix}/{version}/{uuid4()}"
        manifest = ArtifactManifest(
            object_key=f"{folder}/artifact",
            sha256=file_checksum(path),
            size=path.stat().st_size,
        )
        self.client.fput_object(
            self.bucket, manifest.object_key, str(path), num_parallel_uploads=1
        )
        with TemporaryDirectory() as directory:
            self._download(manifest, Path(directory) / "verified")
        payload = manifest.model_dump_json().encode()
        key = f"{folder}/manifest.json"
        self.client.put_object(
            self.bucket,
            key,
            io.BytesIO(payload),
            len(payload),
            content_type="application/json",
        )
        return key

    async def restore(self, manifest_key: str, destination: Path) -> None:
        """Восстанавливает проверенный файл, сохраняя прежний файл при ошибке"""
        await self.run(lambda: self._restore(manifest_key, destination))

    def _restore(self, manifest_key: str, destination: Path) -> None:
        """Читает manifest с освобождением HTTP-соединения"""
        try:
            response = self.client.get_object(self.bucket, manifest_key)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                raise FileNotFoundError("Выпуск не найден в MinIO") from exc
            raise
        try:
            manifest = ArtifactManifest.model_validate_json(response.read())
        finally:
            response.close()
            response.release_conn()
        if manifest.object_key != manifest_key.rsplit("/", 1)[0] + "/artifact":
            raise ValueError("Файл manifest ссылается на другой выпуск")
        self._download(manifest, destination)

    def _download(self, manifest: ArtifactManifest, destination: Path) -> None:
        """Скачивает во временный файл и заменяет итоговый только после проверки"""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4()}.download")
        partial = temporary.with_suffix(".part")
        try:
            self.client.fget_object(
                self.bucket,
                manifest.object_key,
                str(temporary),
                tmp_file_path=str(partial),
            )
            if (
                temporary.stat().st_size != manifest.size
                or file_checksum(temporary) != manifest.sha256
            ):
                raise ValueError("Содержимое файла не совпадает с manifest")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)

    def close(self) -> None:
        """Освобождает пул после завершения операций хранилища"""
        self.http.clear()
