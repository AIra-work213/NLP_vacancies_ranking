"""Проверки выпусков и восстановления на настоящем MinIO"""

import asyncio
import io
import os
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest
import pytest_asyncio
from minio.error import S3Error
from pydantic import SecretStr

from nlp_ranker.core.config import Settings
from nlp_ranker.server.repository.s3.artifacts import ArtifactStore

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def storage():
    """Создаёт отдельный тестовый bucket и удаляет только его объекты"""
    endpoint = os.environ.get("NLP_RANKER_TEST_S3_ENDPOINT_URL")
    if not endpoint:
        pytest.skip("Запустите make test-storage")
    settings = Settings(
        _env_file=None,
        s3_endpoint_url=endpoint,
        s3_bucket=f"nlp-ranker-test-{uuid4()}",
        AWS_ACCESS_KEY_ID="nlp_ranker",
        AWS_SECRET_ACCESS_KEY="test-only",
    )
    store = ArtifactStore(settings)
    try:
        await store.ensure_bucket()
        yield store, settings
    finally:

        def cleanup():
            """Удаляет bucket этой проверки без обращения к рабочим данным"""
            for item in store.client.list_objects(store.bucket, recursive=True):
                store.client.remove_object(store.bucket, item.object_name)
            store.client.remove_bucket(store.bucket)

        await store.run(cleanup)
        store.close()


async def test_parallel_publish_and_restore(storage, tmp_path):
    """Параллельные выпуски одной версии получают независимые адреса"""
    store, _ = storage
    source = tmp_path / "index.bin"
    source.write_bytes(b"test index" * 1024)
    keys = await asyncio.gather(
        store.publish(source, "indexes", "v1"), store.publish(source, "indexes", "v1")
    )
    assert keys[0] != keys[1]
    await store.restore(keys[0], tmp_path / "empty" / "index.bin")
    assert (tmp_path / "empty" / "index.bin").read_bytes() == source.read_bytes()


async def test_corrupted_object_preserves_destination(storage, tmp_path):
    """Повреждённый объект не заменяет существующий локальный файл"""
    store, _ = storage
    source = tmp_path / "model.bin"
    source.write_bytes(b"weights")
    key = await store.publish(source, "models", "v1")
    object_key = key.rsplit("/", 1)[0] + "/artifact"
    await store.run(
        lambda: store.client.put_object(
            store.bucket, object_key, io.BytesIO(b"broken!"), 7
        )
    )
    target = tmp_path / "restored.bin"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="manifest"):
        await store.restore(key, target)
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".*.download"))
    assert not list(tmp_path.glob(".*.part"))


async def test_missing_release_and_denied_access(storage, tmp_path):
    """Отсутствующий выпуск и неверный ключ доступа возвращают явные ошибки"""
    store, settings = storage
    with pytest.raises(FileNotFoundError):
        await store.restore("indexes/missing/manifest.json", tmp_path / "missing.bin")
    denied = ArtifactStore(
        settings.model_copy(update={"s3_secret_key": SecretStr("wrong-password")})
    )
    try:
        with pytest.raises(S3Error):
            await denied.ensure_bucket()
    finally:
        denied.close()


async def test_interrupted_transfer_preserves_destination(
    storage, tmp_path, monkeypatch
):
    """Незавершённая передача удаляет временный файл и сохраняет прежний"""
    store, _ = storage
    source = tmp_path / "index.bin"
    source.write_bytes(b"index")
    key = await store.publish(source, "indexes", "v1")
    target = tmp_path / "restored.bin"
    target.write_bytes(b"previous")

    def interrupted(*args, **kwargs):
        """Прерывает передачу после записи её начальной части"""
        Path(kwargs["tmp_file_path"]).write_bytes(b"partial")
        raise TimeoutError("Передача прервана")

    monkeypatch.setattr(store.client, "fget_object", interrupted)
    with pytest.raises(TimeoutError):
        await store.restore(key, target)
    assert target.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".*.part"))


async def test_failed_publish_has_no_manifest(storage, tmp_path, monkeypatch):
    """Незавершённый выпуск не виден читателям, повторная попытка восстанавливается"""
    store, _ = storage
    source = tmp_path / "model.bin"
    source.write_bytes(b"weights")
    original_put = store.client.put_object

    def interrupted(bucket, key, *args, **kwargs):
        """Прерывает публикацию manifest после передачи проверенного файла"""
        if key.endswith("/manifest.json"):
            raise TimeoutError("Передача прервана")
        return original_put(bucket, key, *args, **kwargs)

    monkeypatch.setattr(store.client, "put_object", interrupted)
    with pytest.raises(TimeoutError):
        await store.publish(source, "models", "v1")
    objects = await store.run(
        lambda: list(store.client.list_objects(store.bucket, recursive=True))
    )
    assert len(objects) == 1
    assert objects[0].object_name.endswith("/artifact")
    monkeypatch.setattr(store.client, "put_object", original_put)
    key = await store.publish(source, "models", "v1")
    await store.restore(key, tmp_path / "restored.bin")
    assert (tmp_path / "restored.bin").read_bytes() == source.read_bytes()


async def test_cancelled_operation_keeps_concurrency_limit(storage):
    """Отмена ожидания не освобождает слот ещё работающего потока"""
    _, settings = storage
    store = ArtifactStore(settings.model_copy(update={"s3_concurrency": 1}))
    started = Event()
    release = Event()
    second_started = Event()

    def blocked():
        """Удерживает поток до разрешения продолжить"""
        started.set()
        release.wait(timeout=3)

    first = asyncio.create_task(store.run(blocked))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 1)
        first.cancel()
        second = asyncio.create_task(store.run(second_started.set))
        await asyncio.sleep(0.02)
        assert not second_started.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert second_started.is_set()
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second else []), return_exceptions=True
        )
        store.close()
