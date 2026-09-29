"""Поиск через настоящий HTTP-сервер с E5, тестовыми PostgreSQL и Redis"""

import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep
from uuid import UUID, uuid4

import httpx
import pyarrow.parquet as pq
import pytest
from redis.asyncio import Redis
from sqlalchemy import text

from nlp_ranker.core.config import Settings
from nlp_ranker.domain.documents import DocumentInput
from nlp_ranker.pipelines.documents import import_documents
from nlp_ranker.server.repository.postgres.connection import (
    create_engine,
    create_session_factory,
)
from nlp_ranker.server.repository.postgres.embeddings import import_embeddings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Загружает выпуск только в указанную тестовую БД и запускает процесс API"""
    dsn = os.environ.get("NLP_RANKER_TEST_POSTGRES_DSN")
    redis_url = os.environ.get("NLP_RANKER_TEST_REDIS_URL")
    if not dsn or not redis_url:
        pytest.skip("Нужны тестовые PostgreSQL и Redis")
    assert dsn.endswith("/nlp_ranker_test")
    settings = Settings(_env_file=None, postgres_dsn=dsn)
    rows = pq.read_table("data/processed/documents.parquet").to_pylist()
    documents = [
        DocumentInput.model_validate(
            {
                "id": row["document_id"],
                **{
                    name: row.get(name)
                    for name in DocumentInput.model_fields
                    if name != "id"
                },
            }
        )
        for row in rows
    ]

    async def prepare():
        """Очищает тестовые таблицы и импортирует документы и эмбеддинги"""
        engine = create_engine(settings)
        try:
            async with engine.begin() as connection:
                await connection.execute(text("TRUNCATE documents CASCADE"))
            await import_documents(documents, settings)
            async with create_session_factory(engine)() as session, session.begin():
                await import_embeddings(
                    session, Path("artifacts/indexes/embeddings.parquet")
                )
        finally:
            await engine.dispose()

    asyncio.run(prepare())
    directory = tmp_path_factory.mktemp("search-api")
    log_path = directory / "server.log"
    with socket.socket() as listener, log_path.open("w+") as log:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        environment = dict(
            os.environ,
            NLP_RANKER_ENVIRONMENT="test",
            NLP_RANKER_SEARCH_ENABLED="true",
            NLP_RANKER_POSTGRES_DSN=dsn,
            NLP_RANKER_REDIS_URL=redis_url,
            NLP_RANKER_LOG_FORMAT="json",
            OPENBLAS_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
        )
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "nlp_ranker.server.app:create_app",
                "--factory",
                "--fd",
                str(listener.fileno()),
            ],
            pass_fds=(listener.fileno(),),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = monotonic() + 60
            with httpx.Client(base_url=url, timeout=1) as client:
                while monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    try:
                        if client.get("/health/ready").status_code == 200:
                            yield url, log_path
                            return
                    except httpx.TransportError:
                        pass
                    sleep(0.1)
            log.seek(0)
            pytest.fail(log.read()[-12000:])
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


async def test_search_matches_offline_hybrid_and_opens_document(server):
    """Ранги совпадают с офлайн-поиском, документ читается по ID из БД"""
    url, _ = server
    reference = json.loads(
        Path("artifacts/reports/metrics-hybrid-validation.json").read_text()
    )["per_query"]["java"]
    async with httpx.AsyncClient(base_url=url) as client:
        response = await client.post("/v1/search", json={"query": reference["text"]})
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["release_version"]) == 64
        assert [hit["document_id"] for hit in body["hits"]] == [
            hit["document_id"] for hit in reference["hits"][:10]
        ]
        assert [hit["rank"] for hit in body["hits"]] == list(range(1, 11))
        first = body["hits"][0]
        opened = await client.get(f"/v1/documents/{first['document_id']}")
        assert opened.status_code == 200
        assert opened.json() == {
            name: value
            for name, value in first.items()
            if name not in {"rank", "score"}
        }
        assert (await client.get(f"/v1/documents/{uuid4()}")).status_code == 404
        assert (await client.get("/v1/documents/invalid")).status_code == 422


async def test_filters_empty_results_and_validation(server):
    """Фильтры применяются к кандидатам, неизвестный город возвращает пустую выдачу"""
    url, _ = server
    async with httpx.AsyncClient(base_url=url) as client:
        response = await client.post(
            "/v1/search",
            json={"query": "тестировщик", "filters": {"work_format": "remote"}},
        )
        assert response.status_code == 200
        assert response.json()["hits"]
        assert all(hit["work_format"] == "remote" for hit in response.json()["hits"])
        empty = await client.post(
            "/v1/search",
            json={"query": "Java", "filters": {"city": "Нет такого города"}},
        )
        assert empty.status_code == 200 and empty.json()["hits"] == []
        for payload in [
            {"query": " "},
            {"query": "Java", "limit": True},
            {"query": "Java", "filters": {"work_format": "unknown"}},
            {"query": "Java", "filters": {"city": " "}},
        ]:
            assert (await client.post("/v1/search", json=payload)).status_code == 422
        schema = (await client.get("/openapi.json")).json()
        assert "/v1/documents/{document_id}" in schema["paths"]


async def test_repeated_and_concurrent_queries_hit_cache(server):
    """Повторная выдача одинакова, параллельные запросы сохраняют отдельные ID"""
    url, _ = server
    async with httpx.AsyncClient(base_url=url, timeout=10) as client:
        payload = {"query": "Frontend-разработчик React и TypeScript", "limit": 5}
        first = await client.post("/v1/search", json=payload)
        second = await client.post("/v1/search", json=payload)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        responses = await asyncio.gather(
            *(client.post("/v1/search", json=payload) for _ in range(4))
        )
        assert all(
            response.status_code == 200 and response.json() == first.json()
            for response in responses
        )
        assert len({response.headers["X-Request-ID"] for response in responses}) == 4
        metrics = (await client.get("/metrics")).text
        assert 'nlp_ranker_search_cache_requests_total{result="hit"}' in metrics
        assert (await client.get("/health/live")).status_code == 200


async def test_corrupted_cache_and_ttl(server):
    """Повреждённая запись заменяется реальной выдачей с ограниченным TTL"""
    url, _ = server
    redis = Redis.from_url(os.environ["NLP_RANKER_TEST_REDIS_URL"])
    try:
        keys = await redis.keys("nlp_ranker:search:v1:*")
        assert keys
        for key in keys:
            assert 0 < await redis.ttl(key) <= 300
            await redis.set(key, b"broken", ex=300)
        async with httpx.AsyncClient(base_url=url) as client:
            response = await client.post(
                "/v1/search",
                json={"query": "Frontend-разработчик React и TypeScript", "limit": 5},
            )
            assert response.status_code == 200
            metrics = (await client.get("/metrics")).text
            assert 'nlp_ranker_search_cache_requests_total{result="invalid"}' in metrics
    finally:
        await redis.aclose()


async def test_changed_document_is_not_served_from_cache(server):
    """Изменённая версия делает поиск неготовым, чтение по ID остаётся актуальным"""
    url, _ = server
    settings = Settings(
        _env_file=None, postgres_dsn=os.environ["NLP_RANKER_TEST_POSTGRES_DSN"]
    )
    engine = create_engine(settings)
    async with httpx.AsyncClient(base_url=url) as client:
        payload = {"query": "Java-разработчик backend со Spring"}
        first = await client.post("/v1/search", json=payload)
        doc = first.json()["hits"][0]
        document_id = UUID(doc["document_id"])
        try:
            async with engine.begin() as connection:
                original = (
                    await connection.execute(
                        text("SELECT content_hash FROM documents WHERE id=:id"),
                        {"id": document_id},
                    )
                ).scalar_one()
                await connection.execute(
                    text(
                        "UPDATE documents SET content_hash=:hash, title=:title, "
                        "version=version+1 WHERE id=:id"
                    ),
                    {
                        "id": document_id,
                        "hash": "0" * 64,
                        "title": "Изменённый документ",
                    },
                )
            assert (await client.post("/v1/search", json=payload)).status_code == 503
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.get("/health/live")).status_code == 200
            opened = await client.get(f"/v1/documents/{document_id}")
            assert opened.json()["title"] == "Изменённый документ"
        finally:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "UPDATE documents SET content_hash=:hash, title=:title, "
                        "version=:version WHERE id=:id"
                    ),
                    {
                        "id": document_id,
                        "hash": original,
                        "title": doc["title"],
                        "version": doc["version"],
                    },
                )
            await engine.dispose()


async def test_redis_timeout_does_not_break_search(server):
    """Остановка ответов тестового Redis не ломает поиск и готовность БД"""
    url, _ = server
    assert os.environ["NLP_RANKER_TEST_REDIS_URL"] == "redis://127.0.0.1:6380/9"
    command = ["docker", "--context", "default"]
    subprocess.run(
        command + ["pause", "nlp-ranker-test-redis-1"], check=True, capture_output=True
    )
    try:
        async with httpx.AsyncClient(base_url=url, timeout=5) as client:
            response = await client.post(
                "/v1/search", json={"query": "Java backend Spring", "limit": 3}
            )
            assert response.status_code == 200 and response.json()["hits"]
            assert (await client.get("/health/ready")).status_code == 200
            metrics = (await client.get("/metrics")).text
            assert 'nlp_ranker_search_cache_requests_total{result="error"}' in metrics
    finally:
        subprocess.run(
            command + ["unpause", "nlp-ranker-test-redis-1"],
            check=True,
            capture_output=True,
        )


async def test_search_snapshot_survives_concurrent_document_change(server, monkeypatch):
    """Обновление во время поиска не смешивает старый индекс и новый текст"""
    import nlp_ranker.server.service.search as serving
    from nlp_ranker.server.repository.cache.search import SearchCache

    dsn = os.environ["NLP_RANKER_TEST_POSTGRES_DSN"]
    settings = Settings(_env_file=None, postgres_dsn=dsn, cache_enabled=False)
    engine = create_engine(settings)
    redis = Redis.from_url(os.environ["NLP_RANKER_TEST_REDIS_URL"])
    entered = asyncio.Event()
    finish = asyncio.Event()
    reference = json.loads(
        Path("artifacts/reports/metrics-hybrid-validation.json").read_text()
    )["per_query"]["java"]
    target_id = UUID(reference["hits"][0]["document_id"])
    vectors = pq.read_table("artifacts/indexes/embeddings.parquet").to_pylist()
    vector = [
        row["embedding"] for row in vectors if row["document_id"] == str(target_id)
    ][0]

    class PausedEncoder:
        """Задерживает запрос после проверки выпуска, не загружая вторую модель"""

        def __init__(self, config):
            """Сохраняет тот же конфиг модели"""
            self.config = config

        async def query(self, text):
            """Возвращает реальный вектор документа после конкурентной записи"""
            entered.set()
            await finish.wait()
            return vector

    from nlp_ranker.ml.inference import release

    monkeypatch.setattr(release, "Encoder", PausedEncoder)
    backend = serving.SearchService(
        serving.load_search_config(Path("configs/search.yaml")),
        settings,
        create_session_factory(engine),
        SearchCache(redis, settings),
    )
    original = None
    task = None
    try:
        await backend.start()
        task = asyncio.create_task(backend.search(reference["text"], 100))
        await asyncio.wait_for(entered.wait(), 5)
        async with engine.begin() as connection:
            original = (
                await connection.execute(
                    text(
                        "SELECT title, content_hash, version FROM documents "
                        "WHERE id=:id"
                    ),
                    {"id": target_id},
                )
            ).one()
            await connection.execute(
                text(
                    "UPDATE documents SET title='Новое название', "
                    "content_hash=:hash, version=version+1 WHERE id=:id"
                ),
                {"id": target_id, "hash": "0" * 64},
            )
        finish.set()
        hits = await task
        found = [hit.document for hit in hits if hit.document_id == str(target_id)]
        assert found and found[0].title == original.title
        assert not await backend.check_ready()
        from nlp_ranker.server.service.documents import DocumentService

        assert (
            await DocumentService(create_session_factory(engine), settings).get(
                target_id
            )
        ).title == "Новое название"
    finally:
        finish.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        if original is not None:
            async with engine.begin() as connection:
                await connection.execute(
                    text(
                        "UPDATE documents SET title=:title, content_hash=:hash, "
                        "version=:version WHERE id=:id"
                    ),
                    {
                        "id": target_id,
                        "title": original.title,
                        "hash": original.content_hash,
                        "version": original.version,
                    },
                )
        await backend.close()
        await redis.aclose()
        await engine.dispose()


async def test_filter_catalog_matches_corpus_and_experience_names(server):
    """Фильтры из API существуют в корпусе, прежнее имя опыта не теряет вакансии"""
    url, _ = server
    async with httpx.AsyncClient(base_url=url, timeout=15) as client:
        options = (await client.get("/v1/search/filters")).raise_for_status().json()
        assert options["city"] == []
        assert options["work_format"] == ["remote"]
        assert "over_6_years" in options["experience"]
        payload = {
            "query": "Java разработчик",
            "filters": {"experience": "over_6_years"},
        }
        response = (
            (await client.post("/v1/search", json=payload)).raise_for_status().json()
        )
        assert response["hits"]
        assert all(hit["experience"] == "over_6_years" for hit in response["hits"])
        payload["filters"]["experience"] = "more_than_6_years"
        legacy = (
            (await client.post("/v1/search", json=payload)).raise_for_status().json()
        )
        assert legacy == response


def test_streamlit_search_document_and_session_isolation(server, monkeypatch):
    """Форма обращается к настоящему API, открытие карточки не повторяет поиск"""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from nlp_ranker.ui.client import SearchClient

    url, _ = server
    monkeypatch.setenv("NLP_RANKER_API_URL", url)
    calls = {"filters": 0, "search": 0, "document": 0}
    for name in calls:
        original = getattr(SearchClient, name)

        def count(self, *args, operation=original, operation_name=name):
            """Считает вызовы, сохраняя реальные HTTP-запросы"""
            calls[operation_name] += 1
            return operation(self, *args)

        monkeypatch.setattr(SearchClient, name, count)
    app = AppTest.from_file(str(Path("src/nlp_ranker/ui/app.py").resolve())).run(
        timeout=30
    )
    assert not app.exception
    assert app.selectbox[0].disabled
    assert calls == {"filters": 1, "search": 0, "document": 0}
    app.button[0].click().run(timeout=30)
    assert app.error and calls["search"] == 0
    app.text_input[0].set_value("Java разработчик со Spring")
    app.selectbox[1].select("remote")
    assert calls["search"] == 0
    app.button[0].click().run(timeout=30)
    assert not app.exception and not app.error, [error.value for error in app.error]
    assert calls["search"] == 1
    hits = app.session_state["search_result"].hits
    assert hits and all(hit.work_format == "remote" for hit in hits)
    app.button[1].click().run(timeout=30)
    assert not app.exception
    assert calls == {"filters": 1, "search": 1, "document": 1}
    assert app.session_state["open_document"].document_id == hits[0].document_id
    app.run()
    assert calls["search"] == 1 and calls["document"] == 1
    second = AppTest.from_file(str(Path("src/nlp_ranker/ui/app.py").resolve())).run(
        timeout=30
    )
    assert not second.exception
    assert "search_result" not in second.session_state
    assert calls["search"] == 1


def test_streamlit_connection_failure_is_readable(monkeypatch):
    """Недоступный API показывает сообщение и кнопку повторного подключения"""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("NLP_RANKER_API_URL", "http://127.0.0.1:1")
    app = AppTest.from_file(str(Path("src/nlp_ranker/ui/app.py").resolve())).run(
        timeout=10
    )
    assert not app.exception
    assert "подключиться" in app.error[0].value
    assert app.button[0].label == "Повторить подключение"


async def test_metrics_cover_stages_readiness_and_unknown_paths(server):
    """Этапы измеряются на живом поиске, неизвестные URL не создают метки"""
    url, _ = server
    async with httpx.AsyncClient(base_url=url) as client:
        response = await client.post(
            "/v1/search", json={"query": "метрики Python PostgreSQL разработчик"}
        )
        assert response.status_code == 200
        await client.get("/unknown-metrics-secret")
        metrics = (await client.get("/metrics")).text
    for stage in ("queue", "validate_release", "bm25", "encoder", "dense", "documents"):
        assert f'nlp_ranker_search_stage_seconds_count{{stage="{stage}"}}' in metrics
    assert "nlp_ranker_search_ready 1.0" in metrics
    assert "nlp_ranker_search_active 0.0" in metrics
    assert "nlp_ranker_search_waiting 0.0" in metrics
    assert "process_resident_memory_bytes" in metrics
    assert response.json()["release_version"] in metrics
    assert "unknown-metrics-secret" not in metrics
    assert "метрики Python" not in metrics
