"""Проверки логов и контекста конкурентных запросов через настоящий HTTP"""

import asyncio
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from time import monotonic, sleep

import httpx
import pytest


@pytest.fixture(scope="module")
def logging_server(tmp_path_factory):
    """Запускает отдельный сервер с маршрутами проверки контекста и ошибок"""
    directory = tmp_path_factory.mktemp("logging")
    module = directory / "probe_app.py"
    module.write_text(
        dedent('''\
        """HTTP-приложение для проверки логов, контекста и ошибок."""

        import asyncio
        import structlog
        from nlp_ranker.server.app import create_app as build_app
        from nlp_ranker.core.config import Settings

        def create_app():
            """Создаёт приложение с тестовыми маршрутами."""
            settings = Settings(_env_file=None, environment="test", log_format="json")
            app = build_app(settings)

            @app.get("/probe")
            async def probe():
                """Возвращает контекст запроса и контекст вычисления в потоке."""
                await asyncio.sleep(0.02)
                context = structlog.contextvars.get_contextvars()
                get_context = structlog.contextvars.get_contextvars
                thread_context = await asyncio.to_thread(get_context)
                structlog.get_logger().info("Контекст проверен")
                return {
                    "request_id": context["request_id"],
                    "thread_id": thread_context["request_id"],
                }

            @app.get("/failure")
            async def failure():
                """Проверяет скрытие содержимого непредвиденного исключения."""
                raise RuntimeError("private-body-secret")

            return app
        '''),
        encoding="utf-8",
    )

    path = directory / "server.log"
    source = str(Path("src").resolve())
    environment = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(directory), source]),
        NLP_RANKER_SEARCH_ENABLED="false",
    )

    with socket.socket() as listener, path.open("w+") as output:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "probe_app:create_app",
                "--factory",
                "--fd",
                str(listener.fileno()),
            ],
            pass_fds=(listener.fileno(),),
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )

        try:
            deadline = monotonic() + 15
            with httpx.Client(base_url=url, timeout=0.5) as client:
                while monotonic() < deadline:
                    if process.poll() is not None:
                        break

                    try:
                        if client.get("/health/live").status_code == 200:
                            yield url, path
                            return
                    except httpx.TransportError:
                        pass

                    sleep(0.1)

            pytest.fail(f"Проверочный API не запустился: {path.read_text()}")

        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


async def test_concurrent_context_and_thread_propagation(logging_server):
    """Проверяет уникальные ID и передачу контекста в потоки без смешения запросов"""
    url, path = logging_server

    async with httpx.AsyncClient(base_url=url) as client:
        responses = await asyncio.gather(
            *(
                client.get("/probe", headers={"X-Request-ID": "untrusted"})
                for _ in range(16)
            )
        )

    ids = set()
    for response in responses:
        assert response.status_code == 200
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 32
        assert response.json() == {"request_id": request_id, "thread_id": request_id}
        ids.add(request_id)

    assert len(ids) == 16
    records = [json.loads(line) for line in path.read_text().splitlines()]
    completed = [
        r
        for r in records
        if r["event"] == "HTTP-запрос завершён" and r["route"] == "/probe"
    ]
    assert {r["request_id"] for r in completed} == ids
    assert all(r["duration_ms"] >= 0 for r in completed)


async def test_error_and_unknown_route_do_not_leak_input(logging_server):
    """Проверяет ID ошибки и отсутствие исключения, тела и неизвестного пути в логах"""
    url, path = logging_server

    async with httpx.AsyncClient(base_url=url) as client:
        failure = await client.get("/failure?token=query-secret")
        unknown = await client.post("/path-secret", content="body-secret")
        following = await client.get("/probe")

    assert failure.status_code == 500
    assert failure.json() == {"detail": "Внутренняя ошибка сервера"}
    assert failure.headers["X-Request-ID"] != following.headers["X-Request-ID"]
    assert unknown.status_code == 404

    output = path.read_text()
    for secret in ("private-body-secret", "query-secret", "body-secret", "path-secret"):
        assert secret not in output

    records = [json.loads(line) for line in output.splitlines()]
    failed = [
        r for r in records if r.get("request_id") == failure.headers["X-Request-ID"]
    ]
    assert any(
        r.get("error_type") == "RuntimeError" and r["status_code"] == 500
        for r in failed
    )


def test_standard_logging_and_structlog_have_one_format():
    """Проверяет общий JSON-формат и читаемый локальный формат в отдельном процессе"""
    code = dedent("""\
        import logging
        import structlog
        from nlp_ranker.core.logging import configure_logging
        from nlp_ranker.core.config import Settings
        configure_logging(Settings(_env_file=None))
        logging.getLogger("standard").info("Стандартный лог")
        structlog.get_logger("structured").info("Структурированный лог")
    """)

    for format_name in ("json", "console"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=dict(os.environ, NLP_RANKER_LOG_FORMAT=format_name),
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.splitlines()
        assert len(lines) == 2

        if format_name == "json":
            assert [json.loads(line)["logger"] for line in lines] == [
                "standard",
                "structured",
            ]
        else:
            assert "Стандартный лог" in lines[0]
            assert "Структурированный лог" in lines[1]
