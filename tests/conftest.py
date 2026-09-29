"""HTTP-клиент и отдельный процесс API для проверок по настоящему HTTP"""

import os
import socket
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from time import monotonic, sleep

import httpx
import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def api_url(tmp_path_factory) -> Iterator[str]:
    """Использует заданный сервер или запускает локальный API на свободном порту"""
    external_url = os.environ.get("NLP_RANKER_TEST_API_URL")
    if external_url:
        yield external_url
        return
    log_path = tmp_path_factory.mktemp("api") / "server.log"
    with socket.socket() as listener, log_path.open("w+") as log:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        environment = dict(
            os.environ, NLP_RANKER_ENVIRONMENT="test", NLP_RANKER_SEARCH_ENABLED="false"
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
            deadline = monotonic() + 15
            with httpx.Client(base_url=url, timeout=0.5) as probe:
                while monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    try:
                        if probe.get("/health/live").status_code == 200:
                            yield url
                            return
                    except httpx.TransportError:
                        pass
                    sleep(0.1)
            log.seek(0)
            pytest.fail(f"API не запустился:\n{log.read()[-12000:]}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest_asyncio.fixture
async def client(api_url: str) -> AsyncIterator[httpx.AsyncClient]:
    """Предоставляет асинхронный клиент к запущенному HTTP-серверу"""
    async with httpx.AsyncClient(base_url=api_url, timeout=5) as client:
        yield client
