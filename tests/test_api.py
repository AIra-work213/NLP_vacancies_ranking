"""Проверки HTTP-контракта и подключения поискового механизма"""

import httpx
import pytest

from nlp_ranker.core.config import Settings
from nlp_ranker.domain.search import SearchHit
from nlp_ranker.server.app import create_app
from nlp_ranker.server.model.schemas import SearchRequest


async def test_liveness_and_readiness_are_distinct(client: httpx.AsyncClient):
    """Проверяет отдельно доступность процесса и готовность поиска"""
    assert (await client.get("/health/live")).status_code == 200
    assert (await client.get("/health/ready")).status_code == 503


async def test_unconfigured_search_is_explicit(client: httpx.AsyncClient):
    """Проверяет явный отказ до подключения поискового механизма"""
    response = await client.post("/v1/search", json={"query": "ML engineer"})
    assert response.status_code == 503


@pytest.mark.parametrize("payload", [{"query": "   "}, {"query": "ML", "limit": 101}])
async def test_invalid_search_is_rejected(client: httpx.AsyncClient, payload):
    """Проверяет отказ для пустого запроса и недопустимого лимита результатов"""
    assert (await client.post("/v1/search", json=payload)).status_code == 422


async def test_metrics_use_route_labels(client: httpx.AsyncClient):
    """Проверяет, что неизвестные пути не попадают в метки метрик"""
    await client.get("/health/live")
    await client.get("/unknown-private-input")
    output = await client.get("/metrics")
    assert output.status_code == 200
    assert 'handler="/health/live"' in output.text
    assert 'handler="none"' in output.text
    assert "unknown-private-input" not in output.text


async def test_backend_can_be_injected():
    """Проверяет передачу механизма в приложение и нормализацию схемой запроса"""

    class StubEngine:
        """Возвращает фиксированную выдачу для проверки зависимости"""

        ready = True

        async def search(self, query: str, limit: int) -> list[SearchHit]:
            """Проверяет параметры и возвращает один документ"""
            assert query == "ML engineer"
            assert limit == 1
            return [SearchHit(document_id="doc-1", score=0.7, text="Пример")]

    app = create_app(Settings(environment="test", _env_file=None), StubEngine())
    payload = SearchRequest(query=" ML engineer ", limit=1)
    assert app.state.search_backend.ready
    hits = await app.state.search_backend.search(payload.query, payload.limit)
    assert hits[0].document_id == "doc-1"
