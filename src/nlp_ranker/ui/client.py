"""HTTP-клиент интерфейса без доступа к моделям и хранилищам"""

from uuid import UUID

import httpx
from pydantic import BaseModel, ValidationError

from nlp_ranker.server.model.schemas import (
    DocumentResponse,
    FilterOptions,
    SearchRequest,
    SearchResponse,
)


class ApiError(Exception):
    """Содержит понятное пользователю сообщение об ошибке API"""


class SearchClient:
    """Отправляет запросы серверу и проверяет публичные схемы ответов"""

    def __init__(self, api_url: str) -> None:
        """Сохраняет адрес, соединения открываются только на время запроса"""
        self.api_url = api_url

    def request[T: BaseModel](
        self,
        method: str,
        path: str,
        schema: type[T],
        payload: dict[str, object] | None = None,
    ) -> T:
        """Ограничивает время ожидания и преобразует ошибки в сообщения интерфейса"""
        try:
            with httpx.Client(
                base_url=self.api_url, timeout=httpx.Timeout(15, connect=2)
            ) as client:
                response = client.request(method, path, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ApiError("Сервис не ответил вовремя, попробуйте ещё раз") from exc
        except httpx.HTTPStatusError as exc:
            messages = {
                404: "Документ больше не доступен",
                422: "Проверьте запрос и значения фильтров",
                429: "Сервис занят, повторите поиск немного позже",
                503: "Поиск временно недоступен, попробуйте позже",
                504: "Поиск занял слишком много времени, попробуйте ещё раз",
            }
            raise ApiError(
                messages.get(exc.response.status_code, "Не удалось выполнить запрос")
            ) from exc
        except httpx.RequestError as exc:
            raise ApiError("Не удалось подключиться к сервису поиска") from exc
        try:
            return schema.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ApiError("Сервис вернул некорректный ответ") from exc

    def filters(self) -> FilterOptions:
        """Получает доступные категории из API"""
        return self.request("GET", "/v1/search/filters", FilterOptions)

    def search(self, payload: SearchRequest) -> SearchResponse:
        """Отправляет запрос и фильтры только после подтверждения формы"""
        return self.request(
            "POST", "/v1/search", SearchResponse, payload.model_dump(mode="json")
        )

    def document(self, document_id: UUID) -> DocumentResponse:
        """Открывает актуальный документ по ID независимо от поисковой выдачи"""
        return self.request("GET", f"/v1/documents/{document_id}", DocumentResponse)
