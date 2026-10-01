"""Сообщения HTTP-клиента при отказах реального сервера"""

import pytest

from nlp_ranker.server.model.schemas import SearchRequest
from nlp_ranker.ui.client import ApiError, SearchClient


def test_client_reports_unconfigured_search(api_url):
    """HTTP 503 превращается в понятное сообщение интерфейса"""
    client = SearchClient(api_url)
    with pytest.raises(ApiError, match="временно недоступен"):
        client.search(SearchRequest(query="Java разработчик"))


def test_client_reports_unavailable_connection():
    """Отказ соединения не раскрывает traceback пользователю"""
    with pytest.raises(ApiError, match="подключиться"):
        SearchClient("http://127.0.0.1:1").filters()
