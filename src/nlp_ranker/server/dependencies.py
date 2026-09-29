"""Зависимости HTTP-контроллеров, созданные в lifespan приложения"""

from typing import Annotated

from fastapi import Depends, Request

from nlp_ranker.server.service.contracts import SearchBackend
from nlp_ranker.server.service.documents import DocumentService


def get_search_service(request: Request) -> SearchBackend:
    """Возвращает готовый сервис текущего процесса"""
    service: SearchBackend = request.app.state.search_backend
    return service


def get_document_service(request: Request) -> DocumentService:
    """Получает сервис чтения документов с фабрикой независимых сессий"""
    service: DocumentService = request.app.state.document_service
    return service


SearchDependency = Annotated[SearchBackend, Depends(get_search_service)]
DocumentDependency = Annotated[DocumentService, Depends(get_document_service)]
