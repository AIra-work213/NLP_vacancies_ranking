"""Единое преобразование исключений в HTTP-ответы"""

from fastapi import Request
from fastapi.responses import JSONResponse

from nlp_ranker.domain.errors import SearchOverloadedError, SearchTimeoutError


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Возвращает общий ответ с request_id, не раскрывая детали исключения"""
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервера"},
        headers={"X-Request-ID": request.state.request_id},
    )


async def handle_search_error(request: Request, exc: Exception) -> JSONResponse:
    """Преобразует ошибки сервиса в единый HTTP-контракт"""
    if isinstance(exc, SearchOverloadedError):
        return JSONResponse(
            status_code=429, content={"detail": str(exc)}, headers={"Retry-After": "1"}
        )
    status = 504 if isinstance(exc, SearchTimeoutError) else 503
    return JSONResponse(status_code=status, content={"detail": str(exc)})
