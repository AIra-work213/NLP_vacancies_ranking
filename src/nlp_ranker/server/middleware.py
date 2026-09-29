"""Контекст HTTP-запроса и логирование ответа"""

from time import perf_counter
from uuid import uuid4

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestLoggingMiddleware:
    """Изолирует request_id и логирует завершение настоящего ASGI-запроса"""

    def __init__(self, app: ASGIApp) -> None:
        """Получает приложение, не создавая общих изменяемых данных запроса"""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Создаёт контекст, добавляет заголовок и очищает контекст при завершении"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        structlog.contextvars.clear_contextvars()
        request_id = uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        scope.setdefault("state", {})["request_id"] = request_id

        started = perf_counter()
        status_code = 500
        error_type: str | None = None

        async def send_response(message: Message) -> None:
            """Добавляет идентификатор к ответу и запоминает HTTP-статус"""
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message = {**message, "headers": headers}

            await send(message)

        try:
            await self.app(scope, receive, send_response)
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            route = getattr(scope.get("route"), "path", "none")
            logger = structlog.stdlib.get_logger("nlp_ranker.http")
            log = logger.error if error_type else logger.info

            try:
                log(
                    "HTTP-запрос завершён",
                    method=scope["method"],
                    route=route,
                    status_code=status_code,
                    duration_ms=round((perf_counter() - started) * 1000, 3),
                    error_type=error_type,
                )
            finally:
                structlog.contextvars.clear_contextvars()
