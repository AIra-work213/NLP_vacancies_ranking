"""Выполнение блокирующей операции с ожиданием при отмене"""

import asyncio
from collections.abc import Callable


async def run_blocking[T](operation: Callable[[], T]) -> T:
    """При отмене ждёт фактического завершения операции в потоке"""
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise
