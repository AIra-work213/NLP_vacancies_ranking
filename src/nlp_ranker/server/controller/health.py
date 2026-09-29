"""HTTP-проверки доступности процесса и готовности поиска"""

from fastapi import APIRouter, HTTPException

from nlp_ranker.server.dependencies import SearchDependency

router = APIRouter(tags=["Состояние"])


@router.get("/health/live")
async def live() -> dict[str, str]:
    """Сообщает о доступности процесса"""
    return {"status": "alive"}


@router.get("/health/ready")
async def ready(service: SearchDependency) -> dict[str, str]:
    """Возвращает HTTP-статус готовности поискового сервиса"""
    if not await service.check_ready():
        raise HTTPException(status_code=503, detail="Поисковые ресурсы не готовы")
    return {"status": "ready", "release_version": service.release_version}
