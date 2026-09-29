"""HTTP-контроллер открытия документов по идентификатору"""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from nlp_ranker.server.dependencies import DocumentDependency
from nlp_ranker.server.model.schemas import DocumentResponse

router = APIRouter(tags=["Документы"])


@router.get(
    "/v1/documents/{document_id}",
    response_model=DocumentResponse,
    responses={
        404: {"description": "Документ не найден"},
        503: {"description": "PostgreSQL недоступен"},
    },
)
async def document(document_id: UUID, service: DocumentDependency) -> DocumentResponse:
    """Сериализует документ или возвращает HTTP 404"""
    result = await service.get(document_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    return DocumentResponse.model_validate(result)
