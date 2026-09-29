"""HTTP-контроллер поиска без алгоритмов и доступа к хранилищам"""

from fastapi import APIRouter

from nlp_ranker.domain.search import SearchFilters
from nlp_ranker.server.dependencies import DocumentDependency, SearchDependency
from nlp_ranker.server.model.schemas import FilterOptions, SearchRequest, SearchResponse

router = APIRouter(tags=["Поиск"])


@router.post(
    "/v1/search",
    response_model=SearchResponse,
    responses={
        429: {"description": "Перегрузка или истечение ожидания"},
        503: {"description": "Поисковый выпуск недоступен"},
        504: {"description": "Истекло время выполнения"},
    },
)
async def search(payload: SearchRequest, service: SearchDependency) -> SearchResponse:
    """Передаёт проверенный запрос сервису и сериализует результат"""
    hits = await service.search(
        payload.query, payload.limit, SearchFilters(**payload.filters.model_dump())
    )
    return SearchResponse.from_hits(service.release_version, hits)


@router.get(
    "/v1/search/filters",
    response_model=FilterOptions,
    responses={503: {"description": "PostgreSQL недоступен"}},
)
async def filter_options(service: DocumentDependency) -> FilterOptions:
    """Возвращает доступные значения фильтров текущего корпуса"""
    return FilterOptions.model_validate(await service.filter_options())
