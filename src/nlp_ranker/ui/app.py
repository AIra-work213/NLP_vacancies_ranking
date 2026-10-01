"""Поиск вакансий через HTTP с состоянием отдельной пользовательской сессии"""

import os

import streamlit as st
from pydantic import ValidationError

from nlp_ranker.server.model.schemas import (
    DocumentResponse,
    FilterOptions,
    SearchRequest,
    SearchResponse,
)
from nlp_ranker.ui.client import ApiError, SearchClient

WORK_FORMATS = {"remote": "Удалённо", "office": "В офисе", "hybrid": "Гибридный формат"}
EXPERIENCE = {
    "no_experience": "Без опыта",
    "1_to_3_years": "От 1 до 3 лет",
    "3_to_6_years": "От 3 до 6 лет",
    "over_6_years": "Более 6 лет",
}


def load_filters(client: SearchClient) -> FilterOptions:
    """Получает фильтры один раз для сессии, повторяет только по кнопке"""
    if (
        "filter_options" not in st.session_state
        and "filter_error" not in st.session_state
    ):
        try:
            st.session_state.filter_options = client.filters()
        except ApiError as exc:
            st.session_state.filter_error = str(exc)
    if "filter_error" in st.session_state:
        st.error(st.session_state.filter_error)
        if st.button("Повторить подключение"):
            del st.session_state.filter_error
            st.rerun()
        st.stop()
    options: FilterOptions = st.session_state.filter_options
    return options


def search_form(options: FilterOptions) -> SearchRequest | None:
    """Собирает параметры в форме без запросов при изменении виджетов"""
    with st.form("search"):
        query = st.text_input(
            "Что ищете?",
            placeholder="Например, Java-разработчик со Spring",
            max_chars=2000,
        )
        city_column, format_column, experience_column = st.columns(3)
        with city_column:
            city = st.selectbox(
                "Город",
                [None, *options.city],
                disabled=not options.city,
                format_func=lambda value: value or "Любой город",
            )
        with format_column:
            work_format = st.selectbox(
                "Формат работы",
                [None, *options.work_format],
                disabled=not options.work_format,
                format_func=lambda value: WORK_FORMATS.get(value, "Любой формат"),
            )
        with experience_column:
            experience = st.selectbox(
                "Опыт",
                [None, *options.experience],
                disabled=not options.experience,
                format_func=lambda value: EXPERIENCE.get(value, "Любой опыт"),
            )
        if not options.city:
            st.caption("В текущем наборе вакансий нет данных о городе")
        limit = st.selectbox("Количество результатов", [5, 10, 20, 50], index=1)
        submitted = st.form_submit_button("Найти вакансии", type="primary")
    if not submitted:
        return None
    try:
        return SearchRequest.model_validate(
            {
                "query": query,
                "limit": limit,
                "filters": {
                    "city": city,
                    "work_format": work_format,
                    "experience": experience,
                },
            }
        )
    except ValidationError:
        st.error("Введите непустой поисковый запрос")
        return None


def show_results(client: SearchClient) -> None:
    """Показывает сохранённую выдачу без повторного поиска при открытии карточки"""
    result: SearchResponse | None = st.session_state.get("search_result")
    if result is None:
        st.info("Опишите роль, навыки или технологии и нажмите «Найти вакансии»")
        return
    st.subheader(f"Результаты по запросу: {st.session_state.submitted_query}")
    if not result.hits:
        st.info("Вакансии не найдены, измените запрос или ослабьте фильтры")
        return
    st.caption(f"Показано вакансий: {len(result.hits)}")
    for hit in result.hits:
        with st.container(border=True):
            st.subheader(f"{hit.rank}. {hit.title}")
            details = [
                hit.employer,
                hit.city or hit.region,
                WORK_FORMATS.get(hit.work_format or ""),
                EXPERIENCE.get(hit.experience or ""),
            ]
            st.caption(" · ".join(value for value in details if value))
            st.text(hit.text[:300] + ("…" if len(hit.text) > 300 else ""))
            if st.button("Открыть полное описание", key=f"open_{hit.document_id}"):
                st.session_state.open_document = None
                st.session_state.document_error = None
                st.session_state.selected_document = hit.document_id
                try:
                    with st.spinner("Открываем вакансию…"):
                        st.session_state.open_document = client.document(
                            hit.document_id
                        )
                except ApiError as exc:
                    st.session_state.document_error = str(exc)
            if st.session_state.get("selected_document") == hit.document_id:
                if st.session_state.get("document_error"):
                    st.error(st.session_state.document_error)
                document: DocumentResponse | None = st.session_state.get(
                    "open_document"
                )
                if document is not None:
                    st.markdown("#### Полное описание")
                    st.text(document.title)
                    st.text(document.text)
                    if document.version != hit.version:
                        st.info("Вакансия обновилась, показана актуальная версия")


def main() -> None:
    """Создаёт интерфейс, состояние выдачи хранится только в пользовательской сессии"""
    st.set_page_config(page_title="Поиск вакансий", page_icon="🔎", layout="wide")
    st.title("Поиск вакансий")
    st.caption("Найдите подходящую роль по навыкам, технологиям и описанию работы")
    client = SearchClient(os.environ.get("NLP_RANKER_API_URL", "http://localhost:8000"))
    options = load_filters(client)
    payload = search_form(options)
    if payload is not None:
        st.session_state.search_result = None
        st.session_state.selected_document = None
        st.session_state.open_document = None
        st.session_state.document_error = None
        try:
            with st.spinner("Ищем подходящие вакансии…"):
                st.session_state.search_result = client.search(payload)
                st.session_state.submitted_query = payload.query
        except ApiError as exc:
            st.error(str(exc))
    show_results(client)


if __name__ == "__main__":
    main()
