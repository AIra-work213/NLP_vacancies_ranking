# Проверки проекта

`make check` запускает Ruff, проверку форматирования, mypy и pytest.
HTTP-контракт проверяется через HTTPX и отдельный процесс Uvicorn.

- `test_api.py`, `test_server_http.py`: эндпоинты, фильтры, ошибки, реальный поиск,
  отказ Redis и конкурентное изменение документов
- `test_serving.py`: ограничения очереди, отмена вычислений и очистка при сбое старта
- `test_database.py`, `test_dense.py`: импорт документов и векторов, версии, pgvector
- `test_storage.py`: публикация и восстановление MinIO, ограничение параллельности
- `test_data.py`, `test_search.py`, `test_evaluation.py`: подготовка, BM25 и метрики
- `test_config.py`, `test_observability.py`: конфигурация, логи и изоляция request_id

Интеграционные проверки используют `compose.test.yaml` с временными данными.
Команды `make test-database`, `make test-storage` и `make test-server` запускают
нужные сервисы. `make test-down` останавливает тестовое окружение.
Для полного запуска задаются NLP_RANKER_TEST_POSTGRES_DSN,
NLP_RANKER_TEST_REDIS_URL и NLP_RANKER_TEST_S3_ENDPOINT_URL, как в Makefile.


Интерфейс дополнительно проверяется через Streamlit AppTest в test_server_http.py:
отправка формы, фильтрация, открытие документа, отсутствие повторного поиска
и изоляция сессий. HTTP-клиент и сообщения об отказах проверяются в test_ui_client.py.
Для запуска AppTest требуется установленная группа demo.

## Полный интеграционный запуск

```bash
make test-up
NLP_RANKER_POSTGRES_DSN=postgresql+asyncpg://nlp_ranker:test-only@127.0.0.1:5433/nlp_ranker_test uv run --no-sync alembic upgrade head
NLP_RANKER_TEST_POSTGRES_DSN=postgresql+asyncpg://nlp_ranker:test-only@127.0.0.1:5433/nlp_ranker_test NLP_RANKER_TEST_REDIS_URL=redis://127.0.0.1:6380/9 NLP_RANKER_TEST_S3_ENDPOINT_URL=http://127.0.0.1:9002 uv run --no-sync pytest
```

Для нагрузки сначала подготовьте рабочий выпуск и запустите API
`make load-test LOAD_USERS=4 LOAD_DURATION=60s` сохраняет CSV и HTML
429 считаются отказами, а не успешными поисками
Правила предупреждений проверяются командой `make monitoring-check`
CI/CD в проекте не используется
