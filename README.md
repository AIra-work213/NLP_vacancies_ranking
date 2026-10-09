# NLP Vacancies Ranking

Поиск вакансий по описанию задачи, навыкам и условиям работы. Например, по запросу «Python backend, FastAPI, PostgreSQL, удалённо»
сервис подбирает вакансии и позволяет открыть полное описание

В проекте есть FastAPI-сервер, интерфейс Streamlit, пайплайны подготовки данных
и оценки поиска, хранение артефактов и локальный мониторинг
Инференс работает на CPU

## Как работает поиск

```text
Запрос + фильтры → BM25 + E5 → RRF → документы из PostgreSQL
```

BM25 ищет совпадения по словам, multilingual-e5-small подбирает кандидатов
по смыслу, а RRF объединяет две выдачи. Длинные описания разбиваются на токенные фрагменты; в ответ попадает целая вакансия

Фильтры применяются до отбора кандидатов. Redis кеширует ID и оценки повторных запросов на 5 минут. Текст и метаданные читаются из PostgreSQL, кеш не заменяет основное хранилище

Готовый cross-encoder доступен для сравнения, но отключён по умолчанию:
на текущем бенчмарке он оказался медленнее и не улучшил выбранный гибрид

## Стек

| Часть проекта | Инструменты |
| --- | --- |
| Сервер | FastAPI, Pydantic, SQLAlchemy async, asyncpg, Alembic |
| Поиск | bm25s, PyTorch, Transformers, sentence-transformers, pgvector |
| Хранилища | PostgreSQL, Redis, MinIO |
| ML-пайплайны | Pandas, PyArrow, DVC, MLflow, ranx |
| Интерфейс | Streamlit, HTTPX |
| Наблюдаемость | structlog, Prometheus, Grafana |
| Проверки | pytest, pytest-asyncio, HTTPX, Locust, Ruff, mypy |
| Окружение | Python 3.12, uv, Docker Compose |

## Запуск

Потребуются Python 3.12, uv и Docker с Compose

```bash
git clone https://github.com/AIra-work213/NLP_vacancies_ranking.git
cd NLP_vacancies_ranking
cp .env.example .env
make install-all
docker compose --profile storage up -d --wait postgres redis minio
make storage-init
```

### Данные и индексы

Исходные данные: первые 3000 записей из
[Mendeley Data, версия 1](https://data.mendeley.com/datasets/gkfx465zwk/1)
Поместите выборку в `data/raw/vacancies_sample.csv`
CSV использует UTF-8 и разделитель `;`, обязательные поля:
`data_source`, `id`, `name`, `description`

```bash
make prepare
make prepare-labels
make index
make embed
```

Если есть доступ к ранее заполненному MinIO, файлы можно восстановить через
`make data-pull` вместо повторной подготовки
Веса моделей загружаются с Hugging Face при первом запуске,
их revision закреплены в `configs/search.yaml`

### База и сервисы

```bash
make migrate
make load-documents
make load-dense
make up-all
```

DVC восстанавливает файлы, документы и векторы загружаются в PostgreSQL отдельно
Настройки подключений находятся в `.env`
Для Docker с нестандартным контекстом можно передать его явно:
`make DOCKER='docker --context default' up-all`

| Сервис | Адрес |
| --- | --- |
| Поиск в браузере | http://localhost:8501 |
| OpenAPI | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| MinIO | http://localhost:9001 |

Локальный вход в Grafana: `admin` / `local-dev-only`
`make down` останавливает контейнеры, сохраняя данные в volumes

## API

```bash
curl -X POST http://localhost:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Python backend FastAPI PostgreSQL","limit":10,"filters":{"work_format":"remote"}}'
```

Ответ содержит версию поискового выпуска и список вакансий с ID, текстом,
метаданными, рангом и score. Score служит для ранжирования и не является вероятностью

| Метод | Маршрут | Назначение |
| --- | --- | --- |
| POST | `/v1/search` | Поиск с фильтрами |
| GET | `/v1/documents/{document_id}` | Полная вакансия по ID |
| GET | `/v1/search/filters` | Доступные значения фильтров |
| GET | `/health/live` | Проверка процесса |
| GET | `/health/ready` | Проверка выпуска и базы |
| GET | `/metrics` | Метрики Prometheus |

Каждый поиск использует отдельную сессию БД и снимок REPEATABLE READ. По умолчанию выполняется один поиск, ещё восемь могут ожидать в очереди. При перегрузке API возвращает 429, при таймауте вычисления 504. Отказ Redis не останавливает поиск

## Качество и производительность

Корпус содержит 2058 вакансий и 2519 фрагментов. Бенчмарк состоит из 12 запросов и 254 оценок релевантности,
разметку выполнена вручную. Настройки выбираются на validation, итоговая проверка проводится на test. Размер набора позволяет сравнивать варианты, но не делать выводы о качестве
на всём рынке вакансий

| Поиск | NDCG@10, validation | NDCG@10, test |
| --- | ---: | ---: |
| BM25 | 0.701 | 0.855 |
| E5 | 0.599 | 0.873 |
| BM25 + E5 + RRF | 0.871 | 0.901 |

Это сравнение на одном пуле из 254 оценок. Отдельные эксперименты с MiniLM и ruBERT описаны в
[журнале экспериментов](docs/experiments.md)
Все параметры и отчёты записываются в MLflow при запуске с `--track`

Локальный замер на Ryzen 7 7840HS: при четырёх пользователях смешанный сценарий
дал 22,4 ответа/с без отказов, p95 260 мс. При 16 пользователях сработало ограничение очереди, после снятия нагрузки
поиск продолжил отвечать
[Условия замера и результаты](docs/operations.md)

## Проверки

```bash
make check
make test-database
make test-storage
make test-server
make monitoring-check
make load-test LOAD_USERS=4 LOAD_DURATION=60s
```

HTTP-тесты обращаются к настоящему Uvicorn через HTTPX. PostgreSQL, Redis и MinIO для интеграционных тестов запускаются отдельно
[Полный запуск тестов](tests/README.md)
CI/CD не используется

## Структура

```text
src/nlp_ranker/
  server/       controller, service, model, repository
  ml/           retrieval, inference, evaluation, training
  pipelines/    подготовка данных, индексов и оценка
  mlops/        DVC, MinIO, MLflow
  ui/           Streamlit и HTTP-клиент
  domain/       документы, результаты поиска и ошибки
  core/         настройки, логи, ограничение вычислений
configs/        конфигурации пайплайнов
migrations/     Alembic
tests/          тесты и сценарий Locust
deployment/     Docker и мониторинг
docs/           архитектура, эксперименты и эксплуатация
```

[Архитектура](docs/architecture.md) ·
[Оценка релевантности](docs/evaluation.md) ·
[Восстановление и откат](docs/release.md)

Сейчас используются исторические вакансии 2022-2023 годов. Город в исходном корпусе не указан, поэтому выбор города в интерфейсе отключён

Получение свежих вакансий, кеш карточек по ID и дообучение остаются следующими задачами

Источник данных: Evgeny Petrov, Daria Dunaeva, Vyacheslav Goiko, Alexey Blaginin,
Eduard Saifulin, Mendeley Data, лицензия CC BY 4.0
