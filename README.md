# Поиск и ранжирование вакансий

Учебный NLP/MLOps-проект с работающим HTTP API и интерфейсом Streamlit
Поиск объединяет BM25 и эмбеддинги multilingual-e5-small через RRF
Полные вакансии и векторы находятся в PostgreSQL, Redis кеширует ID и оценки
Модели выполняются на CPU, готовый reranker доступен для экспериментов
и выключен по умолчанию по результатам validation

## Структура

```text
src/nlp_ranker/
  core/               Настройки, логи и выполнение блокирующих операций
  domain/             Документы, результаты поиска и ошибки
  server/
    controller/       HTTP-маршруты
    service/          Поиск, документы и ограниченная очередь
    model/            HTTP-схемы и модели SQLAlchemy
    repository/       postgres, cache и s3
  ml/                 inference, retrieval, evaluation и training
  pipelines/          Подготовка, индексы, эмбеддинги и оценка
  mlops/              DVC, MinIO и MLflow
  ui/                 Интерфейс через HTTP API
configs/              Проверяемые YAML-конфигурации
migrations/           Миграции Alembic
tests/                pytest и нагрузочный сценарий Locust
deployment/           Docker и настройки мониторинга
docs/                 План, архитектура, качество и эксплуатация
```

[Архитектура](docs/architecture.md), [план реализации](docs/implementation.md),
[протокол качества](docs/evaluation.md), [эксперименты](docs/experiments.md)
Правила кода находятся в AGENTS.md
CI/CD исключён: все проверки запускаются локально

## Первый запуск

Нужны Python 3.12, uv и Docker Compose
Команды выполняются из корня проекта

```bash
cp .env.example .env
make install-all
docker compose --profile storage up -d --wait postgres redis minio
make storage-init
```

Если MinIO уже содержит DVC-копию проекта:

```bash
make data-pull
```

Если копии нет, положите исходный CSV в data/raw/vacancies_sample.csv,
затем подготовьте файлы:

```bash
make prepare
make prepare-labels
make index
make embed
```

CSV содержит первые 3000 записей [Mendeley Data, версия 1](https://data.mendeley.com/datasets/gkfx465zwk/1)
Формат: UTF-8, разделитель `;`, поля data_source, id, name, description
Авторы: Evgeny Petrov, Daria Dunaeva, Vyacheslav Goiko, Alexey Blaginin,
Eduard Saifulin; лицензия публикации CC BY 4.0
Подготовка оставляет 2058 документов и 2519 фрагментов
Это исторические вакансии 2022–2023 годов; город не указан

DVC восстанавливает файлы, но не состояние базы данных:

```bash
make migrate
make load-documents
make load-dense
make up-all
```

Первый запуск моделей требует загрузки весов с Hugging Face
Revision моделей фиксируется в configs/search.yaml
Docker сохраняет веса в отдельном volume model-cache
При ошибке совместимости корпуса и индексов API не объявляет себя готовым
Если локальный Docker использует другой контекст, передавайте его явно:
`make DOCKER='docker --context default' up-all`

| Сервис | Адрес |
| --- | --- |
| Интерфейс | http://localhost:8501 |
| API и OpenAPI | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |
| MinIO | http://localhost:9001 |

Для Grafana локальные значения по умолчанию: admin / local-dev-only
Учётные данные хранилищ задаются в .env; файл не входит в репозиторий
`make down` останавливает сервисы, сохраняя volumes

## API и интерфейс

```bash
curl -X POST http://localhost:8000/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Python backend FastAPI PostgreSQL","limit":10,"filters":{"work_format":"remote"}}'
```

Ответ содержит release_version и ранжированные hits с document_id,
заголовком, текстом, метаданными и score
Score является оценкой ранжирования, а не вероятностью
`GET /v1/documents/{document_id}` читает актуальный документ из PostgreSQL
`GET /v1/search/filters` возвращает доступные значения фильтров
В текущем корпусе город недоступен, поэтому интерфейс отключает его выбор

В Streamlit отправка формы запускает поиск; изменение полей не отправляет запрос
Открытие полного описания отдельно обращается к API по document_id
Состояние выдачи хранится в пользовательской сессии

`/health/live` проверяет процесс, `/health/ready` проверяет выпуск и PostgreSQL
Ошибки: 422 для неверного ввода, 404 для отсутствующего документа,
429 для переполнения/ожидания очереди, 503 для недоступного поиска,
504 для таймаута вычисления

Каждая поисковая задача владеет отдельной AsyncSession и снимком REPEATABLE READ
Проверка корпуса, dense-поиск и чтение документов проходят в одном снимке
По умолчанию один активный поиск и восемь ожидающих; ожидание 1 с, выполнение 10 с
После отмены HTTP слот остаётся занятым до окончания CPU-вычисления
Один процесс API является поддерживаемым режимом мониторинга

Redis необязателен: отказ кеша приводит к обычному поиску
TTL 300 с, время обращения 100 мс, размер записи до 64 KiB
Ключ учитывает выпуск, запрос, фильтры и limit; текст каждый раз читается из БД
После изменения корпуса требуется пересборка индексов и перезапуск API

## Оценка и эксперименты

```bash
make evaluate
make evaluate-dense
make evaluate-hybrid
make evaluate-rerank
uv run --no-sync nlp-ranker-pipeline evaluate --config configs/evaluation-hybrid.yaml --split test --track
```

`--track` сохраняет конфигурацию, версии, метрики и отчёт в MLflow
Названия экспериментов содержат модель и цель, запусков: метод и разбиение
`make track-index` и `make track-embed` публикуют артефакты в MinIO

Основная метрика: NDCG@10; дополнительные: MRR@10 и Recall@100
Recall оценивается относительно известных положительных документов
Бенчмарк содержит 12 запросов, по 4 на train/validation/test, разметка выполнена Codex
Он пригоден для демонстрации сравнения, но слишком мал для уверенных выводов
о качестве на реальном потоке запросов
Подбор проводится на validation, test используется для итоговой проверки

## Проверки и мониторинг

```bash
make check
make test-database
make test-storage
make test-server
make monitoring-up
make monitoring-check
make load-test LOAD_USERS=4 LOAD_DURATION=60s LOAD_MODE=mixed
```

Для полного набора интеграционных тестов используйте окружение из tests/README.md
HTTPX обращается к настоящему Uvicorn; сервисы тестов изолированы от рабочей БД
Ruff проверяет стиль и форматирование, mypy проверяет типы
Сценарий Locust смешивает новые запросы, повторы и фильтры
LOAD_MODE=cached повторяет один запрос; 429 учитываются как отказы
CSV и HTML сохраняются в artifacts/reports

Grafana автоматически получает dashboard «Поиск вакансий: API и модели»
Он показывает RPS, ошибки, p50/p95/p99, этапы, очередь, Redis, CPU и память
На каждом сборе /metrics проверяется готовность PostgreSQL и выпуска
Запросы, ID документов и произвольные URL не входят в метки
Правила предупреждений проверяются promtool; внешняя доставка не настроена
[Условия нагрузки и результаты](docs/operations.md)

## Выпуск и демонстрация

[Проверка кандидата, восстановление и откат](docs/release.md)
Сценарий демонстрации: поиск в UI, фильтр remote, открытие документа,
отчёт качества, эксперимент MLflow, dashboard Grafana и восстановление через DVC

Дообучение, брокер, автоматическое обновление корпуса и несколько API workers
остаются отдельными задачами; Kubernetes, Airflow и Kafka не используются
