# Архитектура проекта

Проект состоит из HTTP-сервера, ML-компонентов и запускаемых отдельно пайплайнов.
Сервер следует структуре Model / Controller / Service / Repository. Представление
данных для API задаётся схемами Pydantic; интерфейс Streamlit обращается
к серверу через HTTP

## Путь запроса

```text
HTTP → Controller → Service → Repository → PostgreSQL / Redis
                        ↓
                  ML-инференс и поиск
```

| Слой | Обязанность | Примеры |
| --- | --- | --- |
| controller | Проверенные параметры HTTP, вызов сервиса, код ответа | search.py, documents.py, health.py |
| service | Сценарий поиска, согласованная транзакция, получение документов | SearchService, DocumentService, DenseSearch |
| repository | Конкретные обращения к хранилищам | DocumentRepository, EmbeddingRepository, ReleaseRepository, SearchCache, ArtifactStore |
| model | Таблицы SQLAlchemy и публичные схемы запросов/ответов | Document, DocumentEmbedding, SearchRequest, SearchResponse |
| domain | Данные и ошибки между компонентами без зависимости от HTTP и ORM | SearchHit, SearchFilters, DocumentInput |
| ml | Загрузка и инференс моделей, BM25, объединение результатов, метрики | Encoder, Reranker, BM25Index, fuse_results |
| pipelines | Подготовка файлов, построение индексов, импорт и offline-оценка | data.py, embeddings.py, index.py, documents.py, evaluate.py |
| mlops | Регистрация экспериментов и управление версиями артефактов | tracking.py, dvc.py, artifacts.py |

Контроллеры не выполняют SQL, инференс или операции Redis. Ошибки сервисов
преобразуются в HTTP-ответы централизованно в `server/errors.py`.
Сериализация и номера позиций выдачи находятся в схеме ответа, поиск и получение
полных документов выполняются сервисом

`server/app.py` создаёт зависимости в lifespan: пул PostgreSQL, клиент Redis,
репозиторий кеша, сервис чтения документов и сервис поиска. `server/dependencies.py`
передаёт их контроллерам через FastAPI Depends. Ресурсы закрываются в обратном
порядке, включая ошибку инициализации

## Поиск и согласованность

1. `ml/inference/release.py` загружает совместимые BM25, эмбеддинги и CPU-модели
   один раз на процесс; обращения к БД в загрузчике отсутствуют.
2. `SearchQueue` ограничивает активные и ожидающие запросы и время ответа.
3. `SearchService` открывает собственную транзакцию REPEATABLE READ для запроса.
4. `ReleaseRepository` проверяет версии документов и наличие векторов в этом снимке.
5. Сервис получает ID и оценки из Redis либо выполняет BM25, кодирование запроса
   и pgvector. `DenseSearch` передаёт готовый вектор в `EmbeddingRepository`.
6. Результаты объединяются через RRF; опционально включается reranker.
7. `DocumentRepository` получает полные документы одной выборкой в той же
   транзакции. Redis хранит только ID и оценки, успешные результаты имеют TTL

При обновлении документа текущий поиск видит согласованный старый снимок.
Следующий поиск отклоняет устаревший выпуск до пересборки. Отдельный
`DocumentService` позволяет открыть актуальный документ независимо от состояния
индекса. Недоступность Redis даёт промах кеша и не отключает поиск

Каждая параллельная задача владеет отдельной AsyncSession. CPU-инференс и SDK MinIO
выполняются в потоках с ограничением параллельности. Отмена запроса не освобождает
слот раньше реального завершения вычисления. Закрытие сервера дожидается принятых
задач перед освобождением клиентов

## ML-процессы

```text
Подготовка корпуса → импорт документов → BM25 и эмбеддинги → импорт pgvector
                                                       ↓
                                    offline-оценка → MLflow и отчёт
```

`pipelines` управляет этапами, входными файлами и ресурсами. В `ml/inference`
находятся готовые модели и инференс, в `ml/retrieval` - алгоритмы поиска,
в `ml/evaluation` - метрики и сравнение результатов. Эти ML-компоненты не
импортируют FastAPI, репозитории сервера или ORM-модели

`ml/training` содержит проверку конфигурации и явно отложенные функции
дообучения и отбора отрицательных примеров. Обучение не реализовано.
Подготовка эмбеддингов готовой моделью не считается обучением

CLI `nlp-ranker-pipeline` остаётся общим входом для этапов. DVC запускает эти
команды и отслеживает новые пути исходников. MLflow сохраняет эффективную
конфигурацию, входные версии и все исходники установленного пакета.
MinIO обслуживается репозиторием `server/repository/s3/artifacts.py`, доступным
также пайплайнам через CLI `python -m nlp_ranker.mlops.artifacts`

## Полное дерево исходников

Файлы `__init__.py` и скомпилированные файлы Python в дереве опущены

```text
src/nlp_ranker/
├── core/
│   ├── concurrency.py
│   ├── config.py
│   └── logging.py
├── domain/
│   ├── documents.py
│   ├── errors.py
│   └── search.py
├── ml/
│   ├── evaluation/
│   │   ├── config.py
│   │   ├── metrics.py
│   │   └── pool.py
│   ├── inference/
│   │   ├── encoder.py
│   │   ├── release.py
│   │   └── reranker.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   └── hybrid.py
│   ├── training/
│   │   ├── config.py
│   │   └── trainer.py
│   ├── artifacts.py
│   └── config.py
├── mlops/
│   ├── artifacts.py
│   ├── dvc.py
│   └── tracking.py
├── pipelines/
│   ├── data.py
│   ├── documents.py
│   ├── embeddings.py
│   ├── evaluate.py
│   └── index.py
├── server/
│   ├── controller/
│   │   ├── documents.py
│   │   ├── health.py
│   │   └── search.py
│   ├── model/
│   │   ├── documents.py
│   │   └── schemas.py
│   ├── repository/
│   │   ├── cache/
│   │   │   └── search.py
│   │   ├── postgres/
│   │   │   ├── connection.py
│   │   │   ├── documents.py
│   │   │   ├── embeddings.py
│   │   │   └── releases.py
│   │   └── s3/
│   │       └── artifacts.py
│   ├── service/
│   │   ├── contracts.py
│   │   ├── dense.py
│   │   ├── documents.py
│   │   ├── search.py
│   │   └── search_queue.py
│   ├── app.py
│   ├── dependencies.py
│   ├── errors.py
│   ├── metrics.py
│   └── middleware.py
├── ui/
│   ├── app.py
│   └── client.py
└── cli.py
```

## Границы

Запуск API рассчитан на один процесс. Очередь и блокировки инференса локальны
процессу; при добавлении процессов отдельно проверяются память и сбор метрик
Prometheus. PostgreSQL и MinIO не образуют общей атомарной транзакции.
Дообучение и автоматическое обновление документов остаются отдельными
предстоящими блоками. Kubernetes, Airflow и Kafka в этот этап не входят


## Интерфейс Streamlit

`ui/app.py` отвечает за форму, карточки и состояние сессии.
`ui/client.py` отправляет HTTP-запросы и проверяет публичные схемы ответа;
UI не получает клиентов БД, Redis, MinIO или объекты ML-моделей.
Схемы из server/model/schemas.py переиспользуются как контракт HTTP

Доступные значения фильтров читаются из API один раз на сессию. Сервер выбирает
их из текущих документов и оставляет поддерживаемые категории. Форма группирует
поля и запускает поиск только по кнопке. Открытие полного текста выполняет
отдельный GET документа и не запускает новый поиск. Соединения HTTPX закрываются
после каждого запроса, результаты не помещаются в общий кеш Streamlit
