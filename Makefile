DOCKER ?= docker
COMPOSE = $(DOCKER) compose

.PHONY: install install-ml install-demo install-all demo test-up test-http test-down serve lint format typecheck test check dry-run build up up-all down status logs

install:
	uv sync --locked

install-ml:
	uv sync --locked --extra ml

install-demo:
	uv sync --locked --extra demo

install-all:
	uv sync --locked --extra ml --extra demo

demo:
	uv run --no-sync streamlit run src/nlp_ranker/ui/app.py

serve:
	OPENBLAS_NUM_THREADS=1 uv run --no-sync uvicorn nlp_ranker.server.app:create_app --factory --limit-concurrency 64

lint:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .

format:
	uv run --no-sync ruff check --fix .
	uv run --no-sync ruff format .

typecheck:
	uv run --no-sync mypy

test:
	uv run --no-sync pytest

check: lint typecheck test

dry-run:
	uv run --no-sync nlp-ranker-pipeline prepare --config configs/data.yaml --dry-run
	uv run --no-sync nlp-ranker-pipeline index --config configs/search.yaml --dry-run
	uv run --no-sync nlp-ranker-pipeline train --config configs/training.yaml --dry-run
	uv run --no-sync nlp-ranker-pipeline evaluate --config configs/evaluation.yaml --dry-run

build:
	uv build

up:
	$(COMPOSE) --profile storage up -d --build --wait --wait-timeout 180

up-all:
	$(COMPOSE) --profile tracking --profile monitoring --profile storage --profile demo up -d --build --wait --wait-timeout 180

status:
	$(COMPOSE) --profile tracking --profile monitoring --profile storage --profile demo ps

logs:
	$(COMPOSE) --profile tracking --profile monitoring --profile storage --profile demo logs --tail 100

down:
	$(COMPOSE) --profile tracking --profile monitoring --profile storage --profile demo down

.PHONY: migrate migration-sql
migrate:
	uv run --no-sync alembic upgrade head

migration-sql:
	uv run --no-sync alembic upgrade head --sql

test-up:
	$(COMPOSE) -f compose.test.yaml up -d --build --wait --wait-timeout 180

test-http:
	NLP_RANKER_TEST_API_URL=http://localhost:8001 uv run --no-sync pytest tests/test_api.py

test-down:
	$(COMPOSE) -f compose.test.yaml down

.PHONY: prepare
prepare:
	uv run --no-sync nlp-ranker-pipeline prepare --config configs/data.yaml

.PHONY: prepare-labels
prepare-labels:
	uv run --no-sync python -c 'from nlp_ranker.pipelines.data import prepare_labels; prepare_labels()'

TEST_POSTGRES_DSN = postgresql+asyncpg://nlp_ranker:test-only@127.0.0.1:5433/nlp_ranker_test

.PHONY: load-documents test-database
load-documents:
	uv run --no-sync python -m nlp_ranker.pipelines.documents

test-database:
	$(COMPOSE) -f compose.test.yaml up -d --wait postgres
	NLP_RANKER_POSTGRES_DSN=$(TEST_POSTGRES_DSN) uv run --no-sync alembic upgrade head
	NLP_RANKER_POSTGRES_DSN=$(TEST_POSTGRES_DSN) uv run --no-sync alembic check
	NLP_RANKER_TEST_POSTGRES_DSN=$(TEST_POSTGRES_DSN) uv run --no-sync pytest tests/test_database.py tests/test_dense.py

.PHONY: storage-init test-storage data-repro data-push data-pull
storage-init:
	uv run --no-sync python -m nlp_ranker.mlops.artifacts init

TEST_S3_ENDPOINT_URL = http://127.0.0.1:9002

test-storage:
	$(COMPOSE) -f compose.test.yaml up -d --wait minio
	NLP_RANKER_TEST_S3_ENDPOINT_URL=$(TEST_S3_ENDPOINT_URL) uv run --no-sync pytest tests/test_storage.py

data-repro:
	uv run --no-sync python -m nlp_ranker.mlops.artifacts dvc repro

data-push:
	uv run --no-sync python -m nlp_ranker.mlops.artifacts dvc push

data-pull:
	uv run --no-sync python -m nlp_ranker.mlops.artifacts dvc pull

.PHONY: index evaluate track-index track-evaluate
index:
	uv run --no-sync nlp-ranker-pipeline index --config configs/search.yaml

evaluate:
	uv run --no-sync nlp-ranker-pipeline evaluate --config configs/evaluation.yaml

track-index:
	uv run --no-sync nlp-ranker-pipeline index --config configs/search.yaml --track

track-evaluate:
	uv run --no-sync nlp-ranker-pipeline evaluate --config configs/evaluation.yaml --track

.PHONY: embed load-dense track-embed evaluate-dense evaluate-hybrid evaluate-rerank
embed:
	uv run --no-sync nlp-ranker-pipeline embed --config configs/search.yaml

load-dense:
	uv run --no-sync nlp-ranker-pipeline load-dense --config configs/search.yaml

track-embed:
	uv run --no-sync nlp-ranker-pipeline embed --config configs/search.yaml --track

evaluate-dense:
	uv run --no-sync nlp-ranker-pipeline evaluate --config configs/evaluation-dense.yaml

evaluate-hybrid:
	uv run --no-sync nlp-ranker-pipeline evaluate --config configs/evaluation-hybrid.yaml

evaluate-rerank:
	uv run --no-sync nlp-ranker-pipeline evaluate --config configs/evaluation-rerank.yaml

.PHONY: install-serving test-server
install-serving:
	uv sync --locked --extra serving

test-server:
	$(COMPOSE) -f compose.test.yaml up -d --wait postgres redis
	NLP_RANKER_POSTGRES_DSN=$(TEST_POSTGRES_DSN) uv run --no-sync alembic upgrade head
	NLP_RANKER_TEST_POSTGRES_DSN=$(TEST_POSTGRES_DSN) NLP_RANKER_TEST_REDIS_URL=redis://127.0.0.1:6380/9 uv run --no-sync pytest tests/test_serving.py tests/test_server_http.py

.PHONY: monitoring-up monitoring-check load-test
monitoring-up:
	$(COMPOSE) --profile storage --profile monitoring up -d --build --wait --wait-timeout 180

monitoring-check:
	$(COMPOSE) --profile monitoring run --rm --no-deps --entrypoint promtool -w /etc/prometheus prometheus check rules alerts.yaml
	$(COMPOSE) --profile monitoring run --rm --no-deps --entrypoint promtool -w /etc/prometheus -v ./deployment/prometheus/alerts.test.yaml:/etc/prometheus/alerts.test.yaml:ro prometheus test rules alerts.test.yaml

LOAD_USERS ?= 1
LOAD_DURATION ?= 60s
LOAD_SPAWN_RATE ?= 1
LOAD_MODE ?= mixed
LOAD_REPORT ?= artifacts/reports/load
load-test:
	NLP_RANKER_LOAD_MODE=$(LOAD_MODE) uv run --no-sync locust -f tests/load/locustfile.py --headless --host http://127.0.0.1:8000 --users $(LOAD_USERS) --spawn-rate $(LOAD_SPAWN_RATE) --run-time $(LOAD_DURATION) --csv $(LOAD_REPORT) --html $(LOAD_REPORT).html --exit-code-on-error 1
