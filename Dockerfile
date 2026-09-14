FROM ghcr.io/astral-sh/uv:0.12.23 AS uv

FROM python:3.12-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_NO_CACHE=1

WORKDIR /app

ARG INSTALL_DEMO=false

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs

RUN if [ "$INSTALL_DEMO" = "true" ]; then \
        uv sync --frozen --no-dev --no-editable --extra serving --extra demo; \
    else \
        uv sync --frozen --no-dev --no-editable --extra serving; \
    fi

FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=4 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src /app/src
COPY --from=builder --chown=app:app /app/configs /app/configs

USER app

EXPOSE 8000

HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"

CMD ["uvicorn", "nlp_ranker.server.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-graceful-shutdown", "15", "--limit-concurrency", "64"]
