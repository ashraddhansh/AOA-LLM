FROM ghcr.io/astral-sh/uv:0.13.8 AS uv

FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

COPY --from=uv /uv /uvx /bin/
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

COPY backend ./backend
COPY alembic.ini ./alembic.ini
COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN mkdir -p /app/.api_data/chroma /app/.api_uploads \
    && chmod +x /app/docker/entrypoint.sh \
    && chown -R app:app /app

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8000') + '/health', timeout=3)" || exit 1
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["serve"]
