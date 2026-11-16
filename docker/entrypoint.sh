#!/bin/sh
set -eu

mode="${1:-serve}"
shift || true

case "$mode" in
  serve)
    exec /app/.venv/bin/uvicorn backend.api.main:app \
      --host 0.0.0.0 --port "${PORT:-8000}" "$@"
    ;;
  migrate)
    exec /app/.venv/bin/alembic "$@"
    ;;
  *)
    echo "Unknown container command: $mode" >&2
    exit 2
    ;;
esac
