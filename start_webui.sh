#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

HOST="${ANIMA_WEB_HOST:-127.0.0.1}"
PORT="${ANIMA_WEB_PORT:-20103}"
PY=".venv/bin/python"

echo "[Anima LoRA] Start WebUI"
echo "[Anima LoRA] Project: $(pwd)"

if [ ! -x "$PY" ]; then
  echo "[Anima LoRA] Existing environment was not found at $PY." >&2
  echo "[Anima LoRA] Run ./setup_and_start.sh first, or run uv sync manually." >&2
  exit 1
fi

if ! "$PY" -c "import aiohttp" >/dev/null 2>&1; then
  echo "[Anima LoRA] aiohttp is missing from the venv." >&2
  echo "[Anima LoRA] Run ./setup_and_start.sh first, or run uv sync manually." >&2
  exit 1
fi

echo "[Anima LoRA] Starting WebUI at http://$HOST:$PORT/"
exec "$PY" -m web --host "$HOST" --port "$PORT" "$@"
