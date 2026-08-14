#!/usr/bin/env bash
# WebUI 快捷启动：等待服务就绪后打开 Dragon UI。
# 环境变量覆盖:
#   ANIMA_WEB_HOST / ANIMA_WEB_PORT
#   ANIMA_WEB_BROWSER_HOST（外部绑定时指定本机访问地址）
#   ANIMA_WEB_OPEN_BROWSER=0（只启动服务，不打开浏览器）
#   ANIMA_WEB_READY_TIMEOUT（等待秒数，默认 45）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export ANIMA_WEB_HOST="${ANIMA_WEB_HOST:-127.0.0.1}"
export ANIMA_WEB_PORT="${ANIMA_WEB_PORT:-20203}"

BROWSER_HOST="${ANIMA_WEB_BROWSER_HOST:-$ANIMA_WEB_HOST}"
case "$BROWSER_HOST" in
  0.0.0.0|::|"[::]") BROWSER_HOST="127.0.0.1" ;;
esac

DRAGON_URL="http://${BROWSER_HOST}:${ANIMA_WEB_PORT}/?ui=dragon"
OPEN_BROWSER="${ANIMA_WEB_OPEN_BROWSER:-1}"
READY_TIMEOUT="${ANIMA_WEB_READY_TIMEOUT:-45}"

if ! [[ "$READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "[Anima LoRA] ANIMA_WEB_READY_TIMEOUT 必须是正整数。" >&2
  exit 1
fi

open_url() {
  local url="$1"
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1
  elif command -v gio >/dev/null 2>&1; then
    gio open "$url" >/dev/null 2>&1
  elif command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1
  else
    return 1
  fi
}

dragon_ready() {
  local html
  html="$(curl -fsS --max-time 1 "$DRAGON_URL" 2>/dev/null)" || return 1
  [[ "$html" == *'id="dragon-root"'* ]]
}

if ! command -v curl >/dev/null 2>&1; then
  echo "[Anima LoRA] 未找到 curl，无法自动检测和打开页面。"
  echo "[Anima LoRA] 服务启动后请手动打开：$DRAGON_URL"
  exec ./start_webui.sh "$@"
fi

if dragon_ready; then
  echo "[Anima LoRA] Dragon UI 已在运行：$DRAGON_URL"
  if [ "$OPEN_BROWSER" != "0" ]; then
    open_url "$DRAGON_URL" || echo "[Anima LoRA] 请手动打开：$DRAGON_URL"
  fi
  exit 0
fi

echo "[Anima LoRA] WebUI 快捷启动 → $DRAGON_URL"
./start_webui.sh "$@" &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

READY=0
for ((attempt = 0; attempt < READY_TIMEOUT * 2; attempt++)); do
  if dragon_ready; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

if [ "$READY" = "1" ]; then
  echo "[Anima LoRA] Dragon UI 已启动：$DRAGON_URL"
  if [ "$OPEN_BROWSER" != "0" ]; then
    open_url "$DRAGON_URL" || echo "[Anima LoRA] 请手动打开：$DRAGON_URL"
  fi
else
  echo "[Anima LoRA] Dragon UI 未在 ${READY_TIMEOUT}s 内就绪。" >&2
  echo "[Anima LoRA] 可稍后手动打开：$DRAGON_URL" >&2
fi

set +e
wait "$SERVER_PID"
STATUS=$?
set -e
trap - INT TERM EXIT
exit "$STATUS"
