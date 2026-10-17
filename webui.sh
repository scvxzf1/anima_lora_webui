#!/usr/bin/env bash
# WebUI 快捷启动：等待服务就绪后打开 Next UI。
# 环境变量覆盖:
#   ANIMA_WEB_HOST / ANIMA_WEB_PORT
#   ANIMA_WEB_BROWSER_HOST（外部绑定时指定本机访问地址）
#   ANIMA_WEB_OPEN_BROWSER=0（只启动服务，不打开浏览器）
#   ANIMA_WEB_READY_TIMEOUT（等待秒数，默认 45）
#   ANIMA_WEB_BUILD_FRONTEND=0（跳过启动前的前端构建）
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export ANIMA_WEB_HOST="${ANIMA_WEB_HOST:-127.0.0.1}"
export ANIMA_WEB_PORT="${ANIMA_WEB_PORT:-20203}"

# The WebUI serves the compiled bundle from web/static/dragon-next rather than
# the TypeScript source tree.  Build it on every launcher start so the Ubuntu
# application-menu shortcut always picks up source changes before opening the UI.
if [ "${ANIMA_WEB_BUILD_FRONTEND:-1}" != "0" ]; then
  # Ubuntu .desktop launches do not load the interactive shell's nvm PATH.
  # Prefer the project's Node 24 installation (pnpm 11 requires Node >=22.13)
  # before resolving pnpm; otherwise /usr/bin/pnpm may run under Node 20.
  for node_bin in "$HOME"/.nvm/versions/node/*/bin; do
    if [ -x "$node_bin/node" ] && [ -x "$node_bin/pnpm" ]; then
      node_version="$($node_bin/node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
      if [ "$node_version" -ge 22 ]; then
        export PATH="$node_bin:$PATH"
        break
      fi
    fi
  done
  if ! command -v pnpm >/dev/null 2>&1; then
    echo "[Anima LoRA] 未找到 pnpm，无法更新前端静态包。设置 ANIMA_WEB_BUILD_FRONTEND=0 可跳过构建。" >&2
    exit 1
  fi
  echo "[Anima LoRA] 正在构建并发布 Next 前端..."
  pnpm --dir "$PROJECT_DIR/web/frontend-next" build
fi

BROWSER_HOST="${ANIMA_WEB_BROWSER_HOST:-$ANIMA_WEB_HOST}"
case "$BROWSER_HOST" in
  0.0.0.0|::|"[::]") BROWSER_HOST="127.0.0.1" ;;
esac

NEXT_URL="http://${BROWSER_HOST}:${ANIMA_WEB_PORT}/next"
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

next_ready() {
  local html
  html="$(curl -fsS --max-time 1 "$NEXT_URL" 2>/dev/null)" || return 1
  [[ "$html" == *'id="root"'* ]]
}

if ! command -v curl >/dev/null 2>&1; then
  echo "[Anima LoRA] 未找到 curl，无法自动检测和打开页面。"
  echo "[Anima LoRA] 服务启动后请手动打开：$NEXT_URL"
  exec ./start_webui.sh "$@"
fi
echo "[Anima LoRA] WebUI 快捷启动 → $NEXT_URL"
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
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[Anima LoRA] WebUI 服务已提前退出。" >&2
    wait "$SERVER_PID" || true
    exit 1
  fi
  if next_ready; then
    READY=1
    break
  fi
  sleep 0.5
done

if [ "$READY" = "1" ]; then
  echo "[Anima LoRA] Next UI 已启动：$NEXT_URL"
  if [ "$OPEN_BROWSER" != "0" ]; then
    open_url "$NEXT_URL" || echo "[Anima LoRA] 请手动打开：$NEXT_URL"
  fi
else
  echo "[Anima LoRA] Next UI 未在 ${READY_TIMEOUT}s 内就绪。" >&2
  echo "[Anima LoRA] 可稍后手动打开：$NEXT_URL" >&2
fi

set +e
wait "$SERVER_PID"
STATUS=$?
set -e
trap - INT TERM EXIT
exit "$STATUS"
