#!/usr/bin/env bash
# Anima LoRA WebUI 一键启动（终端保留日志，不自动打开浏览器）
set -u

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${ANIMA_WEB_HOST:-127.0.0.1}"
PORT="${ANIMA_WEB_PORT:-20103}"
URL="http://${HOST}:${PORT}/"
LOG_DIR="${PROJECT_DIR}/logs"
PY="${PROJECT_DIR}/.venv/bin/python"

cd "$PROJECT_DIR" || {
  echo "[Anima LoRA] 找不到项目目录: $PROJECT_DIR" >&2
  read -r -p "按回车关闭终端..." _
  exit 1
}

mkdir -p "$LOG_DIR"

echo "[Anima LoRA] 一键启动 WebUI"
echo "[Anima LoRA] 项目: $PROJECT_DIR"
echo "[Anima LoRA] 地址: $URL"
echo "[Anima LoRA] 不会自动打开浏览器，请手动访问上面的地址"
echo "[Anima LoRA] 终端会保留，方便看日志"
echo

port_in_use() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :$PORT )" 2>/dev/null | rg -q ":${PORT}\\b"
    return $?
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 1
}

if port_in_use; then
  echo "[Anima LoRA] 端口 ${PORT} 已在监听，视为 WebUI 已启动。"
  echo "[Anima LoRA] 请手动打开: $URL"
  echo
  read -r -p "按回车关闭本终端（不会停止已运行的 WebUI）..." _
  exit 0
fi

if [ -x "$PY" ] && "$PY" -c "import aiohttp" >/dev/null 2>&1; then
  STARTER=("./start_webui.sh")
else
  echo "[Anima LoRA] 环境不完整，先走 setup_and_start.sh"
  STARTER=("./setup_and_start.sh")
fi

ANIMA_WEB_HOST="$HOST" ANIMA_WEB_PORT="$PORT" "${STARTER[@]}" "$@"
status=$?

echo
if [ "$status" -eq 0 ]; then
  echo "[Anima LoRA] WebUI 已退出。"
else
  echo "[Anima LoRA] WebUI 异常退出，退出码: $status" >&2
fi
read -r -p "按回车关闭终端..." _
exit "$status"
