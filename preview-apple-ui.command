#!/bin/bash

set -u

HOST="127.0.0.1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATIC_DIR="${SCRIPT_DIR}/web/static"
PIDFILE="/tmp/anima-apple-ui-preview.pid"
PORTFILE="/tmp/anima-apple-ui-preview.port"

pause_before_exit() {
    read -n 1 -r -p "按任意键退出..." _
}

PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [ ! -x "${PYTHON}" ]; then
    PYTHON="$(command -v python3 || true)"
fi
if [ -z "${PYTHON}" ]; then
    echo "未找到项目 Python 环境，无法启动 WebUI。"
    pause_before_exit
    exit 1
fi

if [ ! -f "${STATIC_DIR}/index.html" ]; then
    echo "未找到前端文件：${STATIC_DIR}/index.html"
    pause_before_exit
    exit 1
fi

if [ -f "${PIDFILE}" ] && [ -f "${PORTFILE}" ]; then
    OLD_PID="$(cat "${PIDFILE}" 2>/dev/null || true)"
    OLD_PORT="$(cat "${PORTFILE}" 2>/dev/null || true)"
    if [ -n "${OLD_PID}" ] && kill -0 "${OLD_PID}" 2>/dev/null && [ -n "${OLD_PORT}" ]; then
        URL="http://${HOST}:${OLD_PORT}/?ui=apple"
        echo "预览已在运行：${URL}"
        open "${URL}" 2>/dev/null || true
        exit 0
    fi
    rm -f "${PIDFILE}" "${PORTFILE}"
fi

PORT=""
for CANDIDATE in $(seq 20102 20120); do
    if ! lsof -nP -iTCP:"${CANDIDATE}" -sTCP:LISTEN >/dev/null 2>&1; then
        PORT="${CANDIDATE}"
        break
    fi
done

if [ -z "${PORT}" ]; then
    echo "20102-20120 端口均被占用，无法启动预览。"
    pause_before_exit
    exit 1
fi

URL="http://${HOST}:${PORT}/?ui=apple"
"${PYTHON}" -m web --host "${HOST}" --port "${PORT}" &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PIDFILE}"
echo "${PORT}" > "${PORTFILE}"

cleanup() {
    kill "${SERVER_PID}" 2>/dev/null || true
    rm -f "${PIDFILE}" "${PORTFILE}"
}
trap cleanup INT TERM EXIT

sleep 1
echo "训练器正式 WebUI 已启动：${URL}"
echo "关闭此终端窗口或按 Ctrl+C 可停止预览。"
open "${URL}" 2>/dev/null || true
wait "${SERVER_PID}"
