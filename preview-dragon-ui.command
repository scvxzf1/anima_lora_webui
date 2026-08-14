#!/bin/bash

set -u

HOST="127.0.0.1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}" || {
    echo "无法进入项目目录：${SCRIPT_DIR}"
    read -n 1 -r -p "按任意键退出..." _
    exit 1
}

STATIC_DIR="${SCRIPT_DIR}/web/static"
PIDFILE="/tmp/anima-dragon-ui-preview.pid"
PORTFILE="/tmp/anima-dragon-ui-preview.port"

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
        URL="http://${HOST}:${OLD_PORT}/?ui=dragon"
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

URL="http://${HOST}:${PORT}/?ui=dragon"
"${PYTHON}" -m web --host "${HOST}" --port "${PORT}" &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PIDFILE}"
echo "${PORT}" > "${PORTFILE}"

cleanup() {
    kill "${SERVER_PID}" 2>/dev/null || true
    rm -f "${PIDFILE}" "${PORTFILE}"
}
trap cleanup INT TERM EXIT

# 首次启动需加载 torch，可能耗时 10~40 秒，等服务器真正就绪后再打开浏览器。
echo "正在启动 Dragon trainer WebUI，请稍候…"
READY=0
for _ in $(seq 1 90); do
    if curl -s -o /dev/null -m 1 "${URL}" 2>/dev/null; then
        READY=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "WebUI 进程提前退出，启动失败，请检查上方输出。"
        break
    fi
    sleep 0.5
done

if [ "${READY}" = "1" ]; then
    echo "Dragon trainer WebUI 已启动：${URL}"
    echo "关闭此终端窗口或按 Ctrl+C 可停止预览。"
    open "${URL}" 2>/dev/null || true
else
    echo "启动超时或失败。可稍后手动访问：${URL}"
fi

wait "${SERVER_PID}"
