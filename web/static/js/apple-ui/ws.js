/* WebSocket helper for live training log streaming.
 * Connects to /ws/training and dispatches messages to callbacks.
 * Auto-reconnects with exponential backoff.
 */

const WS_URL = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/training`;
const MAX_LOG_LINES = 500;

let ws = null;
let reconnectDelay = 1000;
let reconnectTimer = null;
let intentionallyClosed = false;

const handlers = {
    log: [],
    metrics: [],
    system: [],
    progress: [],
    status: [],
    open: [],
    close: [],
};

export function onMessage(type, callback) {
    if (!handlers[type]) handlers[type] = [];
    handlers[type].push(callback);
    return () => {
        const list = handlers[type];
        const index = list.indexOf(callback);
        if (index >= 0) list.splice(index, 1);
    };
}

export function onOpen(callback) {
    handlers.open.push(callback);
}

export function onClose(callback) {
    handlers.close.push(callback);
}

export function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return;
    }

    intentionallyClosed = false;
    try {
        ws = new WebSocket(WS_URL);
    } catch {
        scheduleReconnect();
        return;
    }

    ws.onopen = () => {
        reconnectDelay = 1000;
        handlers.open.forEach((cb) => cb());
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            const type = msg?.type;
            if (type && handlers[type]) {
                handlers[type].forEach((cb) => cb(msg));
            }
        } catch { /* ignore parse errors */ }
    };

    ws.onerror = () => { /* errors handled by onclose */ };

    ws.onclose = () => {
        handlers.close.forEach((cb) => cb());
        if (!intentionallyClosed) scheduleReconnect();
    };
}

export function disconnectWebSocket() {
    intentionallyClosed = true;
    if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }
    if (ws) {
        ws.close();
        ws = null;
    }
}

function scheduleReconnect() {
    if (intentionallyClosed) return;
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(() => connectWebSocket(), reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 1.5, 10000);
}

/* Append a log line to a container element, keeping max lines. */
export function appendLogLine(container, line, level) {
    if (!container) return;
    const div = document.createElement('div');
    div.className = 'apple-log-line';
    if (level) div.dataset.level = level;
    div.textContent = line;
    container.appendChild(div);

    // Trim old lines
    while (container.children.length > MAX_LOG_LINES) {
        container.removeChild(container.firstChild);
    }

    container.scrollTop = container.scrollHeight;
}

export { MAX_LOG_LINES };
