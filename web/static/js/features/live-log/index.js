/**
 * Live training log stream / websocket helpers.
 * Moved out of anima-app mechanical chunks.
 */
import { isLiveRunningState, parseMetricsFromProgressLine } from '../live-training/index.js?v=module-bootstrap-20260711-ir1';
import {
    MAX_LOG_LINES,
} from '../../config/catalog.js?v=module-bootstrap-20260711-ir1';
import {
    markTrainingActivity,
    updateMetrics,
    updateProgress,
    updateStatus,
    updateSystem,
} from '../anima-app/helpers/live-status-bridge.js?v=module-bootstrap-20260711-ir1';
import { api } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir1';
import { showAppConfirmDialog } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir1';
import { isHistoryReviewMode } from '../anima-app/helpers/history-detail-bridge.js?v=module-bootstrap-20260711-ir1';
import { renderResumePanelState } from '../anima-app/helpers/history-timeline-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadTrainingQueue, updateTrainingQueueFromPayload } from '../anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260711-ir1';
import { loadTrainingHistoryList } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260711-ir1';
import { pollStatus, scheduleStatusPoll } from '../anima-app/helpers/status-polling-bridge.js?v=module-bootstrap-20260711-ir1';
import { configureLiveLogBridge } from '../anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260711-ir1';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir1';

const LOG_RENDER_BATCH_SIZE = 250;
const trainingState = getTrainingState();
const trainingRuntime = trainingState.trainingRuntime;

    export async function stopTraining() {
        const stopBtn = document.getElementById('btn-stop-training');
        const ok = await showAppConfirmDialog({
            title: '停止训练',
            description: '当前运行中的训练任务',
            message: '确定要停止训练吗？停止后当前训练过程会立即中断。',
            confirmText: '停止训练',
            danger: true,
        });
        if (!ok) return;
        const wasDisabled = Boolean(stopBtn?.disabled);
        if (stopBtn) stopBtn.disabled = true;
        try {
            const res = await api('/api/training/stop', { method: 'POST' });
            if (!res.ok) {
                const message = res.error || '停止训练失败';
                appendLog(`[状态] ${message}`);
                setLogStatus('停止训练失败', 'error');
                setTrainingHealthNotice(message, 'error');
                return;
            }
            appendLog(`[状态] ${res.message || '训练停止请求已发送'}`);
            await pollStatus();
            await loadTrainingQueue();
        } catch (e) {
            const message = `停止训练请求失败: ${e.message}`;
            appendLog(`[状态] ${message}`);
            setLogStatus('停止训练失败', 'error');
            setTrainingHealthNotice(message, 'error');
        } finally {
            if (stopBtn) stopBtn.disabled = wasDisabled || !isLiveRunningState(trainingRuntime.state);
        }
    }

    // ── WebSocket ──
    export function connectWebSocket() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        setLogStatus('连接中', 'warning');
        trainingState.ws = new WebSocket(`${proto}//${location.host}/ws/training`);
        trainingState.ws.onopen = () => {
            setLogStatus('已连接', 'ok');
            recoverLiveTrainingState();
        };
        trainingState.ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            handleWsMessage(msg);
        };
        trainingState.ws.onclose = () => {
            setLogStatus('已断开，准备重连', 'warning');
            scheduleStatusPoll({ immediate: true });
            setTimeout(connectWebSocket, 3000);
        };
        trainingState.ws.onerror = () => {
            setLogStatus('连接异常', 'error');
            trainingState.ws.close();
        };
    }

    export function handleWsMessage(msg) {
        switch (msg.type) {
            case 'log':
                if (isHistoryReviewMode()) break;
                markTrainingActivity(msg.ts);
                appendLogRecord(msg);
                break;
            case 'progress':
                if (isHistoryReviewMode()) break;
                updateProgress(msg);
                break;
            case 'metrics':
                if (isHistoryReviewMode()) break;
                updateMetrics(msg);
                break;
            case 'status':
                if (isHistoryReviewMode()) {
                    loadTrainingHistoryList();
                    renderResumePanelState();
                    break;
                }
                updateStatus(msg);
                loadTrainingQueue();
                loadTrainingHistoryList();
                break;
            case 'queue':
                updateTrainingQueueFromPayload(msg);
                loadTrainingHistoryList();
                break;
            case 'system':
                if (isHistoryReviewMode()) break;
                updateSystem(msg);
                break;
        }
    }

    export function appendLog(line) {
        appendLogRecord({ line });
    }

    export function appendLogRecord(record) {
        if (record?.id && record.id <= trainingRuntime.lastLogId) return;
        if (record?.id) trainingRuntime.lastLogId = record.id;

        const line = record?.line ?? '';
        const prefix = record?.kind === 'progress' ? '[进度] ' : '';
        trainingRuntime.logBuffer.push(prefix + line);
        trainingRuntime.logLineCount += 1;
        scheduleLogFlush();
    }

    export function renderLogOutputLines(lines, options = {}) {
        const el = document.getElementById('log-output');
        if (!el) return;
        const normalized = (lines || [])
            .map((line) => String(line || ''))
            .filter((line) => line.length);
        trainingRuntime.logOutputLines = normalized;
        trainingRuntime.logRenderToken = (trainingRuntime.logRenderToken || 0) + 1;
        const token = trainingRuntime.logRenderToken;
        const stickToBottom = Boolean(options.stickToBottom);
        el.textContent = '';

        const appendBatch = (start = 0) => {
            if (token !== trainingRuntime.logRenderToken) return;
            const end = Math.min(start + LOG_RENDER_BATCH_SIZE, normalized.length);
            const fragment = document.createDocumentFragment();
            for (let index = start; index < end; index += 1) {
                const line = normalized[index];
                const span = document.createElement('span');
                span.className = `log-line ${logLineTone(line)}`;
                span.textContent = line;
                fragment.append(span, document.createTextNode('\n'));
            }
            el.appendChild(fragment);
            if (stickToBottom) el.scrollTop = el.scrollHeight;
            if (end < normalized.length) {
                scheduleLogRenderBatch(() => appendBatch(end));
            }
        };

        appendBatch();
    }

    export function currentLogOutputLines() {
        if (Array.isArray(trainingRuntime.logOutputLines)) return [...trainingRuntime.logOutputLines];
        const el = document.getElementById('log-output');
        if (!el) return [];
        return el.textContent.split('\n').filter(Boolean);
    }

    export function resetLogOutputLines() {
        trainingRuntime.logOutputLines = [];
        trainingRuntime.logRenderToken = (trainingRuntime.logRenderToken || 0) + 1;
        const el = document.getElementById('log-output');
        if (el) el.textContent = '';
    }

    export function scheduleLogRenderBatch(callback) {
        const schedule = window.requestAnimationFrame
            ? (fn) => window.requestAnimationFrame(fn)
            : (fn) => window.setTimeout(fn, 16);
        schedule(callback);
    }

    export function logLineTone(line) {
        const text = String(line || '').toLowerCase();
        if (text.includes('traceback') || text.includes('exception') || text.includes('error') || text.includes('错误') || text.includes('异常') || text.includes('失败')) {
            return 'error';
        }
        if (text.includes('warn') || text.includes('warning') || text.includes('警告') || text.includes('跳过')) {
            return 'warning';
        }
        if (text.startsWith('[进度]') || text.includes('progress')) {
            return 'progress';
        }
        if (text.startsWith('[状态]') || text.startsWith('[提示]')) {
            return 'status';
        }
        return 'info';
    }

    export function scheduleLogFlush() {
        if (trainingRuntime.logFlushPending) return;
        trainingRuntime.logFlushPending = true;
        const schedule = window.requestAnimationFrame
            ? (fn) => window.requestAnimationFrame(fn)
            : (fn) => window.setTimeout(fn, 16);
        schedule(flushLogBuffer);
    }

    export function flushLogBuffer() {
        trainingRuntime.logFlushPending = false;
        if (!trainingRuntime.logBuffer.length) return;
        const el = document.getElementById('log-output');
        const nextLines = [...currentLogOutputLines(), ...trainingRuntime.logBuffer];
        trainingRuntime.logBuffer = [];
        const lines = nextLines.filter(Boolean).slice(-MAX_LOG_LINES);
        renderLogOutputLines(lines, { stickToBottom: true });
        trainingRuntime.logLineCount = lines.length;
        el.scrollTop = el.scrollHeight;
        updateLogStatusText();
    }

    export async function replayTrainingLogs(options = {}) {
        if (isHistoryReviewMode()) return;
        const includeMetrics = options.includeMetrics !== false;
        try {
            const payload = await api(`/api/training/logs?after=${trainingRuntime.lastLogId}&limit=1000`);
            for (const record of payload.records || []) {
                if (record.ts) markTrainingActivity(record.ts);
                appendLogRecord(record);
                replayMetricsFromLogRecord(record);
            }
            if (includeMetrics) await replayMetricsHistory();
            updateLogStatusText();
        } catch (e) {
            setLogStatus('日志回放失败', 'error');
        }
    }

    export async function replayMetricsHistory() {
        if (isHistoryReviewMode()) return;
        try {
            const records = await api('/api/training/metrics');
            for (const record of records || []) {
                updateMetrics(record, { replay: true });
            }
        } catch (e) {
            // 历史指标不是训练控制关键路径，失败时保留日志回放。
        }
    }

    export function replayMetricsFromLogRecord(record) {
        const line = record?.line || '';
        const parsed = parseMetricsFromProgressLine(line);
        if (!parsed || parsed.loss === undefined) return;
        // tqdm 日志里的 s/it 是累计均速，会被前期编译/慢步骤拖高；实时监控速度只信结构化 metrics。
        const metrics = { ...parsed };
        delete metrics.rate;
        updateMetrics({ ...metrics, ts: record.ts });
    }

    export function setLogStatus(text, state = '') {
        const el = document.getElementById('log-status');
        if (!el) return;
        el.textContent = text;
        el.className = `log-status ${state}`.trim();
    }

    export function updateLogStatusText() {
        const state = trainingState.ws?.readyState === WebSocket.OPEN ? 'ok' : 'warning';
        const text = trainingState.ws?.readyState === WebSocket.OPEN
            ? `已连接 · ${trainingRuntime.logLineCount} 行`
            : `${trainingRuntime.logLineCount} 行`;
        setLogStatus(text, state);
    }

    export function setTrainingHealthNotice(message, state = 'warning') {
        const el = document.getElementById('training-health');
        if (!el) return;
        el.className = `training-health ${state}`.trim();
        el.textContent = message;
    }

    export async function recoverLiveTrainingState() {
        if (isHistoryReviewMode() || location.protocol === 'file:') return;
        await pollStatus({ forceReplayMetrics: true });
        await loadTrainingQueue();
    }

configureLiveLogBridge({
    stopTraining,
    connectWebSocket,
    handleWsMessage,
    appendLog,
    appendLogRecord,
    renderLogOutputLines,
    currentLogOutputLines,
    resetLogOutputLines,
    scheduleLogRenderBatch,
    logLineTone,
    scheduleLogFlush,
    flushLogBuffer,
    replayTrainingLogs,
    replayMetricsHistory,
    replayMetricsFromLogRecord,
    setLogStatus,
    updateLogStatusText,
    setTrainingHealthNotice,
    recoverLiveTrainingState,
});
