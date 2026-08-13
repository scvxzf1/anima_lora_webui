/* Pure state helpers for Dragon's live training workspace. */

const RUNNING_STATES = new Set(['running', 'training', 'compiling', 'caching', 'saving']);

export function createLiveModel(status = {}, metricsPayload = [], logsPayload = {}) {
    const progress = status?.latest_progress || {};
    const latestMetric = status?.latest_metric || {};
    const system = status?.latest_system || {};
    const metrics = normalizeMetrics(metricsPayload);
    const logs = Array.isArray(logsPayload?.records) ? logsPayload.records : [];
    const step = numberOr(progress.current ?? progress.step ?? latestMetric.step, 0);
    const total = numberOr(progress.total ?? progress.total_steps, 0);
    const progressPct = progressPercent(step, total);

    const model = {
        state: status?.status || 'idle',
        step,
        total,
        progressPct,
        loss: latestMetric.loss ?? progress.loss,
        lr: latestMetric.lr ?? progress.lr,
        epoch: latestMetric.epoch ?? progress.epoch ?? 0,
        rate: progress.rate ?? latestMetric.rate ?? '',
        vram: readFirst(system, ['vram_used_gb', 'gpu_memory_used', 'vram_used']),
        vramTotal: system.vram_total_gb,
        gpuTemp: system.gpu_temp,
        gpuUtil: system.gpu_util,
        peakVram: readFirst(system, ['peak_vram_used_gb', 'vram_peak_gb', 'vram_used_peak_gb']),
        peakGpuUtil: readFirst(system, ['peak_gpu_util', 'gpu_util_peak']),
        peakGpuTemp: readFirst(system, ['peak_gpu_temp', 'gpu_temp_peak']),
        metrics,
        logs,
        logClearBeforeId: 0,
        logQuery: '',
        autoScroll: true,
        apiConnected: true,
        apiError: '',
        wsState: 'connecting',
        wsError: '',
        configLabel: formatConfigLabel(status),
        runDir: status?.run_dir || status?.output_dir || status?.training_output_dir || '尚未启动训练',
        lastActivity: status?.anomaly_message || status?.error_hint || status?.last_log_line || stateText(status?.status),
        lastLogId: Number(status?.last_log_id || logs.at?.(-1)?.id || 0),
    };
    seedSystemPeaks(model);
    return model;
}

export function mergeStatusSnapshot(model, status = {}, metricsPayload = [], logsPayload = {}) {
    const next = createLiveModel(status, metricsPayload, logsPayload);
    const sameRun = liveRunIdentity(model) === liveRunIdentity(next);
    const preserved = {
        autoScroll: model.autoScroll,
        logQuery: model.logQuery,
        logClearBeforeId: model.logClearBeforeId,
        wsState: model.wsState,
        wsError: model.wsError,
    };
    Object.assign(model, next, preserved);
    if (sameRun) {
        model.peakVram = maximum(model.peakVram, next.peakVram, next.vram);
        model.peakGpuUtil = maximum(model.peakGpuUtil, next.peakGpuUtil, next.gpuUtil);
        model.peakGpuTemp = maximum(model.peakGpuTemp, next.peakGpuTemp, next.gpuTemp);
    } else {
        seedSystemPeaks(model);
        model.logClearBeforeId = 0;
    }
    return model;
}

export function applyProgress(model, message = {}) {
    const progress = message.progress || message;
    model.step = numberOr(progress.current ?? progress.step, model.step);
    model.total = numberOr(progress.total ?? progress.total_steps, model.total);
    if (progress.loss != null) model.loss = progress.loss;
    if (progress.lr != null) model.lr = progress.lr;
    if (progress.epoch != null) model.epoch = progress.epoch;
    if (progress.rate != null) model.rate = progress.rate;
    model.progressPct = progressPercent(model.step, model.total);
}

export function applySystem(model, message = {}) {
    const system = message.system || message;
    const vram = readFirst(system, ['vram_used_gb', 'gpu_memory_used', 'vram_used']);
    if (vram != null) model.vram = vram;
    if (system.vram_total_gb != null) model.vramTotal = system.vram_total_gb;
    if (system.gpu_temp != null) model.gpuTemp = system.gpu_temp;
    if (system.gpu_util != null) model.gpuUtil = system.gpu_util;
    model.peakVram = maximum(model.peakVram, readFirst(system, ['peak_vram_used_gb', 'vram_peak_gb']), model.vram);
    model.peakGpuUtil = maximum(model.peakGpuUtil, readFirst(system, ['peak_gpu_util', 'gpu_util_peak']), model.gpuUtil);
    model.peakGpuTemp = maximum(model.peakGpuTemp, readFirst(system, ['peak_gpu_temp', 'gpu_temp_peak']), model.gpuTemp);
}

export function seedSystemPeaks(model) {
    model.peakVram = maximum(model.peakVram, model.vram);
    model.peakGpuUtil = maximum(model.peakGpuUtil, model.gpuUtil);
    model.peakGpuTemp = maximum(model.peakGpuTemp, model.gpuTemp);
}

export function visibleLogs(model) {
    const clearBefore = Number(model.logClearBeforeId || 0);
    const records = clearBefore > 0
        ? model.logs.filter((record) => Number(record.id || 0) > clearBefore)
        : model.logs;
    const query = String(model.logQuery || '').trim().toLocaleLowerCase();
    if (!query) return records;
    return records.filter((record) => logRecordText(record).toLocaleLowerCase().includes(query));
}

export function logRecordText(record = {}) {
    return String(record.message || record.text || record.line || '');
}

export function connectionState(model) {
    if (!model.apiConnected) {
        return { tone: 'error', label: '监控接口异常', detail: model.apiError || '无法读取训练状态，请检查 WebUI 服务后重试。' };
    }
    if (model.wsState === 'closed') {
        return { tone: 'warning', label: '实时连接已断开', detail: model.wsError || '正在自动重连；页面仍会定时刷新状态。' };
    }
    if (model.wsState === 'connecting') {
        return { tone: 'info', label: '正在连接实时通道…', detail: '状态接口可用，正在建立日志和指标推送连接。' };
    }
    return { tone: 'success', label: '监控连接正常', detail: '状态接口与实时推送均可用。' };
}

export function isRunningState(state) {
    return RUNNING_STATES.has(String(state || '').toLowerCase());
}

export function stateText(state) {
    const map = {
        idle: '空闲', running: '训练中', training: '训练中', compiling: '编译中',
        caching: '缓存中', saving: '保存中', queued: '排队中', completed: '已完成',
        error: '错误', stopped: '已停止', unavailable: '连接异常', unknown: '未知',
    };
    return map[String(state || '').toLowerCase()] || '未知';
}

export function visualState(state) {
    return String(state || '').toLowerCase() === 'unavailable' ? 'error' : String(state || 'unknown').toLowerCase();
}

export function formatConfigLabel(status = {}) {
    const values = [status.variant, status.preset].filter(Boolean);
    return values.join(' · ') || status.history_source_config_file || status.runtime_config_file || '尚未选择训练配置';
}

function normalizeMetrics(payload) {
    if (Array.isArray(payload?.metrics)) return payload.metrics;
    return Array.isArray(payload) ? payload : [];
}

function progressPercent(step, total) {
    return total > 0 ? Math.max(0, Math.min(100, (step / total) * 100)) : 0;
}

function liveRunIdentity(model = {}) {
    return `${model.runDir || ''}\u0000${model.configLabel || ''}`;
}

function readFirst(source, keys) {
    for (const key of keys) {
        if (source?.[key] != null) return source[key];
    }
    return undefined;
}

function numberOr(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
}

function maximum(...values) {
    const numbers = values.map(Number).filter(Number.isFinite);
    return numbers.length ? Math.max(...numbers) : undefined;
}
