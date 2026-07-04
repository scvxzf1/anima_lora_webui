export function parseProgressRateSeconds(value) {
    const text = String(value || '').trim().toLowerCase();
    if (!text) return null;
    const compact = text.replace(/\s+/g, '');
    const match = compact.match(/([\d.]+)(ms\/it|s\/it|s\/step|it\/s)/);
    if (!match) return null;
    const amount = Number(match[1]);
    if (!Number.isFinite(amount) || amount <= 0) return null;
    const unit = match[2];
    if (unit === 'it/s') return 1 / amount;
    if (unit === 'ms/it') return amount / 1000;
    return amount;
}

export function formatEtaClock(date) {
    const pad = (value) => String(value).padStart(2, '0');
    const time = `${pad(date.getHours())}:${pad(date.getMinutes())}`;
    const now = new Date();
    if (isSameDate(date, now)) return time;
    const tomorrow = new Date(now);
    tomorrow.setDate(now.getDate() + 1);
    if (isSameDate(date, tomorrow)) return `明日 ${time}`;
    return `${date.getMonth() + 1}/${date.getDate()} ${time}`;
}

export function calculateTrainingEtaMetricInfo({
    isRunning,
    current,
    total,
    progressSecondsPerStep,
    progressRate,
    nowMs,
    formatDuration,
} = {}) {
    if (!isRunning) {
        return { text: '待计算', empty: true, title: '训练开始并收到进度后显示预计完成时间。' };
    }
    const currentStep = Number(current || 0);
    const totalSteps = Number(total || 0);
    if (!Number.isFinite(currentStep) || !Number.isFinite(totalSteps) || totalSteps <= 0) {
        return { text: '待计算', empty: true, title: '等待进度总数。' };
    }
    const remaining = Math.max(0, totalSteps - currentStep);
    if (remaining <= 0) {
        return { text: '即将完成', empty: false, title: '当前进度已到达总步数。' };
    }
    const secondsPerStep = progressSecondsPerStep ?? parseProgressRateSeconds(progressRate);
    if (!Number.isFinite(secondsPerStep) || secondsPerStep <= 0) {
        return { text: '待计算', empty: true, title: '等待速度数据后计算预计完成时间。' };
    }
    const remainingSeconds = Math.ceil(remaining * secondsPerStep);
    if (!Number.isFinite(remainingSeconds) || remainingSeconds <= 0) {
        return { text: '即将完成', empty: false, title: '按当前速度估算，剩余不足 1 秒。' };
    }
    const now = Number.isFinite(nowMs) ? nowMs : Date.now();
    const describeDuration = typeof formatDuration === 'function'
        ? formatDuration
        : (seconds) => `${seconds} 秒`;
    const eta = new Date(now + remainingSeconds * 1000);
    return {
        text: formatEtaClock(eta),
        empty: false,
        title: `按当前速度估算，剩余约 ${describeDuration(remainingSeconds)}。`,
    };
}

export function isSameDate(a, b) {
    return a.getFullYear() === b.getFullYear()
        && a.getMonth() === b.getMonth()
        && a.getDate() === b.getDate();
}

export function parseMetricsFromProgressLine(line) {
    const text = String(line || '');
    const metricNumberToken = '([+\\-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+\\-]?\\d+)?|[+\\-]?nan|[+\\-]?inf(?:inity)?)';
    const stepMatch = text.match(/\|\s*(\d+)\/\d+\s*\[/) || text.match(/step[=:/\s]+(\d+)/i);
    const lossMatch = text.match(new RegExp(`(?:avr_)?loss[=:/\\s]+${metricNumberToken}`, 'i'));
    const lrMatch = text.match(new RegExp(`(?:^|[\\s,])(?:lr|learning_rate)[=:/\\s]+${metricNumberToken}`, 'i'));
    const rateMatch = text.match(/([\d.]+\s*(?:s\/it|it\/s|s\/step))/i);
    const out = {};
    if (stepMatch) out.step = Number(stepMatch[1]);
    if (lossMatch) out.loss = lossMatch[1];
    if (lrMatch) out.lr = Number(lrMatch[1]);
    if (rateMatch) out.rate = rateMatch[1].replace(/\s+/g, '');
    if (Object.keys(out).length === 0) return null;
    if (out.step !== undefined && !Number.isFinite(out.step)) delete out.step;
    if (out.lr !== undefined && !Number.isFinite(out.lr)) delete out.lr;
    return Object.keys(out).length ? out : null;
}

export function lastValue(records, key) {
    for (let i = records.length - 1; i >= 0; i -= 1) {
        const value = records[i]?.[key];
        if (value !== undefined && value !== null && value !== '') return value;
    }
    return undefined;
}

export function readConfigNumber(configText, key) {
    const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const match = String(configText || '').match(new RegExp(`^\\s*${escapedKey}\\s*=\\s*([^\\n#]+)`, 'm'));
    if (!match) return undefined;
    const value = Number(match[1].trim().replace(/^["']|["']$/g, ''));
    return Number.isFinite(value) ? value : undefined;
}

export function formatLr(value) {
    if (value === undefined || value === null || value === '') return '-';
    const n = Number(value);
    return Number.isFinite(n) ? n.toExponential(2) : '-';
}
