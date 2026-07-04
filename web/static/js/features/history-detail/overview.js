import { historySystemSummary, formatSystemPercent, formatSystemVram } from './system.js?v=module-bootstrap-20260704-2';
import {
    createHistoryDetailCopyButton,
    historyDetailEmptyText,
    historyDetailRow as createHistoryDetailRow,
    historyDetailRunRoot,
    historyDetailSection,
    normalizedHistoryDetailPath,
} from './ui.js?v=module-bootstrap-20260704-2';

export function createHistoryOverviewRenderer({ ctx, state, deps, renderHistoryDetailResume }) {
    const {
        configGroupLabel,
        formatLr,
        formatStepRange,
        historyContinueLabel,
        historyQueueLabel,
        historyResumeLabel,
        historyStateLabel,
        lastValue,
        metricsWithProgressFallback,
        runtimePathItems,
    } = deps;
    const historyDetailRow = (label, value, options = {}) => createHistoryDetailRow(label, value, options, {
        copyButton: (copyValue, labelText) => createHistoryDetailCopyButton(ctx.dom.copyText, copyValue, labelText),
    });

    function renderHistoryDetailOverview(payload) {
        const box = document.createElement('div');
        box.className = 'history-detail-overview history-detail-overview-dashboard';
        box.classList.toggle('config-group', payload.mode === 'config_group');
        const task = payload.task || {};
        if (task.job === 'preprocess') {
            return renderPreprocessHistoryOverview(payload, box);
        }
        const summary = payload.summary || {};
        const metrics = payload.metrics || [];
        const logs = payload.logs || [];
        const lossPoints = metricsWithProgressFallback(metrics, logs).filter((item) => item.loss !== undefined);
        const lastLoss = lossPoints[lossPoints.length - 1]?.loss;
        const systemSummary = historySystemSummary(payload);
        const stats = document.createElement('div');
        stats.className = 'history-detail-stat-grid';
        const rows = payload.mode === 'config_group'
            ? [
                ['训练数', summary.task_count || 0, 'steps'],
                ['Loss 点', summary.loss_count || 0, 'curve'],
                ['日志', summary.log_count || 0, 'log'],
                ['真实步数', formatStepRange(summary.start_display_step, summary.end_display_step), 'steps'],
                ['系统', '合并视图暂不汇总', 'chip'],
            ]
            : [
                ['状态', historyStateLabel(task.state), 'gauge'],
                ['Loss', lastLoss !== undefined ? Number(lastLoss).toFixed(5) : '-', 'curve'],
                ['学习率', formatLr(lastValue(metrics, 'lr')), 'bolt'],
                ['步数', lastValue(metrics, 'step') ?? '-', 'steps'],
                ['Loss 点', task.metric_count || lossPoints.length || 0, 'curve'],
                ['日志', task.log_count || logs.length || 0, 'log'],
                ['峰值 VRAM', systemSummary.hasSystem ? formatSystemVram(systemSummary.peakVramRecord) : '无系统采样记录', 'chip'],
                ['峰值 GPU', systemSummary.hasSystem ? formatSystemPercent(systemSummary.peakGpu) : '无系统采样记录', 'gpu'],
                ['平均速度', formatHistoryAverageSpeed(task), 'gauge'],
                ['训练总时间', formatHistoryTaskDuration(task), 'time'],
            ];
        for (const [label, value, iconName] of rows) stats.appendChild(renderHistoryStat(label, value, iconName));

        const info = document.createElement('div');
        info.className = 'history-detail-kv';
        const kvRows = payload.mode === 'config_group'
            ? [
                ['配置文件', configGroupLabel(payload.group || {})],
                ['源配置', payload.group?.history_source_config_file || '-'],
                ['时间范围', `${summary.started_at_text || '-'} → ${summary.finished_at_text || '未结束'}`],
                ['包含归档', summary.include_archived ? '是' : '否'],
            ]
            : [
                ['任务 ID', task.id],
                ['来源配置', task.history_source_config_file || '-'],
                ['运行标签', task.history_run_label || '-'],
                ['时间', `${task.started_at_text || '-'} → ${task.finished_at_text || '未结束'}`],
                ['消息', task.message || '-'],
                ['队列', historyQueueLabel(task) || '-'],
                ['续训', historyResumeLabel(task) || historyContinueLabel(task) || '-'],
            ];
        const taskFinished = !['running', 'compiling', 'queued'].includes(task.state);
        for (const [label, value] of kvRows) {
            info.appendChild(historyDetailRow(label, value, {
                muted: taskFinished && ['队列', '续训'].includes(label) && value === '-',
            }));
        }
        const infoBlock = document.createElement('div');
        infoBlock.className = 'history-detail-info-block';
        const infoTitle = document.createElement('h5');
        infoTitle.textContent = '任务信息';
        infoBlock.append(infoTitle, info);
        const metricsBody = document.createElement('div');
        metricsBody.className = 'history-detail-metrics-body';
        metricsBody.append(stats, infoBlock);
        box.append(
            renderHistoryDetailProgress(payload),
            historyDetailSection('实时指标', metricsBody, 'history-detail-section metrics'),
        );
        if (payload.mode !== 'config_group') {
            box.appendChild(historyDetailSection('关键文件', renderHistoryDetailPathSummary(payload), 'history-detail-section paths-summary'));
        }
        if (task.job === 'training') {
            box.appendChild(renderHistoryDetailResume(payload));
        }
        return box;
    }

    function renderPreprocessHistoryOverview(payload, box) {
        const task = payload.task || {};
        const logs = payload.logs || [];
        const systemSummary = historySystemSummary(payload);
        box.classList.add('preprocess-task');

        const stats = document.createElement('div');
        stats.className = 'history-detail-stat-grid history-preprocess-stat-grid';
        [
            ['状态', historyStateLabel(task.state), 'gauge'],
            ['日志', task.log_count || logs.length || 0, 'log'],
            ['峰值 VRAM', systemSummary.hasSystem ? formatSystemVram(systemSummary.peakVramRecord) : '无系统采样记录', 'chip'],
            ['峰值 GPU', systemSummary.hasSystem ? formatSystemPercent(systemSummary.peakGpu) : '无系统采样记录', 'gpu'],
            ['预处理用时', formatHistoryTaskDuration(task), 'time'],
            ['缓存目录', compactHistoryPathName(task.dataset_cache_dir || task.run_dir), 'chip'],
        ].forEach(([label, value, iconName]) => stats.appendChild(renderHistoryStat(label, value, iconName)));

        const info = document.createElement('div');
        info.className = 'history-detail-kv history-preprocess-info';
        [
            ['任务 ID', task.id],
            ['来源配置', task.history_source_config_file || '-'],
            ['运行标签', task.history_run_label || '-'],
            ['时间', `${task.started_at_text || '-'} → ${task.finished_at_text || '未结束'}`],
            ['源图目录', task.source_image_dir || '-'],
            ['消息', task.message || '-'],
            ['队列', historyQueueLabel(task) || '-'],
        ].forEach(([label, value]) => {
            info.appendChild(historyDetailRow(label, value, {
                copyValue: label.endsWith('目录') && value !== '-' ? value : '',
            }));
        });

        const infoBlock = document.createElement('div');
        infoBlock.className = 'history-detail-info-block';
        const infoTitle = document.createElement('h5');
        infoTitle.textContent = '预处理信息';
        infoBlock.append(infoTitle, info);

        const summaryBody = document.createElement('div');
        summaryBody.className = 'history-detail-metrics-body history-preprocess-summary-body';
        summaryBody.append(stats, infoBlock);
        box.append(
            renderHistoryDetailProgress(payload),
            historyDetailSection('预处理摘要', summaryBody, 'history-detail-section metrics preprocess-summary'),
            historyDetailSection('预处理文件', renderHistoryDetailPathSummary(payload), 'history-detail-section paths-summary preprocess-paths'),
        );
        return box;
    }

    function renderHistoryStat(label, value, iconName) {
        const item = document.createElement('div');
        item.className = 'history-detail-stat';
        historyStatClasses(label, value).forEach((className) => item.classList.add(className));
        const strong = document.createElement('strong');
        strong.textContent = value;
        const caption = document.createElement('span');
        caption.className = 'history-detail-stat-label';
        if (iconName) {
            const icon = document.createElement('span');
            icon.className = `metric-icon metric-icon-${iconName}`;
            icon.setAttribute('aria-hidden', 'true');
            caption.appendChild(icon);
        }
        caption.appendChild(document.createTextNode(label));
        item.append(strong, caption);
        return item;
    }

    function historyStatClasses(label, value) {
        const text = String(label || '').trim();
        const valueText = String(value || '').trim();
        const classes = [];
        const primaryLabels = new Set(['状态', '训练数', 'Loss', '步数', '真实步数', '平均速度']);
        const secondaryLabels = new Set(['学习率', 'Loss 点', '日志', '系统', '峰值 VRAM', '峰值 GPU', '训练总时间', '预处理用时', '缓存目录']);
        if (primaryLabels.has(text)) classes.push('is-primary');
        if (secondaryLabels.has(text)) classes.push('is-secondary');
        if (text === '状态') {
            classes.push('stat-status', historyStateToneClass(valueText));
        } else if (text === 'Loss') {
            classes.push('stat-loss');
        } else if (text.includes('速度')) {
            classes.push('stat-speed');
        } else if (text.includes('VRAM') || text.includes('GPU') || text === '系统') {
            classes.push('stat-system');
        } else if (text.includes('时间') || text.includes('用时')) {
            classes.push('stat-time');
        } else if (text.includes('步')) {
            classes.push('stat-steps');
        }
        return classes.filter(Boolean);
    }

    function historyStateToneClass(valueText) {
        if (/失败|错误|中断|取消/.test(valueText)) return 'state-error';
        if (/运行|训练中|编译|处理中/.test(valueText)) return 'state-running';
        if (/排队|等待/.test(valueText)) return 'state-queued';
        if (/完成|成功|已结束|done|completed/i.test(valueText)) return 'state-complete';
        return 'state-neutral';
    }

    function renderHistoryDetailProgress(payload) {
        const task = payload.task || {};
        const summary = payload.summary || {};
        const progress = historyProgressStepSummary(payload);
        const section = document.createElement('section');
        section.className = 'history-detail-progress history-detail-section';
        const title = document.createElement('div');
        title.className = 'history-detail-section-title';
        const name = document.createElement('h4');
        name.textContent = '进度';
        const time = document.createElement('span');
        time.textContent = payload.mode === 'config_group'
            ? `${summary.started_at_text || '-'} → ${summary.finished_at_text || '未结束'}`
            : `${task.started_at_text || '-'} → ${task.finished_at_text || '未结束'}`;
        title.append(name, time);

        const meter = document.createElement('div');
        meter.className = 'history-detail-progress-meter';
        const bar = document.createElement('div');
        bar.className = 'history-detail-progress-bar';
        const fill = document.createElement('span');
        const terminal = payload.mode === 'config_group' || ['idle', 'done', 'completed'].includes(task.state);
        const finished = terminal && (progress.percent === null || progress.percent >= 99.95 || payload.mode === 'config_group');
        section.classList.toggle('is-complete', finished);
        section.classList.add(historyProgressToneClass(payload.mode === 'config_group' ? 'completed' : task.state));
        fill.style.width = progress.percent !== null
            ? `${progress.percent.toFixed(1)}%`
            : (finished ? '100%' : (task.state === 'running' ? '68%' : '0%'));
        bar.setAttribute('role', 'progressbar');
        bar.setAttribute('tabindex', '0');
        bar.setAttribute('aria-valuemin', '0');
        bar.setAttribute('aria-valuemax', '100');
        if (progress.percent !== null) {
            bar.setAttribute('aria-valuenow', progress.percent.toFixed(1));
        }
        bar.setAttribute('aria-valuetext', progress.summaryText);
        bar.title = progress.summaryText;
        bar.setAttribute('aria-label', progress.percent !== null
            ? `进度 ${progress.percent.toFixed(1)}%，已完成 ${progress.completeLabel}，总步数 ${progress.totalLabel}，剩余 ${progress.remainingLabel}`
            : `进度 ${fill.style.width || '0%'}`);
        bar.appendChild(fill);

        const stats = document.createElement('div');
        stats.className = 'history-detail-progress-stats';
        stats.title = progress.summaryText;
        [
            ['已完成', progress.completeLabel, 'complete'],
            ['总步数', progress.totalLabel, 'total'],
            ['剩余', progress.remainingLabel, 'remaining'],
        ].forEach(([label, value, kind]) => {
            const item = document.createElement('span');
            item.className = `history-detail-progress-stat ${kind}`;
            item.setAttribute('aria-label', `${label} ${value}`);
            item.title = `${label}: ${value}`;
            const key = document.createElement('b');
            key.textContent = label;
            const val = document.createElement('strong');
            val.textContent = value;
            item.append(key, val);
            stats.appendChild(item);
        });
        const copySummary = createHistoryDetailCopyButton(ctx.dom.copyText, progress.summaryText, '进度摘要');
        copySummary.classList.add('history-detail-progress-copy');
        stats.appendChild(copySummary);
        meter.append(bar, stats);

        const message = document.createElement('p');
        message.textContent = payload.mode === 'config_group'
            ? `已合并 ${summary.task_count || 0} 次训练，${summary.loss_count || 0} 个 Loss 点。`
            : (task.message || historyStateLabel(task.state) || '历史任务记录');
        section.append(title, meter, message);
        return section;
    }

    function historyProgressStepSummary(payload) {
        const task = payload.task || {};
        const summary = payload.summary || {};
        const metrics = metricsWithProgressFallback(payload.metrics || [], payload.logs || []);
        const progressPayload = task.latest_progress || payload.latest_progress || {};
        const logSteps = historyProgressStepsFromLogs(payload.logs || []);
        let current = historyProgressNumber(progressPayload.current)
            ?? logSteps.current
            ?? historyProgressNumber(lastValue(metrics, 'step'))
            ?? historyProgressNumber(task.average_step_end_step)
            ?? historyProgressNumber(summary.end_display_step);
        let total = historyProgressNumber(progressPayload.total)
            ?? logSteps.total
            ?? historyProgressNumber(task.total_steps)
            ?? historyProgressNumber(summary.total_steps);

        if (payload.mode === 'config_group') {
            current = historyProgressNumber(summary.end_display_step)
                ?? historyProgressNumber(summary.loss_count)
                ?? current;
            total = total ?? current;
        }
        if (current !== null && total !== null && current > total) total = current;

        const remaining = current !== null && total !== null ? Math.max(0, total - current) : null;
        const percent = current !== null && total !== null && total > 0
            ? Math.max(0, Math.min(100, current / total * 100))
            : null;
        return {
            current,
            total,
            remaining,
            percent,
            completeLabel: historyProgressNumberLabel(current),
            totalLabel: historyProgressNumberLabel(total),
            remainingLabel: historyProgressNumberLabel(remaining),
            summaryText: historyProgressSummaryText({ current, total, remaining, percent }, payload),
        };
    }

    function historyProgressSummaryText(progress, payload) {
        const task = payload.task || {};
        const summary = payload.summary || {};
        const percentText = progress.percent !== null ? `${progress.percent.toFixed(1)}%` : '-';
        const message = payload.mode === 'config_group'
            ? `合并 ${summary.task_count || 0} 次训练`
            : (task.message || historyStateLabel(task.state) || '历史任务记录');
        return [
            `进度 ${percentText}`,
            `已完成 ${historyProgressNumberLabel(progress.current)}`,
            `总步数 ${historyProgressNumberLabel(progress.total)}`,
            `剩余 ${historyProgressNumberLabel(progress.remaining)}`,
            message,
        ].filter(Boolean).join(' · ');
    }

    function historyProgressStepsFromLogs(logs) {
        for (let i = (logs || []).length - 1; i >= 0; i -= 1) {
            const record = logs[i] || {};
            if (record.kind && record.kind !== 'progress') continue;
            const match = String(record.line || '').match(/\|\s*([\d,]+)\s*\/\s*([\d,]+)\s*\[/);
            if (!match) continue;
            const current = historyProgressNumber(match[1]);
            const total = historyProgressNumber(match[2]);
            if (current !== null && total !== null) return { current, total };
        }
        return { current: null, total: null };
    }

    function historyProgressNumber(value) {
        if (value === undefined || value === null || value === '') return null;
        const number = Number(String(value).replace(/,/g, ''));
        return Number.isFinite(number) && number >= 0 ? Math.trunc(number) : null;
    }

    function historyProgressNumberLabel(value) {
        return value === null || value === undefined ? '-' : String(value);
    }

    function historyProgressToneClass(state) {
        const value = String(state || '').toLowerCase();
        if (['done', 'completed', 'idle'].includes(value)) return 'state-complete';
        if (['running', 'compiling'].includes(value)) return 'state-running';
        if (value === 'queued') return 'state-queued';
        if (['error', 'failed', 'canceled'].includes(value)) return 'state-error';
        return 'state-neutral';
    }

    function formatHistoryTaskDuration(record) {
        const startedAt = Number(record?.started_at);
        if (!Number.isFinite(startedAt) || startedAt <= 0) return '-';
        const finishedAt = Number(record?.finished_at);
        const endAt = Number.isFinite(finishedAt) && finishedAt > 0
            ? finishedAt
            : Date.now() / 1000;
        return ctx.format.formatDuration(Math.round(Math.max(0, endAt - startedAt)));
    }

    function formatHistoryAverageSpeed(record) {
        const rate = String(record?.average_step_rate || '').trim();
        if (rate) return rate;
        const seconds = Number(record?.average_step_seconds);
        return Number.isFinite(seconds) && seconds > 0 ? `${seconds.toFixed(2)}s/step` : '-';
    }

    function compactHistoryPathName(value) {
        const text = String(value || '').replace(/\\/g, '/').trim();
        if (!text) return '-';
        const parts = text.split('/').filter(Boolean);
        if (!parts.length) return text;
        const last = parts[parts.length - 1] || '';
        const prev = parts[parts.length - 2] || '';
        return prev ? `${prev}/${last}` : last;
    }

    function renderHistoryDetailPathSummary(payload) {
        const box = document.createElement('div');
        box.className = 'history-detail-kv history-detail-path-summary';
        const task = payload.task || {};
        const rootPath = historyDetailRunRoot(task);
        const preferredLabels = task.job === 'preprocess'
            ? new Set([
                '历史目录',
                '本次运行目录',
                '实际运行配置',
                '运行时数据集配置',
                '模型缓存目录',
                '数据集缓存目录',
                '日志目录',
            ])
            : new Set([
                '历史目录',
                '本次运行目录',
                '实际运行配置',
                '训练结果目录',
                '样张目录',
            ]);
        if (rootPath) {
            box.appendChild(historyDetailRow('运行根目录', rootPath, {
                className: 'history-detail-path-root',
                copyValue: rootPath,
            }));
        }
        for (const [label, value] of runtimePathItems(task).filter(([label]) => preferredLabels.has(label))) {
            if (label === '本次运行目录' && normalizedHistoryDetailPath(value) === normalizedHistoryDetailPath(rootPath)) {
                continue;
            }
            box.appendChild(historyDetailRow(label, value, {
                className: 'history-detail-path-row',
                copyValue: value,
            }));
        }
        if (!box.childElementCount) {
            box.appendChild(historyDetailEmptyText('这个任务没有记录可展示的关键文件路径。'));
        }
        return box;
    }

    return { renderHistoryDetailOverview, renderHistoryDetailProgress, renderHistoryDetailPathSummary };
}
