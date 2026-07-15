/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import { ensureQueueFeature } from '../helpers/feature-ensurers.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    renderLiveTrainingDashboard,
    renderTrainingRunSummary,
    resetLiveSystemPeaks,
} from '../helpers/live-status-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { closeHistoryDetailDialog, renderHistoryManagerDetail } from '../helpers/history-task-actions-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { ensureHistoryDetailFeature } from '../helpers/history-detail-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getHistoryState } from '../helpers/history-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { configureHistoryTimelineBridge } from '../helpers/history-timeline-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { formatLr, lastValue, parseMetricsFromProgressLine } from '../../live-training/index.js?v=module-bootstrap-20260714-stage-dataset5';
import { recoverLiveTrainingState, renderLogOutputLines, resetLogOutputLines, setLogStatus } from '../helpers/live-log-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { resetLiveMetricPlaceholders, setEtaMetricText, setMetricText, setText, setTrainingDashboardHeadState, syncLossChartEmptyState, updateDashboardProgressIdleState, updateTrainingToolbarState } from './03-parse-network-arg-entry.js?v=module-bootstrap-20260714-stage-dataset5';
import { showTrainingView } from '../helpers/queue-view-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { loadTrainingHistoryList, renderTrainingHistoryList } from '../helpers/history-list-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';

const historyState = getHistoryState();
const trainingState = getTrainingState();


    export function renderConfigGroupTimeline(payload) {
        const group = payload.group || {};
        const summary = payload.summary || {};
        const banner = document.getElementById('history-view-banner');
        const bannerTitle = document.getElementById('history-view-title');
        if (banner) banner.hidden = false;
        if (bannerTitle) {
            bannerTitle.textContent = `合并查看: ${configGroupLabel(group)} · ${summary.task_count || 0} 次训练`;
        }
        setText('training-run-state', '合并');
        const stateEl = document.getElementById('training-run-state');
        if (stateEl) stateEl.className = 'training-run-state history';
        setTrainingDashboardHeadState('history');
        updateTrainingToolbarState('history', '合并');
        setText('training-run-title', `合并查看: ${configGroupLabel(group)}`);
        setText('training-run-meta', [
            group.methods_subdir ? `方法目录 ${group.methods_subdir}` : '',
            group.variant ? `配置 ${group.variant}` : '',
            group.preset ? `预设 ${group.preset}` : '',
        ].filter(Boolean).join(' · ') || '配置组训练结果');
        renderTrainingRunSummary([
            ['训练次数', `${summary.task_count || 0} 次`],
            ['Loss 点', `${summary.loss_count || 0} 个`],
            ['日志', `${summary.log_count || 0} 行`],
            ['时间', `${summary.started_at_text || '-'} → ${summary.finished_at_text || '未结束'}`],
        ], '配置组训练结果');

        document.getElementById('train-variant').textContent = group.variant || '-';
        document.getElementById('train-preset').textContent = group.preset || '-';
        document.getElementById('progress-bar').style.width = '100%';
        document.getElementById('progress-text').textContent =
            `${summary.started_at_text || '-'} → ${summary.finished_at_text || '持续/未结束'}`;
        updateDashboardProgressIdleState(true);
        setMetricText('metric-vram', 'N/A');
        setMetricText('metric-vram-peak', 'N/A');
        setMetricText('metric-gpu', 'N/A');
        setMetricText('metric-gpu-peak', 'N/A');
        setMetricText('metric-temp', 'N/A');
        setMetricText('metric-temp-peak', 'N/A');
        setMetricText('metric-log-age', '分组合并');
        setMetricText('metric-rate', 'N/A');
        setEtaMetricText({ text: '分组合并', empty: false, title: '合并视图不计算单次训练预计完成时间。' });

        const metrics = payload.metrics || [];
        const lossPoints = metrics.filter((item) => item.loss !== undefined);
        trainingState.lossChart?.setXLabel?.('step');
        trainingState.lossChart?.setScaleMode?.('step', {
            xRange: {
                min: summary.start_display_step ?? lossPoints[0]?.display_step ?? lossPoints[0]?.step,
                max: summary.end_display_step ?? lossPoints[lossPoints.length - 1]?.display_step ?? lossPoints[lossPoints.length - 1]?.step,
            },
        });
        trainingState.lossChart?.setData(lossPoints.map((item) => ({
            step: item.display_step || item.step || 0,
            loss: item.loss,
            lr: item.lr,
            rawStep: item.step,
            displayStepOffset: item.display_step_offset || 0,
            sourceTaskLabel: item.source_task_label || '',
            sourceTaskIndex: item.source_task_index || 0,
            stageBreakBefore: Boolean(item.stage_break_before),
            stageLabel: item.stage_break_before ? `任务${item.source_task_index || ''}` : '',
        })), { keepAll: true });
        syncLossChartEmptyState();
        const lastMetric = metrics[metrics.length - 1] || {};
        const lastLossMetric = lossPoints[lossPoints.length - 1] || {};
        setMetricText('metric-loss',
            lastMetric.loss !== undefined ? Number(lastMetric.loss).toFixed(5) : 'N/A');
        setMetricText('metric-lr', formatLr(lastValue(metrics, 'lr')));
        setMetricText('metric-step',
            lastMetric.display_step ?? lastMetric.step ?? lastLossMetric.display_step ?? lastLossMetric.step ?? 'N/A');

        const logs = payload.logs || [];
        const logEl = document.getElementById('log-output');
        renderLogOutputLines(logs.map(formatGroupTimelineLogRecord));
        logEl.scrollTop = logEl.scrollHeight;
        setLogStatus(`手动合并 · ${logs.length} 行日志 · ${summary.loss_count || 0} Loss 点 · 已隐藏 ${summary.progress_count || 0} 条进度记录`, 'warning');

        const health = document.getElementById('training-health');
        health.className = 'training-health ok';
        health.textContent = [
            `已手动合并 ${summary.task_count || 0} 次训练`,
            `${summary.loss_count || 0} 个 Loss 点`,
            `${summary.log_count || 0} 行日志`,
            `${summary.progress_count || 0} 条进度记录未显示`,
            summary.include_archived ? '包含归档任务' : '',
        ].filter(Boolean).join(' · ');

        const configPanel = document.getElementById('history-config-panel');
        const configTitle = document.getElementById('history-config-title');
        const configOutput = document.getElementById('history-config-output');
        if (configPanel) configPanel.hidden = false;
        if (configTitle) configTitle.textContent = '分组训练明细';
        if (configOutput) configOutput.textContent = configGroupTimelineSummary(payload);
        renderConfigGroupPaths(payload);
        renderResumePanelState();
        if (trainingState.trainingViewMode === 'history') renderHistoryManagerDetail(payload);
    }

    export function formatGroupTimelineLogRecord(record) {
        const taskPrefix = record.source_task_index ? `[任务${record.source_task_index}] ` : '';
        const kindPrefix = record.kind === 'progress' ? '[进度] ' : '';
        return `${taskPrefix}${kindPrefix}${record.line || ''}`;
    }

    export function configGroupTimelineSummary(payload) {
        const group = payload.group || {};
        const lines = [`# 手动合并查看: ${configGroupLabel(group)}`, ''];
        for (const segment of payload.segments || []) {
            const task = segment.task || {};
            const segmentLines = [
                `任务 ${segment.index}: ${task.label || task.id || '-'}`,
                `  ID: ${task.id || '-'}`,
                `  状态: ${historyStateLabel(task.state)}`,
                `  时间: ${task.started_at_text || '-'} -> ${task.finished_at_text || '未结束'}`,
                `  输出目录: ${task.output_dir || '-'}`,
                `  真实 Step: ${formatStepRange(segment.start_display_step, segment.end_display_step)}`,
                segment.display_step_offset ? `  续训偏移: +${segment.display_step_offset}` : '',
                `  日志: ${segment.log_count || 0} 行`,
                `  进度记录: ${segment.progress_count || 0} 条`,
                `  Loss/指标: ${segment.metric_count || 0} 条`,
                '',
            ].filter(Boolean);
            lines.push(...segmentLines);
        }
        return lines.join('\n');
    }

    export function formatStepRange(start, end) {
        if (start === undefined || start === null || end === undefined || end === null) return '-';
        return `${start} -> ${end}`;
    }

    export function renderConfigGroupPaths(payload) {
        const group = payload.group || {};
        const summary = payload.summary || {};
        const el = document.getElementById('history-paths');
        if (!el) return;
        el.innerHTML = '';
        const items = [
            ['配置文件', configGroupLabel(group)],
            ['源配置', group.history_source_config_file || '-'],
            ['合并训练数', `${summary.task_count || 0}`],
            ['时间范围', `${summary.started_at_text || '-'} -> ${summary.finished_at_text || '未结束'}`],
            ['真实步数', formatStepRange(summary.start_display_step, summary.end_display_step)],
            ['归档任务', summary.include_archived ? '已包含' : '未包含'],
        ];
        for (const [label, value] of items) {
            const row = document.createElement('div');
            const key = document.createElement('span');
            key.textContent = label;
            const valEl = document.createElement('code');
            valEl.textContent = value;
            valEl.title = value;
            installSelectableHistoryPathText(valEl);
            row.append(key, valEl);
            el.appendChild(row);
        }
    }

    export function configGroupLabel(group) {
        if (group.methods_subdir === '手动选择') {
            return group.variant || '手动选择';
        }
        return group.history_run_label || group.history_group_label || group.label || `${group.methods_subdir || '-'} / ${group.variant || '-'} / ${group.preset || 'default'}`;
    }

    export function metricsWithProgressFallback(metrics, logs) {
        const out = [...metrics];
        const seen = new Set(out.map(metricIdentity));
        for (const record of logs || []) {
            if (record.kind !== 'progress') continue;
            const parsed = parseMetricsFromProgressLine(record.line);
            if (!parsed) continue;
            const item = { ...parsed, ts: record.ts };
            const key = metricIdentity(item);
            if (seen.has(key)) continue;
            seen.add(key);
            out.push(item);
        }
        out.sort((a, b) => (Number(a.ts || 0) - Number(b.ts || 0)) || (Number(a.step || 0) - Number(b.step || 0)));
        return out;
    }

    export function metricIdentity(item) {
        return [
            item.step ?? '',
            item.loss != null ? Number(item.loss).toFixed(8) : '',
            item.lr != null ? Number(item.lr).toFixed(12) : '',
        ].join('|');
    }

    export function returnToLiveTraining(options = {}) {
        const refresh = options.refresh !== false;
        showTrainingView('live');
        historyState.viewingHistoryTaskId = '';
        historyState.historyViewMode = 'live';
        historyState.currentHistoryConfigGroup = null;
        historyState.currentHistoryTimelineSelection = [];
        historyState.currentHistoryTaskForResume = null;
        ensureHistoryDetailFeature().clearHistoryDetailState();
        closeHistoryDetailDialog();
        clearResumeOptions();
        const banner = document.getElementById('history-view-banner');
        if (banner) banner.hidden = true;
        const resumePanel = document.getElementById('history-resume-panel');
        if (resumePanel) resumePanel.hidden = true;
        const configPanel = document.getElementById('history-config-panel');
        if (configPanel) configPanel.hidden = true;
        const configOutput = document.getElementById('history-config-output');
        if (configOutput) configOutput.textContent = '';
        const paths = document.getElementById('history-paths');
        if (paths) paths.innerHTML = '';
        resetLogOutputLines();
        trainingState.trainingRuntime.logBuffer = [];
        trainingState.trainingRuntime.logFlushPending = false;
        trainingState.trainingRuntime.lastLogId = 0;
        trainingState.trainingRuntime.logLineCount = 0;
        trainingState.trainingRuntime.progressCurrent = 0;
        trainingState.trainingRuntime.progressTotal = 0;
        trainingState.trainingRuntime.progressLabel = '';
        trainingState.trainingRuntime.progressRate = '';
        trainingState.trainingRuntime.progressSecondsPerStep = null;
        trainingState.trainingRuntime.progressUpdatedAt = 0;
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('progress-text').textContent = '暂无正在运行的任务目录...';
        resetLiveMetricPlaceholders();
        resetLiveSystemPeaks();
        renderLiveTrainingDashboard();
        trainingState.stepCounter = 0;
        trainingState.lossChart?.clear();
        syncLossChartEmptyState();
        trainingState.lossChart?.setXLabel?.('step');
        trainingState.lossChart?.setScaleMode?.('index');
        renderTrainingHistoryList();
        if (refresh) {
            recoverLiveTrainingState();
        }
    }

    export async function loadResumeOptionsForTask(taskId = historyState.viewingHistoryTaskId) {
        return ensureHistoryDetailFeature().loadResumeOptionsForTask(taskId);
    }

    export function clearResumeOptions() {
        return ensureHistoryDetailFeature().clearResumeOptions();
    }

    export function renderResumePanelState() {
        return ensureHistoryDetailFeature().renderResumePanelState();
    }

    export function selectedResumeCheckpoint() {
        return ensureHistoryDetailFeature().selectedResumeCheckpoint();
    }

    export async function resumeTrainingFromCheckpoint() {
        return ensureHistoryDetailFeature().resumeTrainingFromCheckpoint();
    }

    export async function queueResumeTrainingFromCheckpoint() {
        return ensureQueueFeature().queueResumeTrainingFromCheckpoint();
    }

    export function setResumeStatus(text, state = '') {
        return ensureHistoryDetailFeature().setResumeStatus(text, state);
    }

    export function renderHistoryPaths(task, options = {}) {
        const el = document.getElementById('history-paths');
        if (!el) return;
        el.innerHTML = '';
        const items = runtimePathItems(task, options);
        for (const [label, value] of items) {
            const row = document.createElement('div');
            const key = document.createElement('span');
            key.textContent = label;
            const valEl = document.createElement('code');
            valEl.textContent = value;
            valEl.title = value;
            installSelectableHistoryPathText(valEl);
            row.append(key, valEl);
            el.appendChild(row);
        }
    }

    export function runtimePathItems(task, options = {}) {
        const includeHistory = options.includeHistory !== false;
        const absolutePath = (value, basePath = '') => historyAbsolutePath(value, task, basePath);
        const runDir = absolutePath(task.run_dir_abs || task.run_dir);
        return [
            includeHistory ? ['历史目录', absolutePath(task.history_dir_abs || task.history_dir)] : null,
            task.training_mode === 'continue_lora' ? ['基于权重', absolutePath(task.continue_from_weight_abs_path)] : null,
            ['本次运行目录', runDir],
            ['实际运行配置', absolutePath(task.runtime_config_file, runDir), 'runtime-config'],
            ['原始配置副本', absolutePath(task.original_config_file, runDir), 'original-config'],
            ['运行时数据集配置', absolutePath(task.dataset_config_file, runDir), 'dataset-config'],
            ['模型缓存目录', absolutePath(task.model_cache_dir, runDir)],
            ['数据集缓存目录', absolutePath(task.dataset_cache_dir, runDir)],
            ['训练结果目录', absolutePath(task.training_output_dir || task.output_dir, runDir)],
            ['样张目录', absolutePath(task.sample_dir, runDir)],
            ['日志目录', absolutePath(task.logs_dir, runDir)],
            includeHistory ? ['历史日志文件', absolutePath(task.logs_path), 'logs'] : null,
            includeHistory ? ['历史指标文件', absolutePath(task.metrics_path), 'metrics'] : null,
            includeHistory ? ['系统指标文件', absolutePath(task.system_path), 'system'] : null,
            includeHistory ? ['历史 TOML 快照', absolutePath(task.config_snapshot), 'config-snapshot'] : null,
        ].filter((item) => item && item[1]);
    }

    export function historyAbsolutePath(value, task = {}, basePath = '') {
        const raw = historyCleanPath(value);
        if (!raw) return '';
        if (historyIsSpecialPath(raw) || historyIsAbsolutePath(raw)) return historyTrimPath(raw);
        const clean = raw.replace(/^\.\//, '').replace(/^\/+/, '');
        if (!clean || clean === '.') {
            return historyAbsolutePath(basePath || task.run_dir_abs || task.run_dir || task.history_dir_abs || '', task);
        }
        const projectRoot = historyProjectRoot(task);
        if (projectRoot && historyLooksProjectRelativePath(clean)) {
            return historyJoinPath(projectRoot, clean);
        }
        if (basePath) {
            const base = historyAbsolutePath(basePath, task);
            if (base) return historyJoinPath(base, clean);
        }
        if (projectRoot) return historyJoinPath(projectRoot, clean);
        return raw;
    }

    export function historyProjectRoot(task = {}) {
        const explicit = historyCleanPath(task.project_root_abs);
        if (historyIsAbsolutePath(explicit)) return historyTrimPath(explicit);
        const historyDir = historyCleanPath(task.history_dir_abs);
        const historyMarker = '/configs/web-training-history/';
        const historyIndex = historyDir.indexOf(historyMarker);
        if (historyIndex > 0) return historyDir.slice(0, historyIndex);
        const runDir = historyCleanPath(task.run_dir_abs || task.run_dir);
        const outputMarker = '/output/runs/';
        const outputIndex = runDir.indexOf(outputMarker);
        if (historyIsAbsolutePath(runDir) && outputIndex > 0) return runDir.slice(0, outputIndex);
        return '';
    }

    export function historyLooksProjectRelativePath(value) {
        return /^(configs|image_dataset|library|logs|models|networks|output|post_image_dataset|scripts|tests|web)(\/|$)/.test(String(value || ''));
    }

    export function historyCleanPath(value) {
        return String(value || '').trim().replace(/\\/g, '/');
    }

    export function historyIsAbsolutePath(value) {
        return value.startsWith('/') || /^[A-Za-z]:\//.test(value);
    }

    export function historyIsSpecialPath(value) {
        return /^[a-z][a-z0-9+.-]*:\/\//i.test(value);
    }

    export function historyTrimPath(value) {
        if (value === '/' || /^[A-Za-z]:\/?$/.test(value)) return value;
        return value.replace(/\/+$/, '');
    }

    export function historyJoinPath(base, path) {
        return `${historyTrimPath(base)}/${String(path || '').replace(/^\/+/, '')}`;
    }

    export function installSelectableHistoryPathText(el) {
        if (!el) return;
        el.classList.add('history-detail-select-all');
        el.addEventListener('dblclick', (event) => {
            event.preventDefault();
            const selection = window.getSelection?.();
            if (!selection || !document.createRange) return;
            const range = document.createRange();
            range.selectNodeContents(el);
            selection.removeAllRanges();
            selection.addRange(range);
        });
    }

    export function historyArtifactUrl(task, artifactKey, options = {}) {
        const taskId = String(task?.id || '').trim();
        const key = String(artifactKey || '').trim();
        if (!taskId || !key) return '#';
        const params = new URLSearchParams();
        if (options.download) params.set('download', '1');
        const suffix = params.toString() ? `?${params.toString()}` : '';
        return `/api/training/history/${encodeURIComponent(taskId)}/artifacts/${encodeURIComponent(key)}${suffix}`;
    }

    export function historyStateLabel(state) {
        return {
            running: '运行中',
            idle: '完成',
            error: '异常',
            interrupted: '已中断',
        }[state] || state || '未知';
    }

    configureHistoryTimelineBridge({
        renderConfigGroupTimeline,
        formatGroupTimelineLogRecord,
        configGroupTimelineSummary,
        formatStepRange,
        renderConfigGroupPaths,
        configGroupLabel,
        metricsWithProgressFallback,
        metricIdentity,
        returnToLiveTraining,
        loadResumeOptionsForTask,
        clearResumeOptions,
        renderResumePanelState,
        selectedResumeCheckpoint,
        resumeTrainingFromCheckpoint,
        queueResumeTrainingFromCheckpoint,
        setResumeStatus,
        renderHistoryPaths,
        runtimePathItems,
        historyAbsolutePath,
        historyProjectRoot,
        historyLooksProjectRelativePath,
        historyCleanPath,
        historyIsAbsolutePath,
        historyIsSpecialPath,
        historyTrimPath,
        historyJoinPath,
        installSelectableHistoryPathText,
        historyArtifactUrl,
        historyStateLabel,
    });

    // ── 事件绑定 ──
