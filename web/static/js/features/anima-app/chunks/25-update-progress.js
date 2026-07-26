/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    calculateTrainingEtaMetricInfo,
    formatLr,
    isLiveRunningState,
    parseProgressRateSeconds,
} from '../../live-training/index.js?v=module-bootstrap-20260714-stage-dataset5';
import { ensurePreviewFeature } from '../helpers/feature-ensurers.js?v=module-bootstrap-20260714-stage-dataset5';
import { configureLiveStatusBridge } from '../helpers/live-status-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { formatLossValue } from '../../history-detail/curve/data.js?v=module-bootstrap-20260714-stage-dataset5';
import { formatSystemPercent, formatSystemTemperature } from '../../history-detail/system.js?v=module-bootstrap-20260714-stage-dataset5';
import { formatCompactNumber, numberOrNull } from '../../history-detail/ui.js?v=module-bootstrap-20260714-stage-dataset5';
import { getAppContext } from '../helpers/app-context-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { isHistoryReviewMode } from '../helpers/history-detail-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { appendLog, logLineTone } from '../helpers/live-log-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { copyText } from '../helpers/preview-view-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { renderHistoryPaths, runtimePathItems } from '../helpers/history-timeline-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { renderLiveChartPanel, resetLiveMetricPlaceholders, setEtaMetricText, setMetricText, setText, setTrainingDashboardHeadState, syncLossChartEmptyState, updateDashboardProgressIdleState, updateTrainingToolbarState } from './03-parse-network-arg-entry.js?v=module-bootstrap-20260714-stage-dataset5';
import { refreshQueueRunningProgressViews } from '../helpers/queue-view-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import { getTrainingState } from '../helpers/training-state-bridge.js?v=module-bootstrap-20260714-stage-dataset5';

const ctx = getAppContext();
const trainingState = getTrainingState();
const trainingRuntime = trainingState.trainingRuntime;

    export function updateProgress(msg, options = {}) {
        if (isHistoryReviewMode()) return;
        markTrainingActivity(msg.ts, { resetQuietHint: options.replay !== true });
        const previousCurrent = Number(trainingRuntime.progressCurrent || 0);
        const previousUpdatedAt = Number(trainingRuntime.progressUpdatedAt || 0);
        const now = Date.now();
        const pct = msg.total > 0 ? (msg.current / msg.total * 100) : 0;
        trainingRuntime.progressCurrent = Number(msg.current) || 0;
        trainingRuntime.progressTotal = Number(msg.total) || 0;
        trainingRuntime.progressLabel = msg.label || '';
        trainingRuntime.progressRate = msg.rate || '';
        const progressAdvanced = trainingRuntime.progressCurrent > previousCurrent;
        const rateSeconds = parseProgressRateSeconds(msg.rate);
        if (rateSeconds !== null) {
            trainingRuntime.progressSecondsPerStep = rateSeconds;
        } else if (previousUpdatedAt && progressAdvanced) {
            const elapsedSeconds = (now - previousUpdatedAt) / 1000;
            const stepDelta = trainingRuntime.progressCurrent - previousCurrent;
            const inferredSeconds = elapsedSeconds / stepDelta;
            if (Number.isFinite(inferredSeconds) && inferredSeconds > 0) {
                trainingRuntime.progressSecondsPerStep = inferredSeconds;
            }
        }
        if (!options.replay || progressAdvanced || !previousUpdatedAt) {
            trainingRuntime.progressUpdatedAt = now;
        }
        document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
        let text = `${msg.label}: ${msg.current}/${msg.total} (${pct.toFixed(1)}%)`;
        if (msg.rate) text += ` — ${msg.rate}`;
        document.getElementById('progress-text').textContent = text;
        updateDashboardProgressIdleState(true);
        setMetricText('metric-step', msg.current);
        if (msg.rate) setMetricText('metric-rate', msg.rate);
        renderLiveTrainingDashboard();
        refreshQueueRunningProgressViews();
    }

    export function updateMetrics(msg, options = {}) {
        if (isHistoryReviewMode()) return;
        markTrainingActivity(msg.ts, { resetQuietHint: options.replay !== true });
        const lrText = msg.lr !== undefined ? formatLr(msg.lr) : '';
        const lrNumber = msg.lr !== undefined ? Number(msg.lr) : null;
        const lossNumber = msg.loss !== undefined ? Number(msg.loss) : null;
        if (msg.loss !== undefined) {
            if (Number.isFinite(lossNumber)) {
                setMetricText('metric-loss', lossNumber.toFixed(5));
            } else {
                setMetricText('metric-loss', formatLossValue(msg.loss));
            }
        }
        if (msg.loss !== undefined && Number.isFinite(lossNumber)) {
            const step = msg.step || ++trainingState.stepCounter;
            const metadata = { rawStep: msg.step ?? step };
            if (Number.isFinite(lrNumber)) metadata.lr = lrNumber;
            trainingState.lossChart?.push(step, lossNumber, metadata);
            syncLossChartEmptyState();
        }
        if (msg.lr !== undefined) {
            setMetricText('metric-lr', lrText);
            if ((msg.loss === undefined || !Number.isFinite(lossNumber)) && Number.isFinite(lrNumber)) {
                trainingState.lossChart?.updatePointMetadata?.(msg.step, { lr: lrNumber });
            }
        }
        if (msg.step !== undefined) {
            setMetricText('metric-step', msg.step);
        }
        if (msg.rate) {
            trainingRuntime.progressRate = msg.rate;
            const rateSeconds = parseProgressRateSeconds(msg.rate);
            if (rateSeconds !== null) trainingRuntime.progressSecondsPerStep = rateSeconds;
            setMetricText('metric-rate', msg.rate);
        }
        renderLiveChartPanel();
        renderLiveTrainingDashboard();
    }

    export function updateStatus(msg) {
        if (isHistoryReviewMode()) return;
        const dot = document.querySelector('.dot');
        const text = document.getElementById('status-text');
        const stopBtn = document.getElementById('btn-stop-training');
        const state = liveStatusState(msg);
        const terminalMessage = terminalStatusMessage(msg);

        dot.className = 'dot ' + state;
        const stateMap = { idle: '空闲', running: '训练中', error: '错误', compiling: '编译中' };
        const jobLabel = msg.job === 'preprocess' ? '预处理中' : (stateMap[state] || state);
        text.textContent = state === 'running' ? jobLabel : (stateMap[state] || state);
        updateTrainingToolbarState(state, text.textContent);
        trainingRuntime.state = state;
        trainingRuntime.job = msg.job || trainingRuntime.job || '';
        trainingRuntime.variant = msg.variant || trainingRuntime.variant || '';
        trainingRuntime.preset = msg.preset || trainingRuntime.preset || '';
        trainingRuntime.methodsSubdir = msg.methods_subdir || trainingRuntime.methodsSubdir || '';
        trainingRuntime.lastTerminalMessage = state === 'error' ? terminalMessage : '';
        trainingRuntime.lastTerminalHint = state === 'error' ? String(msg.error_hint || '').trim() : '';
        if (Object.prototype.hasOwnProperty.call(msg, 'anomaly_message')) {
            trainingRuntime.lastAnomalyMessage = String(msg.anomaly_message || '').trim();
        } else if (state === 'running' || (state === 'idle' && !terminalMessage)) {
            trainingRuntime.lastAnomalyMessage = '';
        }
        if (msg.last_output_at) {
            markTrainingActivity(msg.last_output_at);
        }
        if (!isLiveRunningState(state)) {
            trainingRuntime.lastOutputAt = 0;
            trainingRuntime.lastUiActivityAt = 0;
            resetLiveSystemPeaks();
        }
        if (msg.output_dir !== undefined) {
            trainingRuntime.outputDir = msg.output_dir || '';
        }
        const shouldHydrateRuntimeSample = state !== 'idle';
        if (msg.sample_dir !== undefined) {
            trainingRuntime.sampleDir = msg.sample_dir || '';
            if (shouldHydrateRuntimeSample) {
                ensurePreviewFeature().updateRuntimeSampleState({ sampleDir: trainingRuntime.sampleDir });
            }
        }
        if (msg.sample_config !== undefined) {
            trainingRuntime.sampleConfig = msg.sample_config || null;
            if (shouldHydrateRuntimeSample) {
                ensurePreviewFeature().updateRuntimeSampleState({ sampleConfig: trainingRuntime.sampleConfig });
            }
        }
        applyRuntimeInfoToState(msg);

        const canStop = isLiveRunningState(state);
        stopBtn.disabled = !canStop;
        stopBtn.classList.toggle('is-emergency', canStop);

        if (msg.variant) document.getElementById('train-variant').textContent = msg.variant;
        if (msg.preset) document.getElementById('train-preset').textContent = msg.preset;

        if (msg.message) appendLog(`[状态] ${msg.message}`);

        if (state === 'idle' || state === 'error') {
            document.getElementById('progress-bar').style.width = '0%';
            trainingRuntime.progressCurrent = 0;
            trainingRuntime.progressTotal = 0;
            trainingRuntime.progressLabel = '';
            trainingRuntime.progressRate = '';
            trainingRuntime.progressSecondsPerStep = null;
            trainingRuntime.progressUpdatedAt = 0;
            document.getElementById('progress-text').textContent = '暂无正在运行的任务目录...';
            updateDashboardProgressIdleState(false);
            trainingRuntime.quietHintShown = false;
            trainingRuntime.job = '';
            refreshQueueRunningProgressViews();
            if (state === 'idle') {
                resetLiveRuntimeSnapshot();
            }
        }
        renderCurrentRuntimePaths();
        renderLiveTrainingDashboard();
        refreshTrainingHealth();
    }

    export function liveStatusState(msg = {}) {
        const state = String(msg.state || 'idle');
        if (state === 'idle' && terminalStatusMessage(msg)) return 'error';
        return state;
    }

    export function terminalStatusMessage(msg = {}) {
        const state = String(msg.state || '');
        const hint = String(msg.error_hint || '').trim();
        const line = String(msg.message || msg.last_log_line || '').trim();
        const lineIsError = logLineTone(line) === 'error';
        if (state !== 'error' && !lineIsError) return '';
        if (hint) {
            if (!line || line === hint) return hint;
            return line.includes(hint) ? line : `${line}；${hint}`;
        }
        return lineIsError ? line : '';
    }

    export function resetLiveSystemPeaks() {
        trainingRuntime.lastGpuUtil = null;
        trainingRuntime.lastGpuTemp = null;
        trainingRuntime.lastVramUsedGb = null;
        trainingRuntime.lastVramTotalGb = null;
        trainingRuntime.peakGpuUtil = null;
        trainingRuntime.peakGpuTemp = null;
        trainingRuntime.peakVramUsedGb = null;
        resetLiveMetricPlaceholders({ primary: false });
    }

    export function clearRuntimeInfo() {
        trainingRuntime.outputDir = '';
        trainingRuntime.sampleDir = '';
        trainingRuntime.sampleConfig = null;
        trainingRuntime.runDir = '';
        trainingRuntime.runtimeConfigFile = '';
        trainingRuntime.originalConfigFile = '';
        trainingRuntime.datasetConfigFile = '';
        trainingRuntime.modelCacheDir = '';
        trainingRuntime.datasetCacheDir = '';
        trainingRuntime.trainingOutputDir = '';
        trainingRuntime.logsDir = '';
        ensurePreviewFeature().updateRuntimeSampleState({
            sampleDir: '',
            sampleConfig: null,
        });
    }

    export function resetLiveRuntimeSnapshot() {
        clearRuntimeInfo();
        trainingRuntime.variant = '';
        trainingRuntime.preset = '';
        trainingRuntime.methodsSubdir = '';
        trainingState.stepCounter = 0;
        resetLiveMetricPlaceholders();
        trainingState.lossChart?.clear?.();
        trainingState.lossChart?.setXLabel?.('step');
        trainingState.lossChart?.setScaleMode?.('index');
        syncLossChartEmptyState();
        setText('train-variant', '-');
        setText('train-preset', '-');
    }

    export function applyRuntimeInfoToState(msg) {
        const fields = {
            run_dir: 'runDir',
            runtime_config_file: 'runtimeConfigFile',
            original_config_file: 'originalConfigFile',
            dataset_config_file: 'datasetConfigFile',
            model_cache_dir: 'modelCacheDir',
            dataset_cache_dir: 'datasetCacheDir',
            training_output_dir: 'trainingOutputDir',
            logs_dir: 'logsDir',
        };
        for (const [wireKey, stateKey] of Object.entries(fields)) {
            if (msg[wireKey] !== undefined) {
                trainingRuntime[stateKey] = msg[wireKey] || '';
            }
        }
    }

    export function renderCurrentRuntimePaths() {
        if (isHistoryReviewMode()) return;
        const configPanel = document.getElementById('history-config-panel');
        const configTitle = document.getElementById('history-config-title');
        const configOutput = document.getElementById('history-config-output');
        const task = currentRuntimeTaskInfo();
        const hasRuntimePaths = runtimePathItems(task, { includeHistory: false }).length > 0;
        if (configPanel) configPanel.hidden = !hasRuntimePaths;
        if (!hasRuntimePaths) {
            const paths = document.getElementById('history-paths');
            if (paths) paths.innerHTML = '';
            if (configOutput) configOutput.textContent = '';
            return;
        }
        if (configTitle) {
            configTitle.textContent = trainingRuntime.job === 'preprocess'
                ? '当前预处理运行目录'
                : '当前任务运行目录';
        }
        if (configOutput) {
            configOutput.textContent = [
                task.runtime_config_file ? `实际运行配置: ${task.runtime_config_file}` : '',
                task.original_config_file ? `原始配置: ${task.original_config_file}` : '',
            ].filter(Boolean).join('\n');
        }
        renderHistoryPaths(task, { includeHistory: false });
    }

    export function currentRuntimeTaskInfo() {
        return {
            run_dir: trainingRuntime.runDir,
            runtime_config_file: trainingRuntime.runtimeConfigFile,
            original_config_file: trainingRuntime.originalConfigFile,
            dataset_config_file: trainingRuntime.datasetConfigFile,
            model_cache_dir: trainingRuntime.modelCacheDir,
            dataset_cache_dir: trainingRuntime.datasetCacheDir,
            training_output_dir: trainingRuntime.trainingOutputDir,
            logs_dir: trainingRuntime.logsDir,
            output_dir: trainingRuntime.outputDir,
            sample_dir: trainingRuntime.sampleDir,
        };
    }

    export function updateSystem(msg, options = {}) {
        if (isHistoryReviewMode()) return;
        if (msg.last_output_at) {
            markTrainingActivity(msg.last_output_at, { resetQuietHint: options.replay !== true });
        }
        if (msg.vram_used_gb !== undefined) {
            trainingRuntime.lastVramUsedGb = Number(msg.vram_used_gb);
            trainingRuntime.lastVramTotalGb = Number(msg.vram_total_gb);
            if (Number.isFinite(trainingRuntime.lastVramUsedGb)) {
                trainingRuntime.peakVramUsedGb = Math.max(
                    trainingRuntime.peakVramUsedGb ?? 0,
                    trainingRuntime.lastVramUsedGb
                );
            }
            setMetricText('metric-vram', formatRuntimeVram(
                trainingRuntime.lastVramUsedGb,
                trainingRuntime.lastVramTotalGb
            ));
            setMetricText('metric-vram-peak', formatRuntimeVram(
                trainingRuntime.peakVramUsedGb,
                trainingRuntime.lastVramTotalGb
            ));
        }
        if (msg.gpu_util !== undefined) {
            trainingRuntime.lastGpuUtil = Number(msg.gpu_util);
            if (Number.isFinite(trainingRuntime.lastGpuUtil)) {
                trainingRuntime.peakGpuUtil = Math.max(trainingRuntime.peakGpuUtil ?? 0, trainingRuntime.lastGpuUtil);
            }
            setMetricText('metric-gpu', formatSystemPercent(trainingRuntime.lastGpuUtil));
            setMetricText('metric-gpu-peak', formatSystemPercent(trainingRuntime.peakGpuUtil));
        }
        if (msg.gpu_temp !== undefined) {
            trainingRuntime.lastGpuTemp = Number(msg.gpu_temp);
            if (Number.isFinite(trainingRuntime.lastGpuTemp)) {
                trainingRuntime.peakGpuTemp = Math.max(trainingRuntime.peakGpuTemp ?? 0, trainingRuntime.lastGpuTemp);
            }
            setMetricText('metric-temp', formatSystemTemperature(trainingRuntime.lastGpuTemp));
            setMetricText('metric-temp-peak', formatSystemTemperature(trainingRuntime.peakGpuTemp));
        }
        renderLiveTrainingDashboard();
        refreshTrainingHealth();
    }

    export function formatRuntimeVram(used, total) {
        const usedNumber = numberOrNull(used);
        if (usedNumber === null) return '-';
        const usedText = formatCompactNumber(usedNumber);
        if (usedText === '-') return '-';
        const totalNumber = numberOrNull(total);
        const totalText = totalNumber === null ? '-' : formatCompactNumber(totalNumber);
        return totalText === '-' ? `${usedText} GB` : `${usedText} / ${totalText} GB`;
    }

    export function renderTrainingRunSummary(items, fallback) {
        const el = document.getElementById('training-run-summary');
        if (!el) return;
        const entries = (Array.isArray(items) ? items : [])
            .map((item) => {
                if (Array.isArray(item)) {
                    return {
                        label: String(item[0] || '').trim(),
                        value: String(item[1] || '').trim(),
                    };
                }
                return {
                    label: String(item?.label || '').trim(),
                    value: String(item?.value || '').trim(),
                };
            })
            .filter((item) => item.label && item.value);
        // Dirty-check: path fields almost never change between progress/metrics ticks.
        // Skip full DOM rebuild (4 buttons + rebind click) when content is identical.
        const signature = entries.length
            ? entries.map((item) => `${item.label}${item.value}`).join('')
            : ` ${fallback || '运行目录和配置快照会在任务启动后显示。'}`;
        if (el.dataset.summarySignature === signature) return;
        el.dataset.summarySignature = signature;
        el.innerHTML = '';
        el.classList.toggle('is-empty', entries.length === 0);
        el.classList.remove('has-copy-feedback');
        el.removeAttribute('data-copy-feedback');
        if (!entries.length) {
            el.textContent = fallback || '运行目录和配置快照会在任务启动后显示。';
            return;
        }
        for (const item of entries) {
            const wrap = document.createElement('button');
            wrap.type = 'button';
            wrap.className = 'training-run-summary-item';
            wrap.title = `复制${item.label}: ${item.value}`;
            wrap.setAttribute('aria-label', `复制${item.label}: ${item.value}`);
            const label = document.createElement('span');
            label.className = 'training-run-summary-label';
            label.textContent = item.label;
            const value = document.createElement('code');
            value.className = 'training-run-summary-value';
            value.textContent = item.value;
            value.title = item.value;
            wrap.append(label, value);
            wrap.addEventListener('click', async () => {
                const feedback = el.querySelector('.training-run-summary-feedback');
                let message = '';
                try {
                    await copyText(item.value);
                    wrap.classList.add('is-copied');
                    message = `${item.label}已复制`;
                } catch (_) {
                    message = '复制失败，请手动选择';
                }
                if (feedback) feedback.textContent = message;
                el.dataset.copyFeedback = message;
                el.classList.add('has-copy-feedback');
                window.setTimeout(() => {
                    wrap.classList.remove('is-copied');
                    el.classList.remove('has-copy-feedback');
                    el.removeAttribute('data-copy-feedback');
                    if (feedback) feedback.textContent = '';
                }, 1600);
            });
            el.appendChild(wrap);
        }
        const feedback = document.createElement('span');
        feedback.className = 'training-run-summary-feedback';
        feedback.setAttribute('aria-live', 'polite');
        el.appendChild(feedback);
    }

    export function renderLiveTrainingDashboard() {
        if (isHistoryReviewMode()) return;
        const stateMap = { idle: '空闲', running: '运行中', error: '错误', compiling: '编译中' };
        const jobLabel = trainingRuntime.job === 'preprocess' ? '预处理' : '训练';
        const stateText = isLiveRunningState(trainingRuntime.state)
            ? `${jobLabel}中`
            : (stateMap[trainingRuntime.state] || trainingRuntime.state || '空闲');
        setText('training-run-state', stateText);
        const stateEl = document.getElementById('training-run-state');
        if (stateEl) stateEl.className = `training-run-state ${trainingRuntime.state || 'idle'}`;
        setTrainingDashboardHeadState(trainingRuntime.state || 'idle');
        updateTrainingToolbarState(trainingRuntime.state || 'idle', stateText);
        updateDashboardProgressIdleState(isLiveRunningState(trainingRuntime.state));
        setText('training-run-title', isLiveRunningState(trainingRuntime.state) ? `当前${jobLabel}` : '当前监控');
        setText('training-run-meta', [
            trainingRuntime.methodsSubdir ? `方法目录 ${trainingRuntime.methodsSubdir}` : '',
            trainingRuntime.variant ? `配置 ${trainingRuntime.variant}` : '',
            trainingRuntime.preset ? `预设 ${trainingRuntime.preset}` : '',
        ].filter(Boolean).join(' · ') || '等待训练任务启动。');
        renderTrainingRunSummary([
            ['运行目录', trainingRuntime.runDir],
            ['实际配置', trainingRuntime.runtimeConfigFile],
            ['输出', trainingRuntime.outputDir],
            ['样张', trainingRuntime.sampleDir],
        ], '运行目录和配置快照会在任务启动后显示。');
        setEtaMetricText(trainingEtaMetricInfo());
    }

    export function trainingEtaMetricInfo() {
        return calculateTrainingEtaMetricInfo({
            isRunning: isLiveRunningState(trainingRuntime.state),
            current: trainingRuntime.progressCurrent,
            total: trainingRuntime.progressTotal,
            progressSecondsPerStep: trainingRuntime.progressSecondsPerStep,
            progressRate: trainingRuntime.progressRate,
            nowMs: Date.now(),
            formatDuration,
        });
    }

    export function markTrainingActivity(ts, options = {}) {
        const value = Number(ts);
        const ms = value > 100000000000 ? value : value * 1000;
        if (Number.isFinite(ms) && ms > 0) {
            trainingRuntime.lastOutputAt = Math.max(trainingRuntime.lastOutputAt, ms);
        } else {
            trainingRuntime.lastOutputAt = Date.now();
        }
        trainingRuntime.lastUiActivityAt = Date.now();
        if (options.resetQuietHint !== false) {
            trainingRuntime.quietHintShown = false;
        }
    }

    export function refreshTrainingHealth() {
        const el = document.getElementById('training-health');
        const ageEl = document.getElementById('metric-log-age');
        if (!el || !ageEl) return;

        if (isHistoryReviewMode()) {
            el.className = 'training-health';
            el.removeAttribute('title');
            return;
        }

        const isRunning = isLiveRunningState(trainingRuntime.state);
        if (trainingRuntime.lastAnomalyMessage) {
            if (!isRunning) {
                setMetricText('metric-log-age', 'N/A');
                setEtaMetricText({ text: '待计算', empty: true, title: '训练开始并收到进度后显示预计完成时间。' });
            }
            const headline = trainingRuntime.lastAnomalyMessage.split('\n', 1)[0].trim();
            el.className = 'training-health error';
            el.textContent = headline || '训练异常：Loss 数值异常';
            el.title = trainingRuntime.lastAnomalyMessage;
            return;
        }
        el.removeAttribute('title');

        if (!isRunning) {
            setMetricText('metric-log-age', 'N/A');
            setEtaMetricText({ text: '待计算', empty: true, title: '训练开始并收到进度后显示预计完成时间。' });
            if (trainingRuntime.state === 'error' && trainingRuntime.lastTerminalMessage) {
                el.className = 'training-health error';
                el.textContent = `最近任务异常: ${trainingRuntime.lastTerminalMessage}`;
                return;
            }
            el.className = 'training-health';
            el.textContent = '未运行任务。';
            return;
        }

        const ageSeconds = trainingRuntime.lastOutputAt
            ? Math.max(0, Math.floor((Date.now() - trainingRuntime.lastOutputAt) / 1000))
            : null;
        setMetricText('metric-log-age', ageSeconds == null ? 'N/A' : formatDuration(ageSeconds));
        setEtaMetricText(trainingEtaMetricInfo());

        const jobName = trainingRuntime.job === 'preprocess' ? '预处理' : '训练';

        const gpu = trainingRuntime.lastGpuUtil;
        const gpuActive = gpu != null && gpu >= 15;
        if (ageSeconds == null) {
            el.className = 'training-health';
            el.textContent = gpuActive
                ? `${jobName}运行中，GPU ${gpu}% 活跃，等待第一条日志。`
                : `${jobName}运行中，等待日志和系统指标。`;
            return;
        }

        if (ageSeconds >= 180 && gpuActive) {
            el.className = 'training-health warning';
            el.textContent = `已有 ${formatDuration(ageSeconds)} 没有新日志，但 GPU ${gpu}% 仍在工作；通常是单步较慢或任务脚本未输出进度。`;
            if (!trainingRuntime.quietHintShown) {
                appendLog(`[提示] ${el.textContent}`);
                trainingRuntime.quietHintShown = true;
            }
            return;
        }

        if (ageSeconds >= 180) {
            el.className = 'training-health error';
            el.textContent = `已有 ${formatDuration(ageSeconds)} 没有新日志，且 GPU 活跃度不高；建议观察进程或检查终端输出。`;
            return;
        }

        el.className = 'training-health ok';
        el.textContent = gpu == null
            ? `${jobName}运行中，最近 ${formatDuration(ageSeconds)} 前收到输出。`
            : `${jobName}运行中，最近 ${formatDuration(ageSeconds)} 前收到输出，GPU ${gpu}%。`;
    }

    export function formatDuration(totalSeconds) {
        return ctx.format.formatDuration(totalSeconds);
    }


    configureLiveStatusBridge({ updateProgress, updateMetrics, updateStatus, liveStatusState, terminalStatusMessage, resetLiveSystemPeaks, clearRuntimeInfo, applyRuntimeInfoToState, renderCurrentRuntimePaths, currentRuntimeTaskInfo, updateSystem, formatRuntimeVram, renderTrainingRunSummary, renderLiveTrainingDashboard, trainingEtaMetricInfo, markTrainingActivity, refreshTrainingHealth, formatDuration });

    // ── 全局设置 ──
