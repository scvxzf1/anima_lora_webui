/**
 * Training launch / preflight orchestration helpers.
 * Moved out of anima-app mechanical chunks.
 */
import { ensureQueueFeature } from '../anima-app/helpers/feature-ensurers.js?v=module-bootstrap-20260711-ir6';
import { isLiveRunningState } from '../live-training/index.js?v=module-bootstrap-20260711-ir6';
import {
    configTrainingSourceMode,
    continueTrainingRequestPayload,
    ensureTrainingSourceReadyForLaunch,
    startConfigFullResumeSource,
    trainingSourceLaunchBlockReason,
    trainingSourceLaunchSummary,
} from '../anima-app/helpers/training-source-bridge.js?v=module-bootstrap-20260711-ir6';
import { showHistoryTaskConfirmDialog } from '../anima-app/helpers/history-task-actions-bridge.js?v=module-bootstrap-20260711-ir6';
import { returnToLiveTraining } from '../anima-app/helpers/history-timeline-bridge.js?v=module-bootstrap-20260711-ir6';
import { getGpuPicker } from '../anima-app/helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260711-ir6';
import { configureTrainingLaunchBridge } from '../anima-app/helpers/training-launch-bridge.js?v=module-bootstrap-20260711-ir6';
import { api, val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    hasPendingConfigChanges,
    showAppConfirmDialog,
} from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    currentTrainingConfigFile,
    isPreflightDialogOpen,
    preflightPlainText,
    renderPreflightPending,
    renderPreflightResult,
    showPreflightPendingDialog,
    showPreflightRequestError,
    startPreprocessFromPreflight,
    waitForPreflightDialogClose,
} from '../anima-app/helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260711-ir6';
import { appendLog, recoverLiveTrainingState } from '../anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    setTomlStatus,
    updateTomlActionState,
} from '../anima-app/helpers/toml-action-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getDatasetState } from '../anima-app/helpers/dataset-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260711-ir6';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir6';

const datasetState = getDatasetState();
const tomlState = getTomlState();
const trainingState = getTrainingState();
const trainingRuntime = trainingState.trainingRuntime;

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

function currentOutputRunState() {
    return datasetState.outputRunState || {};
}

    export async function startTraining() {
        if (configTrainingSourceMode() === 'full_resume') {
            await startConfigFullResumeSource(false);
            return;
        }
        const selectedTrainingConfigFile = currentTrainingConfigFile();
        if (tomlState.tomlManagerMode !== 'output' || !currentOutputRunState().file) {
            if (hasPendingConfigChanges(tomlState.currentTomlFile)) {
                setTomlStatus('error', '当前配置有未保存修改，请先保存更新当前选中配置或另存新配置，再开始训练');
                updateTomlActionState(tomlState.currentTomlFile);
                document.querySelector('[data-tab="config"]')?.click();
                return;
            }
        }
        if (!selectedTrainingConfigFile) {
            const message = tomlState.tomlManagerMode === 'output' && currentOutputRunState().selectedRun
                ? '这个训练输出没有可直接继续训练的 config.runtime.toml，请先另存原始配置或选择其他运行目录'
                : '请选择要训练的配置文件';
            setTomlStatus('error', message);
            return;
        }
        const variant = currentTrainingSourceState().method || val('variant-select');
        const preset = val('preset-select');
        const methodsSubdir = currentTrainingSourceState().methods_subdir || 'gui-methods';
        if (!variant) return alert('请选择变体');
        if (isCliOnlySpdSource(variant, methodsSubdir)) {
            const message = 'SPD 是 CLI 实验配置，只能通过 tasks.py exp-spd / scripts/distill_spd.py 运行；Web 普通训练入口已拦截，避免误用 train.py。';
            setTomlStatus('error', message, { persist: true });
            alert(message);
            return;
        }
        if (!(await ensureTrainingSourceReadyForLaunch())) {
            setTomlStatus('error', trainingSourceLaunchBlockReason(), { persist: true });
            return;
        }
        const preflight = await runPreflight(variant, preset, methodsSubdir);
        if (!preflight) {
            if (isPreflightDialogOpen()) await waitForPreflightDialogClose();
            return;
        }
        const willAutoPreprocess = !currentTrainingConfigIsRuntime();
        if (!preflight.ok) {
            const action = await showPreflightDialog(preflight, false, { willAutoPreprocess });
            if (action === 'preprocess') {
                await startPreprocessFromPreflight(preflight);
            }
            return;
        }
        const action = await showPreflightDialog(preflight, true, { willAutoPreprocess });
        if (action === 'preprocess') {
            await startPreprocessFromPreflight(preflight);
            return;
        }
        if (action !== 'continue') return;
        await startTrainingUnchecked(variant, preset, methodsSubdir, { willAutoPreprocess });
    }

    export async function queueCurrentTrainingFromConfig() {
        if (configTrainingSourceMode() === 'full_resume') {
            await startConfigFullResumeSource(true);
            return;
        }
        return ensureQueueFeature().queueCurrentTrainingFromConfig();
    }

    export async function runPreflight(variant, preset, methodsSubdir) {
        const pending = showPreflightPendingDialog({
            title: '训练前预检测',
            message: '正在检查模型路径、数据集路径和预处理启动环境...',
            detail: '这一步可能需要几秒钟；窗口保持打开表示仍在检查。',
        });
        try {
            const res = await api('/api/training/preflight', {
                method: 'POST',
                signal: pending.signal,
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                }),
            });
            pending.resolve();
            return res;
        } catch (e) {
            pending.resolve();
            if (e.name === 'AbortError') {
                return null;
            }
            showPreflightRequestError(`预检测请求失败: ${e.message}`);
            return null;
        }
    }

    export function isCliOnlySpdSource(variant, methodsSubdir) {
        return String(methodsSubdir || '') === 'methods' && String(variant || '') === 'spd';
    }

    export function currentTrainingConfigIsRuntime() {
        return currentTrainingConfigFile().replace(/\\/g, '/').endsWith('/config.runtime.toml');
    }

    export async function chooseTrainingLaunchMode(options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const isRunning = isLiveRunningState(trainingRuntime.state);
        const sourceDetail = trainingSourceLaunchSummary();
        if (isRunning) {
            const ok = await showAppConfirmDialog({
                title: '加入训练队列',
                description: '当前已有任务在运行',
                message: `确认后会冻结当前配置，并加入队列等待自动执行。${sourceDetail}`,
                confirmText: '加入队列',
                cancelText: '取消',
            });
            return ok ? 'queue' : 'cancel';
        }
        const startNow = await showAppConfirmDialog({
            title: willAutoPreprocess ? '最终确认：预处理并训练' : '最终确认：开始训练',
            description: '可以立即启动，也可以先加入队列',
            message: willAutoPreprocess
                ? `确认后会立即创建本次运行目录并启动预处理。${sourceDetail}`
                : `确认后会立即创建本次运行目录并启动训练进程。${sourceDetail}`,
            confirmText: willAutoPreprocess ? '立即预处理并训练' : '立即开始训练',
            cancelText: '不立即启动',
        });
        if (startNow) return 'start';
        const queue = await showAppConfirmDialog({
            title: '加入训练队列',
            description: '冻结当前配置并等待手动继续',
            message: `确认后会创建独立运行配置并加入队列；队列会保持暂停，等待你手动继续。${sourceDetail}`,
            confirmText: '加入队列',
            cancelText: '取消',
        });
        return queue ? 'queue' : 'cancel';
    }

    export async function confirmTrainingLaunch(options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const sourceDetail = trainingSourceLaunchSummary();
        return showAppConfirmDialog({
            title: willAutoPreprocess ? '最终确认：预处理并训练' : '最终确认：开始训练',
            description: '训练启动前的最后一步',
            message: willAutoPreprocess
                ? `确认后会立即创建本次运行目录并启动预处理；预处理完成后会自动开始训练。${sourceDetail}`
                : `确认后会立即创建本次运行目录并启动训练进程。${sourceDetail}`,
            confirmText: willAutoPreprocess ? '确认预处理并训练' : '确认开始训练',
            cancelText: '返回检查',
        });
    }

    export async function startTrainingUnchecked(variant, preset, methodsSubdir, options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const mode = await chooseTrainingLaunchMode({ willAutoPreprocess });
        if (mode === 'cancel') return;
        if (mode === 'queue') {
            await enqueueTrainingFromConfig(variant, preset, methodsSubdir, { willAutoPreprocess });
            return;
        }
        renderPreflightPending({
            title: willAutoPreprocess ? '启动预处理后训练' : '启动训练',
            message: willAutoPreprocess
                ? '正在创建运行目录并启动预处理...'
                : '正在创建运行目录并启动训练...',
            detail: willAutoPreprocess
                ? '预处理完成后会自动开始训练；成功后会自动切换到训练页。'
                : '后端正在准备训练进程；启动成功后会自动切换到训练页。',
        });
        try {
            const res = await api('/api/training/start', {
                method: 'POST',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                    extra_args: [],
                    gpu_whitelist: getGpuPicker()?.selectedGpuPayload?.() ?? [],
                    confirmed: true,
                    confirm_preprocess: willAutoPreprocess,
                    ...continueTrainingRequestPayload(),
                }),
            });
            if (res.ok) {
                const dialog = document.getElementById('preflight-dialog');
                if (dialog?.open) dialog.close('training-started');
                enterLiveTrainingForNewRun();
                appendLog(`[状态] ${res.message || '任务已启动'}`);
            } else {
                if (res.preflight) {
                    const action = await showPreflightDialog(res.preflight, false);
                    if (action === 'preprocess') {
                        await startPreprocessFromPreflight(res.preflight);
                    }
                } else {
                    showPreflightRequestError(res.error || '启动失败');
                }
            }
        } catch (e) {
            showPreflightRequestError('请求失败: ' + e.message);
        }
    }

    export async function enqueueTrainingFromConfig(variant, preset, methodsSubdir, options = {}) {
        return ensureQueueFeature().enqueueTrainingFromConfig(variant, preset, methodsSubdir, options);
    }

    export async function enqueueTrainingQueueRequest(options = {}) {
        return ensureQueueFeature().enqueueTrainingQueueRequest(options);
    }

    export async function enqueueTrainingQueueBatchRequest(options = {}) {
        return ensureQueueFeature().enqueueTrainingQueueBatchRequest(options);
    }

    export function enterLiveTrainingForNewRun() {
        returnToLiveTraining({ refresh: false });
        document.querySelector('[data-tab="training"]')?.click();
        recoverLiveTrainingState();
    }

    export function showPreflightDialog(result, allowContinue, options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        if (!dialog) {
            if (!allowContinue) return Promise.resolve('cancel');
            const confirmText = options.willAutoPreprocess ? '确认预处理并训练' : '确认开始训练';
            return showAppConfirmDialog({
                title: '训练前预检测',
                description: '检测到训练前提示',
                message: `${preflightPlainText(result)}\n\n是否继续下一步？`,
                confirmText,
            }).then((ok) => ok ? 'continue' : 'cancel');
        }
        renderPreflightResult(result, allowContinue, options);
        if (!dialog.open) dialog.showModal();
        return new Promise((resolve) => {
            dialog.addEventListener('close', () => {
                resolve(dialog.returnValue || 'cancel');
            }, { once: true });
        });
    }


configureTrainingLaunchBridge({
    startTraining,
    queueCurrentTrainingFromConfig,
    runPreflight,
    isCliOnlySpdSource,
    currentTrainingConfigIsRuntime,
    chooseTrainingLaunchMode,
    confirmTrainingLaunch,
    startTrainingUnchecked,
    enqueueTrainingFromConfig,
    enqueueTrainingQueueRequest,
    enqueueTrainingQueueBatchRequest,
    enterLiveTrainingForNewRun,
    showPreflightDialog,
});
