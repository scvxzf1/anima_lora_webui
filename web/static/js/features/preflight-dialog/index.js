/**
 * Preflight dialog + preprocess launch helpers.
 * Moved out of anima-app mechanical chunks.
 */
import { isLiveRunningState } from '../live-training/index.js?v=module-bootstrap-20260711-ir2';
import {
    FIELD_LABEL_ZH,
} from '../../config/catalog.js?v=module-bootstrap-20260711-ir2';
import {
    continueTrainingRequestPayload,
    ensureTrainingSourceReadyForLaunch,
    trainingSourceLaunchBlockReason,
} from '../anima-app/helpers/training-source-bridge.js?v=module-bootstrap-20260711-ir2';
import {
    chooseTrainingLaunchMode,
    enqueueTrainingFromConfig,
    enterLiveTrainingForNewRun,
} from '../anima-app/helpers/training-launch-bridge.js?v=module-bootstrap-20260711-ir2';
import { showAppConfirmDialog } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260711-ir2';
import {
    markTrainingActivity,
    updateStatus,
} from '../anima-app/helpers/live-status-bridge.js?v=module-bootstrap-20260711-ir2';
import { api, val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir2';
import { getGpuPicker } from '../anima-app/helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260711-ir2';
import { outputRunRuntimeFile } from '../output-run/runtime-file.js?v=module-bootstrap-20260711-ir2';
import { loadTrainingQueue, updateTrainingQueueFromPayload } from '../anima-app/helpers/queue-view-bridge.js?v=module-bootstrap-20260711-ir2';
import { scheduleStatusPoll } from '../anima-app/helpers/status-polling-bridge.js?v=module-bootstrap-20260711-ir2';
import { loadTrainingHistoryList } from '../anima-app/helpers/history-list-bridge.js?v=module-bootstrap-20260711-ir2';
import { configurePreflightDialogBridge } from '../anima-app/helpers/preflight-dialog-bridge.js?v=module-bootstrap-20260711-ir2';
import { appendLog, setTrainingHealthNotice } from '../anima-app/helpers/live-log-bridge.js?v=module-bootstrap-20260711-ir2';
import { getTomlState } from '../anima-app/helpers/toml-state-bridge.js?v=module-bootstrap-20260711-ir2';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir2';

const tomlState = getTomlState();
const trainingState = getTrainingState();
const trainingRuntime = trainingState.trainingRuntime;

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

    export function showPreflightPendingDialog(options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        const controller = new AbortController();
        if (!dialog) {
            return { signal: controller.signal, resolve: () => {} };
        }
        renderPreflightPending(options);
        let settled = false;
        const cleanup = () => {
            dialog.removeEventListener('close', handleClose);
        };
        const handleClose = () => {
            cleanup();
            if (!settled) {
                controller.abort();
            }
        };
        dialog.addEventListener('close', handleClose);
        if (!dialog.open) {
            try {
                dialog.showModal();
            } catch (e) {
                dialog.setAttribute('open', 'open');
            }
        }
        return {
            signal: controller.signal,
            resolve: () => {
                settled = true;
                cleanup();
            },
        };
    }

    export function renderPreflightPending(options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        const heading = dialog?.querySelector('.preflight-header h2');
        const summary = document.getElementById('preflight-summary');
        const list = document.getElementById('preflight-results');
        const continueBtn = document.getElementById('btn-preflight-continue');
        const preprocessBtn = document.getElementById('btn-preflight-preprocess');
        const cancelBtn = document.getElementById('btn-preflight-cancel');
        if (heading) heading.textContent = options.title || '训练前预检测';
        if (summary) {
            summary.className = 'preflight-summary pending';
            summary.setAttribute('aria-live', 'polite');
            summary.textContent = options.message || '正在预检测...';
        }
        if (list) {
            list.innerHTML = '';
            const row = document.createElement('div');
            row.className = 'preflight-item pending';
            row.setAttribute('aria-busy', 'true');

            const badge = document.createElement('span');
            badge.className = 'preflight-badge preflight-spinner';
            badge.setAttribute('aria-label', '正在检查');
            row.appendChild(badge);

            const body = document.createElement('div');
            body.className = 'preflight-body';
            const title = document.createElement('div');
            title.className = 'preflight-message';
            title.textContent = options.detail || '正在连接后端并执行轻量检查...';
            const path = document.createElement('div');
            path.className = 'preflight-path';
            path.textContent = '请稍等，预检测返回后会在这里显示每一项结果。';
            body.append(title, path);
            row.appendChild(body);
            list.appendChild(row);
        }
        if (preprocessBtn) {
            preprocessBtn.hidden = true;
            preprocessBtn.disabled = true;
        }
        if (continueBtn) {
            continueBtn.hidden = false;
            continueBtn.disabled = true;
            continueBtn.textContent = '正在检查...';
        }
        if (cancelBtn) {
            cancelBtn.disabled = false;
            cancelBtn.textContent = '取消';
        }
    }

    export function showPreflightRequestError(message) {
        const result = {
            ok: false,
            summary: { errors: 1, warnings: 0, checks: 1 },
            checks: [{
                level: 'error',
                key: 'preflight',
                message,
            }],
            errors: [{
                level: 'error',
                key: 'preflight',
                message,
            }],
            warnings: [],
        };
        const dialog = document.getElementById('preflight-dialog');
        if (dialog) {
            renderPreflightResult(result, false);
            if (!dialog.open) dialog.showModal();
        } else {
            alert(message);
        }
    }

    export function isPreflightDialogOpen() {
        const dialog = document.getElementById('preflight-dialog');
        return Boolean(dialog?.open);
    }

    export function waitForPreflightDialogClose() {
        const dialog = document.getElementById('preflight-dialog');
        if (!dialog?.open) return Promise.resolve();
        return new Promise((resolve) => {
            dialog.addEventListener('close', resolve, { once: true });
        });
    }

    export function renderPreflightResult(result, allowContinue, options = {}) {
        const dialog = document.getElementById('preflight-dialog');
        const heading = dialog?.querySelector('.preflight-header h2');
        const summary = document.getElementById('preflight-summary');
        const list = document.getElementById('preflight-results');
        const continueBtn = document.getElementById('btn-preflight-continue');
        const preprocessBtn = document.getElementById('btn-preflight-preprocess');
        const cancelBtn = document.getElementById('btn-preflight-cancel');
        const errors = result.summary?.errors || 0;
        const warnings = result.summary?.warnings || 0;
        const checks = result.summary?.checks || 0;
        const canPreprocess = preflightCanStartPreprocess(result);
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);

        if (heading) heading.textContent = '训练前预检测';
        summary.className = `preflight-summary ${errors ? 'error' : warnings ? 'warning' : 'ok'}`;
        summary.removeAttribute('aria-live');
        if (errors && canPreprocess) {
            summary.textContent = `发现 ${errors} 个错误：当前数据需要先预处理。点击下方按钮后，还会出现最终确认；确认后才会启动预处理并在完成后训练。`;
        } else {
            summary.textContent = errors
                ? `发现 ${errors} 个错误，已阻止训练。`
                : warnings
                    ? (willAutoPreprocess
                        ? `通过基础检查，但有 ${warnings} 个警告。点击下方按钮后，还需要最终确认才会预处理并训练。`
                        : `通过基础检查，但有 ${warnings} 个警告。点击下方按钮后，还需要最终确认才会开始训练。`)
                    : willAutoPreprocess
                        ? `预检测通过，共 ${checks} 项。点击下方按钮后，还需要最终确认才会创建运行目录、预处理并自动训练。`
                        : `预检测通过，共 ${checks} 项。点击下方按钮后，还需要最终确认才会开始训练。`;
        }

        list.innerHTML = '';
        for (const item of result.checks || []) {
            const row = document.createElement('div');
            row.className = `preflight-item ${item.level}`;

            const badge = document.createElement('span');
            badge.className = 'preflight-badge';
            badge.textContent = item.level === 'ok' ? '通过' :
                item.level === 'warning' ? '警告' : '错误';
            row.appendChild(badge);

            const body = document.createElement('div');
            body.className = 'preflight-body';
            const title = document.createElement('div');
            title.className = 'preflight-message';
            title.textContent = `${FIELD_LABEL_ZH[item.key] || item.key}: ${item.message}`;
            body.appendChild(title);
            if (item.path) {
                const path = document.createElement('div');
                path.className = 'preflight-path';
                path.textContent = item.path;
                body.appendChild(path);
            }
            row.appendChild(body);
            list.appendChild(row);
        }

        preprocessBtn.hidden = !canPreprocess;
        preprocessBtn.disabled = !canPreprocess;
        continueBtn.hidden = !allowContinue;
        continueBtn.disabled = !allowContinue;
        continueBtn.textContent = warnings
            ? (willAutoPreprocess ? '查看最终确认' : '查看最终确认')
            : (willAutoPreprocess ? '下一步：最终确认' : '下一步：最终确认');
        if (cancelBtn) {
            cancelBtn.disabled = false;
            cancelBtn.textContent = '取消';
        }
    }

    export function preflightCanStartPreprocess(result) {
        const checks = result.checks || [];
        const errors = result.errors || [];
        const allowedErrorKeys = new Set(['training_images', 'resized_image_dir']);
        if (errors.some((item) => !allowedErrorKeys.has(item.key))) return false;
        const sourceOk = checks.some((item) => item.key === 'source_image_dir' && item.level === 'ok');
        if (!sourceOk) return false;
        return checks.some((item) =>
            ['training_images', 'resized_image_dir', 'lora_cache_dir', 'latent_cache', 'text_cache'].includes(item.key)
            && ['error', 'warning'].includes(item.level)
        );
    }

    export async function startPreprocessFromPreflight(result) {
        const currentTrainingSource = currentTrainingSourceState();
        const variant = result.variant || currentTrainingSource.method || val('variant-select');
        const preset = result.preset || val('preset-select');
        const methodsSubdir = result.methods_subdir || currentTrainingSource.methods_subdir || 'gui-methods';
        if (!(await ensureTrainingSourceReadyForLaunch())) {
            showPreflightRequestError(trainingSourceLaunchBlockReason());
            return;
        }
        const mode = await chooseTrainingLaunchMode({ willAutoPreprocess: true });
        if (mode === 'cancel') return;
        if (mode === 'queue') {
            await enqueueTrainingFromConfig(variant, preset, methodsSubdir, { willAutoPreprocess: true });
            return;
        }
        renderPreflightPending({
            title: '启动预处理',
            message: '正在创建运行目录并启动预处理...',
            detail: '正在把任务交给后端；成功后会自动切换到训练页。',
        });
        try {
            const res = await api('/api/training/preprocess', {
                method: 'POST',
                body: JSON.stringify({
                    variant,
                    preset,
                    methods_subdir: methodsSubdir,
                    config_file: currentTrainingConfigFile(),
                    extra_args: [],
                    train_after: true,
                    confirmed: true,
                    confirm_train_after: true,
                    confirm_preprocess: true,
                    gpu_whitelist: getGpuPicker()?.selectedGpuPayload?.() ?? [],
                    ...continueTrainingRequestPayload(),
                }),
            });
            if (!res.ok) {
                showPreflightRequestError(res.error || '预处理启动失败');
                return;
            }
            const dialog = document.getElementById('preflight-dialog');
            if (dialog?.open) dialog.close('preprocess-started');
            enterLiveTrainingForNewRun();
            appendLog(`[状态] ${res.message || '预处理已启动'}`);
        } catch (e) {
            showPreflightRequestError('预处理请求失败: ' + e.message);
        }
    }

    export function currentTrainingConfigFile() {
        const currentTrainingSource = currentTrainingSourceState();
        if (tomlState.tomlManagerMode === 'output') {
            return outputRunRuntimeFile();
        }
        return currentTrainingSource.file || tomlState.currentTomlFile || val('toml-file-select') || '';
    }

    export function preflightPlainText(result) {
        return (result.checks || [])
            .map((item) => `[${item.level}] ${item.key}: ${item.message}${item.path ? ` (${item.path})` : ''}`)
            .join('\n');
    }

configurePreflightDialogBridge({
    showPreflightPendingDialog,
    renderPreflightPending,
    showPreflightRequestError,
    isPreflightDialogOpen,
    waitForPreflightDialogClose,
    renderPreflightResult,
    preflightCanStartPreprocess,
    startPreprocessFromPreflight,
    currentTrainingConfigFile,
    preflightPlainText,
});
