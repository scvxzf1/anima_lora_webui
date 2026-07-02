import {
    enqueueTrainingQueue,
    enqueueTrainingQueueBatch,
    enqueueTrainingQueueBatchAlias,
    enqueueTrainingQueueBatchRoot,
    resumeTrainingQueue,
} from './api.js?v=module-bootstrap-20260702-1';

export function createQueueEnqueue({ ctx, deps, updateTrainingQueueFromPayload }) {
    async function queueCurrentTrainingFromConfig() {
        if (deps.getTrainingSourceMode?.() === 'full_resume') {
            await deps.queueConfigFullResumeSource?.();
            return;
        }
        const selectedTrainingConfigFile = deps.currentTrainingConfigFile();
        if (deps.getTomlManagerMode() !== 'output' || !deps.getOutputRunFile()) {
            if (deps.hasPendingConfigChanges(deps.getCurrentTomlFile())) {
                deps.setTomlStatus('error', '当前配置有未保存修改，请先保存更新当前选中配置或另存新配置，再加入队列');
                deps.updateTomlActionState(deps.getCurrentTomlFile());
                document.querySelector('[data-tab="config"]')?.click();
                return;
            }
        }
        if (!selectedTrainingConfigFile) {
            const message = deps.getTomlManagerMode() === 'output' && deps.getOutputRunSelectedRun()
                ? '这个训练输出没有可直接继续训练的 config.runtime.toml，请先另存原始配置或选择其他运行目录'
                : '请选择要加入队列的配置文件';
            deps.setTomlStatus('error', message);
            return;
        }
        const source = deps.getCurrentTrainingSource();
        const variant = source.method || ctx.dom.val('variant-select');
        const preset = ctx.dom.val('preset-select');
        const methodsSubdir = source.methods_subdir || 'gui-methods';
        if (!variant) return alert('请选择变体');
        if (deps.isCliOnlySpdSource(variant, methodsSubdir)) {
            const message = 'SPD 是 CLI 实验配置，只能通过 tasks.py exp-spd / scripts/distill_spd.py 运行；Web 普通训练入口已拦截，避免误用 train.py。';
            deps.setTomlStatus('error', message, { persist: true });
            alert(message);
            return;
        }
        if (deps.ensureTrainingSourceReadyForLaunch && !(await deps.ensureTrainingSourceReadyForLaunch())) {
            deps.setTomlStatus('error', deps.trainingSourceLaunchBlockReason?.() || '训练来源审查未通过', { persist: true });
            return;
        }
        await enqueueTrainingFromConfig(variant, preset, methodsSubdir, {
            willAutoPreprocess: !deps.currentTrainingConfigIsRuntime(),
        });
    }

    async function enqueueTrainingFromConfig(variant, preset, methodsSubdir, options = {}) {
        const willAutoPreprocess = Boolean(options.willAutoPreprocess);
        const configFile = options.configFile || deps.currentTrainingConfigFile();
        deps.renderPreflightPending({
            title: options.title || '加入训练队列',
            message: '正在冻结当前配置并加入队列...',
            detail: '队列会保存独立运行配置并保持暂停；之后修改当前 TOML 不会影响这个队列任务。',
        });
        try {
            const startPaused = options.startPaused !== false;
            const res = await enqueueTrainingQueueRequest({
                variant,
                preset,
                methodsSubdir,
                configFile,
                willAutoPreprocess,
                startPaused,
                continuePayload: options.includeContinueSource === false ? {} : deps.continueTrainingRequestPayload(),
            });
            if (!res.ok) {
                if (res.preflight) {
                    await deps.showPreflightDialog(res.preflight, false, { willAutoPreprocess });
                } else {
                    deps.showPreflightRequestError(res.error || '加入队列失败');
                }
                return;
            }
            const dialog = document.getElementById('preflight-dialog');
            if (dialog?.open) dialog.close('queued');
            updateTrainingQueueFromPayload(res);
            document.querySelector('[data-tab="training"]')?.click();
            deps.showTrainingView('queue');
            deps.appendLog(`[状态] ${res.message || '已加入训练队列'}`);
        } catch (e) {
            deps.showPreflightRequestError('加入队列失败: ' + e.message);
        }
    }

    async function enqueueTrainingQueueRequest(options = {}) {
        return enqueueTrainingQueue(ctx, {
            ...options,
            gpuWhitelist: deps.selectedGpuPayload(),
        });
    }

    async function enqueueTrainingQueueBatchRequest(options = {}) {
        const requestOptions = {
            ...options,
            gpuWhitelist: deps.selectedGpuPayload(),
        };
        let res;
        try {
            res = await enqueueTrainingQueueBatch(ctx, requestOptions);
        } catch (e) {
            const unsupported = {
                ok: false,
                error: e?.message || String(e || ''),
                status: e?.status,
                status_code: e?.status_code,
            };
            if (!queueBatchApiUnsupported(unsupported)) throw e;
            return enqueueTrainingQueueBatchCompat(requestOptions, unsupported);
        }
        if (!queueBatchApiUnsupported(res)) return res;
        return enqueueTrainingQueueBatchCompat(requestOptions, res);
    }

    async function enqueueTrainingQueueBatchCompat(options = {}, unsupported = {}) {
        let aliasRes;
        try {
            aliasRes = await enqueueTrainingQueueBatchAlias(ctx, options);
        } catch (e) {
            const aliasUnsupported = {
                ok: false,
                error: e?.message || String(e || ''),
                status: e?.status,
                status_code: e?.status_code,
            };
            if (!queueBatchApiUnsupported(aliasUnsupported)) throw e;
            return enqueueTrainingQueueBatchRootCompat(options, aliasUnsupported);
        }
        if (!queueBatchApiUnsupported(aliasRes)) return aliasRes;
        return enqueueTrainingQueueBatchRootCompat(options, aliasRes || unsupported);
    }

    async function enqueueTrainingQueueBatchRootCompat(options = {}, unsupported = {}) {
        let rootRes;
        try {
            rootRes = await enqueueTrainingQueueBatchRoot(ctx, options);
        } catch (e) {
            const rootUnsupported = {
                ok: false,
                error: e?.message || String(e || ''),
                status: e?.status,
                status_code: e?.status_code,
            };
            if (!queueBatchApiUnsupported(rootUnsupported)) throw e;
            return enqueueTrainingQueueBatchFallback(options, rootUnsupported);
        }
        if (!queueBatchApiUnsupported(rootRes)) return rootRes;
        return enqueueTrainingQueueBatchFallback(options, rootRes || unsupported);
    }

    function queueBatchApiUnsupported(res = {}) {
        if (res?.ok !== false) return false;
        const status = Number(res.status || res.status_code || res.code || 0);
        if (status === 404 || status === 405) return true;
        const message = String(res.error || res.message || '').trim();
        return /\b(404|405)\b|method not allowed|not found/i.test(message);
    }

    async function enqueueTrainingQueueBatchFallback(options = {}, unsupported = {}) {
        const items = Array.isArray(options.items) ? options.items : [];
        const queuedItems = [];
        let latestPayload = null;
        for (let index = 0; index < items.length; index += 1) {
            const item = items[index] || {};
            const res = await enqueueTrainingQueue(ctx, {
                variant: item.variant,
                preset: item.preset || options.preset || 'default',
                methodsSubdir: item.methods_subdir || item.methodsSubdir || 'gui-methods',
                configFile: item.config_file || item.configFile || '',
                willAutoPreprocess: item.confirm_preprocess !== false,
                startPaused: options.startPaused !== false,
                continuePayload: item.continuePayload || item.continue_info || {},
                gpuWhitelist: options.gpuWhitelist || [],
            });
            latestPayload = res;
            if (!res.ok) {
                return {
                    ...res,
                    ok: false,
                    message: queuedItems.length
                        ? `已加入 ${queuedItems.length} 个配置，批量加入在第 ${index + 1} 项停止`
                        : (res.error || unsupported.error || '批量加入队列失败'),
                    queued_count: queuedItems.length,
                    requested_count: items.length,
                    queued_items: queuedItems,
                    failed_index: index,
                    failed_item: item,
                    failures: [{
                        index,
                        variant: item.variant || '',
                        preset: item.preset || options.preset || 'default',
                        methods_subdir: item.methods_subdir || item.methodsSubdir || '',
                        config_file: item.config_file || item.configFile || '',
                        label: item.label || item.filename || '',
                        error: res.error || unsupported.error || '加入队列失败',
                    }],
                };
            }
            queuedItems.push(res.item || {});
        }
        return {
            ...(latestPayload || unsupported || {}),
            ok: true,
            message: `已将 ${queuedItems.length} 个配置加入训练队列`,
            queued_count: queuedItems.length,
            requested_count: items.length,
            queued_items: queuedItems,
            items: Array.isArray(latestPayload?.items) ? latestPayload.items : queuedItems,
        };
    }

    async function queueResumeTrainingFromCheckpoint() {
        const taskId = deps.getViewingHistoryTaskId();
        if (!taskId) return;
        const selected = deps.selectedResumeCheckpoint();
        if (!selected) {
            deps.setResumeStatus('请先选择一个可续训状态目录。', 'error');
            return;
        }
        const taskName = deps.historyTaskLabel(deps.getCurrentHistoryTaskForResume() || {});
        const ok = await deps.showHistoryTaskConfirmDialog({
            title: '续训加入队列',
            description: taskName,
            message: `将使用这个历史任务的配置快照，并从 ${selected.name} 续训。任务会排队等待当前训练结束后自动启动。`,
            confirmText: '加入队列',
        });
        if (!ok) return;

        deps.setResumeStatus('正在加入队列...', '');
        try {
            const res = await resumeTrainingQueue(ctx, {
                taskId,
                checkpoint: selected.path,
                gpuWhitelist: deps.selectedGpuPayload(),
            });
            if (!res.ok) {
                deps.setResumeStatus(res.error || '续训加入队列失败', 'error');
                return;
            }
            updateTrainingQueueFromPayload(res);
            deps.setResumeStatus(res.message || '续训任务已加入队列', 'ok');
            document.querySelector('[data-tab="training"]')?.click();
            deps.showTrainingView('queue');
        } catch (e) {
            deps.setResumeStatus('续训加入队列失败: ' + e.message, 'error');
        }
    }

    return {
        queueCurrentTrainingFromConfig,
        enqueueTrainingFromConfig,
        enqueueTrainingQueueRequest,
        enqueueTrainingQueueBatchRequest,
        queueResumeTrainingFromCheckpoint,
    };
}
