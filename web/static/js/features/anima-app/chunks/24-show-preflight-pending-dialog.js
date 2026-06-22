/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.showPreflightPendingDialog = function showPreflightPendingDialog(options = {}) {
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

    globalThis.renderPreflightPending = function renderPreflightPending(options = {}) {
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

    globalThis.showPreflightRequestError = function showPreflightRequestError(message) {
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

    globalThis.isPreflightDialogOpen = function isPreflightDialogOpen() {
        const dialog = document.getElementById('preflight-dialog');
        return Boolean(dialog?.open);
    }

    globalThis.waitForPreflightDialogClose = function waitForPreflightDialogClose() {
        const dialog = document.getElementById('preflight-dialog');
        if (!dialog?.open) return Promise.resolve();
        return new Promise((resolve) => {
            dialog.addEventListener('close', resolve, { once: true });
        });
    }

    globalThis.renderPreflightResult = function renderPreflightResult(result, allowContinue, options = {}) {
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

    globalThis.preflightCanStartPreprocess = function preflightCanStartPreprocess(result) {
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

    globalThis.startPreprocessFromPreflight = async function startPreprocessFromPreflight(result) {
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
                    gpu_whitelist: gpuPicker.selectedGpuPayload(),
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

    globalThis.currentTrainingConfigFile = function currentTrainingConfigFile() {
        if (tomlManagerMode === 'output') {
            return outputRunRuntimeFile();
        }
        return currentTrainingSource.file || currentTomlFile || val('toml-file-select') || '';
    }

    globalThis.preflightPlainText = function preflightPlainText(result) {
        return (result.checks || [])
            .map((item) => `[${item.level}] ${item.key}: ${item.message}${item.path ? ` (${item.path})` : ''}`)
            .join('\n');
    }

    globalThis.stopTraining = async function stopTraining() {
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
            if (stopBtn) stopBtn.disabled = wasDisabled || !isLiveRunningState();
        }
    }

    // ── WebSocket ──
    globalThis.connectWebSocket = function connectWebSocket() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        setLogStatus('连接中', 'warning');
        ws = new WebSocket(`${proto}//${location.host}/ws/training`);
        ws.onopen = () => {
            setLogStatus('已连接', 'ok');
            recoverLiveTrainingState();
        };
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            handleWsMessage(msg);
        };
        ws.onclose = () => {
            setLogStatus('已断开，准备重连', 'warning');
            scheduleStatusPoll({ immediate: true });
            setTimeout(connectWebSocket, 3000);
        };
        ws.onerror = () => {
            setLogStatus('连接异常', 'error');
            ws.close();
        };
    }

    globalThis.handleWsMessage = function handleWsMessage(msg) {
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

    globalThis.appendLog = function appendLog(line) {
        appendLogRecord({ line });
    }

    globalThis.appendLogRecord = function appendLogRecord(record) {
        if (record?.id && record.id <= trainingRuntime.lastLogId) return;
        if (record?.id) trainingRuntime.lastLogId = record.id;

        const line = record?.line ?? '';
        const prefix = record?.kind === 'progress' ? '[进度] ' : '';
        trainingRuntime.logBuffer.push(prefix + line);
        trainingRuntime.logLineCount += 1;
        scheduleLogFlush();
    }

    globalThis.renderLogOutputLines = function renderLogOutputLines(lines) {
        const el = document.getElementById('log-output');
        if (!el) return;
        const normalized = (lines || [])
            .map((line) => String(line || ''))
            .filter((line) => line.length);
        const fragment = document.createDocumentFragment();
        for (const line of normalized) {
            const span = document.createElement('span');
            span.className = `log-line ${logLineTone(line)}`;
            span.textContent = line;
            fragment.append(span, document.createTextNode('\n'));
        }
        el.replaceChildren(fragment);
    }

    globalThis.currentLogOutputLines = function currentLogOutputLines() {
        const el = document.getElementById('log-output');
        if (!el) return [];
        return el.textContent.split('\n').filter(Boolean);
    }

    globalThis.logLineTone = function logLineTone(line) {
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

    globalThis.scheduleLogFlush = function scheduleLogFlush() {
        if (trainingRuntime.logFlushPending) return;
        trainingRuntime.logFlushPending = true;
        const schedule = window.requestAnimationFrame
            ? (fn) => window.requestAnimationFrame(fn)
            : (fn) => window.setTimeout(fn, 16);
        schedule(flushLogBuffer);
    }

    globalThis.flushLogBuffer = function flushLogBuffer() {
        trainingRuntime.logFlushPending = false;
        if (!trainingRuntime.logBuffer.length) return;
        const el = document.getElementById('log-output');
        const nextLines = [...currentLogOutputLines(), ...trainingRuntime.logBuffer];
        trainingRuntime.logBuffer = [];
        const lines = nextLines.filter(Boolean).slice(-MAX_LOG_LINES);
        renderLogOutputLines(lines);
        trainingRuntime.logLineCount = lines.length;
        el.scrollTop = el.scrollHeight;
        updateLogStatusText();
    }

    globalThis.replayTrainingLogs = async function replayTrainingLogs(options = {}) {
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

    globalThis.replayMetricsHistory = async function replayMetricsHistory() {
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

    globalThis.replayMetricsFromLogRecord = function replayMetricsFromLogRecord(record) {
        const line = record?.line || '';
        const parsed = parseMetricsFromProgressLine(line);
        if (!parsed || parsed.loss === undefined) return;
        updateMetrics({ ...parsed, ts: record.ts });
    }

    globalThis.setLogStatus = function setLogStatus(text, state = '') {
        const el = document.getElementById('log-status');
        if (!el) return;
        el.textContent = text;
        el.className = `log-status ${state}`.trim();
    }

    globalThis.updateLogStatusText = function updateLogStatusText() {
        const state = ws?.readyState === WebSocket.OPEN ? 'ok' : 'warning';
        const text = ws?.readyState === WebSocket.OPEN
            ? `已连接 · ${trainingRuntime.logLineCount} 行`
            : `${trainingRuntime.logLineCount} 行`;
        setLogStatus(text, state);
    }

    globalThis.setTrainingHealthNotice = function setTrainingHealthNotice(message, state = 'warning') {
        const el = document.getElementById('training-health');
        if (!el) return;
        el.className = `training-health ${state}`.trim();
        el.textContent = message;
    }

    globalThis.recoverLiveTrainingState = async function recoverLiveTrainingState() {
        if (isHistoryReviewMode() || location.protocol === 'file:') return;
        await pollStatus({ forceReplayMetrics: true });
        await replayTrainingLogs({ includeMetrics: false });
        await replayMetricsHistory();
        await loadTrainingQueue();
    }
