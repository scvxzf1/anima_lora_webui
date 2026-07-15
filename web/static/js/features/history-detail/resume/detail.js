import {
    createHistoryDetailCopyButton,
    fileNameFromPath,
    historyDetailEmptyText,
    historyDetailRow,
    historyDetailSection,
    optionNode,
} from '../ui.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    resumeCheckpointOptionLabel,
    resumeCheckpointRemainingText,
    resumeSummaryLine,
} from './state.js?v=module-bootstrap-20260714-stage-dataset5';

export function createHistoryResumeDetailRenderer({ ctx, state, deps, slots, actions }) {
    const historyDetailCopyButton = (value, label) => createHistoryDetailCopyButton(ctx.dom.copyText, value, label);

    function renderHistoryDetailResume(payload) {
        const box = document.createElement('div');
        box.className = 'history-detail-resume';
        const task = payload.task || {};
        if (!task.id || task.job !== 'training') {
            box.appendChild(historyDetailEmptyText('只有训练任务可以从检查点续训。'));
            return box;
        }

        const fullResume = document.createElement('div');
        fullResume.className = 'history-resume-block';
        const fullHint = document.createElement('p');
        fullHint.className = 'history-resume-hint';
        fullHint.textContent = '完整续训会恢复 optimizer、scheduler 和已完成步数，需要 checkpoint-state 目录里包含 train_state.json。';
        fullResume.appendChild(fullHint);
        const select = document.createElement('select');
        select.id = 'history-manager-resume-select';
        select.className = 'history-resume-select';
        if (state.resumeOptions.loading) {
            select.appendChild(optionNode('', '正在读取检查点...'));
        } else if (state.resumeOptions.checkpoints.length) {
            for (const item of state.resumeOptions.checkpoints) {
                select.appendChild(optionNode(item.path, resumeCheckpointOptionLabel(item)));
            }
            select.value = state.resumeOptions.defaultCheckpoint || state.resumeOptions.checkpoints[0]?.path || '';
        } else {
            select.appendChild(optionNode('', '未找到可续训状态目录'));
        }
        const actionsRow = document.createElement('div');
        actionsRow.className = 'history-detail-action-row';
        const refresh = document.createElement('button');
        refresh.type = 'button';
        refresh.className = 'btn btn-small';
        refresh.textContent = '刷新检查点';
        refresh.addEventListener('click', () => actions.loadResumeOptionsForTask(task.id));
        const start = document.createElement('button');
        start.type = 'button';
        start.className = 'btn btn-primary';
        start.textContent = '从检查点继续训练';
        start.addEventListener('click', () => actions.resumeTrainingFromHistoryDetail(false));
        const queue = document.createElement('button');
        queue.type = 'button';
        queue.className = 'btn';
        queue.textContent = '加入队列';
        queue.addEventListener('click', () => actions.resumeTrainingFromHistoryDetail(true));
        actionsRow.append(refresh, start, queue);
        const controls = document.createElement('div');
        controls.className = 'history-resume-control-row';
        controls.append(select, actionsRow);

        const summary = document.createElement('div');
        summary.className = 'resume-checkpoint-summary';
        const selectedCheckpoint = () => {
            const value = select.value || '';
            if (!value) return null;
            return state.resumeOptions.checkpoints.find((item) => item.path === value) || null;
        };
        const fillSummary = () => {
            summary.innerHTML = '';
            const selected = selectedCheckpoint();
            if (!selected) {
                const p = document.createElement('p');
                p.textContent = state.resumeOptions.error || state.resumeOptions.message || '该任务没有可续训状态目录。';
                summary.appendChild(p);
                return;
            }
            summary.append(
                resumeSummaryLine('状态目录', selected.path),
                resumeSummaryLine('续训进度', resumeCheckpointRemainingText(selected)),
                resumeSummaryLine('保存时间', selected.mtime_text || '-'),
                resumeSummaryLine('关联权重', selected.paired_weight || '无或未找到'),
            );
            if (selected.unavailable_reason) {
                summary.appendChild(resumeSummaryLine('不可用原因', selected.unavailable_reason));
            } else if (selected.estimate_error) {
                summary.appendChild(resumeSummaryLine('步数估算', `无法确认剩余步数: ${selected.estimate_error}`));
            }
        };
        const updateActionState = () => {
            const selected = selectedCheckpoint();
            const selectedAvailable = Boolean(selected && selected.resume_available !== false);
            const isRunning = deps.getTrainingRuntime().state === 'running' || deps.getTrainingRuntime().state === 'compiling';
            select.disabled = state.resumeOptions.loading || !state.resumeOptions.checkpoints.length || isRunning;
            start.disabled = state.resumeOptions.loading || !selectedAvailable || isRunning;
            queue.disabled = state.resumeOptions.loading || !selectedAvailable;
        };
        select.addEventListener('change', () => {
            fillSummary();
            updateActionState();
        });
        fullResume.append(controls, summary);
        fillSummary();
        updateActionState();
        box.appendChild(historyDetailSection('完整续训', fullResume, 'history-detail-section resume-full'));

        if (!state.resumeOptions.checkpoints.length || state.resumeOptions.error) {
            box.appendChild(historyDetailSection('为什么不可用', renderResumeDiagnosticBlock(), 'history-detail-section resume-diagnostic'));
        }
        box.appendChild(historyDetailSection('可选续接', renderHistoryResumeWeightOptions(), 'history-detail-section resume-hotstart'));
        return box;
    }

    function renderResumeDiagnosticBlock() {
        const box = document.createElement('div');
        box.className = 'history-resume-diagnostic-body';
        const diagnostic = state.resumeOptions.diagnostic || {};
        const reason = state.resumeOptions.loading
            ? '正在扫描输出目录和 checkpoint-state。'
            : (state.resumeOptions.error || diagnostic.reason || state.resumeOptions.message || '没有找到包含 train_state.json 的完整续训状态目录。');
        const reasonNode = document.createElement('p');
        reasonNode.className = state.resumeOptions.error ? 'history-resume-diagnostic-error' : '';
        reasonNode.textContent = reason;
        box.appendChild(reasonNode);

        const kv = document.createElement('div');
        kv.className = 'history-detail-kv';
        [
            ['输出目录', diagnostic.output_dir_resolved || diagnostic.output_dir || '-'],
            ['输出目录存在', diagnostic.output_dir_valid === false ? '路径不合法' : formatDiagnosticBool(diagnostic.output_dir_exists)],
            ['状态子目录数', diagnostic.state_dir_count ?? '-'],
            ['train_state.json 数', diagnostic.train_state_count ?? '-'],
            ['完整续训候选', diagnostic.checkpoint_count ?? state.resumeOptions.checkpoints.length],
        ].forEach(([label, value]) => kv.appendChild(historyDetailRow(label, value)));
        box.appendChild(kv);

        const recommendation = document.createElement('p');
        recommendation.textContent = diagnostic.recommendation || '推荐从配置页选择原配置和可用权重重新续接训练。';
        box.appendChild(recommendation);
        return box;
    }

    function renderHistoryResumeWeightOptions() {
        const box = document.createElement('div');
        box.className = 'history-resume-hotstart-body';
        const hint = document.createElement('p');
        hint.className = 'history-resume-hint';
        hint.textContent = '权重热启动只加载 LoRA/LoHa/LoKr/GLoRA 权重，不恢复 optimizer、scheduler 和已完成步数；适合 checkpoint-state 已丢失但还有 safetensors 权重时继续训练。';
        box.appendChild(hint);

        if (state.resumeWeights.loading) {
            box.appendChild(historyDetailEmptyText(state.resumeWeights.message || '正在读取历史权重...'));
            return box;
        }
        if (state.resumeWeights.error) {
            box.appendChild(historyDetailEmptyText(state.resumeWeights.error));
        }
        if (state.resumeWeights.weights.length) {
            const checkpointWeightPaths = new Set(
                state.resumeOptions.checkpoints
                    .filter((item) => item.resume_available !== false)
                    .map((item) => String(item.paired_weight || '').trim())
                    .filter(Boolean),
            );
            if (!checkpointWeightPaths.size) {
                const note = document.createElement('p');
                note.className = 'history-resume-hint warning';
                note.textContent = '这些权重没有对应的完整续训状态，不能从这里一键续训；可复制路径后到配置页做权重热启动。';
                box.appendChild(note);
            }
            const list = document.createElement('div');
            list.className = 'history-resume-weight-list';
            for (const item of state.resumeWeights.weights) {
                const row = document.createElement('div');
                row.className = 'continue-lora-weight-item history-resume-weight-item';
                const info = document.createElement('div');
                const name = document.createElement('strong');
                const weightPath = item.abs_path || item.file || '';
                name.textContent = fileNameFromPath(item.name || weightPath) || '未命名权重';
                info.title = weightPath;
                const meta = document.createElement('span');
                meta.textContent = [
                    item.scope_label || '',
                    item.epoch != null ? `Epoch ${item.epoch}` : '',
                    item.steps != null ? `Step ${item.steps}` : '',
                    item.inspect_status === 'ok' ? '审查通过' : (item.inspect_message || '等待审查'),
                    item.mtime_text || '',
                ].filter(Boolean).join(' · ');
                info.append(name, meta);
                const actionsRow = document.createElement('div');
                actionsRow.className = 'history-resume-weight-actions';
                if (weightPath) {
                    actionsRow.appendChild(historyDetailCopyButton(weightPath, `${name.textContent} 完整路径`));
                }
                const useBtn = document.createElement('button');
                useBtn.type = 'button';
                useBtn.className = 'btn btn-small btn-primary';
                const hasPairedCheckpoint = checkpointWeightPaths.has(String(weightPath || '').trim());
                const inspected = item.inspect_status === 'ok';
                const canUseWeightDirectly = hasPairedCheckpoint && inspected && item.inspect_compatible !== false;
                useBtn.textContent = canUseWeightDirectly
                    ? '用权重热启动'
                    : (hasPairedCheckpoint ? '审查未通过' : '缺少检查点');
                useBtn.disabled = !canUseWeightDirectly;
                if (!canUseWeightDirectly) {
                    useBtn.title = hasPairedCheckpoint
                        ? (item.inspect_message || '权重审查未通过，不能从历史详情一键续训。')
                        : '缺少对应的 checkpoint-state/train_state.json，不能从历史详情一键续训。';
                }
                useBtn.addEventListener('click', async () => {
                    if (useBtn.disabled) return;
                    const ok = await deps.selectContinueLoraWeight(item.abs_path || item.file || '');
                    if (ok) slots.closeHistoryDetailDialog();
                });
                actionsRow.appendChild(useBtn);
                row.append(info, actionsRow);
                list.appendChild(row);
            }
            box.appendChild(list);
        } else {
            const empty = historyDetailEmptyText(state.resumeWeights.message || '没有在该历史输出目录找到可热启动的 safetensors 权重。');
            box.appendChild(empty);
        }

        const actionsRow = document.createElement('div');
        actionsRow.className = 'history-detail-action-row';
        const configBtn = document.createElement('button');
        configBtn.type = 'button';
        configBtn.className = 'btn';
        configBtn.textContent = '回配置页手动续接训练';
        configBtn.addEventListener('click', () => {
            slots.closeHistoryDetailDialog();
            document.querySelector('[data-tab="config"]')?.click();
        });
        actionsRow.appendChild(configBtn);
        box.appendChild(actionsRow);
        return box;
    }

    function formatDiagnosticBool(value) {
        if (value === true) return '是';
        if (value === false) return '否';
        return '-';
    }

    return {
        renderHistoryDetailResume,
    };
}
