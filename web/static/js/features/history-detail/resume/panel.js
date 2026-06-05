import { optionNode } from '../ui.js?v=module-bootstrap-20260604-8';
import {
    resumeCheckpointOptionLabel,
    resumeCheckpointProgressText,
    resumeSummaryLine,
    selectedResumeCheckpointFromState,
} from './state.js?v=module-bootstrap-20260604-8';

export function createHistoryResumePanelRenderer({ state, deps, slots }) {
    function syncHistoryDetailResumeContent() {
        if (slots.isHistoryDetailDialogOpen() && state.detailTab === 'overview') {
            slots.renderHistoryDetailContent();
        }
    }

    function resetInlineResumePanel(panel, select, btn, queueBtn, summary, status) {
        if (panel) panel.hidden = true;
        if (select) {
            select.innerHTML = '<option value="">选择历史训练任务后读取</option>';
            select.disabled = true;
        }
        if (btn) btn.disabled = true;
        if (queueBtn) queueBtn.disabled = true;
        if (summary) summary.textContent = '';
        if (status) {
            status.textContent = '';
            status.className = 'resume-status';
        }
    }

    function renderResumePanelState() {
        const panel = document.getElementById('history-resume-panel');
        const select = document.getElementById('resume-checkpoint-select');
        const btn = document.getElementById('btn-resume-training');
        const queueBtn = document.getElementById('btn-queue-resume-training');
        const summary = document.getElementById('resume-checkpoint-summary');
        const status = document.getElementById('resume-training-status');
        if (!panel || !select || !btn || !summary || !status) {
            syncHistoryDetailResumeContent();
            return;
        }
        if (deps.shouldRenderInlineResumePanel?.() !== true) {
            resetInlineResumePanel(panel, select, btn, queueBtn, summary, status);
            syncHistoryDetailResumeContent();
            return;
        }

        const isTrainingTask = Boolean(deps.getViewingHistoryTaskId() && deps.getCurrentHistoryTaskForResume()?.job === 'training');
        panel.hidden = !isTrainingTask;
        if (!isTrainingTask) {
            resetInlineResumePanel(panel, select, btn, queueBtn, summary, status);
            syncHistoryDetailResumeContent();
            return;
        }

        const isRunning = deps.getTrainingRuntime().state === 'running' || deps.getTrainingRuntime().state === 'compiling';
        select.innerHTML = '';
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

        const hasCheckpoint = Boolean(select.value);
        select.disabled = state.resumeOptions.loading || !hasCheckpoint || isRunning;
        btn.disabled = state.resumeOptions.loading || !hasCheckpoint || isRunning;
        if (queueBtn) queueBtn.disabled = state.resumeOptions.loading || !hasCheckpoint;
        summary.innerHTML = '';
        const selected = selectedResumeCheckpointFromState(state);
        if (selected) {
            summary.append(
                resumeSummaryLine('状态目录', selected.path),
                resumeSummaryLine('已训练到', resumeCheckpointProgressText(selected)),
                resumeSummaryLine('保存时间', selected.mtime_text || '-'),
                resumeSummaryLine('关联权重', selected.paired_weight || '无或未找到'),
            );
        } else {
            const note = document.createElement('p');
            note.textContent = state.resumeOptions.message || state.resumeOptions.error || '该任务还没有可续训状态。需要训练配置启用 checkpointing_epochs，训练中才会写出状态目录。';
            summary.appendChild(note);
        }

        status.textContent = isRunning
            ? '当前已有训练或预处理在运行，续训按钮暂不可用。'
            : (state.resumeOptions.error || state.resumeOptions.message || '');
        status.className = [
            'resume-status',
            isRunning ? 'warning' : (state.resumeOptions.error ? 'error' : ''),
        ].filter(Boolean).join(' ');
        syncHistoryDetailResumeContent();
    }

    return { renderResumePanelState };
}
