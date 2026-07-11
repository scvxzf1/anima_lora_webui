import { previewSourceLabel } from './state.js?v=module-bootstrap-20260711-ir1';

export function createPreviewWorkspace({ state, deps, closePreviewImageDialog }) {
    function previewWorkspace() {
        return document.getElementById('preview-workspace');
    }

    function mountPreviewWorkspace(target) {
        const workspace = previewWorkspace();
        if (!workspace || !target || workspace.parentElement === target) return;
        target.appendChild(workspace);
    }

    function mountPreviewWorkspaceInPage() {
        mountPreviewWorkspace(document.getElementById('preview-page-mount'));
    }

    function mountPreviewWorkspaceInDialog() {
        mountPreviewWorkspace(document.getElementById('preview-dialog-mount'));
    }

    function applyPreviewPanelMode(mode = 'default') {
        const normalizedMode = mode === 'sampling' ? 'sampling' : 'default';
        const dialog = document.getElementById('preview-panel-dialog');
        const workspace = previewWorkspace();
        state.panel.mode = normalizedMode;
        dialog?.classList.toggle('preview-panel-dialog-sampling', normalizedMode === 'sampling');
        workspace?.classList.toggle('preview-workspace-sampling', normalizedMode === 'sampling');
    }

    function resetPreviewPanelMode() {
        applyPreviewPanelMode('default');
    }

    function mountPreviewWorkspaceInHistoryDetail() {
        mountPreviewWorkspace(document.getElementById('history-detail-preview-mount'));
    }

    function restorePreviewWorkspaceFromHistoryDetail() {
        const workspace = previewWorkspace();
        if (!workspace?.closest('#history-detail-content')) return;
        closePreviewImageDialog();
        mountPreviewWorkspaceInPage();
    }

    function activeTabName() {
        return document.querySelector('.tab-btn.active')?.dataset.tab || '';
    }

    function previewPanelSourceSummary() {
        if (state.source !== 'training') return previewSourceLabel(state.source);
        if (state.selectedGroup) {
            return `配置组: ${state.selectedGroup.label || state.selectedGroup.history_group_label || '-'}`;
        }
        if (state.selectedTaskId) {
            const task = deps.getHistoryTasks().find((item) => item.id === state.selectedTaskId);
            return `历史任务: ${task?.name || state.selectedTaskId}`;
        }
        return '当前任务 / 最新运行目录';
    }

    function syncPreviewPanelSubtitle() {
        const subtitle = document.getElementById('preview-panel-subtitle');
        if (!subtitle) return;
        const title = document.getElementById('preview-title')?.textContent || previewSourceLabel(state.source);
        const count = document.getElementById('preview-count')?.textContent || '';
        const dir = document.getElementById('preview-current-dir')?.textContent || '';
        const parts = [
            previewPanelSourceSummary(),
            title,
            count && count !== '0 张' ? count : '',
            dir && dir !== '-' ? `目录: ${dir}` : '',
        ].filter(Boolean);
        subtitle.textContent = parts.join(' · ') || '训练样张、权重文件和路径设置。';
    }

    function openPreviewPanel(options = {}) {
        const dialog = document.getElementById('preview-panel-dialog');
        if (!dialog) return;
        if (!state.panel.open) {
            state.panel.previousTab = activeTabName();
            state.panel.restoreTrainingView = deps.getTrainingViewMode();
        }
        mountPreviewWorkspaceInDialog();
        applyPreviewPanelMode(options.mode);
        state.panel.open = true;
        syncPreviewPanelSubtitle();
        try {
            if (dialog.showModal && !dialog.open) {
                dialog.showModal();
            } else if (!dialog.open) {
                dialog.setAttribute('open', 'open');
            }
        } catch (e) {
            dialog.setAttribute('open', 'open');
        }
        window.requestAnimationFrame(() => {
            document.getElementById('btn-close-preview-panel')?.focus();
        });
    }

    function restorePreviewWorkspaceAfterPanelClose() {
        closePreviewImageDialog();
        mountPreviewWorkspaceInPage();
        if (state.panel.previousTab === 'training' && activeTabName() === 'training' && state.panel.restoreTrainingView) {
            deps.showTrainingView(state.panel.restoreTrainingView);
        }
        state.panel.open = false;
        resetPreviewPanelMode();
        state.panel.previousTab = '';
        state.panel.restoreTrainingView = '';
    }

    function closePreviewPanel() {
        const dialog = document.getElementById('preview-panel-dialog');
        closePreviewImageDialog();
        if (dialog?.open) {
            dialog.close();
            return;
        }
        dialog?.removeAttribute('open');
        restorePreviewWorkspaceAfterPanelClose();
    }

    return {
        previewWorkspace,
        mountPreviewWorkspaceInPage,
        mountPreviewWorkspaceInDialog,
        mountPreviewWorkspaceInHistoryDetail,
        restorePreviewWorkspaceFromHistoryDetail,
        syncPreviewPanelSubtitle,
        openPreviewPanel,
        closePreviewPanel,
        restorePreviewWorkspaceAfterPanelClose,
    };
}
