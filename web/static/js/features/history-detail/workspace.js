import { historyDetailEmptyText } from './ui.js?v=module-bootstrap-20260714-stage-dataset5';

export function createHistoryDetailWorkspace({ deps }) {
    function renderHistoryDetailPreview(payload) {
        const box = document.createElement('div');
        box.className = 'history-detail-preview';
        const target = historyDetailPreviewTarget(payload);
        if (!target) {
            box.appendChild(historyDetailEmptyText(
                payload?.task?.job === 'preprocess'
                    ? '预处理任务没有样张与权重预览。'
                    : '这个合并视图不能直接汇总预览，请选择自动配置组或单个训练任务。',
            ));
            return box;
        }
        const mount = document.createElement('div');
        mount.id = 'history-detail-preview-mount';
        mount.className = 'history-detail-preview-mount';
        box.appendChild(mount);
        return box;
    }

    function historyDetailPreviewTarget(payload) {
        const task = payload?.task || null;
        if (task) {
            return task.job === 'training' && task.id
                ? { taskId: String(task.id) }
                : null;
        }
        const group = payload?.group || null;
        if (payload?.mode === 'config_group' && deps.canPreviewHistoryConfigGroup(group)) {
            return { group: deps.normalizePreviewGroup(group) };
        }
        return null;
    }

    return {
        renderHistoryDetailPreview,
        historyDetailPreviewTarget,
    };
}
