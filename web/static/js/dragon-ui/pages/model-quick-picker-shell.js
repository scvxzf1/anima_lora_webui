import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

export const MODEL_QUICK_PATH_KEYS = Object.freeze([
    'pretrained_model_name_or_path',
    'qwen3',
    'vae',
]);

export function renderModelQuickPickerTrigger() {
    return `<button class="dragon-btn dragon-btn-secondary dragon-model-quick-trigger" type="button" data-model-quick-action="open">${renderIcon('layers', 'dragon-btn-icon')}<span>快速配置模型</span></button>`;
}

export function renderModelQuickPickerDialog() {
    return `
        <dialog class="dragon-model-quick-dialog" data-model-quick-dialog aria-labelledby="dragon-model-quick-title">
            <div class="dragon-model-quick-shell">
                <div class="dragon-model-quick-head">
                    <div>
                        <span class="dragon-eyebrow">模型配置库</span>
                        <h2 id="dragon-model-quick-title">快速配置模型</h2>
                        <p>全局模型配置 · 分组顺序同步</p>
                    </div>
                    <div class="dragon-model-quick-head-actions">
                        <button class="dragon-icon-button" type="button" data-model-quick-action="refresh" aria-label="刷新模型配置" title="刷新">${renderIcon('refresh')}</button>
                        <button class="dragon-icon-button" type="button" data-model-quick-action="close" aria-label="关闭快速配置" title="关闭">${renderIcon('x')}</button>
                    </div>
                </div>
                <div class="dragon-model-quick-status" data-model-quick-status role="status" aria-live="polite"></div>
                <div class="dragon-model-quick-workspace">
                    <aside class="dragon-model-quick-library" aria-label="模型配置列表">
                        <label class="dragon-model-quick-search">
                            <span class="visually-hidden">搜索模型配置</span>
                            <input class="dragon-input" type="search" autocomplete="off" data-model-quick-search placeholder="搜索名称、模型族或路径…">
                        </label>
                        <div class="dragon-model-quick-groups" data-model-quick-list></div>
                    </aside>
                    <section class="dragon-model-quick-preview" data-model-quick-preview aria-label="模型路径预览"></section>
                </div>
                <div class="dragon-model-quick-footer">
                    <div class="dragon-model-quick-selection" data-model-quick-selection></div>
                    <button class="dragon-btn dragon-btn-primary" type="button" data-model-quick-action="apply">应用此配置</button>
                </div>
            </div>
        </dialog>
    `;
}

export function bindLazyModelQuickPicker(root, options = {}) {
    const trigger = root.querySelector('[data-model-quick-action="open"]');
    if (!trigger) return null;
    let disposed = false;
    let loading = false;
    const open = async () => {
        if (loading) return;
        loading = true;
        trigger.disabled = true;
        try {
            const { bindModelQuickPicker } = await import('./model-quick-picker.js?v=dragon-ui-20260826v4');
            if (disposed || !trigger.isConnected) return;
            trigger.removeEventListener('click', open);
            trigger.disabled = false;
            bindModelQuickPicker(root, options);
            trigger.click();
        } catch {
            if (!disposed) {
                trigger.disabled = false;
                const dialog = root.querySelector('[data-model-quick-dialog]');
                const status = dialog?.querySelector('[data-model-quick-status]');
                if (status) {
                    status.dataset.tone = 'error';
                    status.textContent = '模型配置模块加载失败，请重试。';
                }
                const close = dialog?.querySelector('[data-model-quick-action="close"]');
                if (close) close.onclick = () => dialog.close('cancel');
                if (dialog && !dialog.open) dialog.showModal();
            }
        } finally {
            loading = false;
        }
    };
    trigger.addEventListener('click', open);
    return () => {
        disposed = true;
        trigger.removeEventListener('click', open);
    };
}
