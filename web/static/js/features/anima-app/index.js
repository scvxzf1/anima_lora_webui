/**
 * Anima LoRA Web UI — modular application entry.
 */
export async function createAnimaApp(ctx) {
    globalThis.ctx = ctx;
    globalThis.__animaAppContext = ctx;
    await import('./imports.js?v=module-bootstrap-20260608-3');
    await import('./chunks/01-scope-state.js?v=module-bootstrap-20260608-3');
    await import('./chunks/02-ensure-history-detail-feature.js?v=module-bootstrap-20260608-3');
    await import('./chunks/03-parse-network-arg-entry.js?v=module-bootstrap-20260608-3');
    await import('./chunks/04-create-config-group-entry.js?v=module-bootstrap-20260608-3');
    await import('./chunks/05-create-stage-resolution-summary.js?v=module-bootstrap-20260608-3');
    await import('./chunks/06-stronger-selective-checkpoint-value.js?v=module-bootstrap-20260608-3');
    await import('./chunks/07-render-config-dataset-picker-dialog.js?v=module-bootstrap-20260608-3');
    await import('./chunks/08-origin-closest.js?v=module-bootstrap-20260608-3');
    await import('./chunks/09-setup-config-group-drop-target.js?v=module-bootstrap-20260608-3');
    await import('./chunks/10-create-dataset-config-input.js?v=module-bootstrap-20260608-3');
    await import('./chunks/11-create-dataset-editor-row.js?v=module-bootstrap-20260608-3');
    await import('./chunks/12-create-dataset-row-caption-source-mode-editor.js?v=module-bootstrap-20260608-3');
    await import('./chunks/13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260608-3');
    await import('./chunks/14-lora-adapter-kind-from-config.js?v=module-bootstrap-20260608-3');
    await import('./chunks/15-append-sample-prompt-row.js?v=module-bootstrap-20260608-3');
    await import('./chunks/16-load-output-run-config.js?v=module-bootstrap-20260608-3');
    await import('./chunks/17-apply-selected-dataset-preset-to-current-config.js?v=module-bootstrap-20260608-3');
    await import('./chunks/18-delete-dataset-preset-group.js?v=module-bootstrap-20260608-3');
    await import('./chunks/19-current-sample-prompt-text.js?v=module-bootstrap-20260608-3');
    await import('./chunks/20-can-drop-toml-file-to-group.js?v=module-bootstrap-20260608-3');
    await import('./chunks/21-update-toml-selection-ui.js?v=module-bootstrap-20260608-3');
    await import('./chunks/22-update-toml-action-state.js?v=module-bootstrap-20260608-3');
    await import('./chunks/23-move-current-toml-to-group.js?v=module-bootstrap-20260608-3');
    await import('./chunks/24-show-preflight-pending-dialog.js?v=module-bootstrap-20260608-3');
    await import('./chunks/25-update-progress.js?v=module-bootstrap-20260608-3');
    await import('./chunks/26-load-global-settings.js?v=module-bootstrap-20260608-3');
    await import('./chunks/27-render-history-collections-workbench.js?v=module-bootstrap-20260608-3');
    await import('./chunks/28-history-collection-search-text.js?v=module-bootstrap-20260608-3');
    await import('./chunks/29-start-history-config-group-pointer-drag.js?v=module-bootstrap-20260608-3');
    await import('./chunks/30-start-history-collection-pointer-drag.js?v=module-bootstrap-20260608-3');
    await import('./chunks/31-create-history-collection-workbench-card.js?v=module-bootstrap-20260608-3');
    await import('./chunks/32-history-task-collection-label.js?v=module-bootstrap-20260608-3');
    await import('./chunks/33-create-history-task-item.js?v=module-bootstrap-20260608-3');
    await import('./chunks/34-show-history-collection-select-dialog.js?v=module-bootstrap-20260608-3');
    await import('./chunks/35-render-config-group-timeline.js?v=module-bootstrap-20260608-3');
    await import('./chunks/36-setup-event-listeners.js?v=module-bootstrap-20260608-3');
    return globalThis.startAnimaApp();
}
