/**
 * Anima LoRA Web UI — modular application entry.
 */
import { configureAppShellStateBridge } from './helpers/app-shell-state-bridge.js?v=module-bootstrap-20260707-93';
import { configureAppContextBridge } from './helpers/app-context-bridge.js?v=module-bootstrap-20260707-93';
import { configureConfigStateBridge } from './helpers/config-state-bridge.js?v=module-bootstrap-20260707-93';
import { configureDatasetStateBridge } from './helpers/dataset-state-bridge.js?v=module-bootstrap-20260707-93';
import { configureHistoryStateBridge } from './helpers/history-state-bridge.js?v=module-bootstrap-20260707-93';
import { configureImageTestBridge } from './helpers/image-test-bridge.js?v=module-bootstrap-20260707-93';
import { configureRuntimeBridge } from './helpers/runtime-bridge.js?v=module-bootstrap-20260707-93';
import { configureStatusPollingBridge } from './helpers/status-polling-bridge.js?v=module-bootstrap-20260707-93';
import { configureGlobalSettingsBridge } from './helpers/global-settings-bridge.js?v=module-bootstrap-20260707-93';
import { configurePreviewViewBridge } from './helpers/preview-view-bridge.js?v=module-bootstrap-20260707-93';
import { configureQueueViewBridge } from './helpers/queue-view-bridge.js?v=module-bootstrap-20260707-93';
import { configureHistoryListBridge } from './helpers/history-list-bridge.js?v=module-bootstrap-20260707-93';
import { configureTomlStateBridge } from './helpers/toml-state-bridge.js?v=module-bootstrap-20260707-93';
import { configureTrainingStateBridge } from './helpers/training-state-bridge.js?v=module-bootstrap-20260707-93';
import { createAnimaRuntime } from './runtime.js?v=module-bootstrap-20260707-93';

export async function createAnimaApp(ctx) {
    const runtime = createAnimaRuntime(ctx);
    configureAppContextBridge(runtime.ctx);
    configureAppShellStateBridge(runtime.state.appShell);
    configureConfigStateBridge(runtime.state.config);
    configureDatasetStateBridge(runtime.state.dataset);
    configureHistoryStateBridge(runtime.state.history);
    configureRuntimeBridge(runtime);
    configureTomlStateBridge(runtime.state.toml);
    configureTrainingStateBridge(runtime.state.training);
    await import('./chunks/01-scope-state.js?v=module-bootstrap-20260707-93');
    const imageTestFeatureModule = await import('./chunks/01a-image-test-feature.js?v=module-bootstrap-20260707-93');
    const imageTestFeatureBridge = imageTestFeatureModule.createImageTestFeatureBridge(runtime);
    runtime.features.imageTest = imageTestFeatureBridge;
    configureImageTestBridge(imageTestFeatureBridge.ensureImageTestFeature);
    const appShellModule = await import('./chunks/02-ensure-history-detail-feature.js?v=module-bootstrap-20260707-93');
    await import('./chunks/03-parse-network-arg-entry.js?v=module-bootstrap-20260707-93');
    await import('./chunks/04-create-config-group-entry.js?v=module-bootstrap-20260707-93');
    await import('./chunks/05-create-stage-resolution-summary.js?v=module-bootstrap-20260707-93');
    await import('./chunks/05a-no-dataset-regularization-mode.js?v=module-bootstrap-20260707-93');
    await import('./chunks/06-stronger-selective-checkpoint-value.js?v=module-bootstrap-20260707-93');
    await import('./chunks/07-render-config-dataset-picker-dialog.js?v=module-bootstrap-20260707-93');
    await import('./chunks/08-origin-closest.js?v=module-bootstrap-20260707-93');
    await import('./chunks/09-setup-config-group-drop-target.js?v=module-bootstrap-20260707-93');
    await import('./chunks/10-create-dataset-config-input.js?v=module-bootstrap-20260707-93');
    await import('./chunks/10a-dataset-inline-help.js?v=module-bootstrap-20260707-93');
    await import('./chunks/11-create-dataset-editor-row.js?v=module-bootstrap-20260707-93');
    await import('./chunks/12-create-dataset-row-caption-source-mode-editor.js?v=module-bootstrap-20260707-93');
    await import('./chunks/13-update-dataset-editor-rows-setting-value.js?v=module-bootstrap-20260707-93');
    await import('./chunks/14-lora-adapter-kind-from-config.js?v=module-bootstrap-20260707-93');
    await import('./chunks/15-append-sample-prompt-row.js?v=module-bootstrap-20260707-93');
    await import('./chunks/16-load-output-run-config.js?v=module-bootstrap-20260707-93');
    await import('./chunks/17-apply-selected-dataset-preset-to-current-config.js?v=module-bootstrap-20260707-93');
    await import('./chunks/18-delete-dataset-preset-group.js?v=module-bootstrap-20260707-93');
    await import('./chunks/19-current-sample-prompt-text.js?v=module-bootstrap-20260707-93');
    await import('./chunks/20-can-drop-toml-file-to-group.js?v=module-bootstrap-20260707-93');
    await import('./chunks/21-update-toml-selection-ui.js?v=module-bootstrap-20260707-93');
    await import('./chunks/22-update-toml-action-state.js?v=module-bootstrap-20260711-c6');
    await import('./chunks/23-move-current-toml-to-group.js?v=module-bootstrap-20260707-93');
    await import('./chunks/24-show-preflight-pending-dialog.js?v=module-bootstrap-20260707-93');
    await import('./chunks/25-update-progress.js?v=module-bootstrap-20260707-93');
    const globalSettingsModule = await import('./chunks/26a-global-settings.js?v=module-bootstrap-20260707-93');
    const previewViewModule = await import('./chunks/26b-preview-view.js?v=module-bootstrap-20260707-93');
    const queueViewModule = await import('./chunks/26c-queue-view.js?v=module-bootstrap-20260707-93');
    const historyListModule = await import('./chunks/26d-history-list.js?v=module-bootstrap-20260707-93');
    // Keep the compatibility barrel reachable for module-graph tests and older imports.
    await import('./chunks/26-load-global-settings.js?v=module-bootstrap-20260707-93');
    configureGlobalSettingsBridge(globalSettingsModule);
    configurePreviewViewBridge(previewViewModule);
    configureQueueViewBridge({
        ...queueViewModule,
        resetTrainingExpandedStateOnLeave: historyListModule.resetTrainingExpandedStateOnLeave,
    });
    configureHistoryListBridge(historyListModule);
    const statusPollingModule = await import('./chunks/26a-status-polling.js?v=module-bootstrap-20260707-93');
    const statusPollingBridge = statusPollingModule.createStatusPollingBridge(runtime.state.training);
    runtime.features.statusPolling = statusPollingBridge;
    configureStatusPollingBridge(statusPollingBridge);
    await import('./chunks/27-render-history-collections-workbench.js?v=module-bootstrap-20260707-93');
    await import('./chunks/28-history-collection-search-text.js?v=module-bootstrap-20260707-93');
    await import('./chunks/29-start-history-config-group-pointer-drag.js?v=module-bootstrap-20260707-93');
    await import('./chunks/30-start-history-collection-pointer-drag.js?v=module-bootstrap-20260707-93');
    await import('./chunks/31-create-history-collection-workbench-card.js?v=module-bootstrap-20260707-93');
    await import('./chunks/32-history-task-collection-label.js?v=module-bootstrap-20260707-93');
    await import('./chunks/33-create-history-task-item.js?v=module-bootstrap-20260707-93');
    await import('./chunks/34-show-history-collection-select-dialog.js?v=module-bootstrap-20260707-93');
    await import('./chunks/35-render-config-group-timeline.js?v=module-bootstrap-20260707-93');
    await import('./chunks/36-setup-event-listeners.js?v=module-bootstrap-20260707-93');
    await import('./chunks/37-config-training-source.js?v=module-bootstrap-20260707-93');
    return appShellModule.startAnimaApp();
}
