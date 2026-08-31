import { createAppShellState } from './state/app-shell-state.js?v=module-bootstrap-20260831-release-v1';
import { createConfigState } from './state/config-state.js?v=module-bootstrap-20260831-release-v1';
import { createDatasetState } from './state/dataset-state.js?v=module-bootstrap-20260831-release-v1';
import { createHistoryState } from './state/history-state.js?v=module-bootstrap-20260831-release-v1';
import { createTomlState } from './state/toml-state.js?v=module-bootstrap-20260831-release-v1';
import { createTrainingState } from './state/training-state.js?v=module-bootstrap-20260831-release-v1';
import { createRuntimeApi } from './runtime/api.js?v=module-bootstrap-20260831-release-v1';
import { createRuntimeDom } from './runtime/dom.js?v=module-bootstrap-20260831-release-v1';
import { createRuntimeEvents } from './runtime/events.js?v=module-bootstrap-20260831-release-v1';
import { createFeatureRegistry } from './runtime/feature-registry.js?v=module-bootstrap-20260831-release-v1';

export function createAnimaRuntime(ctx) {
    return {
        ctx,
        app: {},
        api: createRuntimeApi(ctx),
        state: {
            appShell: createAppShellState(),
            config: createConfigState(),
            training: createTrainingState(),
            toml: createTomlState(),
            dataset: createDatasetState(),
            history: createHistoryState(),
        },
        features: createFeatureRegistry(),
        timers: {},
        dom: createRuntimeDom(ctx),
        events: createRuntimeEvents(),
    };
}
