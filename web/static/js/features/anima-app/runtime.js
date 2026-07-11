import { createAppShellState } from './state/app-shell-state.js?v=module-bootstrap-20260711-ir6';
import { createConfigState } from './state/config-state.js?v=module-bootstrap-20260711-ir6';
import { createDatasetState } from './state/dataset-state.js?v=module-bootstrap-20260711-ir6';
import { createHistoryState } from './state/history-state.js?v=module-bootstrap-20260711-ir6';
import { createTomlState } from './state/toml-state.js?v=module-bootstrap-20260711-ir6';
import { createTrainingState } from './state/training-state.js?v=module-bootstrap-20260711-ir6';
import { createRuntimeApi } from './runtime/api.js?v=module-bootstrap-20260711-ir6';
import { createRuntimeDom } from './runtime/dom.js?v=module-bootstrap-20260711-ir6';
import { createRuntimeEvents } from './runtime/events.js?v=module-bootstrap-20260711-ir6';
import { createFeatureRegistry } from './runtime/feature-registry.js?v=module-bootstrap-20260711-ir6';

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
