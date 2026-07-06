import { createAppShellState } from './state/app-shell-state.js?v=module-bootstrap-20260706-1';
import { createConfigState } from './state/config-state.js?v=module-bootstrap-20260706-1';
import { createDatasetState } from './state/dataset-state.js?v=module-bootstrap-20260706-1';
import { createHistoryState } from './state/history-state.js?v=module-bootstrap-20260706-1';
import { createTomlState } from './state/toml-state.js?v=module-bootstrap-20260706-1';
import { createTrainingState } from './state/training-state.js?v=module-bootstrap-20260706-1';

export function createAnimaRuntime(ctx) {
    return {
        ctx,
        app: {},
        state: {
            appShell: createAppShellState(),
            config: createConfigState(),
            training: createTrainingState(),
            toml: createTomlState(),
            dataset: createDatasetState(),
            history: createHistoryState(),
        },
        features: {},
        timers: {},
        dom: {
            byId(id) {
                return document.getElementById(id);
            },
        },
    };
}
