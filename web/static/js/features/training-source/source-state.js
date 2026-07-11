/**
 * Training source method/variant path helpers.
 * Extracted from anima-app chunk 13.
 */
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260711-ir1';
import { val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir1';

const trainingState = getTrainingState();

export function setCurrentTrainingSourceFromVariant(variant) {
    if (!variant) {
        clearCurrentTrainingSource();
        return;
    }
    if (val('method-select') === 'spd' || variant === 'spd') {
        trainingState.currentTrainingSource = {
            method: 'spd',
            methods_subdir: 'methods',
            file: 'configs/methods/spd.toml',
        };
        return;
    }
    trainingState.currentTrainingSource = {
        method: variant,
        methods_subdir: 'gui-methods',
        file: `configs/gui-methods/${variant}.toml`,
    };
}

export function clearCurrentTrainingSource() {
    trainingState.currentTrainingSource = {
        method: '',
        methods_subdir: '',
        file: '',
    };
}
