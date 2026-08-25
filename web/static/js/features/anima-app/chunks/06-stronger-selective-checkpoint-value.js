/**
 * Compatibility barrel for former chunk 06.
 * Implementations live in config-form/* and training-source/continue-lora.js.
 */
export {
    strongerSelectiveCheckpointValue,
    resourceQuickCurrentValue,
    fillGlobalModelPathsIntoConfigForm,
} from '../../config-form/resource-values.js?v=module-bootstrap-20260824-zimage-defaults-v1';
export {
    appendFieldRows,
} from '../../config-form/field-rows.js?v=module-bootstrap-20260809-nf4-v2';
export {
    createConfigDatasetPicker,
    renderConfigDatasetPicker,
    isConfigDatasetPickerDialogOpen,
    closeConfigDatasetPickerDialog,
} from '../../config-form/dataset-picker.js?v=module-bootstrap-20260809-nf4-v2';
export {
    clearContinueTrainingSource,
    openContinueLoraDialog,
    loadContinueLoraWeights,
    requestContinueLoraInspection,
    selectContinueLoraWeight,
    refreshContinueTrainingSourceCompatibility,
} from '../../training-source/continue-lora.js?v=module-bootstrap-20260809-nf4-v2';
