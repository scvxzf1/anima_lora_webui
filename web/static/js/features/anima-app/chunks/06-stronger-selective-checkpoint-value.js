/**
 * Compatibility barrel for former chunk 06.
 * Implementations live in config-form/* and training-source/continue-lora.js.
 */
export {
    strongerSelectiveCheckpointValue,
    resourceQuickCurrentValue,
    fillGlobalModelPathsIntoConfigForm,
} from '../../config-form/resource-values.js?v=module-bootstrap-20260711-ir2';
export {
    appendFieldRows,
} from '../../config-form/field-rows.js?v=module-bootstrap-20260711-ir2';
export {
    createConfigDatasetPicker,
    renderConfigDatasetPicker,
    isConfigDatasetPickerDialogOpen,
    closeConfigDatasetPickerDialog,
} from '../../config-form/dataset-picker.js?v=module-bootstrap-20260711-ir2';
export {
    clearContinueTrainingSource,
    openContinueLoraDialog,
    loadContinueLoraWeights,
    requestContinueLoraInspection,
    selectContinueLoraWeight,
    refreshContinueTrainingSourceCompatibility,
} from '../../training-source/continue-lora.js?v=module-bootstrap-20260711-ir2';
