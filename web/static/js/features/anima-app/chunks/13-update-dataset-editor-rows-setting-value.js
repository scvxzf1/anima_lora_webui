/**
 * Compatibility barrel for former chunk 13.
 * Implementations live in dataset-editor/mutations.js and config-form helpers.
 */
export {
    setFieldInputValue,
    escapeHtml,
} from '../../config-form/field-input.js?v=module-bootstrap-20260831-release-v1';
export {
    setCurrentTrainingSourceFromVariant,
    clearCurrentTrainingSource,
} from '../../training-source/source-state.js?v=module-bootstrap-20260831-release-v1';
export { outputRunRuntimeFile } from '../../output-run/runtime-file.js?v=module-bootstrap-20260831-release-v1';
export {
    activeMethodKey,
    inferMethodFromConfig,
} from '../../config-form/method-key.js?v=module-bootstrap-20260831-release-v1';
export {
    updateDatasetEditorRowsSettingValue,
    updateDatasetEditorRowNlTagMix,
    updateDatasetEditorRowsNlTagMix,
    updateDatasetEditorRowTriggerClone,
    datasetExperimentalScopeIndices,
    setDatasetExperimentalScopeIndices,
    datasetValidTargetIndices,
    moveDatasetEditorRow,
    moveDatasetEditorRowToIndex,
    markDatasetEditorDirty,
    addDatasetEditorRow,
    removeDatasetEditorRow,
    syncDatasetEditorToCompatFields,
} from '../../dataset-editor/mutations.js?v=module-bootstrap-20260831-release-v1';
export {
    rememberSelectionSnapshot,
    restoreSelectionSnapshot,
    confirmBeforeConfigSelectionChange,
    updateChoiceGuide,
    createChoiceCard,
    methodGuideFromConfig,
    configGuideFromCurrentSource,
    presetGuideFromConfig,
} from '../../config-form/choice-guide-ui.js?v=module-bootstrap-20260831-release-v1';
