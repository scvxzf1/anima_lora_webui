/**
 * Compatibility barrel for former chunk 12.
 * Implementations live in dataset-editor/row-fields.js and preview.js.
 */
export {
    createDatasetPathField,
    createDatasetRowCaptionSourceModeEditor,
    createDatasetRowSettingInput,
    updateDatasetDefault,
    updateDatasetEditorRow,
    updateDatasetEditorRowSettingValue,
} from '../../dataset-editor/row-fields.js?v=module-bootstrap-20260809-nf4-v2';
export {
    loadDatasetPreviewImages,
    openDatasetPreview,
} from '../../dataset-editor/preview.js?v=module-bootstrap-20260809-nf4-v2';
