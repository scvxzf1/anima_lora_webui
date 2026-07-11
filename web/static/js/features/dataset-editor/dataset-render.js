/**
 * Dataset render public surface and bridge registration.
 * Implementation: dataset-preset-groups.js + dataset-editor-panel.js.
 */
import { configureDatasetRenderBridge } from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    createDatasetPresetGroupNode,
    readDatasetPresetGroupState,
    renderDatasetPresetHeader,
} from './dataset-preset-groups.js?v=module-bootstrap-20260711-ir6';
import {
    activeDatasetDirty,
    activeDatasetFileLabel,
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    refreshDatasetEditorItem,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    setActiveDatasetRows,
} from './dataset-editor-panel.js?v=module-bootstrap-20260711-ir6';

export {
    createDatasetPresetGroupNode,
    readDatasetPresetGroupState,
    renderDatasetPresetHeader,
} from './dataset-preset-groups.js?v=module-bootstrap-20260711-ir6';

export {
    activeDatasetDirty,
    activeDatasetFileLabel,
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    refreshDatasetEditorItem,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    setActiveDatasetRows,
} from './dataset-editor-panel.js?v=module-bootstrap-20260711-ir6';

configureDatasetRenderBridge({
    createDatasetPresetGroupNode,
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    readDatasetPresetGroupState,
    refreshDatasetEditorItem,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    renderDatasetPresetHeader,
});
