/**
 * Dataset render public surface and bridge registration.
 * Implementation: dataset-preset-groups.js + dataset-editor-panel.js.
 */
import { configureDatasetRenderBridge } from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    createDatasetPresetGroupNode,
    renderDatasetPresetHeader,
} from './dataset-preset-groups.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    activeDatasetDirty,
    activeDatasetFileLabel,
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    refreshDatasetEditorItem,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    setActiveDatasetRows,
} from './dataset-editor-panel.js?v=module-bootstrap-20260714-stage-dataset5';

export {
    createDatasetPresetGroupNode,
    renderDatasetPresetHeader,
} from './dataset-preset-groups.js?v=module-bootstrap-20260714-stage-dataset5';

export {
    activeDatasetDirty,
    activeDatasetFileLabel,
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    refreshDatasetEditorItem,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    setActiveDatasetRows,
} from './dataset-editor-panel.js?v=module-bootstrap-20260714-stage-dataset5';

configureDatasetRenderBridge({
    createDatasetPresetGroupNode,
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    refreshDatasetEditorItem,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    renderDatasetPresetHeader,
});
