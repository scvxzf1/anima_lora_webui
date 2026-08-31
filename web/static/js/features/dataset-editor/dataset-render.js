/**
 * Dataset render public surface and bridge registration.
 * Implementation: dataset-preset-groups.js + dataset-editor-panel.js.
 */
import { configureDatasetRenderBridge } from '../anima-app/helpers/dataset-render-bridge.js?v=module-bootstrap-20260831-release-v1';
import {
    createDatasetPresetGroupNode,
    renderDatasetPresetHeader,
} from './dataset-preset-groups.js?v=module-bootstrap-20260831-release-v1';
import {
    activeDatasetDirty,
    activeDatasetFileLabel,
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    refreshDatasetEditorItem,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    setActiveDatasetRows,
} from './dataset-editor-panel.js?v=module-bootstrap-20260831-release-v1';

export {
    createDatasetPresetGroupNode,
    renderDatasetPresetHeader,
} from './dataset-preset-groups.js?v=module-bootstrap-20260831-release-v1';

export {
    activeDatasetDirty,
    activeDatasetFileLabel,
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    refreshDatasetEditorItem,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    setActiveDatasetRows,
} from './dataset-editor-panel.js?v=module-bootstrap-20260831-release-v1';

configureDatasetRenderBridge({
    createDatasetPresetGroupNode,
    datasetEditorStateForActivePanel,
    isDatasetTabActive,
    refreshDatasetEditorItem,
    refreshDatasetEditorItems,
    renderDatasetEditor,
    renderDatasetPresetHeader,
});
