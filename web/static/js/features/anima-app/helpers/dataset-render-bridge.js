const datasetRenderers = {
    createDatasetPresetGroupNode: () => null,
    datasetEditorStateForActivePanel: () => ({ datasets: [], defaults: {} }),
    renderConfigDatasetPickerDialog: () => {},
    readDatasetPresetGroupState: () => ({}),
    ensureConfigDatasetPreview: () => {},
    isDatasetTabActive: () => false,
    refreshDatasetEditorItem: () => false,
    refreshDatasetEditorItems: () => false,
    renderDatasetEditor: () => {},
    renderDatasetPresetHeader: () => {},
    renderDatasetPresetList: () => {},
};

export function configureDatasetRenderBridge(renderers = {}) {
    if (typeof renderers.createDatasetPresetGroupNode === 'function') {
        datasetRenderers.createDatasetPresetGroupNode = renderers.createDatasetPresetGroupNode;
    }
    if (typeof renderers.datasetEditorStateForActivePanel === 'function') {
        datasetRenderers.datasetEditorStateForActivePanel = renderers.datasetEditorStateForActivePanel;
    }
    if (typeof renderers.renderConfigDatasetPickerDialog === 'function') {
        datasetRenderers.renderConfigDatasetPickerDialog = renderers.renderConfigDatasetPickerDialog;
    }
    if (typeof renderers.readDatasetPresetGroupState === 'function') {
        datasetRenderers.readDatasetPresetGroupState = renderers.readDatasetPresetGroupState;
    }
    if (typeof renderers.ensureConfigDatasetPreview === 'function') {
        datasetRenderers.ensureConfigDatasetPreview = renderers.ensureConfigDatasetPreview;
    }
    if (typeof renderers.isDatasetTabActive === 'function') {
        datasetRenderers.isDatasetTabActive = renderers.isDatasetTabActive;
    }
    if (typeof renderers.refreshDatasetEditorItem === 'function') {
        datasetRenderers.refreshDatasetEditorItem = renderers.refreshDatasetEditorItem;
    }
    if (typeof renderers.refreshDatasetEditorItems === 'function') {
        datasetRenderers.refreshDatasetEditorItems = renderers.refreshDatasetEditorItems;
    }
    if (typeof renderers.renderDatasetEditor === 'function') {
        datasetRenderers.renderDatasetEditor = renderers.renderDatasetEditor;
    }
    if (typeof renderers.renderDatasetPresetHeader === 'function') {
        datasetRenderers.renderDatasetPresetHeader = renderers.renderDatasetPresetHeader;
    }
    if (typeof renderers.renderDatasetPresetList === 'function') {
        datasetRenderers.renderDatasetPresetList = renderers.renderDatasetPresetList;
    }
}

export function createDatasetPresetGroupNode(...args) {
    return datasetRenderers.createDatasetPresetGroupNode(...args);
}

export function datasetEditorStateForActivePanel(...args) {
    return datasetRenderers.datasetEditorStateForActivePanel(...args);
}

export function renderConfigDatasetPickerDialog(...args) {
    return datasetRenderers.renderConfigDatasetPickerDialog(...args);
}

export function readDatasetPresetGroupState(...args) {
    return datasetRenderers.readDatasetPresetGroupState(...args);
}

export function ensureConfigDatasetPreview(...args) {
    return datasetRenderers.ensureConfigDatasetPreview(...args);
}

export function isDatasetTabActive(...args) {
    return datasetRenderers.isDatasetTabActive(...args);
}

export function refreshDatasetEditorItem(...args) {
    return datasetRenderers.refreshDatasetEditorItem(...args);
}

export function refreshDatasetEditorItems(...args) {
    return datasetRenderers.refreshDatasetEditorItems(...args);
}

export function renderDatasetEditor(...args) {
    return datasetRenderers.renderDatasetEditor(...args);
}

export function renderDatasetPresetHeader(...args) {
    return datasetRenderers.renderDatasetPresetHeader(...args);
}

export function renderDatasetPresetList(...args) {
    return datasetRenderers.renderDatasetPresetList(...args);
}
