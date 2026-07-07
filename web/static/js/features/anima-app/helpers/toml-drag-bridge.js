const legacyRoot = globalThis;

const tomlDragBridge = {
    canDropTomlFileToGroup: (...args) => legacyRoot.canDropTomlFileToGroup?.(...args),
    isTomlFileDraggable: (...args) => legacyRoot.isTomlFileDraggable?.(...args),
    createTomlGroupDragHandle: (...args) => legacyRoot.createTomlGroupDragHandle?.(...args),
    placeTomlGroup: (...args) => legacyRoot.placeTomlGroup?.(...args),
    placeTomlFile: (...args) => legacyRoot.placeTomlFile?.(...args),
    tomlFileDragOptions: (...args) => legacyRoot.tomlFileDragOptions?.(...args),
    tomlGroupDragOptions: (...args) => legacyRoot.tomlGroupDragOptions?.(...args),
    populateTomlFileSelect: (...args) => legacyRoot.populateTomlFileSelect?.(...args),
    renderTomlFileGroups: (...args) => legacyRoot.renderTomlFileGroups?.(...args),
    renderTomlFileGroupList: (...args) => legacyRoot.renderTomlFileGroupList?.(...args),
    createTomlGroupActions: (...args) => legacyRoot.createTomlGroupActions?.(...args),
    exportableTomlGroupFiles: (...args) => legacyRoot.exportableTomlGroupFiles?.(...args),
    exportTomlGroup: (...args) => legacyRoot.exportTomlGroup?.(...args),
    exportTomlGroupFilename: (...args) => legacyRoot.exportTomlGroupFilename?.(...args),
    queueableTomlGroupFiles: (...args) => legacyRoot.queueableTomlGroupFiles?.(...args),
    tomlItemQueueVariant: (...args) => legacyRoot.tomlItemQueueVariant?.(...args),
    tomlItemQueueEntry: (...args) => legacyRoot.tomlItemQueueEntry?.(...args),
    tomlGroupQueueFailureLabel: (...args) => legacyRoot.tomlGroupQueueFailureLabel?.(...args),
    showTomlGroupQueueConfirmDialog: (...args) => legacyRoot.showTomlGroupQueueConfirmDialog?.(...args),
    enqueueTomlGroupToQueue: (...args) => legacyRoot.enqueueTomlGroupToQueue?.(...args),
    createTomlGroupActionButton: (...args) => legacyRoot.createTomlGroupActionButton?.(...args),
    runTomlGroupAction: (...args) => legacyRoot.runTomlGroupAction?.(...args),
    createTomlFileButton: (...args) => legacyRoot.createTomlFileButton?.(...args),
};

export function configureTomlDragBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in tomlDragBridge) {
            tomlDragBridge[key] = handler;
        }
    }
}

export const canDropTomlFileToGroup = (...args) => tomlDragBridge.canDropTomlFileToGroup(...args);
export const isTomlFileDraggable = (...args) => tomlDragBridge.isTomlFileDraggable(...args);
export const createTomlGroupDragHandle = (...args) => tomlDragBridge.createTomlGroupDragHandle(...args);
export const placeTomlGroup = (...args) => tomlDragBridge.placeTomlGroup(...args);
export const placeTomlFile = (...args) => tomlDragBridge.placeTomlFile(...args);
export const tomlFileDragOptions = (...args) => tomlDragBridge.tomlFileDragOptions(...args);
export const tomlGroupDragOptions = (...args) => tomlDragBridge.tomlGroupDragOptions(...args);
export const populateTomlFileSelect = (...args) => tomlDragBridge.populateTomlFileSelect(...args);
export const renderTomlFileGroups = (...args) => tomlDragBridge.renderTomlFileGroups(...args);
export const renderTomlFileGroupList = (...args) => tomlDragBridge.renderTomlFileGroupList(...args);
export const createTomlGroupActions = (...args) => tomlDragBridge.createTomlGroupActions(...args);
export const exportableTomlGroupFiles = (...args) => tomlDragBridge.exportableTomlGroupFiles(...args);
export const exportTomlGroup = (...args) => tomlDragBridge.exportTomlGroup(...args);
export const exportTomlGroupFilename = (...args) => tomlDragBridge.exportTomlGroupFilename(...args);
export const queueableTomlGroupFiles = (...args) => tomlDragBridge.queueableTomlGroupFiles(...args);
export const tomlItemQueueVariant = (...args) => tomlDragBridge.tomlItemQueueVariant(...args);
export const tomlItemQueueEntry = (...args) => tomlDragBridge.tomlItemQueueEntry(...args);
export const tomlGroupQueueFailureLabel = (...args) => tomlDragBridge.tomlGroupQueueFailureLabel(...args);
export const showTomlGroupQueueConfirmDialog = (...args) => tomlDragBridge.showTomlGroupQueueConfirmDialog(...args);
export const enqueueTomlGroupToQueue = (...args) => tomlDragBridge.enqueueTomlGroupToQueue(...args);
export const createTomlGroupActionButton = (...args) => tomlDragBridge.createTomlGroupActionButton(...args);
export const runTomlGroupAction = (...args) => tomlDragBridge.runTomlGroupAction(...args);
export const createTomlFileButton = (...args) => tomlDragBridge.createTomlFileButton(...args);
