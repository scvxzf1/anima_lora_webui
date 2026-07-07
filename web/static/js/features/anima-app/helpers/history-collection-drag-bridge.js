const legacyRoot = globalThis;

const historyCollectionDragBridge = {
    startHistoryCollectionPointerDrag: (...args) => legacyRoot.startHistoryCollectionPointerDrag?.(...args),
    startHistoryCollectionMouseDrag: (...args) => legacyRoot.startHistoryCollectionMouseDrag?.(...args),
    startHistoryCollectionTouchDrag: (...args) => legacyRoot.startHistoryCollectionTouchDrag?.(...args),
    readHistoryDraggedCollectionValue: (...args) => legacyRoot.readHistoryDraggedCollectionValue?.(...args),
    historyCollectionDropPosition: (...args) => legacyRoot.historyCollectionDropPosition?.(...args),
    setHistoryCollectionSortTarget: (...args) => legacyRoot.setHistoryCollectionSortTarget?.(...args),
    clearHistoryCollectionSortTarget: (...args) => legacyRoot.clearHistoryCollectionSortTarget?.(...args),
    historyCollectionOrderDragEnter: (...args) => legacyRoot.historyCollectionOrderDragEnter?.(...args),
    historyCollectionOrderDragLeave: (...args) => legacyRoot.historyCollectionOrderDragLeave?.(...args),
    moveItemNearList: (...args) => legacyRoot.moveItemNearList?.(...args),
    reorderHistoryCollectionValue: (...args) => legacyRoot.reorderHistoryCollectionValue?.(...args),
    dropHistoryCollectionToSort: (...args) => legacyRoot.dropHistoryCollectionToSort?.(...args),
    dropHistoryTasksToCollection: (...args) => legacyRoot.dropHistoryTasksToCollection?.(...args),
    defaultHistoryCollectionName: (...args) => legacyRoot.defaultHistoryCollectionName?.(...args),
    uniqueHistoryCollectionName: (...args) => legacyRoot.uniqueHistoryCollectionName?.(...args),
    openHistoryNewCollectionPopover: (...args) => legacyRoot.openHistoryNewCollectionPopover?.(...args),
    renderHistoryDropPopover: (...args) => legacyRoot.renderHistoryDropPopover?.(...args),
    closeHistoryDropPopover: (...args) => legacyRoot.closeHistoryDropPopover?.(...args),
    submitHistoryDropPopover: (...args) => legacyRoot.submitHistoryDropPopover?.(...args),
    setHistoryDropFeedback: (...args) => legacyRoot.setHistoryDropFeedback?.(...args),
    startHistoryConfigGroupPointerDrag: (...args) => legacyRoot.startHistoryConfigGroupPointerDrag?.(...args),
    startHistoryConfigGroupMouseDrag: (...args) => legacyRoot.startHistoryConfigGroupMouseDrag?.(...args),
    startHistoryConfigGroupTouchDrag: (...args) => legacyRoot.startHistoryConfigGroupTouchDrag?.(...args),
    reorderHistoryConfigGroupValue: (...args) => legacyRoot.reorderHistoryConfigGroupValue?.(...args),
    dropHistoryConfigGroupToSort: (...args) => legacyRoot.dropHistoryConfigGroupToSort?.(...args),
    readHistoryDraggedTaskIds: (...args) => legacyRoot.readHistoryDraggedTaskIds?.(...args),
    setHistoryDropTarget: (...args) => legacyRoot.setHistoryDropTarget?.(...args),
    clearHistoryDropTarget: (...args) => legacyRoot.clearHistoryDropTarget?.(...args),
    clearHistoryDropIndicators: (...args) => legacyRoot.clearHistoryDropIndicators?.(...args),
    historyTasksByIds: (...args) => legacyRoot.historyTasksByIds?.(...args),
    historyDraggedTasksAlreadyInCollection: (...args) => legacyRoot.historyDraggedTasksAlreadyInCollection?.(...args),
    historyDropTargetDragEnter: (...args) => legacyRoot.historyDropTargetDragEnter?.(...args),
    historyDropTargetDragLeave: (...args) => legacyRoot.historyDropTargetDragLeave?.(...args),
    canBeginHistoryCollectionSort: (...args) => legacyRoot.canBeginHistoryCollectionSort?.(...args),
    beginHistoryCollectionDrag: (...args) => legacyRoot.beginHistoryCollectionDrag?.(...args),
    finishHistoryCollectionDrag: (...args) => legacyRoot.finishHistoryCollectionDrag?.(...args),
    clearHistoryCollectionSortIndicators: (...args) => legacyRoot.clearHistoryCollectionSortIndicators?.(...args),
    createHistoryCollectionPointerDragImage: (...args) => legacyRoot.createHistoryCollectionPointerDragImage?.(...args),
    moveHistoryCollectionPointerDragImage: (...args) => legacyRoot.moveHistoryCollectionPointerDragImage?.(...args),
    historyCollectionForPointerCard: (...args) => legacyRoot.historyCollectionForPointerCard?.(...args),
    historyCollectionPointerTargetForCard: (...args) => legacyRoot.historyCollectionPointerTargetForCard?.(...args),
    nearestHistoryCollectionPointerTarget: (...args) => legacyRoot.nearestHistoryCollectionPointerTarget?.(...args),
    historyCollectionPointerTargetFromPoint: (...args) => legacyRoot.historyCollectionPointerTargetFromPoint?.(...args),
    findHistoryCollectionPointerScroller: (...args) => legacyRoot.findHistoryCollectionPointerScroller?.(...args),
    autoScrollHistoryCollectionPointerDrag: (...args) => legacyRoot.autoScrollHistoryCollectionPointerDrag?.(...args),
    cleanupHistoryCollectionPointerDrag: (...args) => legacyRoot.cleanupHistoryCollectionPointerDrag?.(...args),
    historyCollectionEventPoint: (...args) => legacyRoot.historyCollectionEventPoint?.(...args),
    finishHistoryCollectionPointerDrag: (...args) => legacyRoot.finishHistoryCollectionPointerDrag?.(...args),
    historyDragTaskIdsForGroup: (...args) => legacyRoot.historyDragTaskIdsForGroup?.(...args),
    createHistoryDragImage: (...args) => legacyRoot.createHistoryDragImage?.(...args),
    removeHistoryDragImage: (...args) => legacyRoot.removeHistoryDragImage?.(...args),
    canBeginHistoryConfigGroupDrag: (...args) => legacyRoot.canBeginHistoryConfigGroupDrag?.(...args),
    beginHistoryConfigGroupDrag: (...args) => legacyRoot.beginHistoryConfigGroupDrag?.(...args),
    finishHistoryDrag: (...args) => legacyRoot.finishHistoryDrag?.(...args),
    readHistoryDraggedConfigGroup: (...args) => legacyRoot.readHistoryDraggedConfigGroup?.(...args),
    historyConfigGroupDropPosition: (...args) => legacyRoot.historyConfigGroupDropPosition?.(...args),
    removeHistoryConfigGroupDropPreview: (...args) => legacyRoot.removeHistoryConfigGroupDropPreview?.(...args),
    ensureHistoryConfigGroupDropPreview: (...args) => legacyRoot.ensureHistoryConfigGroupDropPreview?.(...args),
    placeHistoryConfigGroupDropPreview: (...args) => legacyRoot.placeHistoryConfigGroupDropPreview?.(...args),
    setHistoryConfigGroupSortTarget: (...args) => legacyRoot.setHistoryConfigGroupSortTarget?.(...args),
    clearHistoryConfigGroupSortTarget: (...args) => legacyRoot.clearHistoryConfigGroupSortTarget?.(...args),
    clearHistoryConfigGroupSortIndicators: (...args) => legacyRoot.clearHistoryConfigGroupSortIndicators?.(...args),
    historyConfigGroupOrderDragEnter: (...args) => legacyRoot.historyConfigGroupOrderDragEnter?.(...args),
    historyConfigGroupOrderDragLeave: (...args) => legacyRoot.historyConfigGroupOrderDragLeave?.(...args),
    historyConfigGroupForPointerCard: (...args) => legacyRoot.historyConfigGroupForPointerCard?.(...args),
    historyConfigGroupPointerTargetForCard: (...args) => legacyRoot.historyConfigGroupPointerTargetForCard?.(...args),
    nearestHistoryConfigGroupPointerTarget: (...args) => legacyRoot.nearestHistoryConfigGroupPointerTarget?.(...args),
    historyConfigGroupPointerTargetFromPoint: (...args) => legacyRoot.historyConfigGroupPointerTargetFromPoint?.(...args),
    historyCollectionDropTargetFromPoint: (...args) => legacyRoot.historyCollectionDropTargetFromPoint?.(...args),
    cleanupHistoryConfigGroupPointerDrag: (...args) => legacyRoot.cleanupHistoryConfigGroupPointerDrag?.(...args),
    finishHistoryConfigGroupPointerDrag: (...args) => legacyRoot.finishHistoryConfigGroupPointerDrag?.(...args),
    dropHistoryTasksToCollectionLikePointer: (...args) => legacyRoot.dropHistoryTasksToCollectionLikePointer?.(...args),
};

export function configureHistoryCollectionDragBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function' && key in historyCollectionDragBridge) {
            historyCollectionDragBridge[key] = handler;
        }
    }
}

export const startHistoryCollectionPointerDrag = (...args) => historyCollectionDragBridge.startHistoryCollectionPointerDrag(...args);
export const startHistoryCollectionMouseDrag = (...args) => historyCollectionDragBridge.startHistoryCollectionMouseDrag(...args);
export const startHistoryCollectionTouchDrag = (...args) => historyCollectionDragBridge.startHistoryCollectionTouchDrag(...args);
export const readHistoryDraggedCollectionValue = (...args) => historyCollectionDragBridge.readHistoryDraggedCollectionValue(...args);
export const historyCollectionDropPosition = (...args) => historyCollectionDragBridge.historyCollectionDropPosition(...args);
export const setHistoryCollectionSortTarget = (...args) => historyCollectionDragBridge.setHistoryCollectionSortTarget(...args);
export const clearHistoryCollectionSortTarget = (...args) => historyCollectionDragBridge.clearHistoryCollectionSortTarget(...args);
export const historyCollectionOrderDragEnter = (...args) => historyCollectionDragBridge.historyCollectionOrderDragEnter(...args);
export const historyCollectionOrderDragLeave = (...args) => historyCollectionDragBridge.historyCollectionOrderDragLeave(...args);
export const moveItemNearList = (...args) => historyCollectionDragBridge.moveItemNearList(...args);
export const reorderHistoryCollectionValue = (...args) => historyCollectionDragBridge.reorderHistoryCollectionValue(...args);
export const dropHistoryCollectionToSort = (...args) => historyCollectionDragBridge.dropHistoryCollectionToSort(...args);
export const dropHistoryTasksToCollection = (...args) => historyCollectionDragBridge.dropHistoryTasksToCollection(...args);
export const defaultHistoryCollectionName = (...args) => historyCollectionDragBridge.defaultHistoryCollectionName(...args);
export const uniqueHistoryCollectionName = (...args) => historyCollectionDragBridge.uniqueHistoryCollectionName(...args);
export const openHistoryNewCollectionPopover = (...args) => historyCollectionDragBridge.openHistoryNewCollectionPopover(...args);
export const renderHistoryDropPopover = (...args) => historyCollectionDragBridge.renderHistoryDropPopover(...args);
export const closeHistoryDropPopover = (...args) => historyCollectionDragBridge.closeHistoryDropPopover(...args);
export const submitHistoryDropPopover = (...args) => historyCollectionDragBridge.submitHistoryDropPopover(...args);
export const setHistoryDropFeedback = (...args) => historyCollectionDragBridge.setHistoryDropFeedback(...args);
export const startHistoryConfigGroupPointerDrag = (...args) => historyCollectionDragBridge.startHistoryConfigGroupPointerDrag(...args);
export const startHistoryConfigGroupMouseDrag = (...args) => historyCollectionDragBridge.startHistoryConfigGroupMouseDrag(...args);
export const startHistoryConfigGroupTouchDrag = (...args) => historyCollectionDragBridge.startHistoryConfigGroupTouchDrag(...args);
export const reorderHistoryConfigGroupValue = (...args) => historyCollectionDragBridge.reorderHistoryConfigGroupValue(...args);
export const dropHistoryConfigGroupToSort = (...args) => historyCollectionDragBridge.dropHistoryConfigGroupToSort(...args);
export const readHistoryDraggedTaskIds = (...args) => historyCollectionDragBridge.readHistoryDraggedTaskIds(...args);
export const setHistoryDropTarget = (...args) => historyCollectionDragBridge.setHistoryDropTarget(...args);
export const clearHistoryDropTarget = (...args) => historyCollectionDragBridge.clearHistoryDropTarget(...args);
export const clearHistoryDropIndicators = (...args) => historyCollectionDragBridge.clearHistoryDropIndicators(...args);
export const historyTasksByIds = (...args) => historyCollectionDragBridge.historyTasksByIds(...args);
export const historyDraggedTasksAlreadyInCollection = (...args) => historyCollectionDragBridge.historyDraggedTasksAlreadyInCollection(...args);
export const historyDropTargetDragEnter = (...args) => historyCollectionDragBridge.historyDropTargetDragEnter(...args);
export const historyDropTargetDragLeave = (...args) => historyCollectionDragBridge.historyDropTargetDragLeave(...args);
export const canBeginHistoryCollectionSort = (...args) => historyCollectionDragBridge.canBeginHistoryCollectionSort(...args);
export const beginHistoryCollectionDrag = (...args) => historyCollectionDragBridge.beginHistoryCollectionDrag(...args);
export const finishHistoryCollectionDrag = (...args) => historyCollectionDragBridge.finishHistoryCollectionDrag(...args);
export const clearHistoryCollectionSortIndicators = (...args) => historyCollectionDragBridge.clearHistoryCollectionSortIndicators(...args);
export const createHistoryCollectionPointerDragImage = (...args) => historyCollectionDragBridge.createHistoryCollectionPointerDragImage(...args);
export const moveHistoryCollectionPointerDragImage = (...args) => historyCollectionDragBridge.moveHistoryCollectionPointerDragImage(...args);
export const historyCollectionForPointerCard = (...args) => historyCollectionDragBridge.historyCollectionForPointerCard(...args);
export const historyCollectionPointerTargetForCard = (...args) => historyCollectionDragBridge.historyCollectionPointerTargetForCard(...args);
export const nearestHistoryCollectionPointerTarget = (...args) => historyCollectionDragBridge.nearestHistoryCollectionPointerTarget(...args);
export const historyCollectionPointerTargetFromPoint = (...args) => historyCollectionDragBridge.historyCollectionPointerTargetFromPoint(...args);
export const findHistoryCollectionPointerScroller = (...args) => historyCollectionDragBridge.findHistoryCollectionPointerScroller(...args);
export const autoScrollHistoryCollectionPointerDrag = (...args) => historyCollectionDragBridge.autoScrollHistoryCollectionPointerDrag(...args);
export const cleanupHistoryCollectionPointerDrag = (...args) => historyCollectionDragBridge.cleanupHistoryCollectionPointerDrag(...args);
export const historyCollectionEventPoint = (...args) => historyCollectionDragBridge.historyCollectionEventPoint(...args);
export const finishHistoryCollectionPointerDrag = (...args) => historyCollectionDragBridge.finishHistoryCollectionPointerDrag(...args);
export const historyDragTaskIdsForGroup = (...args) => historyCollectionDragBridge.historyDragTaskIdsForGroup(...args);
export const createHistoryDragImage = (...args) => historyCollectionDragBridge.createHistoryDragImage(...args);
export const removeHistoryDragImage = (...args) => historyCollectionDragBridge.removeHistoryDragImage(...args);
export const canBeginHistoryConfigGroupDrag = (...args) => historyCollectionDragBridge.canBeginHistoryConfigGroupDrag(...args);
export const beginHistoryConfigGroupDrag = (...args) => historyCollectionDragBridge.beginHistoryConfigGroupDrag(...args);
export const finishHistoryDrag = (...args) => historyCollectionDragBridge.finishHistoryDrag(...args);
export const readHistoryDraggedConfigGroup = (...args) => historyCollectionDragBridge.readHistoryDraggedConfigGroup(...args);
export const historyConfigGroupDropPosition = (...args) => historyCollectionDragBridge.historyConfigGroupDropPosition(...args);
export const removeHistoryConfigGroupDropPreview = (...args) => historyCollectionDragBridge.removeHistoryConfigGroupDropPreview(...args);
export const ensureHistoryConfigGroupDropPreview = (...args) => historyCollectionDragBridge.ensureHistoryConfigGroupDropPreview(...args);
export const placeHistoryConfigGroupDropPreview = (...args) => historyCollectionDragBridge.placeHistoryConfigGroupDropPreview(...args);
export const setHistoryConfigGroupSortTarget = (...args) => historyCollectionDragBridge.setHistoryConfigGroupSortTarget(...args);
export const clearHistoryConfigGroupSortTarget = (...args) => historyCollectionDragBridge.clearHistoryConfigGroupSortTarget(...args);
export const clearHistoryConfigGroupSortIndicators = (...args) => historyCollectionDragBridge.clearHistoryConfigGroupSortIndicators(...args);
export const historyConfigGroupOrderDragEnter = (...args) => historyCollectionDragBridge.historyConfigGroupOrderDragEnter(...args);
export const historyConfigGroupOrderDragLeave = (...args) => historyCollectionDragBridge.historyConfigGroupOrderDragLeave(...args);
export const historyConfigGroupForPointerCard = (...args) => historyCollectionDragBridge.historyConfigGroupForPointerCard(...args);
export const historyConfigGroupPointerTargetForCard = (...args) => historyCollectionDragBridge.historyConfigGroupPointerTargetForCard(...args);
export const nearestHistoryConfigGroupPointerTarget = (...args) => historyCollectionDragBridge.nearestHistoryConfigGroupPointerTarget(...args);
export const historyConfigGroupPointerTargetFromPoint = (...args) => historyCollectionDragBridge.historyConfigGroupPointerTargetFromPoint(...args);
export const historyCollectionDropTargetFromPoint = (...args) => historyCollectionDragBridge.historyCollectionDropTargetFromPoint(...args);
export const cleanupHistoryConfigGroupPointerDrag = (...args) => historyCollectionDragBridge.cleanupHistoryConfigGroupPointerDrag(...args);
export const finishHistoryConfigGroupPointerDrag = (...args) => historyCollectionDragBridge.finishHistoryConfigGroupPointerDrag(...args);
export const dropHistoryTasksToCollectionLikePointer = (...args) => historyCollectionDragBridge.dropHistoryTasksToCollectionLikePointer(...args);
