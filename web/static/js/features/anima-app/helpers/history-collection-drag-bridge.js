const historyCollectionDragHandlers = Object.create(null);

function requireHistoryCollectionDragHandler(name) {
    const handler = historyCollectionDragHandlers[name];
    if (typeof handler !== 'function') {
        throw new Error(`[history-collection-drag] bridge not configured: ${name}`);
    }
    return handler;
}

export function configureHistoryCollectionDragBridge(handlers = {}) {
    for (const [key, handler] of Object.entries(handlers)) {
        if (typeof handler === 'function') {
            historyCollectionDragHandlers[key] = handler;
        }
    }
}

export function startHistoryCollectionPointerDrag(...args) { return requireHistoryCollectionDragHandler('startHistoryCollectionPointerDrag')(...args); }
export function startHistoryCollectionMouseDrag(...args) { return requireHistoryCollectionDragHandler('startHistoryCollectionMouseDrag')(...args); }
export function startHistoryCollectionTouchDrag(...args) { return requireHistoryCollectionDragHandler('startHistoryCollectionTouchDrag')(...args); }
export function readHistoryDraggedCollectionValue(...args) { return requireHistoryCollectionDragHandler('readHistoryDraggedCollectionValue')(...args); }
export function historyCollectionDropPosition(...args) { return requireHistoryCollectionDragHandler('historyCollectionDropPosition')(...args); }
export function setHistoryCollectionSortTarget(...args) { return requireHistoryCollectionDragHandler('setHistoryCollectionSortTarget')(...args); }
export function clearHistoryCollectionSortTarget(...args) { return requireHistoryCollectionDragHandler('clearHistoryCollectionSortTarget')(...args); }
export function historyCollectionOrderDragEnter(...args) { return requireHistoryCollectionDragHandler('historyCollectionOrderDragEnter')(...args); }
export function historyCollectionOrderDragLeave(...args) { return requireHistoryCollectionDragHandler('historyCollectionOrderDragLeave')(...args); }
export function moveItemNearList(...args) { return requireHistoryCollectionDragHandler('moveItemNearList')(...args); }
export function reorderHistoryCollectionValue(...args) { return requireHistoryCollectionDragHandler('reorderHistoryCollectionValue')(...args); }
export function dropHistoryCollectionToSort(...args) { return requireHistoryCollectionDragHandler('dropHistoryCollectionToSort')(...args); }
export function dropHistoryTasksToCollection(...args) { return requireHistoryCollectionDragHandler('dropHistoryTasksToCollection')(...args); }
export function defaultHistoryCollectionName(...args) { return requireHistoryCollectionDragHandler('defaultHistoryCollectionName')(...args); }
export function uniqueHistoryCollectionName(...args) { return requireHistoryCollectionDragHandler('uniqueHistoryCollectionName')(...args); }
export function openHistoryNewCollectionPopover(...args) { return requireHistoryCollectionDragHandler('openHistoryNewCollectionPopover')(...args); }
export function renderHistoryDropPopover(...args) { return requireHistoryCollectionDragHandler('renderHistoryDropPopover')(...args); }
export function closeHistoryDropPopover(...args) { return requireHistoryCollectionDragHandler('closeHistoryDropPopover')(...args); }
export function submitHistoryDropPopover(...args) { return requireHistoryCollectionDragHandler('submitHistoryDropPopover')(...args); }
export function setHistoryDropFeedback(...args) { return requireHistoryCollectionDragHandler('setHistoryDropFeedback')(...args); }
export function startHistoryConfigGroupPointerDrag(...args) { return requireHistoryCollectionDragHandler('startHistoryConfigGroupPointerDrag')(...args); }
export function startHistoryConfigGroupMouseDrag(...args) { return requireHistoryCollectionDragHandler('startHistoryConfigGroupMouseDrag')(...args); }
export function startHistoryConfigGroupTouchDrag(...args) { return requireHistoryCollectionDragHandler('startHistoryConfigGroupTouchDrag')(...args); }
export function reorderHistoryConfigGroupValue(...args) { return requireHistoryCollectionDragHandler('reorderHistoryConfigGroupValue')(...args); }
export function dropHistoryConfigGroupToSort(...args) { return requireHistoryCollectionDragHandler('dropHistoryConfigGroupToSort')(...args); }
export function readHistoryDraggedTaskIds(...args) { return requireHistoryCollectionDragHandler('readHistoryDraggedTaskIds')(...args); }
export function setHistoryDropTarget(...args) { return requireHistoryCollectionDragHandler('setHistoryDropTarget')(...args); }
export function clearHistoryDropTarget(...args) { return requireHistoryCollectionDragHandler('clearHistoryDropTarget')(...args); }
export function clearHistoryDropIndicators(...args) { return requireHistoryCollectionDragHandler('clearHistoryDropIndicators')(...args); }
export function historyTasksByIds(...args) { return requireHistoryCollectionDragHandler('historyTasksByIds')(...args); }
export function historyDraggedTasksAlreadyInCollection(...args) { return requireHistoryCollectionDragHandler('historyDraggedTasksAlreadyInCollection')(...args); }
export function historyDropTargetDragEnter(...args) { return requireHistoryCollectionDragHandler('historyDropTargetDragEnter')(...args); }
export function historyDropTargetDragLeave(...args) { return requireHistoryCollectionDragHandler('historyDropTargetDragLeave')(...args); }
export function canBeginHistoryCollectionSort(...args) { return requireHistoryCollectionDragHandler('canBeginHistoryCollectionSort')(...args); }
export function beginHistoryCollectionDrag(...args) { return requireHistoryCollectionDragHandler('beginHistoryCollectionDrag')(...args); }
export function finishHistoryCollectionDrag(...args) { return requireHistoryCollectionDragHandler('finishHistoryCollectionDrag')(...args); }
export function clearHistoryCollectionSortIndicators(...args) { return requireHistoryCollectionDragHandler('clearHistoryCollectionSortIndicators')(...args); }
export function createHistoryCollectionPointerDragImage(...args) { return requireHistoryCollectionDragHandler('createHistoryCollectionPointerDragImage')(...args); }
export function moveHistoryCollectionPointerDragImage(...args) { return requireHistoryCollectionDragHandler('moveHistoryCollectionPointerDragImage')(...args); }
export function historyCollectionForPointerCard(...args) { return requireHistoryCollectionDragHandler('historyCollectionForPointerCard')(...args); }
export function historyCollectionPointerTargetForCard(...args) { return requireHistoryCollectionDragHandler('historyCollectionPointerTargetForCard')(...args); }
export function nearestHistoryCollectionPointerTarget(...args) { return requireHistoryCollectionDragHandler('nearestHistoryCollectionPointerTarget')(...args); }
export function historyCollectionPointerTargetFromPoint(...args) { return requireHistoryCollectionDragHandler('historyCollectionPointerTargetFromPoint')(...args); }
export function findHistoryCollectionPointerScroller(...args) { return requireHistoryCollectionDragHandler('findHistoryCollectionPointerScroller')(...args); }
export function autoScrollHistoryCollectionPointerDrag(...args) { return requireHistoryCollectionDragHandler('autoScrollHistoryCollectionPointerDrag')(...args); }
export function cleanupHistoryCollectionPointerDrag(...args) { return requireHistoryCollectionDragHandler('cleanupHistoryCollectionPointerDrag')(...args); }
export function historyCollectionEventPoint(...args) { return requireHistoryCollectionDragHandler('historyCollectionEventPoint')(...args); }
export function finishHistoryCollectionPointerDrag(...args) { return requireHistoryCollectionDragHandler('finishHistoryCollectionPointerDrag')(...args); }
export function historyDragTaskIdsForGroup(...args) { return requireHistoryCollectionDragHandler('historyDragTaskIdsForGroup')(...args); }
export function createHistoryDragImage(...args) { return requireHistoryCollectionDragHandler('createHistoryDragImage')(...args); }
export function removeHistoryDragImage(...args) { return requireHistoryCollectionDragHandler('removeHistoryDragImage')(...args); }
export function canBeginHistoryConfigGroupDrag(...args) { return requireHistoryCollectionDragHandler('canBeginHistoryConfigGroupDrag')(...args); }
export function beginHistoryConfigGroupDrag(...args) { return requireHistoryCollectionDragHandler('beginHistoryConfigGroupDrag')(...args); }
export function finishHistoryDrag(...args) { return requireHistoryCollectionDragHandler('finishHistoryDrag')(...args); }
export function readHistoryDraggedConfigGroup(...args) { return requireHistoryCollectionDragHandler('readHistoryDraggedConfigGroup')(...args); }
export function historyConfigGroupDropPosition(...args) { return requireHistoryCollectionDragHandler('historyConfigGroupDropPosition')(...args); }
export function removeHistoryConfigGroupDropPreview(...args) { return requireHistoryCollectionDragHandler('removeHistoryConfigGroupDropPreview')(...args); }
export function ensureHistoryConfigGroupDropPreview(...args) { return requireHistoryCollectionDragHandler('ensureHistoryConfigGroupDropPreview')(...args); }
export function placeHistoryConfigGroupDropPreview(...args) { return requireHistoryCollectionDragHandler('placeHistoryConfigGroupDropPreview')(...args); }
export function setHistoryConfigGroupSortTarget(...args) { return requireHistoryCollectionDragHandler('setHistoryConfigGroupSortTarget')(...args); }
export function clearHistoryConfigGroupSortTarget(...args) { return requireHistoryCollectionDragHandler('clearHistoryConfigGroupSortTarget')(...args); }
export function clearHistoryConfigGroupSortIndicators(...args) { return requireHistoryCollectionDragHandler('clearHistoryConfigGroupSortIndicators')(...args); }
export function historyConfigGroupOrderDragEnter(...args) { return requireHistoryCollectionDragHandler('historyConfigGroupOrderDragEnter')(...args); }
export function historyConfigGroupOrderDragLeave(...args) { return requireHistoryCollectionDragHandler('historyConfigGroupOrderDragLeave')(...args); }
export function historyConfigGroupForPointerCard(...args) { return requireHistoryCollectionDragHandler('historyConfigGroupForPointerCard')(...args); }
export function historyConfigGroupPointerTargetForCard(...args) { return requireHistoryCollectionDragHandler('historyConfigGroupPointerTargetForCard')(...args); }
export function nearestHistoryConfigGroupPointerTarget(...args) { return requireHistoryCollectionDragHandler('nearestHistoryConfigGroupPointerTarget')(...args); }
export function historyConfigGroupPointerTargetFromPoint(...args) { return requireHistoryCollectionDragHandler('historyConfigGroupPointerTargetFromPoint')(...args); }
export function historyCollectionDropTargetFromPoint(...args) { return requireHistoryCollectionDragHandler('historyCollectionDropTargetFromPoint')(...args); }
export function cleanupHistoryConfigGroupPointerDrag(...args) { return requireHistoryCollectionDragHandler('cleanupHistoryConfigGroupPointerDrag')(...args); }
export function finishHistoryConfigGroupPointerDrag(...args) { return requireHistoryCollectionDragHandler('finishHistoryConfigGroupPointerDrag')(...args); }
export function dropHistoryTasksToCollectionLikePointer(...args) { return requireHistoryCollectionDragHandler('dropHistoryTasksToCollectionLikePointer')(...args); }
