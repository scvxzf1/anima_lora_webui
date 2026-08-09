import { renderItemsInChunks } from './chunked-render.js?v=module-bootstrap-20260809-nf4-v2';

/**
 * Fill config-group and collection card lists with cancelable chunked append.
 */
export function fillHistoryWorkbenchCardLists({
    configList,
    collectionList,
    visibleConfigGroups,
    visibleCollections,
    splitCollections,
    configGroups,
    selectedCollection,
    selectedTasksLength,
    allCollections,
    createHistoryConfigGroupWorkbenchCard,
    createHistoryCollectionWorkbenchCard,
    emptyConfigMessage,
    signal,
    configOnly = false,
    collectionsOnly = false,
}) {
    if (!collectionsOnly) {
        if (!visibleConfigGroups.length) {
            const empty = document.createElement('div');
            empty.className = 'history-current-group-empty';
            empty.textContent = emptyConfigMessage;
            configList.appendChild(empty);
        } else {
            renderItemsInChunks(
                configList,
                visibleConfigGroups,
                (group) => createHistoryConfigGroupWorkbenchCard(group, splitCollections, {
                    groups: configGroups,
                    collection: selectedCollection,
                }),
                { signal },
            );
        }
    }

    if (!configOnly && collectionList) {
        renderItemsInChunks(
            collectionList,
            visibleCollections,
            (collection) => createHistoryCollectionWorkbenchCard(
                collection,
                selectedTasksLength,
                allCollections,
            ),
            { signal },
        );
    }
}
