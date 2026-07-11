import { renderItemsInChunks } from './chunked-render.js?v=module-bootstrap-20260711-ir1';

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
}) {
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
