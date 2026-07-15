/**
 * History collections workbench cards + ordering helpers.
 * Implementation: workbench-collection-card / workbench-config-group-card / workbench-order.
 */
import { configureHistoryCollectionsBridge } from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    createHistoryCollectionWorkbenchCard,
} from './workbench-collection-card.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    createHistoryConfigGroupWorkbenchCard,
} from './workbench-config-group-card.js?v=module-bootstrap-20260714-stage-dataset5';
import {
    historyCollectionNamesForTasks,
    moveItemInList,
    collectionOrderValues,
    moveHistoryCollection,
    moveHistoryCollectionValue,
    ensureHistoryCollectionOrderValue,
    configGroupOrderValues,
    moveHistoryConfigGroup,
    groupHistoryTasksByCollection,
    historyCollectionComparator,
    historyCollectionStorageKey,
    historyCollectionByKey,
    sortedHistoryConfigGroups,
    enrichHistoryCollection,
    sortHistoryManagerGroupTasks,
    historyTaskCollectionValue,
} from './workbench-order.js?v=module-bootstrap-20260714-stage-dataset5';

export {
    createHistoryCollectionWorkbenchCard,
} from './workbench-collection-card.js?v=module-bootstrap-20260714-stage-dataset5';

export {
    createHistoryConfigGroupWorkbenchCard,
} from './workbench-config-group-card.js?v=module-bootstrap-20260714-stage-dataset5';

export {
    historyCollectionNamesForTasks,
    moveItemInList,
    collectionOrderValues,
    moveHistoryCollection,
    moveHistoryCollectionValue,
    ensureHistoryCollectionOrderValue,
    configGroupOrderValues,
    moveHistoryConfigGroup,
    groupHistoryTasksByCollection,
    historyCollectionComparator,
    historyCollectionStorageKey,
    historyCollectionByKey,
    sortedHistoryConfigGroups,
    enrichHistoryCollection,
    sortHistoryManagerGroupTasks,
    historyTaskCollectionValue,
} from './workbench-order.js?v=module-bootstrap-20260714-stage-dataset5';

configureHistoryCollectionsBridge({
    createHistoryCollectionWorkbenchCard,
    createHistoryConfigGroupWorkbenchCard,
    historyCollectionNamesForTasks,
    moveItemInList,
    collectionOrderValues,
    moveHistoryCollection,
    moveHistoryCollectionValue,
    ensureHistoryCollectionOrderValue,
    configGroupOrderValues,
    moveHistoryConfigGroup,
    groupHistoryTasksByCollection,
    historyCollectionComparator,
    historyCollectionStorageKey,
    historyCollectionByKey,
    sortedHistoryConfigGroups,
    enrichHistoryCollection,
    sortHistoryManagerGroupTasks,
    historyTaskCollectionValue,
});
