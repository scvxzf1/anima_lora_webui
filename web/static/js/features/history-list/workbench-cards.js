/**
 * History collections workbench cards + ordering helpers.
 * Implementation: workbench-collection-card / workbench-config-group-card / workbench-order.
 */
import { configureHistoryCollectionsBridge } from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260711-ir1';
import {
    createHistoryCollectionWorkbenchCard,
} from './workbench-collection-card.js?v=module-bootstrap-20260711-ir1';
import {
    createHistoryConfigGroupWorkbenchCard,
} from './workbench-config-group-card.js?v=module-bootstrap-20260711-ir1';
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
} from './workbench-order.js?v=module-bootstrap-20260711-ir1';

export {
    createHistoryCollectionWorkbenchCard,
} from './workbench-collection-card.js?v=module-bootstrap-20260711-ir1';

export {
    createHistoryConfigGroupWorkbenchCard,
} from './workbench-config-group-card.js?v=module-bootstrap-20260711-ir1';

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
} from './workbench-order.js?v=module-bootstrap-20260711-ir1';

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

