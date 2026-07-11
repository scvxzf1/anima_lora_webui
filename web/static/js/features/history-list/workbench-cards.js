/**
 * History collections workbench cards + ordering helpers.
 * Implementation: workbench-collection-card / workbench-config-group-card / workbench-order.
 */
import { configureHistoryCollectionsBridge } from '../anima-app/helpers/history-collections-bridge.js?v=module-bootstrap-20260711-ir6';
import {
    createHistoryCollectionWorkbenchCard,
} from './workbench-collection-card.js?v=module-bootstrap-20260711-ir6';
import {
    createHistoryConfigGroupWorkbenchCard,
} from './workbench-config-group-card.js?v=module-bootstrap-20260711-ir6';
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
} from './workbench-order.js?v=module-bootstrap-20260711-ir6';

export {
    createHistoryCollectionWorkbenchCard,
} from './workbench-collection-card.js?v=module-bootstrap-20260711-ir6';

export {
    createHistoryConfigGroupWorkbenchCard,
} from './workbench-config-group-card.js?v=module-bootstrap-20260711-ir6';

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
} from './workbench-order.js?v=module-bootstrap-20260711-ir6';

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

