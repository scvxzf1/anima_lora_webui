import { fetchTrainingQueue } from './api.js?v=module-bootstrap-20260625-9';
import { createQueueActions } from './actions.js?v=module-bootstrap-20260625-9';
import { createQueueEnqueue } from './enqueue.js?v=module-bootstrap-20260625-9';
import { createQueueRenderer } from './render.js?v=module-bootstrap-20260625-9';
import {
    createQueueState,
    setQueueError,
    setQueueLoading,
    updateQueueStateFromPayload,
} from './state.js?v=module-bootstrap-20260625-9';

export function createQueueFeature(ctx, deps) {
    const state = createQueueState();
    let renderer = null;

    function renderTrainingQueue() {
        renderer?.renderTrainingQueue();
    }

    function updateTrainingQueueFromPayload(payload = {}) {
        updateQueueStateFromPayload(state, payload);
        renderTrainingQueue();
    }

    async function loadTrainingQueue() {
        if (location.protocol === 'file:') return;
        setQueueLoading(state);
        renderTrainingQueue();
        try {
            const payload = await fetchTrainingQueue(ctx);
            updateTrainingQueueFromPayload(payload);
        } catch (e) {
            setQueueError(state, '读取队列失败: ' + e.message);
            renderTrainingQueue();
        }
    }

    const actionSlots = {};
    const actions = new Proxy(actionSlots, {
        get(target, prop) {
            return target[prop];
        },
    });

    renderer = createQueueRenderer({
        state,
        deps,
        actions,
    });

    const queueActions = createQueueActions({
        ctx,
        state,
        deps: {
            ...deps,
            loadTrainingQueue,
        },
        updateTrainingQueueFromPayload,
        renderTrainingQueue,
        queueItemTitle: renderer.queueItemTitle,
    });
    Object.assign(actionSlots, queueActions);

    const enqueue = createQueueEnqueue({
        ctx,
        deps,
        updateTrainingQueueFromPayload,
    });

    return {
        loadTrainingQueue,
        renderTrainingQueue,
        updateTrainingQueueFromPayload,
        updateRunningQueueProgress: renderer.updateRunningQueueProgress,
        queueCurrentTrainingFromConfig: enqueue.queueCurrentTrainingFromConfig,
        enqueueTrainingFromConfig: enqueue.enqueueTrainingFromConfig,
        enqueueTrainingQueueRequest: enqueue.enqueueTrainingQueueRequest,
        enqueueTrainingQueueBatchRequest: enqueue.enqueueTrainingQueueBatchRequest,
        queueResumeTrainingFromCheckpoint: enqueue.queueResumeTrainingFromCheckpoint,
        bindQueueEvents: queueActions.bindQueueEvents,
    };
}
