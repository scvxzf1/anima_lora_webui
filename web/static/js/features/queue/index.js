import { fetchTrainingQueue } from './api.js?v=module-bootstrap-20260831-release-v1';
import { createQueueActions } from './actions.js?v=module-bootstrap-20260831-release-v1';
import { createQueueEnqueue } from './enqueue.js?v=module-bootstrap-20260831-release-v1';
import { createQueueRenderer } from './render.js?v=module-bootstrap-20260831-release-v1';
import {
    createQueueState,
    queueRenderSignature,
    setQueueError,
    setQueueLoading,
    updateQueueStateFromPayload,
} from './state.js?v=module-bootstrap-20260831-release-v1';

export function createQueueFeature(ctx, deps) {
    const state = createQueueState();
    let renderer = null;
    let lastRenderSignature = '';

    function renderTrainingQueue(options = {}) {
        const force = options.force === true;
        const signature = queueRenderSignature(state);
        if (!force && signature && signature === lastRenderSignature) return;
        lastRenderSignature = signature;
        renderer?.renderTrainingQueue();
    }

    function updateTrainingQueueFromPayload(payload = {}, options = {}) {
        updateQueueStateFromPayload(state, payload);
        renderTrainingQueue(options);
    }

    async function loadTrainingQueue() {
        if (location.protocol === 'file:') return;
        setQueueLoading(state);
        renderTrainingQueue({ force: true });
        try {
            const payload = await fetchTrainingQueue(ctx);
            updateTrainingQueueFromPayload(payload, { force: true });
        } catch (e) {
            setQueueError(state, '读取队列失败: ' + e.message);
            renderTrainingQueue({ force: true });
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
