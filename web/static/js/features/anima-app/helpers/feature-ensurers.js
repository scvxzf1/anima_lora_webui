import { createEnvironmentCheckFeature } from '../../environment-check/index.js?v=module-bootstrap-20260809-nf4-v2';
import { createPreviewFeature } from '../../preview/index.js?v=module-bootstrap-20260809-nf4-v2';
import { createQueueFeature } from '../../queue/index.js?v=module-bootstrap-20260809-nf4-v2';
import { createWeightAnalysisFeature } from '../../weight-analysis/index.js?v=module-bootstrap-20260809-nf4-v2';

let previewFeatureConfig = null;
let queueFeatureConfig = null;

export function configurePreviewFeatureEnsurer(ctx, holder, deps) {
    previewFeatureConfig = { ctx, holder, deps };
}

export function ensurePreviewFeature() {
    if (!previewFeatureConfig) {
        throw new Error('preview feature ensurer is not configured');
    }
    const { ctx, holder, deps } = previewFeatureConfig;
    if (holder.previewFeature) return holder.previewFeature;
    holder.previewFeature = createPreviewFeature(ctx, deps);
    return holder.previewFeature;
}

export function configureQueueFeatureEnsurer(ctx, holder, deps) {
    queueFeatureConfig = { ctx, holder, deps };
}

export function ensureQueueFeature() {
    if (!queueFeatureConfig) {
        throw new Error('queue feature ensurer is not configured');
    }
    const { ctx, holder, deps } = queueFeatureConfig;
    if (holder.queueFeature) return holder.queueFeature;
    holder.queueFeature = createQueueFeature(ctx, deps);
    return holder.queueFeature;
}

export function ensureWeightAnalysisFeature(ctx, holder) {
    if (holder.weightAnalysisFeature) return holder.weightAnalysisFeature;
    holder.weightAnalysisFeature = createWeightAnalysisFeature(ctx);
    return holder.weightAnalysisFeature;
}

export function ensureEnvironmentCheckFeature(ctx, holder) {
    if (holder.environmentCheckFeature) return holder.environmentCheckFeature;
    holder.environmentCheckFeature = createEnvironmentCheckFeature(ctx);
    return holder.environmentCheckFeature;
}
