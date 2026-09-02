/**
 * Active method inference for config form scoping.
 * Extracted from anima-app chunk 13.
 */
import { VARIANT_METHOD_FAMILY } from '../../config/catalog.js?v=module-bootstrap-20260902-krea2-pp-v1';
import { isTruthy } from '../anima-app/helpers/config-values.js?v=module-bootstrap-20260831-release-v1';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { getTrainingState } from '../anima-app/helpers/training-state-bridge.js?v=module-bootstrap-20260831-release-v1';
import { val } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260831-release-v1';

const configState = getConfigState();
const trainingState = getTrainingState();

function currentConfigState() {
    return configState.currentConfig || {};
}

function currentTrainingSourceState() {
    return trainingState.currentTrainingSource || {};
}

export function activeMethodKey(config = currentConfigState()) {
    const currentTrainingSource = currentTrainingSourceState();
    const inferred = inferMethodFromConfig(config);
    if (inferred) return inferred;
    if (currentTrainingSource.methods_subdir === 'methods' && currentTrainingSource.method === 'spd') {
        return 'spd';
    }
    if (currentTrainingSource.methods_subdir === 'gui-methods') {
        return VARIANT_METHOD_FAMILY[currentTrainingSource.method] || val('method-select') || 'lora';
    }
    return val('method-select') || 'lora';
}

export function inferMethodFromConfig(config) {
    const currentTrainingSource = currentTrainingSourceState();
    if (!config || typeof config !== 'object') return '';
    const moduleName = String(config.network_module || '');
    if (currentTrainingSource.methods_subdir === 'methods' && currentTrainingSource.method === 'spd') return 'spd';
    if ('dit_path' in config && 'iterations' in config && currentTrainingSource.method === 'spd') return 'spd';
    if (isTruthy(config.use_glora)) return 'glora';
    if (isTruthy(config.use_vera)) return 'vera';
    if (isTruthy(config.use_lokr)) return 'lokr';
    if (isTruthy(config.use_loha)) return 'loha';
    if (isTruthy(config.use_easycontrol) || moduleName.includes('easycontrol')) return 'easycontrol';
    if (isTruthy(config.use_ip_adapter) || moduleName.includes('ip_adapter')) return 'ip_adapter';
    if (moduleName.includes('soft_tokens')) return 'soft_tokens';
    if (isTruthy(config.add_reft) || ('reft_dim' in config && Number(config.reft_dim) > 0)) return 'reft';
    if (
        isTruthy(config.use_hydra) ||
        isTruthy(config.use_sigma_router) ||
        String(config.use_moe_style || 'false') !== 'false' ||
        moduleName.includes('chimera') ||
        moduleName.includes('hydra')
    ) {
        if (moduleName.includes('chimera') || 'content_router_source' in config) return 'chimera';
        return 'hydralora';
    }
    if (isTruthy(config.use_timestep_mask)) return 'tlora';
    if (isTruthy(config.use_ortho)) return 'ortholora';
    return '';
}
