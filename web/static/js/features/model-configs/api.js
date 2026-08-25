import { api } from '../anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { cleanModelConfigItem, modelConfigRequest } from './model-config-data.js?v=module-bootstrap-20260824-zimage-v1';

function normalizeLibrary(payload = {}) {
    return {
        ok: payload.ok !== false,
        items: Array.isArray(payload.items) ? payload.items.map(cleanModelConfigItem) : [],
        defaultId: String(payload.default_id || ''),
        revision: String(payload.revision || ''),
        migrated: Boolean(payload.migrated),
        message: String(payload.message || ''),
    };
}

export async function fetchModelConfigLibrary() {
    const payload = await api('/api/settings/model-configs');
    if (!payload?.ok) throw new Error(payload?.error || '读取全局模型配置失败');
    return normalizeLibrary(payload);
}

export async function saveModelConfigLibrary({ items, defaultId, revision }) {
    const payload = await api('/api/settings/model-configs', {
        method: 'PUT',
        body: JSON.stringify(modelConfigRequest(items, defaultId, revision)),
    });
    if (!payload?.ok) {
        const error = new Error(payload?.error || '保存全局模型配置失败');
        error.status = Number(payload?.status || payload?.status_code || 0);
        throw error;
    }
    return normalizeLibrary(payload);
}
