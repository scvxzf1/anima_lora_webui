import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';

const request = createApiClient();

export async function captioningApi(path, options = {}) {
    const payload = await request(`/api/captioning${path}`, options);
    if (payload?.ok === false) throw new Error(payload.error || '打标服务请求失败');
    return payload;
}

export function jsonOptions(method, body) {
    return { method, body: JSON.stringify(body || {}) };
}
