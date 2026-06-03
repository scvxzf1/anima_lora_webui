export function createApiClient(fetchImpl = window.fetch.bind(window)) {
    return async function api(url, opts = {}) {
        const headers = { 'Content-Type': 'application/json' };
        const res = await fetchImpl(url, { headers, ...opts });
        const text = await res.text();
        let data;
        try {
            data = text ? JSON.parse(text) : {};
        } catch {
            data = { ok: false, error: text || `HTTP ${res.status}` };
        }
        if (!res.ok && data && !Object.prototype.hasOwnProperty.call(data, 'ok')) {
            data.ok = false;
        }
        return data;
    };
}
