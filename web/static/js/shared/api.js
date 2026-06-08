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
        if (!data || typeof data !== 'object') {
            data = { ok: res.ok, value: data };
        }
        if (!res.ok && data && !Object.prototype.hasOwnProperty.call(data, 'ok')) {
            data.ok = false;
        }
        if (!res.ok) {
            if (!Object.prototype.hasOwnProperty.call(data, 'status')) data.status = res.status;
            if (!Object.prototype.hasOwnProperty.call(data, 'status_code')) data.status_code = res.status;
            if (!data.error) data.error = text || `${res.status}: ${res.statusText || 'HTTP error'}`;
        }
        return data;
    };
}
