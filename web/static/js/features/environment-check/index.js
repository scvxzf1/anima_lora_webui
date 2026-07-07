import { fetchEnvironmentCheck } from './api.js?v=module-bootstrap-20260707-93';
import { createEnvironmentCheckRenderer } from './render.js?v=module-bootstrap-20260707-93';
import { createEnvironmentCheckState } from './state.js?v=module-bootstrap-20260707-93';

export function createEnvironmentCheckFeature(ctx) {
    const state = createEnvironmentCheckState();
    const renderer = createEnvironmentCheckRenderer({ state });

    async function loadEnvironmentCheck(options = {}) {
        if (state.loading && !options.force) return;
        if (location.protocol === 'file:') {
            renderer.setStatus('静态打开没有后端 API，无法检测环境。', 'error');
            return;
        }
        state.loading = true;
        renderer.renderPending();
        try {
            const payload = await fetchEnvironmentCheck(ctx);
            renderer.renderAll(payload);
            const s = payload.summary || {};
            if (payload.ok) {
                renderer.setStatus(`检测完成：${s.checks || 0} 项，警告 ${s.warnings || 0}。`, 'ok');
            } else {
                renderer.setStatus(`检测完成：${s.errors || 0} 项错误，${s.warnings || 0} 项警告。`, 'error');
            }
        } catch (e) {
            renderer.setStatus('环境检测失败: ' + e.message, 'error');
        } finally {
            state.loading = false;
        }
    }

    function bindEnvironmentCheckEvents() {
        document.getElementById('btn-refresh-environment-check')?.addEventListener('click', () => loadEnvironmentCheck({ force: true }));
        document.getElementById('btn-copy-environment-report')?.addEventListener('click', copyReport);
    }

    function copyReport() {
        const payload = state.lastPayload;
        if (!payload) return;
        const lines = [
            'Anima LoRA 环境检测报告',
            JSON.stringify(payload.platform, null, 2),
            '',
            ...(payload.checks || []).map((c) => `[${c.level}] ${c.message}${c.hint ? ' — ' + c.hint : ''}`),
        ];
        navigator.clipboard?.writeText(lines.join('\n')).then(() => {
            renderer.setStatus('报告已复制到剪贴板。', 'ok');
        }).catch(() => {
            renderer.setStatus('复制失败，请手动选择结果。', 'error');
        });
    }

    return { loadEnvironmentCheck, bindEnvironmentCheckEvents };
}
