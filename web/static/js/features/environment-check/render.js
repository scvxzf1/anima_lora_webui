export function createEnvironmentCheckRenderer({ state }) {
    function badge(level) {
        const map = { ok: '✓', warning: '!', error: '×' };
        return `<span class="environment-check-badge ${level}">${map[level] || '?'}</span>`;
    }

    function renderSummary(payload) {
        const panel = document.getElementById('environment-check-summary-panel');
        if (!panel) return;
        const s = payload?.summary || {};
        panel.innerHTML = `
            <div class="environment-check-summary-stat"><strong>${s.errors ?? 0}</strong><small>错误</small></div>
            <div class="environment-check-summary-stat"><strong>${s.warnings ?? 0}</strong><small>警告</small></div>
            <div class="environment-check-summary-stat"><strong>${s.checks ?? 0}</strong><small>检查项</small></div>
        `;
    }

    function renderPlatform(payload) {
        const el = document.getElementById('environment-check-platform-meta');
        if (!el || !payload?.platform) return;
        const p = payload.platform;
        el.textContent = [
            `系统: ${p.system} (${p.platform})`,
            `Python: ${p.python_version}`,
            `Web 解释器: ${p.web_executable}`,
            `项目 venv: ${p.venv_python || '—'}`,
            `CUDA 轨道: ${p.cuda_track}`,
        ].join(' · ');
    }

    function renderGroups(payload) {
        const root = document.getElementById('environment-check-groups');
        if (!root) return;
        const groups = payload?.groups || [];
        if (!groups.length) {
            root.innerHTML = '<div class="environment-check-group"><div class="environment-check-results">暂无结果</div></div>';
            return;
        }
        root.innerHTML = groups.map((group) => {
            const rows = (group.checks || []).map((item) => `
                <div class="environment-check-item">
                    ${badge(item.level)}
                    <div>
                        <div class="environment-check-message">${escapeHtml(item.message || '')}</div>
                        ${item.detail ? `<div class="environment-check-detail">${escapeHtml(item.detail)}</div>` : ''}
                        ${item.hint ? `<div class="environment-check-hint">${escapeHtml(item.hint)}</div>` : ''}
                    </div>
                </div>
            `).join('');
            return `<section class="environment-check-group"><div class="environment-check-group-head">${escapeHtml(group.title || group.key)}</div><div class="environment-check-results">${rows}</div></section>`;
        }).join('');
    }

    function renderAll(payload) {
        state.lastPayload = payload;
        renderSummary(payload);
        renderPlatform(payload);
        renderGroups(payload);
    }

    function renderPending() {
        const status = document.getElementById('environment-check-status');
        if (status) {
            status.className = 'environment-check-status';
            status.textContent = '正在检测环境…';
        }
    }

    function setStatus(message, tone) {
        const status = document.getElementById('environment-check-status');
        if (!status) return;
        status.className = `environment-check-status ${tone || ''}`.trim();
        status.textContent = message || '';
    }

    return { renderAll, renderPending, setStatus };
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
