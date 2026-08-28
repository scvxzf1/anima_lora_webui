import { captioningApi, escapeAttribute, escapeHtml, feedback, jsonOptions, panelShell } from './shared.js?v=dragon-ui-20260829v12';

export function renderLogsPanel(state) {
    const page = state.workspaceData.logsPage || {items:[...(state.workspace?.label_logs || [])].reverse().slice(0, 50),page:1,page_size:50,total:(state.workspace?.label_logs || []).length};
    const logs = page.items;
    const query = state.workspaceData.logQuery || '';
    const status = state.workspaceData.logStatus || '';
    const loading = state.workspaceData.logsLoading === true;
    const counts = Object.fromEntries(['success','failed','running','info'].map((key) => [key, logs.filter((entry) => entry.status === key).length]));
    return panelShell('LOGS', '打标日志', `<button class="dragon-btn dragon-btn-secondary" type="button" data-log-refresh ${loading ? 'disabled' : ''}>${loading ? '加载中…' : '刷新'}</button><button class="dragon-btn dragon-btn-secondary dragon-caption-danger-action" type="button" data-log-clear ${loading || !page.total ? 'disabled' : ''}>清空全部日志</button>`, `<div class="dragon-caption-list-toolbar dragon-caption-logs-toolbar"><input class="dragon-input" data-log-search aria-label="搜索日志" value="${escapeAttribute(query)}" placeholder="搜索事件、状态或内容" ${loading ? 'disabled' : ''}><select class="dragon-select" data-log-status aria-label="日志状态" ${loading ? 'disabled' : ''}><option value="" ${status === '' ? 'selected' : ''}>全部状态</option><option value="success" ${status === 'success' ? 'selected' : ''}>成功</option><option value="failed" ${status === 'failed' ? 'selected' : ''}>失败</option><option value="running" ${status === 'running' ? 'selected' : ''}>运行中</option><option value="info" ${status === 'info' ? 'selected' : ''}>信息</option></select><div class="dragon-caption-log-summary" aria-label="当前页状态统计"><span data-tone="success">成功 ${counts.success}</span><span data-tone="error">失败 ${counts.failed}</span><span>运行中 ${counts.running}</span><span>信息 ${counts.info}</span></div></div>${state.workspaceData.logsError ? `<div class="dragon-caption-inline-status" data-tone="error">${escapeHtml(state.workspaceData.logsError)} <button class="dragon-btn dragon-btn-secondary" type="button" data-log-retry-error>重新加载</button></div>` : ''}<div class="dragon-caption-log-list" aria-busy="${loading}">${logs.map((entry, index) => `<article data-log-entry data-status="${entry.status}"><header><strong>${escapeHtml(entry.source)}</strong><span>${statusLabel(entry.status)}${Number.isFinite(entry.duration_ms) ? ` · ${entry.duration_ms}ms` : ''}</span></header><p>${escapeHtml(entry.message)}</p><details><summary>${escapeHtml(formatTimestamp(entry.created_at))}</summary><div class="dragon-caption-log-detail-actions"><button class="dragon-btn dragon-btn-secondary" type="button" data-log-copy="${index}">复制详情</button></div><pre>${escapeHtml(entry.detail)}</pre></details></article>`).join('') || `<div class="dragon-empty-state"><p>${loading ? '正在加载日志…' : query || status ? '当前筛选没有日志。' : '暂无打标日志。'}</p></div>`}</div><div class="dragon-caption-pagination"><button class="dragon-btn dragon-btn-secondary" type="button" data-log-page="${page.page - 1}" ${page.page <= 1 || loading ? 'disabled' : ''}>上一页</button><span>第 ${page.page} 页 · 共 ${page.total} 条</span><button class="dragon-btn dragon-btn-secondary" type="button" data-log-page="${page.page + 1}" ${page.page * page.page_size >= page.total || loading ? 'disabled' : ''}>下一页</button></div>`);
}

export function bindLogsPanel(root, state) {
    const load = async (page = 1) => {
        const requestId = (state.workspaceData.logsRequestId || 0) + 1;
        state.workspaceData.logsRequestId = requestId;
        const query = state.workspaceData.logQuery || '';
        const status = state.workspaceData.logStatus || '';
        state.workspaceData.logsLoading = true; state.workspaceData.logsError = ''; state.suiteRender();
        try {
            const payload = await captioningApi(`/workspace/logs?page=${page}&page_size=50&query=${encodeURIComponent(query)}&status=${encodeURIComponent(status)}`);
            if (!state.active || state.activePanel !== 'logs' || requestId !== state.workspaceData.logsRequestId) return;
            state.workspaceData.logsPage = payload;
        } catch (error) { state.workspaceData.logsError = error.message; }
        finally { if (requestId === state.workspaceData.logsRequestId) { state.workspaceData.logsLoading = false; state.suiteRender(); } }
    };
    root.querySelector('[data-log-search]')?.addEventListener('input', (event) => {
        state.workspaceData.logQuery = event.target.value;
        if (state.workspaceData.logSearchTimer) window.clearTimeout(state.workspaceData.logSearchTimer);
        state.workspaceData.logSearchTimer = window.setTimeout(() => load(1), 250);
    });
    root.querySelector('[data-log-status]')?.addEventListener('change', (event) => { state.workspaceData.logStatus = event.target.value; load(1); });
    root.querySelector('[data-log-refresh]')?.addEventListener('click', () => load(state.workspaceData.logsPage?.page || 1));
    root.querySelector('[data-log-retry-error]')?.addEventListener('click', () => load(state.workspaceData.logsPage?.page || 1));
    root.querySelectorAll('[data-log-page]').forEach((button) => button.addEventListener('click', () => load(Number(button.dataset.logPage))));
    root.querySelectorAll('[data-log-copy]').forEach((button) => button.addEventListener('click', async () => { const entry = (state.workspaceData.logsPage?.items || [])[Number(button.dataset.logCopy)]; if (!entry) return; try { await navigator.clipboard.writeText(String(entry.detail || entry.message || '')); feedback(root, '日志详情已复制', 'success'); } catch { feedback(root, '复制失败，请手动选择详情', 'error'); } }));
    root.querySelector('[data-log-clear]')?.addEventListener('click', async () => {
        if (!window.confirm('确认清空全部打标日志？此操作不可撤销。')) return;
        try { state.workspaceData.logsLoading = true; state.suiteRender(); state.workspace = await captioningApi('/workspace/logs', {method:'DELETE'}); state.workspaceData.logsPage = null; state.workspaceData.logsLoading = false; state.suiteRender(); }
        catch (error) { state.workspaceData.logsLoading = false; state.workspaceData.logsError = error.message; state.suiteRender(); }
    });
}

function statusLabel(status) { return ({success:'成功',failed:'失败',running:'运行中',info:'信息'}[status] || escapeHtml(status)); }
function formatTimestamp(value) { if (!value) return '详情'; const date = new Date(value); return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('zh-CN', {hour12:false}); }
