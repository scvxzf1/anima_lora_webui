import { captioningApi, escapeAttribute, escapeHtml, feedback, jsonOptions, panelShell } from './shared.js?v=dragon-ui-20260829v11';

export function renderRetryPanel(state) {
    const detail = state.workspaceData.retryJob;
    const failed = detail?.results?.filter((item) => item.state === 'failed') || [];
    const filter = state.workspaceData.retryFilter || '';
    const visible = failed.filter((item) => !filter || (item.failure_kind || 'request') === filter);
    const loading = state.workspaceData.retryLoading === true;
    return panelShell('RETRY', '失败任务重试', '', `<div class="dragon-caption-list-toolbar"><select class="dragon-select" data-retry-job aria-label="失败任务" ${loading ? 'disabled' : ''}><option value="">选择任务</option>${(state.jobs || []).map((job) => `<option value="${job.id}" ${job.id === detail?.id ? 'selected' : ''}>${escapeHtml(job.directory)} · 失败 ${job.failed || 0}</option>`).join('')}</select><select class="dragon-select" data-retry-filter aria-label="失败类型" ${loading ? 'disabled' : ''}><option value="" ${filter ? '' : 'selected'}>全部失败</option><option value="request" ${filter === 'request' ? 'selected' : ''}>请求失败</option><option value="parse" ${filter === 'parse' ? 'selected' : ''}>解析失败</option></select><button class="dragon-btn dragon-btn-secondary" type="button" data-retry-load ${loading ? 'disabled' : ''}>${loading ? '加载中…' : '加载'}</button><button class="dragon-btn dragon-btn-primary" type="button" data-retry-all ${failed.length && !loading ? '' : 'disabled'}>重试全部失败项（${failed.length}）</button><span class="dragon-caption-list-summary">当前显示 ${visible.length}/${failed.length} · 筛选不改变批量范围</span></div>
        <div class="dragon-caption-result-list" data-retry-list aria-busy="${loading}">${visible.map((item) => `<article data-retry-entry data-kind="${item.failure_kind || 'request'}"><header><strong title="${escapeAttribute(item.name)}">${escapeHtml(item.name)}</strong><span>${item.failure_kind === 'parse' ? '解析失败' : '请求失败'}</span></header><p>${escapeHtml(item.error || '未知错误')}</p><div class="dragon-caption-form-actions"><button class="dragon-btn dragon-btn-secondary" type="button" data-retry-item="${escapeAttribute(item.id)}">重试此图</button><button class="dragon-btn dragon-btn-secondary" type="button" data-review-item="${escapeAttribute(item.id)}">打开审阅</button></div></article>`).join('') || retryEmptyState({detail, failed, filter, loading, error: state.workspaceData.retryError})}</div>`);
}

export function bindRetryPanel(root, state) {
    const load = async () => {
        const id = root.querySelector('[data-retry-job]').value;
        if (!id) return feedback(root, '请选择任务', 'error');
        if (state.workspaceData.retryLoading) return;
        state.workspaceData.retryLoading = true; state.workspaceData.retryError = ''; state.suiteRender();
        try { const payload = await captioningApi(`/jobs/${encodeURIComponent(id)}`); state.workspaceData.retryJob = payload.job; }
        catch (error) { state.workspaceData.retryError = error.message; }
        finally { state.workspaceData.retryLoading = false; state.suiteRender(); }
    };
    root.querySelector('[data-retry-load]')?.addEventListener('click', load);
    root.querySelector('[data-retry-filter]')?.addEventListener('change', (event) => { state.workspaceData.retryFilter = event.target.value; state.suiteRender(); });
    root.querySelector('[data-retry-all]')?.addEventListener('click', async (event) => {
        if (!state.workspaceData.retryJob) return feedback(root, '请先加载任务', 'error');
        const button = event.currentTarget; button.disabled = true; button.textContent = '正在入队…';
        try { await captioningApi(`/jobs/${encodeURIComponent(state.workspaceData.retryJob.id)}/retry-failed`, jsonOptions('POST', {})); state.workspaceData.retryJob.results.forEach((item) => { if (item.state === 'failed') item.state = 'queued'; }); state.suiteRender(); }
        catch (error) { state.workspaceData.retryError = error.message; state.suiteRender(); }
    });
    root.querySelectorAll('[data-retry-item]').forEach((button) => button.addEventListener('click', async () => {
        if (button.disabled) return; button.disabled = true; button.textContent = '重试中…';
        try { await captioningApi(`/jobs/${encodeURIComponent(state.workspaceData.retryJob.id)}/items/${encodeURIComponent(button.dataset.retryItem)}/retry`, jsonOptions('POST', {})); const item = state.workspaceData.retryJob.results.find((entry) => String(entry.id) === button.dataset.retryItem); if (item) item.state = 'queued'; state.suiteRender(); }
        catch (error) { button.disabled = false; button.textContent = '重试此图'; feedback(root, error.message, 'error'); }
    }));
    root.querySelectorAll('[data-review-item]').forEach((button) => button.addEventListener('click', () => { state.selectedJobId = state.workspaceData.retryJob.id; state.selectedItemId = button.dataset.reviewItem; state.activePanel = 'workbench'; state.suiteRender(); }));
}

function retryEmptyState({detail, failed, filter, loading, error}) {
    if (loading) return '<div class="dragon-empty-state"><p>正在加载任务失败项…</p></div>';
    if (error) return `<div class="dragon-empty-state" data-tone="error"><p>加载或重试失败：${escapeHtml(error)}</p></div>`;
    if (!detail) return '<div class="dragon-empty-state"><p>选择任务后查看失败项。</p></div>';
    if (failed.length && filter) return '<div class="dragon-empty-state"><p>当前筛选没有匹配的失败项。</p></div>';
    return '<div class="dragon-empty-state"><p>该任务没有失败项。</p></div>';
}
