import { captioningApi, escapeAttribute, escapeHtml, feedback, groupOptions, jsonOptions, panelShell, selectedGroup } from './shared.js?v=dragon-ui-20260829v11';

export function renderExportPanel(state) {
    const jobs = state.jobs || [];
    const draft = state.workspaceData.exportDraft || {};
    const detail = state.workspaceData.exportJobDetail;
    const eligible = detail?.results?.filter((item) => ['ready', 'committed'].includes(item.state)) || [];
    const skipped = detail ? detail.results.length - eligible.length : 0;
    const busy = state.workspaceData.exportBusy === true;
    const canExport = jobs.length && detail && eligible.length && !busy;
    return panelShell('EXPORT', 'Caption 导出', '', `<form class="dragon-caption-tool-form" data-export-form>
        <div class="dragon-caption-form-grid"><label><span>打标任务</span><select class="dragon-select" name="job_id" ${jobs.length && !busy ? '' : 'disabled'}><option value="">${jobs.length ? '选择任务' : '暂无可导出任务'}</option>${jobs.map((job) => `<option value="${escapeAttribute(job.id)}" ${job.id === draft.jobId ? 'selected' : ''}>${escapeHtml(job.directory)} · ${escapeHtml(job.model)}</option>`).join('')}</select></label><label><span>目标目录组</span><select class="dragon-select" name="group_id" ${busy ? 'disabled' : ''}><option value="">使用任务目录</option>${groupOptions(state, draft.groupId)}</select></label><label class="dragon-caption-span-2"><span>手动目标目录（优先于目录组）</span><input class="dragon-input" name="manual_directory" placeholder="/data/export" value="${escapeAttribute(draft.manualDirectory || '')}" ${busy ? 'disabled' : ''}></label></div>
        <fieldset class="dragon-caption-check-row"><legend>输出类型</legend><label><input type="checkbox" name="types" value="tag" checked>Tag</label><label><input type="checkbox" name="types" value="mixed_70tag_30nl" checked>70% Tag + 30% NL</label><label><input type="checkbox" name="types" value="pure_nl" checked>纯自然语言</label></fieldset>
        <div class="dragon-caption-export-scope" data-export-scope>${detail ? `<strong>${eligible.length} 项可导出</strong><span>${skipped} 项因状态不符将跳过</span><span>目标：${escapeHtml(resolveTarget(state, detail, draft))}</span>` : '<span>选择任务后先检查导出范围。</span>'}</div>
        <div class="dragon-caption-form-actions"><button class="dragon-btn dragon-btn-secondary" type="button" data-export-inspect ${jobs.length && !busy ? '' : 'disabled'}>${busy && !detail ? '检查中…' : '检查导出范围'}</button><button class="dragon-btn dragon-btn-primary" type="submit" data-export-action="captions-json" ${canExport ? '' : 'disabled'}>写入 captions.json</button><button class="dragon-btn dragon-btn-secondary" type="submit" data-export-action="image-txt" ${canExport ? '' : 'disabled'}>写入同名 .txt</button><button class="dragon-btn dragon-btn-secondary" type="submit" data-export-action="download" ${canExport ? '' : 'disabled'}>下载 captions.json</button></div>
    </form><div class="dragon-caption-export-result" data-export-summary>${state.workspaceData.exportSummary ? escapeHtml(state.workspaceData.exportSummary) : ''}</div>`);
}

export function bindExportPanel(root, state) {
    const form = root.querySelector('[data-export-form]');
    const collectDraft = () => ({jobId: form.elements.job_id.value, groupId: form.elements.group_id.value, manualDirectory: form.elements.manual_directory.value.trim()});
    form?.querySelector('[name="job_id"]')?.addEventListener('change', () => { state.workspaceData.exportDraft = collectDraft(); state.workspaceData.exportJobDetail = null; state.workspaceData.exportSummary = ''; state.suiteRender(); });
    form?.querySelector('[name="group_id"]')?.addEventListener('change', () => { state.workspaceData.exportDraft = collectDraft(); state.suiteRender(); });
    form?.querySelector('[name="manual_directory"]')?.addEventListener('input', () => { state.workspaceData.exportDraft = collectDraft(); const scope = root.querySelector('[data-export-scope]'); if (scope && state.workspaceData.exportJobDetail) scope.lastElementChild.textContent = `目标：${resolveTarget(state, state.workspaceData.exportJobDetail, state.workspaceData.exportDraft)}`; });
    root.querySelector('[data-export-inspect]')?.addEventListener('click', async () => {
        const draft = collectDraft(); if (!draft.jobId) return feedback(root, '请选择打标任务', 'error');
        state.workspaceData.exportBusy = true; state.workspaceData.exportDraft = draft; state.workspaceData.exportJobDetail = null; state.suiteRender();
        try { const payload = await captioningApi(`/jobs/${encodeURIComponent(draft.jobId)}`); state.workspaceData.exportJobDetail = payload.job; }
        catch (error) { state.workspaceData.exportSummary = `检查失败：${error.message}`; }
        finally { state.workspaceData.exportBusy = false; state.suiteRender(); }
    });
    root.querySelector('[data-export-form]')?.addEventListener('submit', async (event) => {
        event.preventDefault(); const button = event.submitter; if (!button?.dataset.exportAction) return; const originalLabel = button.textContent; button.disabled = true; button.textContent = '处理中…';
        try {
            const form = event.currentTarget; const action = event.submitter.dataset.exportAction; const types = [...form.querySelectorAll('[name="types"]:checked')].map((item) => item.value);
            if (!types.length) throw new Error('至少选择一种输出类型');
            const detail = state.workspaceData.exportJobDetail; if (!detail || String(detail.id) !== form.elements.job_id.value) throw new Error('任务范围已变化，请重新检查');
            const group = selectedGroup(state, form.elements.group_id.value);
            const captions = {};
            detail.results.filter((item) => ['ready', 'committed'].includes(item.state)).forEach((item) => {
                const variants = item.caption_variants || {}; const values = types.map((type) => variants[type]).filter(Boolean);
                captions[item.name] = values.length ? values : [item.proposed_caption];
            });
            if (!Object.keys(captions).length) throw new Error('当前任务没有可导出的 Caption');
            if (action === 'download') {
                const payload = await captioningApi('/workspace/download-captions', jsonOptions('POST', {captions}));
                downloadText(payload.filename, payload.content); state.workspaceData.exportSummary = `已下载 ${Object.keys(captions).length} 项，跳过 ${detail.results.length - Object.keys(captions).length} 项`; state.suiteRender(); return;
            }
            const result = await captioningApi('/workspace/export-captions', jsonOptions('POST', {directory: form.elements.manual_directory.value.trim() || group?.path || detail.directory, captions, save_format: action}));
            state.workspaceData.exportSummary = `已写入 ${result.count} 项到 ${result.captions_path || result.target_path}，跳过 ${detail.results.length - Object.keys(captions).length} 项`; state.suiteRender();
        } catch (error) { feedback(root, error.message, 'error'); }
        finally { button.disabled = false; button.textContent = originalLabel; }
    });
}

function resolveTarget(state, detail, draft) { return draft.manualDirectory || selectedGroup(state, draft.groupId)?.path || detail?.directory || '未确定'; }

function downloadText(filename, content) { const blob = new Blob([content], {type:'application/json'}); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = filename; link.click(); URL.revokeObjectURL(link.href); }
