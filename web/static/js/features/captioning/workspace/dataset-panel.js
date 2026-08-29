import { captioningApi, escapeAttribute, escapeHtml, feedback, fileToDataUrl, groupOptions, jsonOptions, panelShell, selectedGroup, uid } from './shared.js?v=dragon-ui-20260829v12';

export function renderDatasetPanel(state) {
    state.workspaceData.datasetResults ||= state.workspace?.dataset_results || [];
    const results = state.workspaceData.datasetResults;
    const running = state.workspaceData.datasetRunning === true;
    const draft = state.workspaceData.datasetDraft || {};
    const compatibleSchedules = geminiSchedules(state);
    const canGenerate = compatibleSchedules.length > 0;
    return panelShell('DATASET', '数据集生成', '', `<form class="dragon-caption-tool-form" data-dataset-form>
        <div class="dragon-caption-form-grid dragon-caption-dataset-fields"><label class="dragon-caption-span-2"><span>参考图片</span><input class="dragon-input" type="file" name="images" accept="image/*" multiple required ${running ? 'disabled' : ''}></label><label><span>调度组</span><select class="dragon-select" name="schedule_id" ${running ? 'disabled' : ''}>${datasetScheduleOptions(state, compatibleSchedules, draft.scheduleId)}</select></label><label><span>并发</span><input class="dragon-input" type="number" name="concurrency" min="1" max="12" value="${escapeAttribute(String(draft.concurrency ?? 2))}" ${running ? 'disabled' : ''}></label><label><span>保存目录组</span><select class="dragon-select" name="group_id" ${running ? 'disabled' : ''}><option value="">稍后选择</option>${groupOptions(state, draft.groupId)}</select></label></div>
        <label><span>Prompt（每行一个任务）</span><textarea class="dragon-textarea" name="prompts" rows="5" required ${running ? 'disabled' : ''}>${escapeHtml(draft.prompts || '')}</textarea></label>
        <div class="dragon-caption-dataset-plan" data-dataset-plan>选择参考图并输入 Prompt 后显示调用规模。</div>
        <div class="dragon-caption-form-actions"><button class="dragon-btn dragon-btn-primary" type="submit" ${running || !canGenerate ? 'disabled' : ''}>开始生成</button><button class="dragon-btn dragon-btn-secondary" type="button" data-dataset-retry ${results.some((item) => ['failed', 'stopped'].includes(item.status)) && !running && canGenerate ? '' : 'disabled'}>重跑失败 / 已停止项</button><button class="dragon-btn dragon-btn-secondary" type="button" data-dataset-stop ${running ? '' : 'disabled'}>${state.workspaceData.datasetStopped && running ? '正在停止排队项…' : '停止等待中任务'}</button><button class="dragon-btn dragon-btn-secondary" type="button" data-dataset-clear ${running ? 'disabled' : ''}>清空结果</button></div>
    </form><div class="dragon-caption-dataset-progress"><progress max="${results.length || 1}" value="${results.filter((item) => ['success', 'failed', 'stopped'].includes(item.status)).length}"></progress><span>${results.filter((item) => item.status === 'success').length} 成功 / ${results.filter((item) => item.status === 'failed').length} 失败 / ${results.filter((item) => item.status === 'stopped').length} 已停止 / ${results.length} 总计</span></div>
    <div class="dragon-caption-dataset-grid">${results.map(renderResult).join('') || '<div class="dragon-empty-state"><p>选择参考图并输入 Prompt。</p></div>'}</div>
    <form class="dragon-caption-save-generated" data-dataset-save><label><span>目标目录组</span><select class="dragon-select" name="group_id">${groupOptions(state)}</select></label><label><input type="checkbox" name="parts" value="generated" checked>生成图</label><label><input type="checkbox" name="parts" value="source">参考图</label><label><input type="checkbox" name="prompt_line" checked>文件名追加 Prompt 行号</label><label><input type="checkbox" name="timestamp">追加时间戳</label><button class="dragon-btn dragon-btn-primary" type="submit" ${results.some((item) => item.status === 'success') ? '' : 'disabled'}>保存图片</button></form>`);
}

export function bindDatasetPanel(root, state) {
    const datasetForm = root.querySelector('[data-dataset-form]');
    const updatePlan = () => {
        const images = datasetForm?.elements.images.files.length || 0;
        const prompts = datasetForm?.elements.prompts.value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).length || 0;
        const total = images * prompts; const plan = root.querySelector('[data-dataset-plan]');
        if (plan) { plan.textContent = total ? `${images} 张参考图 × ${prompts} 条 Prompt = ${total} 次 API 调用` : '选择参考图并输入 Prompt 后显示调用规模。'; plan.dataset.tone = total > 500 ? 'error' : total > 100 ? 'warning' : 'info'; }
    };
    datasetForm?.elements.images.addEventListener('change', updatePlan);
    datasetForm?.elements.prompts.addEventListener('input', updatePlan);
    root.querySelector('[data-dataset-form]')?.addEventListener('submit', async (event) => {
        event.preventDefault(); const form = event.currentTarget;
        try {
            const images = await Promise.all([...form.elements.images.files].map(async (file) => ({name: file.name, dataUrl: await fileToDataUrl(file)})));
            const prompts = form.elements.prompts.value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
            const total = images.length * prompts.length;
            if (!total) throw new Error('请选择参考图并输入 Prompt');
            if (total > 500) throw new Error('单批次最多生成 500 张图片');
            const schedule = state.routing.schedules.find((item) => item.id === form.elements.schedule_id.value);
            const channels = new Map(state.routing.channels.map((item) => [item.id, item]));
            if (!schedule?.steps.some((step) => step.enabled && channels.get(step.channel_id)?.protocol === 'gemini')) throw new Error('所选调度组没有启用 Gemini 渠道，无法生成图片');
            state.workspaceData.datasetResults = images.flatMap((image) => prompts.map((prompt, index) => ({id: uid('generated'), source_name: image.name, source_data_url: image.dataUrl, prompt, prompt_line_index: index, status: 'queued', generated_data_url: '', mime_type: '', error: ''})));
            state.workspaceData.datasetDraft = {scheduleId: form.elements.schedule_id.value, concurrency: Number(form.elements.concurrency.value || 1), groupId: form.elements.group_id.value, prompts: form.elements.prompts.value};
            state.workspaceData.datasetScheduleId = form.elements.schedule_id.value;
            state.workspaceData.datasetConcurrency = Number(form.elements.concurrency.value || 1);
            await runQueue(root, state, state.workspaceData.datasetResults);
        } catch (error) { feedback(root, error.message, 'error'); }
    });
    root.querySelector('[data-dataset-stop]')?.addEventListener('click', () => { state.workspaceData.datasetStopped = true; state.suiteRender(); });
    root.querySelector('[data-dataset-retry]')?.addEventListener('click', () => {
        if (!state.workspaceData.datasetRunning) runQueue(root, state, state.workspaceData.datasetResults.filter((item) => ['failed', 'stopped'].includes(item.status)));
    });
    root.querySelectorAll('[data-dataset-retry-item]').forEach((button) => button.addEventListener('click', () => { const item = state.workspaceData.datasetResults.find((entry) => entry.id === button.dataset.datasetRetryItem); if (item && !state.workspaceData.datasetRunning) runQueue(root, state, [item]); }));
    root.querySelector('[data-dataset-clear]')?.addEventListener('click', () => { if (!state.workspaceData.datasetResults.length || window.confirm(`清空 ${state.workspaceData.datasetResults.length} 条生成结果？`)) { state.workspaceData.datasetResults = []; state.suiteRender(); } });
    root.querySelector('[data-dataset-save]')?.addEventListener('submit', async (event) => {
        event.preventDefault(); const form = event.currentTarget; const button = event.submitter; const group = selectedGroup(state, form.elements.group_id.value);
        if (!group) return feedback(root, '请选择目标目录组', 'error');
        button.disabled = true; button.textContent = '保存中…';
        try {
            const parts = [...form.querySelectorAll('[name="parts"]:checked')].map((item) => item.value);
            if (!parts.length) throw new Error('至少选择生成图或参考图中的一项');
            const payload = await captioningApi('/workspace/save-generated', jsonOptions('POST', {directory: group.path, results: state.workspaceData.datasetResults.filter((item) => item.status === 'success'), parts, options: {include_prompt_line_suffix: form.elements.prompt_line.checked, append_timestamp_suffix: form.elements.timestamp.checked}}));
            feedback(root, `已保存 ${payload.count} 张图片`, 'success');
        } catch (error) { feedback(root, error.message, 'error'); }
        finally { button.disabled = false; button.textContent = '保存图片'; }
    });
}

async function runQueue(root, state, candidates) {
    if (state.workspaceData.datasetRunning || !candidates.length) return;
    state.workspaceData.datasetRunning = true; state.workspaceData.datasetStopped = false; candidates.forEach((item) => { item.status = 'queued'; item.error = ''; }); state.suiteRender();
    let next = 0; const concurrency = Math.max(1, Math.min(12, Number(state.workspaceData.datasetConcurrency || 1)));
    const worker = async () => {
        while (next < candidates.length && !state.workspaceData.datasetStopped) {
            const item = candidates[next++]; item.status = 'running'; state.suiteRender();
            try {
                const payload = await captioningApi('/workspace/image-dispatch', jsonOptions('POST', {schedule_id: state.workspaceData.datasetScheduleId, image_data_url: item.source_data_url, prompt: item.prompt}));
                item.generated_data_url = payload.image.data_url; item.mime_type = payload.image.mime_type; item.status = 'success';
            } catch (error) { item.status = 'failed'; item.error = error.message; }
            state.suiteRender();
        }
    };
    await Promise.all(Array.from({length: Math.min(concurrency, candidates.length)}, worker));
    if (state.workspaceData.datasetStopped) candidates.filter((item) => item.status === 'queued').forEach((item) => { item.status = 'stopped'; item.error = '用户已停止，可重跑此项'; });
    state.workspaceData.datasetRunning = false;
    if (state.workspace) state.workspace.dataset_results = state.workspaceData.datasetResults.map(({source_data_url, ...item}) => item);
    state.suiteRender();
}

function renderResult(item) {
    const src = item.generated_data_url || item.source_data_url;
    return `<article data-tone="${item.status === 'failed' ? 'error' : item.status === 'success' ? 'success' : 'info'}">${src ? `<img src="${escapeAttribute(src)}" alt="${escapeAttribute(item.source_name)}">` : ''}<header><strong>${escapeHtml(item.source_name)}</strong><span>${datasetStatusLabel(item.status)}</span></header><p>${escapeHtml(item.prompt)}</p>${item.error ? `<small>${escapeHtml(item.error)}</small>` : ''}${['failed','stopped'].includes(item.status) ? `<button class="dragon-btn dragon-btn-secondary" type="button" data-dataset-retry-item="${escapeAttribute(item.id)}">仅重跑此项</button>` : ''}</article>`;
}

function datasetStatusLabel(status) {
    return ({queued:'排队中',running:'生成中',success:'成功',failed:'失败',stopped:'已停止'}[status] || escapeHtml(status));
}

function geminiSchedules(state) {
    const channels = new Map((state.routing?.channels || []).map((item) => [item.id, item]));
    return (state.routing?.schedules || []).filter((schedule) => schedule.steps.some((step) => step.enabled && channels.get(step.channel_id)?.protocol === 'gemini'));
}

function datasetScheduleOptions(state, compatible, selected = '') {
    if (!compatible.length) return '<option value="">无可用 Gemini 调度</option>';
    const active = selected || state.workspaceData.datasetScheduleId || compatible[0]?.id;
    return compatible.map((schedule) => `<option value="${escapeAttribute(schedule.id)}" ${schedule.id === active ? 'selected' : ''}>${escapeHtml(schedule.name)}</option>`).join('');
}
