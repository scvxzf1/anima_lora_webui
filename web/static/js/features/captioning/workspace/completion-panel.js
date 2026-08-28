import { captioningApi, escapeAttribute, escapeHtml, feedback, groupOptions, jsonOptions, panelShell, promptOptions, scheduleOptions, selectedGroup, selectedPrompt } from './shared.js?v=dragon-ui-20260829v11';

export function renderCompletionPanel(state) {
    const items = state.workspaceData.completionItems || [];
    const draft = state.workspaceData.completionDraft || {};
    const running = state.workspaceData.completionRunning === true;
    const done = items.filter((item) => ['ready', 'failed', 'saved'].includes(item.state)).length;
    const failed = items.filter((item) => item.state === 'failed').length;
    return panelShell('COMPLETION', '打标补全', '', `<form class="dragon-caption-tool-form" data-completion-form>
        <div class="dragon-caption-form-grid dragon-caption-completion-grid">
            <label><span>目录组</span><select class="dragon-select" name="group_id" required ${running ? 'disabled' : ''}><option value="">选择目录组</option>${groupOptions(state, draft.groupId || state.workspaceData.completionGroupId)}</select></label>
            <label><span>调度组</span><select class="dragon-select" name="schedule_id" ${running ? 'disabled' : ''}>${scheduleOptions(state, draft.scheduleId)}</select></label>
            <label><span>失败阈值</span><input class="dragon-input" type="number" name="threshold" min="0" max="10000" value="${escapeAttribute(String(draft.threshold ?? 40))}" ${running ? 'disabled' : ''}></label>
            <label><span>提示词预设</span><select class="dragon-select" name="prompt_id" ${running ? 'disabled' : ''}>${promptOptions(state, 'system', draft.promptId)}</select></label>
        </div>
        <label><span>补全要求</span><textarea class="dragon-textarea" name="prompt" rows="3" ${running ? 'disabled' : ''}>${escapeHtml(draft.prompt || 'Complete the existing short or missing caption with accurate visible English tags. Return only comma-separated tags.')}</textarea></label>
        <div class="dragon-caption-form-actions dragon-caption-completion-actions"><button class="dragon-btn dragon-btn-secondary" type="button" data-completion-scan ${running ? 'disabled' : ''}>扫描短标注</button><button class="dragon-btn dragon-btn-primary" type="submit" data-completion-start ${items.length && !running ? '' : 'disabled'}>开始补全</button><button class="dragon-btn dragon-btn-secondary" type="button" data-completion-stop ${running ? '' : 'disabled'}>停止后续任务</button><button class="dragon-btn dragon-btn-secondary" type="button" data-completion-save ${items.some((item) => item.proposed) && !running ? '' : 'disabled'}>保存全部候选</button></div>
    </form>${state.workspaceData.completionNotice ? `<div class="dragon-caption-inline-status" role="status">${escapeHtml(state.workspaceData.completionNotice)}</div>` : ''}${items.length ? `<div class="dragon-caption-completion-progress"><progress max="${items.length}" value="${done}"></progress><span>${done}/${items.length} 已处理 · ${failed} 失败${state.workspaceData.completionStopped ? ' · 已停止' : ''}</span></div>` : ''}<div class="dragon-caption-result-list" data-completion-list>${items.map(renderItem).join('') || '<div class="dragon-empty-state"><p>扫描后显示需要补全的图片。</p></div>'}</div>`);
}

export function bindCompletionPanel(root, state) {
    const form = root.querySelector('[data-completion-form]');
    root.querySelector('[data-completion-scan]')?.addEventListener('click', async (event) => {
        const button = event.currentTarget;
        button.disabled = true;
        try {
            const group = selectedGroup(state, form.elements.group_id.value);
            if (!group) throw new Error('请选择目录组');
            const payload = await captioningApi('/workspace/tags', jsonOptions('POST', {directory: group.path}));
            const threshold = Number(form.elements.threshold.value || 40);
            state.workspaceData.completionDraft = collectDraft(form);
            state.workspaceData.completionGroupId = group.id;
            state.workspaceData.completionItems = payload.results.filter((item) => item.text_length < threshold).map((item) => ({...item, state: 'waiting', proposed: '', error: ''}));
            state.suiteRender();
        } catch (error) { feedback(root, error.message, 'error'); }
        finally { button.disabled = false; }
    });
    const runCompletion = async (button, candidates = null) => {
        if (state.workspaceData.completionRunning) return;
        button.disabled = true;
        const draft = collectDraft(form);
        state.workspaceData.completionDraft = draft;
        const group = selectedGroup(state, draft.groupId) || selectedGroup(state, state.workspaceData.completionGroupId);
        const preset = selectedPrompt(state, 'system', draft.promptId);
        try {
            if (!group) throw new Error('请选择目录组');
            state.workspaceData.completionRunning = true; state.workspaceData.completionStopped = false;
            const queue = candidates || (state.workspaceData.completionItems || []).filter((item) => ['waiting', 'failed'].includes(item.state));
            for (const item of queue) {
                if (state.workspaceData.completionStopped) break;
                item.state = 'running'; state.suiteRender();
                try {
                    const payload = await captioningApi('/workspace/dispatch', jsonOptions('POST', {
                        directory: group.path, image_name: item.name, schedule_id: draft.scheduleId,
                        system_prompt: preset?.content || '', prompt: `${draft.prompt}\nExisting caption: ${item.text || '(empty)'}`,
                        min_chars: 1, max_chars: 10000, source: 'completion',
                    }));
                    item.proposed = payload.response.trim(); item.state = 'ready';
                } catch (error) { item.state = 'failed'; item.error = error.message; }
            }
            state.suiteRender();
        } finally { state.workspaceData.completionRunning = false; button.disabled = false; state.suiteRender(); }
    };
    form.addEventListener('submit', (event) => {
        event.preventDefault();
        const button = event.submitter || form.querySelector('[data-completion-start]');
        if (button) runCompletion(button);
        else feedback(root, '无法识别补全提交操作', 'error');
    });
    root.querySelector('[data-completion-start]')?.addEventListener('click', (event) => {
        event.preventDefault();
        runCompletion(event.currentTarget);
    });
    root.querySelector('[data-completion-stop]')?.addEventListener('click', () => { state.workspaceData.completionStopped = true; });
    root.querySelectorAll('[data-completion-retry]').forEach((button) => button.addEventListener('click', () => {
        const item = state.workspaceData.completionItems.find((entry) => entry.name === button.dataset.completionRetry);
        if (item) runCompletion(button, [item]);
    }));
    root.querySelector('[data-completion-save]')?.addEventListener('click', async (event) => {
        const group = selectedGroup(state, state.workspaceData.completionGroupId);
        event.currentTarget.disabled = true; let saved = 0, failed = 0;
        try {
            if (!group) throw new Error('补全目录组已失效，请重新扫描');
            for (const item of state.workspaceData.completionItems.filter((entry) => entry.proposed)) {
                try { await captioningApi('/workspace/tag', jsonOptions('PATCH', {directory: group.path, image_name: item.name, text: item.proposed, expected: item.text})); item.text = item.proposed; item.proposed = ''; item.state = 'saved'; item.error = ''; saved += 1; }
                catch (error) { item.state = 'failed'; item.error = `保存失败：${error.message}`; failed += 1; }
            }
            state.workspaceData.completionNotice = `已保存 ${saved} 个标注${failed ? `，${failed} 个保存失败并保留候选` : ''}`; state.suiteRender();
        } catch (error) { feedback(root, error.message, 'error'); }
        finally { event.currentTarget.disabled = false; }
    });
}

function collectDraft(form) {
    return {
        groupId: form.elements.group_id.value,
        scheduleId: form.elements.schedule_id.value,
        threshold: Number(form.elements.threshold.value || 40),
        promptId: form.elements.prompt_id.value,
        prompt: form.elements.prompt.value,
    };
}

function renderItem(item) {
    return `<article data-tone="${item.state === 'failed' ? 'error' : item.state === 'ready' ? 'success' : 'info'}"><header><strong>${escapeHtml(item.name)}</strong><span>${completionStatus(item.state)}</span></header><p>${escapeHtml(item.proposed || item.text || '空标注')}</p>${item.error ? `<small>${escapeHtml(item.error)}</small>` : ''}${item.state === 'failed' ? `<button class="dragon-btn dragon-btn-secondary" type="button" data-completion-retry="${escapeAttribute(item.name)}">仅重试此图</button>` : ''}</article>`;
}

function completionStatus(status) { return ({waiting:'待处理',running:'推理中',ready:'待保存',failed:'失败',saved:'已保存'}[status] || escapeHtml(status)); }
