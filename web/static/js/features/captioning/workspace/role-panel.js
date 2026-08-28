import { captioningApi, escapeAttribute, escapeHtml, feedback, fileToDataUrl, groupOptions, jsonOptions, panelShell, promptOptions, scheduleOptions, selectedGroup, selectedPrompt } from './shared.js?v=dragon-ui-20260829v11';

const ROLE_SYSTEM = 'Return only a clean comma-separated English Booru tag sequence for the visible character. Do not use Markdown or sentences.';

export function renderRolePanel(state) {
    const result = state.workspaceData.roleResult || '';
    const source = state.workspaceData.roleSource || '';
    const images = state.workspaceData.roleImages || [];
    return panelShell('ROLE TAG', '角色 Tag', '', `<form class="dragon-caption-tool-form" data-role-form>
        <div class="dragon-caption-form-grid dragon-caption-role-source">
            <label><span>目录组</span><select class="dragon-select" name="group_id"><option value="">上传单图</option>${groupOptions(state, state.workspaceData.roleGroupId)}</select></label>
            <label><span>组内图片</span><select class="dragon-select" name="image_name" data-role-image><option value="">选择图片</option>${images.map((item) => `<option value="${escapeAttribute(item.name)}" ${item.name === state.workspaceData.roleImageName ? 'selected' : ''}>${item.name}</option>`).join('')}</select></label>
            <label><span>调度组</span><select class="dragon-select" name="schedule_id">${scheduleOptions(state)}</select></label>
            <label><span>提示词预设</span><select class="dragon-select" name="prompt_id"><option value="">角色 Tag 默认</option>${promptOptions(state, 'system')}</select></label>
            <label class="dragon-caption-span-2"><span>本地图片</span><input class="dragon-input" type="file" name="image_file" accept="image/*"></label>
        </div>
        <div class="dragon-caption-role-preview" data-role-drop tabindex="0">${state.workspaceData.rolePreview ? `<img src="${escapeAttribute(state.workspaceData.rolePreview)}" alt="角色图片预览">` : '<span>选择目录图片，或上传 / 拖入一张图片</span>'}<small>${source ? `当前来源：${escapeHtml(source)}` : '尚未选择图片'}</small></div>
        <details class="dragon-caption-role-details" open><summary>角色约束（可选）</summary><div class="dragon-caption-form-grid">
            <label><span>角色名</span><input class="dragon-input" name="character_name"></label>
            <label><span>作品名</span><input class="dragon-input" name="series_name"></label>
            <label><span>外观特征</span><input class="dragon-input" name="appearance"></label>
            <label><span>必须保留</span><input class="dragon-input" name="required_tags"></label>
            <label><span>禁止加入</span><input class="dragon-input" name="blocked_tags"></label>
        </div></details>
        <div class="dragon-caption-form-actions dragon-caption-role-actions"><button class="dragon-btn dragon-btn-primary" type="submit" data-role-submit ${source ? '' : 'disabled'}>生成角色 Tag</button><button class="dragon-btn dragon-btn-secondary" type="button" data-role-copy ${result ? '' : 'disabled'}>复制结果</button><span>${result ? `${result.length} 字符` : '尚未生成'}</span></div>
        <textarea class="dragon-textarea" rows="8" data-role-result placeholder="生成结果" aria-label="角色 Tag 结果">${result}</textarea>
    </form>`);
}

export function bindRolePanel(root, state) {
    const form = root.querySelector('[data-role-form]');
    form.elements.group_id.addEventListener('change', async () => {
        const group = selectedGroup(state, form.elements.group_id.value);
        if (!group) return;
        try {
            const payload = await captioningApi('/workspace/images', jsonOptions('POST', {directory: group.path}));
            state.workspaceData.roleGroupId = group.id; state.workspaceData.roleImages = payload.images;
            form.elements.image_name.innerHTML = '<option value="">选择图片</option>' + payload.images.map((item) => `<option value="${item.name}">${item.name}</option>`).join('');
        } catch (error) { feedback(root, error.message, 'error'); }
    });
    form.elements.image_name.addEventListener('change', () => { const group = selectedGroup(state, form.elements.group_id.value); if (!group || !form.elements.image_name.value) return; state.workspaceData.roleFile = null; state.workspaceData.roleImageName = form.elements.image_name.value; state.workspaceData.roleSource = `目录图片 · ${form.elements.image_name.value}`; state.workspaceData.rolePreview = `/api/captioning/workspace/image?directory=${encodeURIComponent(group.path)}&name=${encodeURIComponent(form.elements.image_name.value)}`; state.suiteRender(); });
    form.elements.image_file.addEventListener('change', async () => { const file = form.elements.image_file.files[0]; if (file) { state.workspaceData.roleImageName = ''; state.workspaceData.roleFile = file; state.workspaceData.roleSource = `本地图片 · ${file.name}`; state.workspaceData.rolePreview = await fileToDataUrl(file); state.suiteRender(); } });
    const drop = root.querySelector('[data-role-drop]');
    drop?.addEventListener('dragover', (event) => event.preventDefault());
    drop?.addEventListener('drop', async (event) => { event.preventDefault(); const file = event.dataTransfer.files[0]; if (!file) return; state.workspaceData.roleImageName = ''; state.workspaceData.roleFile = file; state.workspaceData.roleSource = `拖入图片 · ${file.name}`; state.workspaceData.rolePreview = await fileToDataUrl(file); state.suiteRender(); });
    form.addEventListener('submit', async (event) => {
        event.preventDefault(); const button = event.submitter; button.disabled = true; const idleText = button.textContent; button.textContent = '正在生成…';
        try {
            const group = selectedGroup(state, form.elements.group_id.value);
            const preset = selectedPrompt(state, 'system', form.elements.prompt_id.value);
            const details = ['character_name', 'series_name', 'appearance', 'required_tags', 'blocked_tags'].map((key) => `${key}: ${form.elements[key].value.trim()}`).filter((line) => !line.endsWith(': ')).join('\n');
            const file = form.elements.image_file.files[0] || state.workspaceData.roleFile;
            const payload = await captioningApi('/workspace/dispatch', jsonOptions('POST', {
                schedule_id: form.elements.schedule_id.value,
                directory: group?.path || '', image_name: form.elements.image_name.value,
                image_data_url: file ? await fileToDataUrl(file) : '',
                system_prompt: preset?.content || ROLE_SYSTEM,
                prompt: `Create character tags using these supplied facts only when visually consistent:\n${details}`,
                min_chars: 1, max_chars: 5000, source: 'role-tag',
            }));
            state.workspaceData.roleResult = payload.response.trim().replace(/^```\w*|```$/g, '').replace(/[\n，、]+/g, ', ');
            form.querySelector('[data-role-result]').value = state.workspaceData.roleResult;
            feedback(root, `生成成功，尝试 ${payload.attempts} 次`, 'success');
        } catch (error) { feedback(root, error.message, 'error'); }
        finally { button.disabled = false; button.textContent = idleText; }
    });
    root.querySelector('[data-role-copy]')?.addEventListener('click', async () => { try { await navigator.clipboard.writeText(root.querySelector('[data-role-result]').value); feedback(root, '已复制角色 Tag', 'success'); } catch { feedback(root, '复制失败，请手动选择结果', 'error'); } });
}
