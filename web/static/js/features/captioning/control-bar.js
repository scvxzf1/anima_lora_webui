import { captioningApi, jsonOptions } from './api.js?v=dragon-ui-20260829v12';
import { findPreset, presetOptions } from './presets.js?v=dragon-ui-20260829v12';
import { escapeAttribute, escapeHtml, showFeedback, stateLabel, withBusy } from './utils.js?v=dragon-ui-20260829v12';

export function renderControlBar(state, prefill = {}) {
    const job = state.selectedJob;
    const total = Number(job?.total || 0);
    const done = Number(job?.completed || 0) + Number(job?.failed || 0);
    return `<header class="dragon-caption-controls">
        <form class="dragon-caption-command" data-caption-create-form>
            <label class="dragon-caption-directory"><span>数据集目录</span><input class="dragon-input" name="directory" value="${escapeAttribute(state.workspaceData.workbenchDirectory || prefill.directory || job?.directory || '')}" placeholder="/data/character_a" required></label>
            <label class="dragon-caption-engine"><span>引擎</span><select class="dragon-select" name="engine">${engineOptions(state.engines)}</select></label>
            <label data-caption-schedule-field><span>调度</span><select class="dragon-select" name="schedule_id">${scheduleOptions(state.routing)}</select></label>
            <label class="dragon-caption-preset"><span>预设</span><select class="dragon-select" name="preset_id">${presetOptions(state.presets)}</select></label>
            <button class="dragon-icon-button" type="button" data-caption-api-settings title="API 设置" aria-label="API 设置">⚙</button>
            <button class="dragon-btn dragon-btn-primary" type="submit">开始打标</button>
            <button class="dragon-btn dragon-btn-secondary" type="button" data-caption-advanced>任务参数</button>
            <div class="dragon-caption-advanced" data-caption-advanced-panel hidden>
                <label><span>标注扩展名</span><input class="dragon-input" name="caption_extension" value="${escapeAttribute(prefill.caption_extension || '.txt')}"></label>
                <label><span>图片上限</span><input class="dragon-input" type="number" name="limit" min="0" max="100000" value="0"></label>
                <label><span>已有标注</span><select class="dragon-select" name="existing_policy"><option value="skip">跳过</option><option value="include">重新生成</option></select></label>
                <label><span>前置词</span><input class="dragon-input" name="prefix" placeholder="my_character, 1girl"></label>
                <label><span>后置词</span><input class="dragon-input" name="suffix" placeholder="white background"></label>
                <label><span>自动过滤</span><input class="dragon-input" name="blacklist" value="high quality, image, description"></label>
                <label class="dragon-caption-check"><input type="checkbox" name="recursive" checked><span>包含子目录</span></label>
                <label class="dragon-caption-local-path"><span>本地模型路径</span><input class="dragon-input" name="local_model_path" placeholder="/path/to/model"></label>
                <label class="dragon-caption-local-path"><span>WD14 标签表 CSV</span><input class="dragon-input" name="local_tags_path" placeholder="/path/to/selected_tags.csv"></label>
                <label><span>本地设备</span><select class="dragon-select" name="local_device"><option value="auto">自动</option><option value="cuda">CUDA</option><option value="cpu">CPU</option></select></label>
                <label><span>通用阈值</span><input class="dragon-input" type="number" name="general_threshold" min="0" max="1" step="0.05" value="0.35"></label>
                <label><span>角色阈值</span><input class="dragon-input" type="number" name="character_threshold" min="0" max="1" step="0.05" value="0.85"></label>
                <label class="dragon-caption-check"><input type="checkbox" name="trust_remote_code"><span>允许本地模型自定义代码</span></label>
                <label><span>输出结果</span><select class="dragon-select" name="output_variant" data-caption-output-variant>${variantOptions(findPreset(state.presets, 'danbooru'))}</select></label>
                <label class="dragon-caption-prompt"><span>模型提示词</span><textarea class="dragon-textarea" name="prompt" rows="3">${escapeHtml(findPreset(state.presets, 'danbooru')?.prompt || '')}</textarea></label>
            </div>
        </form>
        <div class="dragon-caption-runbar">
            <div class="dragon-caption-progress"><progress max="${total || 1}" value="${done}"></progress><strong>${done}/${total}</strong><span>${job ? stateLabel(job.state) : '未创建任务'}</span></div>
            <span>并发 ${job?.config?.concurrency || state.settings.concurrency}</span>
            ${jobActions(job)}
            <span class="dragon-caption-feedback" data-caption-feedback role="status" aria-live="polite"></span>
        </div>
    </header>`;
}

export function bindControlBar(root, state, actions) {
    const form = root.querySelector('[data-caption-create-form]');
    form?.elements.preset_id?.addEventListener('change', () => {
        const preset = findPreset(state.presets, form.elements.preset_id.value);
        form.elements.prompt.value = preset.prompt;
        form.elements.output_variant.innerHTML = variantOptions(preset);
    });
    root.querySelector('[data-caption-advanced]')?.addEventListener('click', () => {
        const panel = root.querySelector('[data-caption-advanced-panel]');
        panel.hidden = !panel.hidden;
    });
    root.querySelector('[data-caption-api-settings]')?.addEventListener('click', () => root.querySelector('[data-caption-settings-dialog]')?.showModal());
    form?.addEventListener('submit', (event) => createJob(event, root, state, actions));
    root.querySelectorAll('[data-caption-job-action]').forEach((button) => button.addEventListener('click', () => runJobAction(button, root, state, actions)));
}

async function createJob(event, root, state, actions) {
    event.preventDefault();
    await withBusy(event.submitter, async () => {
        try {
            const form = event.currentTarget;
            const data = Object.fromEntries(new FormData(form).entries());
            const preset = findPreset(state.presets, data.preset_id);
            data.output_mode = preset.mode;
            data.output_variant = form.elements.output_variant.value || preset.output_variant || preset.mode;
            data.recursive = form.elements.recursive.checked;
            data.trust_remote_code = form.elements.trust_remote_code.checked;
            ['limit', 'general_threshold', 'character_threshold'].forEach((key) => { data[key] = Number(data[key] || 0); });
            const payload = await captioningApi('/jobs', jsonOptions('POST', data));
            state.selectedJobId = payload.job.id;
            await actions.refresh(true);
            showFeedback(root, `已创建 ${payload.job.total} 张图片的任务`, 'success');
        } catch (error) { showFeedback(root, error.message, 'error'); }
    });
}

async function runJobAction(button, root, state, actions) {
    await withBusy(button, async () => {
        try {
            await captioningApi(`/jobs/${encodeURIComponent(state.selectedJobId)}/${button.dataset.captionJobAction}`, jsonOptions('POST', {}));
            await actions.refresh(true);
        } catch (error) { showFeedback(root, error.message, 'error'); }
    });
}

function engineOptions(capabilities) {
    return (capabilities?.engines || []).map((engine) => `<option value="${escapeAttribute(engine.id)}" ${engine.available ? '' : 'disabled'}>${escapeHtml(engine.label)}${engine.available ? '' : '（依赖缺失）'}</option>`).join('');
}

function scheduleOptions(routing) {
    return (routing?.schedules || []).map((schedule) => `<option value="${escapeAttribute(schedule.id)}" ${schedule.id === routing.default_schedule_id ? 'selected' : ''}>${escapeHtml(schedule.name)}</option>`).join('');
}

function variantOptions(preset) {
    const variants = preset?.mode === 'three_format'
        ? [['tag', 'Tag 标签'], ['mixed_70tag_30nl', '70% Tag + 30% 自然语言'], ['pure_nl', '纯自然语言']]
        : [[preset?.output_variant || preset?.mode || 'tags', preset?.label || '默认结果']];
    return variants.map(([value, label]) => `<option value="${escapeAttribute(value)}">${escapeHtml(label)}</option>`).join('');
}

function jobActions(job) {
    if (!job) return '';
    const action = job.state === 'paused' || job.state === 'interrupted' ? ['resume', '继续'] : job.state === 'running' ? ['pause', '暂停'] : null;
    return `${action ? `<button class="dragon-btn dragon-btn-secondary" type="button" data-caption-job-action="${action[0]}">${action[1]}</button>` : ''}${job.failed ? '<button class="dragon-btn dragon-btn-secondary" type="button" data-caption-job-action="retry-failed">重试失败项</button>' : ''}`;
}
