import { captioningApi, jsonOptions } from './api.js?v=dragon-ui-20260829v11';
import { escapeAttribute, escapeHtml } from './utils.js?v=dragon-ui-20260829v11';

export function renderRoutingPanel(routing) {
    return `<section class="dragon-caption-section dragon-caption-routing-section" data-caption-settings-section="routing">
        <header class="dragon-caption-section-head"><div><span class="dragon-eyebrow">ROUTING</span><h2>渠道与调度</h2><p>${routing.channels?.length || 0} 个渠道 · ${routing.schedules?.length || 0} 个调度组 · 默认：${escapeHtml(routing.default_schedule_id || '未设置')}</p></div></header>
        <div data-caption-routing-editor>${renderEditor(makeDraft(routing))}</div>
    </section>`;
}

export function bindRoutingPanel(root, state) {
    const host = root.querySelector('[data-caption-routing-editor]');
    if (!host) return;
    let draft = makeDraft(state.routing);
    host.addEventListener('input', (event) => { updateDraft(draft, event.target); markDirty(host); });
    host.addEventListener('change', (event) => { updateDraft(draft, event.target); markDirty(host); });
    host.addEventListener('click', async (event) => {
        const button = event.target.closest('[data-routing-action]');
        if (!button) return;
        const action = button.dataset.routingAction;
        if (action === 'test-ping' || action === 'test-actual') {
            await testChannel(button, host, draft, action.replace('test-', ''));
            return;
        }
        if (action === 'fetch-models') {
            const payload = await testChannel(button, host, draft, 'ping');
            if (payload?.models?.length) { draft.channels[Number(button.dataset.channelIndex)]._models = payload.models; host.innerHTML = renderEditor(draft, host.closest('dialog')?.dataset.routingDirty === 'true'); }
            return;
        }
        if (action === 'save') {
            await saveDraft(button, host, state, draft);
            draft = makeDraft(state.routing);
            return;
        }
        if (['remove-channel', 'remove-schedule'].includes(action) && !window.confirm(action === 'remove-channel' ? '删除此渠道并将相关步骤切换到备用渠道？' : '删除此调度组？')) return;
        mutateDraft(draft, action, button.dataset);
        host.innerHTML = renderEditor(draft, true);
    });
}

function renderEditor(draft, dirty = false) {
    return `<form class="dragon-caption-routing-form" data-caption-routing-form>
        <div class="dragon-caption-routing-heading"><h3>API 渠道 <small>${draft.channels.length}</small></h3><button class="dragon-btn dragon-btn-secondary" type="button" data-routing-action="add-channel">＋ 添加渠道</button></div>
        <div class="dragon-caption-routing-list">${draft.channels.map(renderChannel).join('')}</div>
        <div class="dragon-caption-routing-heading"><h3>调度组 <small>${draft.schedules.length}</small></h3><button class="dragon-btn dragon-btn-secondary" type="button" data-routing-action="add-schedule">＋ 添加调度组</button></div>
        <div class="dragon-caption-routing-list">${draft.schedules.map((schedule, index) => renderSchedule(schedule, index, draft)).join('')}</div>
        <footer class="dragon-caption-routing-actions"><button class="dragon-btn dragon-btn-primary" type="button" data-routing-action="save">保存渠道与调度</button><span class="dragon-caption-feedback" data-caption-routing-feedback data-tone="${dirty ? 'warning' : 'info'}" role="status" aria-live="polite">${dirty ? '有未保存修改' : ''}</span></footer>
    </form>`;
}

function renderChannel(channel, index) {
    const hints = (channel.api_key_hints || []).join('、') || '未配置';
    return `<section class="dragon-caption-routing-row" data-channel-index="${index}">
        <header><div class="dragon-caption-channel-title"><strong>${escapeHtml(channel.name || `渠道 ${index + 1}`)}</strong><span>${channel.api_key_count || 0} Key · ${escapeHtml(channel.protocol === 'gemini' ? 'Gemini' : 'OpenAI')}</span><small data-channel-feedback="${index}" role="status" aria-live="polite"></small></div><div class="dragon-caption-channel-actions"><button class="dragon-btn dragon-btn-secondary" type="button" data-routing-action="fetch-models" data-channel-index="${index}">获取模型</button><button class="dragon-btn dragon-btn-secondary" type="button" data-routing-action="test-ping" data-channel-index="${index}">Ping</button><button class="dragon-btn dragon-btn-secondary" type="button" data-routing-action="test-actual" data-channel-index="${index}">实际调用</button><button class="dragon-icon-button" type="button" data-routing-action="remove-channel" data-channel-index="${index}" title="删除渠道" aria-label="删除渠道">×</button></div></header>
        <div class="dragon-caption-routing-fields">
            ${field('名称', 'name', channel.name, `data-channel-index="${index}" data-channel-field="name"`)}
            ${field('渠道 ID', 'id', channel.id, `data-channel-index="${index}" data-channel-field="id"`)}
            ${field('API URL', 'base_url', channel.base_url, `data-channel-index="${index}" data-channel-field="base_url"`, 'url', 'dragon-caption-span-2')}
            <label class="dragon-field dragon-caption-span-2"><span>默认模型</span><input class="dragon-input" name="default_model" value="${escapeAttribute(channel.default_model)}" list="caption-models-${index}" data-channel-index="${index}" data-channel-field="default_model"><datalist id="caption-models-${index}">${(channel._models || []).map((model) => `<option value="${escapeAttribute(model)}"></option>`).join('')}</datalist></label>
            <label class="dragon-field"><span>协议</span><select class="dragon-select" data-channel-index="${index}" data-channel-field="protocol"><option value="openai" ${channel.protocol !== 'gemini' ? 'selected' : ''}>OpenAI 兼容</option><option value="gemini" ${channel.protocol === 'gemini' ? 'selected' : ''}>Gemini 图像生成</option></select></label>
            <label class="dragon-field dragon-caption-span-2"><span>新增 API Keys（每行一个）</span><textarea class="dragon-textarea" rows="2" autocomplete="new-password" data-channel-index="${index}" data-channel-field="_apiKeys" placeholder="留空以保留现有密钥"></textarea><small>${channel.api_key_count || 0} 个 · ${escapeHtml(hints)}</small></label>
            <label class="dragon-caption-check"><input type="checkbox" data-channel-index="${index}" data-channel-field="allow_private_network" ${channel.allow_private_network ? 'checked' : ''}><span>允许私有网络 API</span></label>
            <label class="dragon-caption-check"><input type="checkbox" data-channel-index="${index}" data-channel-field="_clearApiKeys"><span>清空密钥池</span></label>
        </div>
    </section>`;
}

function renderSchedule(schedule, index, draft) {
    const checked = schedule.id === draft.default_schedule_id ? 'checked' : '';
    return `<section class="dragon-caption-routing-row" data-schedule-index="${index}">
        <header><label class="dragon-caption-default"><input type="radio" name="default_schedule" value="${escapeAttribute(schedule.id)}" data-default-schedule ${checked}><strong>调度组 ${index + 1}</strong></label><button class="dragon-icon-button" type="button" data-routing-action="remove-schedule" data-schedule-index="${index}" title="删除调度组" aria-label="删除调度组">×</button></header>
        <div class="dragon-caption-routing-fields dragon-caption-schedule-fields">
            ${field('名称', 'name', schedule.name, `data-schedule-index="${index}" data-schedule-field="name"`)}
            ${field('调度 ID', 'id', schedule.id, `data-schedule-index="${index}" data-schedule-field="id"`)}
            ${field('并发上限', 'concurrency', schedule.concurrency, `data-schedule-index="${index}" data-schedule-field="concurrency" min="1" max="4"`, 'number')}
            <label class="dragon-field dragon-caption-span-2"><span>系统提示词注入</span><textarea class="dragon-textarea" rows="2" data-schedule-index="${index}" data-schedule-field="system_prompt">${escapeHtml(schedule.system_prompt || '')}</textarea></label>
            ${field('用户提示词前置', 'user_prefix', schedule.user_prefix || '', `data-schedule-index="${index}" data-schedule-field="user_prefix"`, 'text', 'dragon-caption-span-2')}
            ${field('用户提示词后置', 'user_suffix', schedule.user_suffix || '', `data-schedule-index="${index}" data-schedule-field="user_suffix"`, 'text', 'dragon-caption-span-2')}
        </div>
        <div class="dragon-caption-step-list">${schedule.steps.map((step, stepIndex) => renderStep(step, index, stepIndex, draft.channels)).join('')}</div>
        <button class="dragon-btn dragon-btn-secondary" type="button" data-routing-action="add-step" data-schedule-index="${index}">＋ 添加步骤</button>
    </section>`;
}

function renderStep(step, scheduleIndex, stepIndex, channels) {
    const attrs = `data-schedule-index="${scheduleIndex}" data-step-index="${stepIndex}"`;
    return `<div class="dragon-caption-step" data-step-row>
        <span class="dragon-caption-step-number">${stepIndex + 1}</span>
        <label><span>渠道</span><select class="dragon-select" ${attrs} data-step-field="channel_id">${channels.map((channel) => `<option value="${escapeAttribute(channel.id)}" ${channel.id === step.channel_id ? 'selected' : ''}>${escapeHtml(channel.name)}</option>`).join('')}</select></label>
        ${field('模型覆盖', 'model', step.model, `${attrs} data-step-field="model"`)}
        ${field('重试', 'retry_count', step.retry_count, `${attrs} data-step-field="retry_count" min="0" max="10"`, 'number')}
        ${field('间隔/秒', 'retry_interval_seconds', step.retry_interval_seconds, `${attrs} data-step-field="retry_interval_seconds" min="0" max="300" step="0.5"`, 'number')}
        ${field('超时/秒', 'timeout_seconds', step.timeout_seconds, `${attrs} data-step-field="timeout_seconds" min="1" max="900"`, 'number')}
        <label><span>推理强度</span><select class="dragon-select" ${attrs} data-step-field="reasoning_effort">${['', 'low', 'medium', 'high', 'xhigh'].map((value) => `<option value="${value}" ${value === step.reasoning_effort ? 'selected' : ''}>${value || '默认'}</option>`).join('')}</select></label>
        <label class="dragon-caption-check"><input type="checkbox" ${attrs} data-step-field="enabled" ${step.enabled ? 'checked' : ''}><span>启用</span></label>
        <div class="dragon-caption-step-actions"><button type="button" data-routing-action="move-step-up" ${attrs} title="上移" aria-label="上移">↑</button><button type="button" data-routing-action="move-step-down" ${attrs} title="下移" aria-label="下移">↓</button><button type="button" data-routing-action="remove-step" ${attrs} title="删除步骤" aria-label="删除步骤">×</button></div>
    </div>`;
}

function field(label, name, value, attrs, type = 'text', extraClass = '') {
    return `<label class="dragon-field ${extraClass}"><span>${label}</span><input class="dragon-input" type="${type}" name="${name}" value="${escapeAttribute(value)}" ${attrs}></label>`;
}

function updateDraft(draft, target) {
    const value = target.type === 'checkbox' ? target.checked : target.value;
    if (target.hasAttribute('data-default-schedule')) draft.default_schedule_id = value;
    if (target.dataset.channelField) {
        const channel = draft.channels[Number(target.dataset.channelIndex)];
        const previous = channel[target.dataset.channelField];
        channel[target.dataset.channelField] = value;
        if (target.dataset.channelField === 'id') draft.schedules.forEach((schedule) => schedule.steps.forEach((step) => { if (step.channel_id === previous) step.channel_id = value; }));
    }
    if (target.dataset.scheduleField) {
        const schedule = draft.schedules[Number(target.dataset.scheduleIndex)];
        const previous = schedule[target.dataset.scheduleField];
        schedule[target.dataset.scheduleField] = value;
        if (target.dataset.scheduleField === 'id' && draft.default_schedule_id === previous) draft.default_schedule_id = value;
    }
    if (target.dataset.stepField) draft.schedules[Number(target.dataset.scheduleIndex)].steps[Number(target.dataset.stepIndex)][target.dataset.stepField] = value;
}

function mutateDraft(draft, action, data) {
    const scheduleIndex = Number(data.scheduleIndex);
    const stepIndex = Number(data.stepIndex);
    if (action === 'add-channel') draft.channels.push(newChannel(nextNumber(draft.channels, 'channel')));
    if (action === 'remove-channel' && draft.channels.length > 1) removeChannel(draft, Number(data.channelIndex));
    if (action === 'add-schedule') draft.schedules.push(newSchedule(nextNumber(draft.schedules, 'schedule'), draft.channels[0]?.id));
    if (action === 'remove-schedule' && draft.schedules.length > 1) draft.schedules.splice(scheduleIndex, 1);
    if (action === 'add-step') draft.schedules[scheduleIndex].steps.push(newStep(draft.channels[0]?.id));
    if (action === 'remove-step' && draft.schedules[scheduleIndex].steps.length > 1) draft.schedules[scheduleIndex].steps.splice(stepIndex, 1);
    if (action === 'move-step-up' && stepIndex > 0) move(draft.schedules[scheduleIndex].steps, stepIndex, stepIndex - 1);
    if (action === 'move-step-down' && stepIndex < draft.schedules[scheduleIndex].steps.length - 1) move(draft.schedules[scheduleIndex].steps, stepIndex, stepIndex + 1);
    if (!draft.schedules.some((item) => item.id === draft.default_schedule_id)) draft.default_schedule_id = draft.schedules[0].id;
}

function removeChannel(draft, index) {
    const [removed] = draft.channels.splice(index, 1);
    const fallback = draft.channels[0].id;
    draft.schedules.forEach((schedule) => schedule.steps.forEach((step) => { if (step.channel_id === removed.id) step.channel_id = fallback; }));
}

async function saveDraft(button, host, state, draft) {
    const feedback = host.querySelector(`[data-channel-feedback="${index}"]`) || host.querySelector('[data-caption-routing-feedback]');
    const originalLabel = button.textContent; button.disabled = true; button.textContent = kind === 'ping' ? 'Ping 中…' : '调用中…';
    show(feedback, '正在保存…', 'info');
    try {
        const payload = await captioningApi('/routing', jsonOptions('PUT', serialize(draft)));
        state.routing = payload;
        const dialog = host.closest('dialog'); if (dialog) dialog.dataset.routingDirty = 'false';
        host.innerHTML = renderEditor(makeDraft(payload));
        state.suiteRender?.();
        show(host.querySelector('[data-caption-routing-feedback]'), '渠道与调度已保存', 'success');
    } catch (error) {
        show(feedback, error.message, 'error');
    } finally {
        button.disabled = false; button.textContent = originalLabel;
    }
}

async function testChannel(button, host, draft, kind) {
    const index = Number(button.dataset.channelIndex);
    const channel = draft.channels[index];
    const step = draft.schedules.flatMap((schedule) => schedule.steps).find((item) => item.channel_id === channel.id) || newStep(channel.id);
    const feedback = host.querySelector('[data-caption-routing-feedback]');
    button.disabled = true;
    show(feedback, kind === 'ping' ? '正在 Ping…' : '正在执行实际调用…', 'info');
    try {
        const payload = await captioningApi(`/routing/test/${kind}`, jsonOptions('POST', {
            ...channel,
            api_keys: channel._apiKeys.trim() ? channel._apiKeys.split(/\r?\n/) : [],
            model: step.model || channel.default_model,
            retry_count: Number(step.retry_count),
            retry_interval_seconds: Number(step.retry_interval_seconds),
            timeout_seconds: Number(step.timeout_seconds),
            reasoning_effort: step.reasoning_effort,
        }));
        const message = kind === 'ping' ? `Ping ${payload.elapsed_ms}ms · ${payload.model_available ? '模型可用' : '模型未出现在列表'}` : `实际调用 ${payload.elapsed_ms}ms · ${payload.response}`;
        show(feedback, message, kind === 'ping' && !payload.model_available ? 'warning' : 'success');
        return payload;
    } catch (error) {
        show(feedback, error.message, 'error');
    } finally {
        button.disabled = false;
    }
}

function serialize(draft) {
    return {
        default_schedule_id: draft.default_schedule_id,
        channels: draft.channels.map(({_apiKeys, _clearApiKeys, ...channel}) => ({...channel, ...(_apiKeys.trim() ? {api_keys: _apiKeys.split(/\r?\n/)} : {}), clear_api_keys: _clearApiKeys})),
        schedules: draft.schedules.map((schedule) => ({...schedule, concurrency: Number(schedule.concurrency), steps: schedule.steps.map((step) => ({...step, retry_count: Number(step.retry_count), retry_interval_seconds: Number(step.retry_interval_seconds), timeout_seconds: Number(step.timeout_seconds)}))})),
    };
}

function makeDraft(routing = {}) {
    const channels = (routing.channels || []).map((item) => ({...item, _apiKeys: '', _clearApiKeys: false}));
    const schedules = (routing.schedules || []).map((item) => ({...item, steps: (item.steps || []).map((step) => ({...step}))}));
    if (!channels.length) channels.push(newChannel(1));
    if (!schedules.length) schedules.push(newSchedule(1, channels[0].id));
    return {default_schedule_id: routing.default_schedule_id || schedules[0].id, channels, schedules};
}

function newChannel(index) { return {id: `channel-${index}`, name: `渠道 ${index}`, base_url: '', default_model: '', protocol: 'openai', allow_private_network: false, api_key_count: 0, api_key_hints: [], _apiKeys: '', _clearApiKeys: false}; }
function newSchedule(index, channelId) { return {id: `schedule-${index}`, name: `调度组 ${index}`, concurrency: 1, system_prompt: '', user_prefix: '', user_suffix: '', steps: [newStep(channelId)]}; }
function newStep(channelId) { return {channel_id: channelId || '', model: '', retry_count: 2, retry_interval_seconds: 2, timeout_seconds: 120, reasoning_effort: '', enabled: true}; }
function move(items, from, to) { const [item] = items.splice(from, 1); items.splice(to, 0, item); }
function nextNumber(items, prefix) { let index = items.length + 1; while (items.some((item) => item.id === `${prefix}-${index}`)) index += 1; return index; }
function show(node, message, tone) { if (node) { node.textContent = message; node.dataset.tone = tone; } }
function markDirty(host) { const dialog = host.closest('dialog'); if (dialog) dialog.dataset.routingDirty = 'true'; show(host.querySelector('[data-caption-routing-feedback]'), '有未保存修改', 'warning'); }
