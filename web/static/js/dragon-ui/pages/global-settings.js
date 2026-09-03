/* Structured global settings backed by /api/settings/global. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { applyDragonUIScale, clampUIScale } from '../ui-scale.js?v=dragon-ui-20260814v43';
import { applyDragonMotionSetting } from '../motion.js?v=dragon-ui-20260824v1';
import { applyDragonConfigChromeSettings } from '../config-chrome.js?v=dragon-ui-20260825v1';

const api = createApiClient();
const SETTING_GROUPS = [
    {
        eyebrow: '文件与任务', title: '存储路径', desc: '训练输出、配置、历史与队列的存放位置。长路径可单独占行显示。',
        fields: [
            ['output_root', '训练输出目录', 'text', 'output/runs', '训练运行、日志与检查点的默认根目录。'],
            ['configs_root', '配置根目录', 'text', 'configs', '包含方法、数据集、历史与队列配置。更改后页面会自动刷新。'],
            ['history_root', '训练历史目录', 'text', '留空时跟随配置根目录', '留空时使用“配置根目录/web-training-history”。'],
            ['queue_root', '训练队列目录', 'text', '留空时跟随配置根目录', '留空时使用“配置根目录/web-training-queue”。'],
            ['tagging_max_retained_jobs', '打标任务保留上限', 'integer', '40', '超过上限时自动清理最旧的已结束任务；不会删除图片或已写回的 TXT。'],
        ],
    },
    {
        eyebrow: '模型验证', title: '生图测试', desc: '控制推理结果目录与权重扫描边界。',
        fields: [
            ['image_test_save_root', '生图输出目录', 'text', 'output/tests', '留空时使用默认推理输出目录。'],
            ['image_test_allow_home_search', '允许扫描用户目录中的权重', 'boolean', '', '开启后会扩大权重搜索范围，目录较大时扫描会更慢。'],
        ],
    },
    {
        eyebrow: '显示', title: '界面显示', desc: '管理 Dragon 动态效果、训练参数标记与界面缩放。系统启用“减少动态效果”时始终以系统设置为准。',
        fields: [
            ['dragon_motion_enabled', '启用 Dragon 动态效果', 'boolean', '', '控制页面入场、滚动揭示、视差和平滑过渡；关闭后仍保留必要的加载状态。'],
            ['dragon_config_help_always_visible', '常态显示参数“？”', 'boolean', '', '开启后训练配置积木会始终显示帮助问号；关闭时仅在悬浮、聚焦或展开说明时显示。'],
            ['dragon_config_tags_always_visible', '常态显示参数标签', 'boolean', '', '开启后训练配置积木会始终显示右上角分类标签；关闭时仅在悬浮或聚焦时显示。'],
            ['ui_scale', '全局界面缩放', 'number', '100', '有效范围 25%–400%。'],
            ['ui_scale_config', '训练配置页', 'scale', '', ''],
            ['ui_scale_datasets', '数据集页', 'scale', '', ''],
            ['ui_scale_training', '训练监控页', 'scale', '', ''],
            ['ui_scale_weight_analysis', '权重分析页', 'scale', '', ''],
            ['ui_scale_image_test', '生图测试页', 'scale', '', ''],
            ['ui_scale_settings', '全局设置页', 'scale', '', ''],
            ['ui_scale_model_config', '模型配置页', 'scale', '', ''],
            ['ui_scale_environment', '环境检测页', 'scale', '', ''],
        ],
    },
    {
        eyebrow: '历史详情', title: '历史页面缩放', desc: '分别控制历史详情中的查看区域。',
        fields: [
            ['ui_scale_history_overview', '历史概览', 'scale', '', ''],
            ['ui_scale_history_analysis', '历史分析', 'scale', '', ''],
            ['ui_scale_history_preview', '历史预览', 'scale', '', ''],
            ['ui_scale_history_logs', '历史日志', 'scale', '', ''],
            ['ui_scale_history_config_files', '历史配置文件', 'scale', '', ''],
        ],
    },
];

export async function loadGlobalSettings() {
    let payload;
    try {
        payload = await api('/api/settings/global');
        if (payload?.ok === false) throw new Error(payload.error || '读取全局设置失败');
    } catch (error) {
        return renderError(error.message || '读取全局设置失败');
    }

    const state = createState(payload || {});
    return {
        html: renderPage(state),
        onMount: (root) => bindPage(root, state),
        beforeLeave: () => confirmLeave(state),
        onUnmount: () => cleanupPage(state),
    };
}

function createState(payload) {
    const values = normalizeFormSettings(payload);
    return {
        values,
        savedValues: { ...values },
        defaults: normalizeSettings(payload.defaults || {}),
        effectivePaths: effectivePathValues(payload),
        dirty: false,
        saving: false,
        beforeUnload: null,
    };
}

function normalizeFormSettings(payload, previousValues = null) {
    const source = { ...(payload || {}) };
    if (payload?.path_overrides && typeof payload.path_overrides === 'object') {
        Object.assign(source, payload.path_overrides);
    } else if (previousValues) {
        source.history_root = previousValues.history_root;
        source.queue_root = previousValues.queue_root;
    } else {
        const configsRoot = String(payload?.configs_root || 'configs').replace(/\/$/, '');
        if (String(payload?.history_root || '') === `${configsRoot}/web-training-history`) source.history_root = '';
        if (String(payload?.queue_root || '') === `${configsRoot}/web-training-queue`) source.queue_root = '';
    }
    return normalizeSettings(source);
}

function effectivePathValues(payload) {
    if (payload?.effective_paths && typeof payload.effective_paths === 'object') return { ...payload.effective_paths };
    return Object.fromEntries(['configs_root', 'history_root', 'queue_root'].map((key) => [key, payload?.[key] || '']));
}

function normalizeSettings(source) {
    const values = {};
    for (const group of SETTING_GROUPS) {
        for (const [key, , type] of group.fields) {
            const raw = source?.[key];
            if (type === 'boolean') values[key] = raw == null ? booleanDefault(key) : Boolean(raw);
            else values[key] = raw ?? '';
        }
    }
    return values;
}

function booleanDefault(key) {
    return key === 'dragon_motion_enabled';
}

function renderPage(state) {
    return `
        <div class="dragon-page dragon-page-wide dragon-tool-page dragon-global-settings-page">
            <div class="dragon-global-settings-workspace">
                <aside class="dragon-global-settings-sidebar dragon-reveal" aria-label="全局设置范围">
                    <div class="dragon-global-settings-sidebar-brand">
                        <span class="dragon-eyebrow">GLOBAL FORGE</span>
                        <h1>全局设置</h1>
                        <p>路径、任务存储与界面显示的统一入口。</p>
                    </div>
                    <div class="dragon-global-settings-scope">
                        <span class="dragon-eyebrow">SETTING SCOPE</span>
                        <div class="dragon-global-settings-summary">
                            ${renderSidebarSummary('output_root', '输出根目录', state.values.output_root || '未设置')}
                            ${renderSidebarSummary('configs_root', '配置目录路径', state.values.configs_root || 'configs')}
                            ${renderSidebarSummary('ui_scale', '界面设置', `${clampUIScale(state.values.ui_scale || 100)}%`)}
                        </div>
                    </div>
                    <p class="dragon-global-settings-sidebar-note">修改会作用于本机 WebUI 与后续训练任务。</p>
                </aside>
                <section class="dragon-global-settings-main" aria-labelledby="dragon-global-settings-title">
                    <header class="dragon-tool-hero dragon-global-settings-header dragon-reveal">
                        <div class="dragon-tool-hero-copy">
                            <span class="dragon-eyebrow">GLOBAL SETTINGS</span>
                            <h2 id="dragon-global-settings-title">路径与界面</h2>
                            <p>管理本机存储、推理输出和界面显示。模型路径请在“全局模型配置”中维护。</p>
                        </div>
                        <div class="dragon-global-settings-header-side">
                            ${renderSettingsSummary(state)}
                            <div class="dragon-tool-actions">
                                <button class="dragon-btn dragon-btn-secondary" type="button" data-global-action="reset">恢复默认</button>
                            </div>
                        </div>
                    </header>
                    <form class="dragon-global-settings-form" data-global-settings-form novalidate>
                        <div class="dragon-savebar dragon-global-settings-savebar" data-dirty="false">
                            <div class="dragon-savebar-status">
                                <strong data-global-dirty-label>所有修改已保存</strong>
                                <span data-global-feedback role="status" aria-live="polite">界面显示设置保存后会立即应用。</span>
                            </div>
                            <div class="dragon-savebar-actions">
                                <button class="dragon-btn dragon-btn-secondary" type="button" data-global-action="revert" disabled>还原修改</button>
                                <button class="dragon-btn dragon-btn-primary" type="submit" data-global-action="save" disabled>保存全局设置</button>
                            </div>
                        </div>
                        ${SETTING_GROUPS.map((group, index) => renderGroup(group, state, index)).join('')}
                    </form>
                </section>
            </div>
        </div>
    `;
}

function renderGroup(group, state, index) {
    const sectionNumber = String(index + 1).padStart(2, '0');
    return `
        <section class="dragon-config-entry dragon-settings-entry dragon-global-settings-group dragon-reveal" data-settings-section="${sectionNumber}" data-stagger="${Math.min(index + 1, 6)}">
            <header class="dragon-config-entry-header dragon-global-settings-group-header">
                <span class="dragon-global-settings-group-index" aria-hidden="true">${sectionNumber}</span>
                <div><span class="dragon-eyebrow">${group.eyebrow}</span><h2>${group.title}</h2><p>${group.desc}</p></div>
            </header>
            <div class="dragon-config-entry-fields dragon-global-settings-group-fields"><div class="dragon-settings-fields">${group.fields.map((field) => renderField(field, state.values, state.effectivePaths)).join('')}</div></div>
        </section>
    `;
}

function renderSidebarSummary(key, label, value) {
    return `<div class="dragon-global-settings-summary-card"><span>${label}</span><strong data-global-summary="${key}" title="${escapeAttribute(value)}">${escapeHtml(value)}</strong></div>`;
}

function renderSettingsSummary(state) {
    const output = state.values.output_root || '未设置';
    const configs = state.values.configs_root || 'configs';
    const scale = `${clampUIScale(state.values.ui_scale || 100)}%`;
    return `<div class="dragon-global-settings-stats" aria-label="设置摘要">
        <div class="dragon-global-settings-stat"><span>ROOT</span><strong>01</strong><small data-global-summary="output_root" title="${escapeAttribute(output)}">${escapeHtml(output)}</small></div>
        <div class="dragon-global-settings-stat"><span>CONFIGS</span><strong>01</strong><small data-global-summary="configs_root" title="${escapeAttribute(configs)}">${escapeHtml(configs)}</small></div>
        <div class="dragon-global-settings-stat"><span>UI</span><strong>01</strong><small data-global-summary="ui_scale">${escapeHtml(scale)}</small></div>
    </div>`;
}

function renderField([key, label, type, placeholder, help], values, effectivePaths = {}) {
    const value = values[key];
    if (type === 'boolean') {
        return `<label class="dragon-setting-toggle"><input class="visually-hidden" type="checkbox" name="${key}" data-global-key="${key}" ${value ? 'checked' : ''}><span class="dragon-toggle" data-checked="${Boolean(value)}" aria-hidden="true"></span><span><strong>${label}</strong><small>${help}</small></span></label>`;
    }
    if (type === 'scale') return renderScaleField(key, label, value, values.ui_scale);
    const inputType = ['number', 'integer'].includes(type) ? 'number' : type;
    const minMax = type === 'number'
        ? ' min="25" max="400" step="5" inputmode="numeric"'
        : type === 'integer' ? ' min="1" max="500" step="1" inputmode="numeric"' : '';
    const suffix = type === 'number' ? '<span class="dragon-input-suffix">%</span>' : '';
    const effective = ['history_root', 'queue_root'].includes(key) && effectivePaths[key]
        ? `<small class="dragon-setting-effective">当前实际目录：<span class="dragon-text-mono">${escapeHtml(effectivePaths[key])}</span></small>`
        : '';
    return `<label class="dragon-field dragon-setting-field"><span class="dragon-field-label-text">${label}</span><span class="dragon-setting-input-wrap"><input class="dragon-input" type="${inputType}" name="${key}" autocomplete="off" spellcheck="false" data-global-key="${key}" value="${escapeAttribute(value)}" placeholder="${escapeAttribute(placeholder)}"${minMax}>${suffix}</span>${help ? `<small class="dragon-setting-help">${help}</small>` : ''}${effective}<small class="dragon-field-error" data-global-error="${key}" aria-live="polite"></small></label>`;
}

function renderScaleField(key, label, value, globalScale) {
    const follow = String(value ?? '').trim() === '';
    const displayValue = follow ? clampUIScale(globalScale || 100) : clampUIScale(value);
    return `
        <div class="dragon-field dragon-setting-field dragon-scale-field" data-scale-row="${key}" data-follow="${follow}">
            <div class="dragon-scale-field-head"><label for="dragon-${key}">${label}</label><label class="dragon-follow-default"><input type="checkbox" name="${key}_follow_global" data-scale-follow="${key}" ${follow ? 'checked' : ''}><span>跟随全局</span></label></div>
            <span class="dragon-setting-input-wrap"><input id="dragon-${key}" class="dragon-input" type="number" name="${key}" inputmode="numeric" min="25" max="400" step="5" autocomplete="off" data-global-key="${key}" value="${displayValue}" ${follow ? 'disabled' : ''}><span class="dragon-input-suffix">%</span></span>
            <small class="dragon-setting-help">${follow ? `当前跟随全局 ${clampUIScale(globalScale || 100)}%` : '使用该页面的独立缩放比例。'}</small>
            <small class="dragon-field-error" data-global-error="${key}" aria-live="polite"></small>
        </div>
    `;
}

function bindPage(root, state) {
    state.root = root;
    state.beforeUnload = (event) => {
        if (!state.dirty) return;
        event.preventDefault();
        event.returnValue = '';
    };
    window.addEventListener('beforeunload', state.beforeUnload);

    bindFieldEvents(root, state);
    root.querySelector('[data-global-action="reset"]')?.addEventListener('click', () => resetToDefaults(root, state));
    root.querySelector('[data-global-action="revert"]')?.addEventListener('click', () => revertChanges(root, state));
    root.querySelector('[data-global-settings-form]')?.addEventListener('submit', async (event) => {
        event.preventDefault();
        await saveSettings(root, state);
    });
    updatePageState(root, state);
}

function bindFieldEvents(root, state) {
    root.querySelectorAll('[data-global-key]').forEach((field) => {
        if (field.dataset.bound === 'true') return;
        field.dataset.bound = 'true';
        const update = () => {
            if (field.type === 'checkbox') {
                field.closest('.dragon-setting-toggle')?.querySelector('.dragon-toggle')?.setAttribute('data-checked', String(field.checked));
            }
            clearError(root, field.dataset.globalKey);
            collectValues(root, state);
            markDirty(root, state);
            if (field.dataset.globalKey === 'ui_scale') syncFollowingScales(root, state);
        };
        field.addEventListener('input', update);
        field.addEventListener('change', update);
    });
    root.querySelectorAll('[data-scale-follow]').forEach((toggle) => {
        if (toggle.dataset.bound === 'true') return;
        toggle.dataset.bound = 'true';
        toggle.addEventListener('change', () => {
            const key = toggle.dataset.scaleFollow;
            const input = root.querySelector(`[data-global-key="${key}"]`);
            if (input) {
                input.disabled = toggle.checked;
                if (toggle.checked) input.value = String(clampUIScale(root.querySelector('[data-global-key="ui_scale"]')?.value || 100));
            }
            updateScaleRow(root, key, toggle.checked);
            collectValues(root, state);
            markDirty(root, state);
        });
    });
}

function collectValues(root, state) {
    const next = {};
    root.querySelectorAll('[data-global-key]').forEach((field) => {
        const key = field.dataset.globalKey;
        const follow = root.querySelector(`[data-scale-follow="${key}"]`);
        if (follow?.checked) next[key] = '';
        else if (field.type === 'checkbox') next[key] = field.checked;
        else if (field.type === 'number') next[key] = field.value === '' ? '' : Number(field.value);
        else next[key] = field.value.trim();
    });
    state.values = next;
    return next;
}

function markDirty(root, state) {
    state.dirty = settingsSignature(state.values) !== settingsSignature(state.savedValues);
    updatePageState(root, state);
}

function settingsSignature(values) {
    return JSON.stringify(Object.keys(values).sort().map((key) => [key, values[key]]));
}

function resetToDefaults(root, state) {
    if (!window.confirm('确认将全局设置恢复为项目默认值吗？保存前仍可还原。')) return;
    state.values = { ...state.defaults };
    renderFields(root, state);
    markDirty(root, state);
    showFeedback(root, '已载入项目默认值，保存后生效', 'info');
}

function revertChanges(root, state) {
    if (!state.dirty || !window.confirm('确认还原本页的所有未保存修改吗？')) return;
    state.values = { ...state.savedValues };
    renderFields(root, state);
    state.dirty = false;
    updatePageState(root, state);
    showFeedback(root, '已还原到上次保存的设置', 'info');
}

function renderFields(root, state) {
    const entries = root.querySelectorAll('.dragon-settings-entry');
    entries.forEach((entry, index) => {
        const container = entry.querySelector('.dragon-settings-fields');
        if (container) container.innerHTML = SETTING_GROUPS[index].fields.map((field) => renderField(field, state.values, state.effectivePaths)).join('');
    });
    bindFieldEvents(root, state);
}

function syncFollowingScales(root, state) {
    const globalScale = clampUIScale(root.querySelector('[data-global-key="ui_scale"]')?.value || 100);
    root.querySelectorAll('[data-scale-follow]:checked').forEach((toggle) => {
        const key = toggle.dataset.scaleFollow;
        const input = root.querySelector(`[data-global-key="${key}"]`);
        if (input) input.value = String(globalScale);
        updateScaleRow(root, key, true);
    });
    collectValues(root, state);
}

function updateScaleRow(root, key, follow) {
    const row = root.querySelector(`[data-scale-row="${key}"]`);
    if (!row) return;
    row.dataset.follow = String(follow);
    const help = row.querySelector('.dragon-setting-help');
    if (help) help.textContent = follow
        ? `当前跟随全局 ${clampUIScale(root.querySelector('[data-global-key="ui_scale"]')?.value || 100)}%`
        : '使用该页面的独立缩放比例。';
}

async function saveSettings(root, state) {
    if (state.saving) return;
    collectValues(root, state);
    clearAllErrors(root);
    const invalid = validateSettings(state.values);
    if (invalid) {
        showError(root, invalid.key, invalid.message);
        return showFeedback(root, invalid.message, 'error');
    }
    state.saving = true;
    updatePageState(root, state);
    showFeedback(root, '正在保存…', 'info');
    try {
        const payload = await api('/api/settings/global', { method: 'PUT', body: JSON.stringify(state.values) });
        if (payload?.ok === false) throw new Error(payload.error || '保存全局设置失败');
        state.values = normalizeFormSettings(payload, state.values);
        state.savedValues = { ...state.values };
        state.defaults = normalizeSettings(payload.defaults || state.defaults);
        state.effectivePaths = effectivePathValues(payload);
        state.dirty = false;
        applyDragonUIScale(payload, 'global-settings');
        applyDragonMotionSetting(payload);
        applyDragonConfigChromeSettings(payload);
        renderFields(root, state);
        updatePageState(root, state);
        if (payload.requires_reload) {
            showFeedback(root, '设置已保存，正在切换新的配置根目录…', 'success');
            window.setTimeout(() => window.location.reload(), 250);
            return;
        }
        showFeedback(root, payload.message || '全局设置已保存并应用', 'success');
    } catch (error) {
        showFeedback(root, `${error.message || '保存全局设置失败'}。请检查路径或缩放值后重试。`, 'error');
    } finally {
        state.saving = false;
        updatePageState(root, state);
    }
}

function validateSettings(values) {
    const retained = Number(values.tagging_max_retained_jobs);
    if (!Number.isInteger(retained) || retained < 1 || retained > 500) {
        return { key: 'tagging_max_retained_jobs', message: '打标任务保留上限必须是 1–500 的整数' };
    }
    for (const [key, value] of Object.entries(values)) {
        if (!key.startsWith('ui_scale') || value === '') continue;
        const numeric = Number(value);
        if (!Number.isFinite(numeric) || numeric < 25 || numeric > 400) return { key, message: '界面缩放必须在 25%–400% 之间' };
    }
    return null;
}

function updatePageState(root, state) {
    const savebar = root.querySelector('.dragon-global-settings-savebar');
    if (savebar) savebar.dataset.dirty = String(state.dirty);
    setText(root, '[data-global-dirty-label]', state.dirty ? '有未保存修改' : '所有修改已保存');
    updateSummaryValue(root, 'output_root', state.values.output_root || '未设置');
    updateSummaryValue(root, 'configs_root', state.values.configs_root || 'configs');
    updateSummaryValue(root, 'ui_scale', `${clampUIScale(state.values.ui_scale || 100)}%`);
    setDisabled(root, '[data-global-action="revert"]', !state.dirty || state.saving);
    setDisabled(root, '[data-global-action="save"]', !state.dirty || state.saving);
    setDisabled(root, '[data-global-action="reset"]', state.saving);
}

function updateSummaryValue(root, key, value) {
    root.querySelectorAll(`[data-global-summary="${key}"]`).forEach((element) => {
        const text = String(value ?? '');
        element.textContent = text;
        element.title = text;
    });
}

function confirmLeave(state) {
    return !state.dirty || window.confirm('全局设置有未保存修改，离开会丢失这些修改。是否继续？');
}

function cleanupPage(state) {
    if (state.beforeUnload) window.removeEventListener('beforeunload', state.beforeUnload);
}

function showError(root, key, message) {
    const input = root.querySelector(`[data-global-key="${key}"]`);
    const error = root.querySelector(`[data-global-error="${key}"]`);
    if (input) {
        input.disabled = false;
        input.setAttribute('aria-invalid', 'true');
        input.focus();
    }
    if (error) error.textContent = message;
}

function clearError(root, key) {
    root.querySelector(`[data-global-key="${key}"]`)?.removeAttribute('aria-invalid');
    const error = root.querySelector(`[data-global-error="${key}"]`);
    if (error) error.textContent = '';
}

function clearAllErrors(root) {
    root.querySelectorAll('[data-global-error]').forEach((error) => { error.textContent = ''; });
    root.querySelectorAll('[aria-invalid="true"]').forEach((input) => input.removeAttribute('aria-invalid'));
}

function showFeedback(root, message, tone = '') {
    const element = root.querySelector('[data-global-feedback]');
    if (!element) return;
    element.textContent = message;
    element.dataset.tone = tone;
}

function setText(root, selector, value) {
    const element = root.querySelector(selector);
    if (element) element.textContent = String(value ?? '');
}

function setDisabled(root, selector, disabled) {
    const element = root.querySelector(selector);
    if (element) element.disabled = Boolean(disabled);
}

function renderError(message) {
    return `<div class="dragon-page dragon-tool-page"><header class="dragon-tool-hero"><div><span class="dragon-eyebrow">模型与系统</span><h1>全局设置</h1><p>无法读取本机设置。</p></div></header><div class="dragon-empty-state"><p>${escapeHtml(message)}</p><p>请检查服务连接后刷新页面。</p></div></div>`;
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
