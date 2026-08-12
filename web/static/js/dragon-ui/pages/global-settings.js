/* Structured global settings backed by /api/settings/global. */

import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';

const api = createApiClient();
const SETTING_GROUPS = [
    {
        eyebrow: '文件与任务', title: '存储路径', desc: '控制训练输出、配置、历史和队列的存放位置。',
        fields: [
            ['output_root', '训练输出目录', 'text', 'output/runs'],
            ['configs_root', '配置根目录', 'text', 'configs'],
            ['history_root', '训练历史目录', 'text', '留空时跟随配置根目录'],
            ['queue_root', '训练队列目录', 'text', '留空时跟随配置根目录'],
        ],
    },
    {
        eyebrow: '模型验证', title: '生图测试', desc: '设置推理结果目录和权重扫描范围。',
        fields: [
            ['image_test_save_root', '生图输出目录', 'text', 'output/tests'],
            ['image_test_allow_home_search', '允许扫描用户目录中的权重', 'boolean', ''],
        ],
    },
    {
        eyebrow: '显示', title: '界面缩放', desc: '主缩放应用于全部页面；单页留空表示跟随主缩放。',
        fields: [
            ['ui_scale', '全局界面缩放', 'number', '100'],
            ['ui_scale_config', '训练配置页', 'number', '跟随全局'],
            ['ui_scale_datasets', '数据集页', 'number', '跟随全局'],
            ['ui_scale_training', '训练监控页', 'number', '跟随全局'],
            ['ui_scale_weight_analysis', '权重分析页', 'number', '跟随全局'],
            ['ui_scale_image_test', '生图测试页', 'number', '跟随全局'],
            ['ui_scale_settings', '全局设置页', 'number', '跟随全局'],
            ['ui_scale_model_config', '模型配置页', 'number', '跟随全局'],
            ['ui_scale_environment', '环境检测页', 'number', '跟随全局'],
        ],
    },
    {
        eyebrow: '历史详情', title: '历史页面缩放', desc: '分别控制历史详情的各个查看页；留空表示跟随主缩放。',
        fields: [
            ['ui_scale_history_overview', '历史概览', 'number', '跟随全局'],
            ['ui_scale_history_analysis', '历史分析', 'number', '跟随全局'],
            ['ui_scale_history_preview', '历史预览', 'number', '跟随全局'],
            ['ui_scale_history_logs', '历史日志', 'number', '跟随全局'],
            ['ui_scale_history_config_files', '历史配置文件', 'number', '跟随全局'],
        ],
    },
];

export async function loadGlobalSettings() {
    let payload;
    try {
        payload = await api('/api/settings/global');
        if (payload.ok === false) throw new Error(payload.error || '读取全局设置失败');
    } catch (error) {
        return `<div class="dragon-page"><div class="dragon-page-hero"><h1>全局设置</h1></div><div class="dragon-empty-state"><p>${escapeHtml(error.message || '读取全局设置失败')}</p></div></div>`;
    }

    const settings = payload || {};
    return { html: renderPage(settings), onMount: (root) => bindPage(root, settings) };
}

function renderPage(settings) {
    return `
        <div class="dragon-page dragon-page-wide dragon-global-settings-page">
            <div class="dragon-page-hero dragon-reveal">
                <span class="dragon-eyebrow">模型与系统</span>
                <h1>全局设置</h1>
                <p>管理本机存储位置、推理输出和界面缩放。模型路径在“全局模型配置”中维护。</p>
            </div>
            <form class="dragon-global-settings-form" data-global-settings-form>
                ${SETTING_GROUPS.map((group, index) => renderGroup(group, settings, index)).join('')}
                <div class="dragon-config-actions dragon-config-actions-sticky">
                    <button class="dragon-btn dragon-btn-primary" type="submit" data-global-action="save">保存全局设置</button>
                    <span class="dragon-config-feedback" data-global-feedback role="status" aria-live="polite"></span>
                </div>
            </form>
        </div>
    `;
}

function renderGroup(group, settings, index) {
    return `
        <section class="dragon-config-entry dragon-reveal" data-stagger="${Math.min(index + 1, 6)}">
            <header class="dragon-config-entry-header"><span class="dragon-eyebrow">${group.eyebrow}</span><h2>${group.title}</h2><p>${group.desc}</p></header>
            <div class="dragon-config-entry-fields"><div class="dragon-field-grid-2">${group.fields.map((field) => renderField(field, settings[field[0]])).join('')}</div></div>
        </section>
    `;
}

function renderField([key, label, type, placeholder], value) {
    if (type === 'boolean') {
        return `<div class="dragon-field"><div class="dragon-toggle-row"><button class="dragon-toggle" type="button" data-global-key="${key}" data-checked="${Boolean(value)}" role="switch" aria-checked="${Boolean(value)}"></button><div><div class="dragon-toggle-label">${label}</div><div class="dragon-toggle-desc">默认关闭，只扫描训练器管理的输出路径。</div></div></div></div>`;
    }
    const minMax = type === 'number' ? ' min="25" max="400" step="5"' : '';
    return `<label class="dragon-field"><span class="dragon-field-label-text">${label}</span><input class="dragon-input" type="${type}" data-global-key="${key}" value="${escapeAttribute(value ?? '')}" placeholder="${escapeAttribute(placeholder)}"${minMax}></label>`;
}

function bindPage(root) {
    root.querySelectorAll('.dragon-toggle[data-global-key]').forEach((toggle) => {
        toggle.addEventListener('click', () => {
            const checked = toggle.dataset.checked !== 'true';
            toggle.dataset.checked = String(checked);
            toggle.setAttribute('aria-checked', String(checked));
        });
    });
    root.querySelector('[data-global-settings-form]')?.addEventListener('submit', async (event) => {
        event.preventDefault();
        await saveSettings(root);
    });
}

async function saveSettings(root) {
    const button = root.querySelector('[data-global-action="save"]');
    button.disabled = true;
    try {
        const values = {};
        root.querySelectorAll('[data-global-key]').forEach((field) => {
            const key = field.dataset.globalKey;
            if (field.classList.contains('dragon-toggle')) values[key] = field.dataset.checked === 'true';
            else if (field.type === 'number') values[key] = field.value === '' ? '' : Number(field.value);
            else values[key] = field.value;
        });
        const payload = await api('/api/settings/global', { method: 'PUT', body: JSON.stringify(values) });
        if (payload.ok === false) throw new Error(payload.error || '保存全局设置失败');
        showFeedback(root, payload.requires_reload ? '设置已保存，刷新页面后使用新的配置目录' : '全局设置已保存', 'success');
    } catch (error) {
        showFeedback(root, error.message || '保存全局设置失败', 'error');
    } finally {
        button.disabled = false;
    }
}

function showFeedback(root, message, tone) { const el = root.querySelector('[data-global-feedback]'); if (el) { el.textContent = message; el.dataset.tone = tone; el.classList.add('dragon-config-feedback-visible'); } }
function escapeHtml(value) { return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])); }
function escapeAttribute(value) { return escapeHtml(value); }
