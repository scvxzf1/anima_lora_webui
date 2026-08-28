import { captioningApi, jsonOptions } from './api.js?v=dragon-ui-20260829v11';
import { bindRoutingPanel, renderRoutingPanel } from './routing-panel.js?v=dragon-ui-20260829v11';

const SETTING_KEYS = ['base_url', 'api_key', 'model', 'retry_count', 'retry_interval_seconds', 'concurrency', 'timeout_seconds', 'allow_private_network'];

export function renderProviderPanel(settings, routing) {
    return `<dialog class="dragon-caption-settings-dialog" data-caption-settings-dialog aria-labelledby="caption-provider-title"><section class="dragon-caption-section">
        <header class="dragon-caption-section-head dragon-caption-settings-toolbar"><div><span class="dragon-eyebrow">PROVIDER</span><h2 id="caption-provider-title">外部 API 与调度</h2><p>密钥只写入本机受限文件，页面重新读取时仅返回配置状态。</p></div><div class="dragon-caption-settings-summary"><span class="dragon-caption-key-state" data-key-state="${settings.api_key_configured ? 'ready' : 'missing'}">${escapeHtml(settings.api_key_hint)}</span><span>${routing.channels?.length || 0} 渠道</span><span>${routing.schedules?.length || 0} 调度组</span><button class="dragon-icon-button" type="button" data-caption-settings-close title="关闭" aria-label="关闭 API 与调度设置">×</button></div></header>
        <form class="dragon-caption-provider-form" data-caption-provider-form>
            <label class="dragon-field dragon-caption-wide"><span>API URL</span><input class="dragon-input" type="url" name="base_url" value="${escapeAttribute(settings.base_url)}" placeholder="https://api.example.com" required></label>
            <label class="dragon-field"><span>API Key</span><input class="dragon-input" type="password" name="api_key" value="" autocomplete="new-password" placeholder="${settings.api_key_configured ? '留空以保留已保存密钥' : '输入 API Key'}"></label>
            <label class="dragon-field"><span>模型</span><input class="dragon-input" type="text" name="model" value="${escapeAttribute(settings.model)}" placeholder="gpt-5.6-sol" required></label>
            <label class="dragon-field"><span>失败后重试次数</span><input class="dragon-input" type="number" name="retry_count" min="0" max="10" step="1" value="${settings.retry_count}"></label>
            <label class="dragon-field"><span>重试间隔（秒）</span><input class="dragon-input" type="number" name="retry_interval_seconds" min="0" max="300" step="0.5" value="${settings.retry_interval_seconds}"></label>
            <label class="dragon-field"><span>并发上限</span><input class="dragon-input" type="number" name="concurrency" min="1" max="${settings.max_concurrency || 4}" step="1" value="${settings.concurrency}"></label>
            <label class="dragon-field"><span>最大超时（秒）</span><input class="dragon-input" type="number" name="timeout_seconds" min="1" max="900" step="1" value="${settings.timeout_seconds}"></label>
            <label class="dragon-caption-check"><input type="checkbox" name="allow_private_network" ${settings.allow_private_network ? 'checked' : ''}><span>允许私有网络 API</span></label>
            <div class="dragon-caption-provider-actions dragon-caption-wide">
                <button class="dragon-btn dragon-btn-primary" type="submit">保存配置</button>
                <button class="dragon-btn dragon-btn-secondary" type="button" data-caption-test="ping">单纯 Ping</button>
                <button class="dragon-btn dragon-btn-secondary" type="button" data-caption-test="actual">实际可用性</button>
                <label class="dragon-caption-clear-key"><input type="checkbox" name="clear_api_key"><span>清除已保存 Key</span></label>
                <span class="dragon-caption-feedback" data-caption-provider-feedback role="status" aria-live="polite"></span>
                <button class="dragon-btn dragon-btn-secondary" type="button" data-caption-settings-close>关闭</button>
            </div>
        </form>
    </section>${renderRoutingPanel(routing)}</dialog>`;
}

export function bindProviderPanel(root, state) {
    bindRoutingPanel(root, state);
    const form = root.querySelector('[data-caption-provider-form]');
    form?.addEventListener('submit', async (event) => {
        event.preventDefault();
        await runAction(root, state, 'save');
    });
    root.querySelectorAll('[data-caption-test]').forEach((button) => button.addEventListener('click', () => runAction(root, state, button.dataset.captionTest)));
    root.querySelectorAll('[data-caption-settings-close]').forEach((button) => button.addEventListener('click', () => root.querySelector('[data-caption-settings-dialog]')?.close()));
}

export function collectProviderDraft(root) {
    const form = root.querySelector('[data-caption-provider-form]');
    const data = Object.fromEntries(new FormData(form).entries());
    const numeric = ['retry_count', 'retry_interval_seconds', 'concurrency', 'timeout_seconds'];
    numeric.forEach((key) => { data[key] = Number(data[key]); });
    data.clear_api_key = Boolean(form.elements.clear_api_key?.checked);
    data.allow_private_network = Boolean(form.elements.allow_private_network?.checked);
    return Object.fromEntries(SETTING_KEYS.concat('clear_api_key').map((key) => [key, data[key]]));
}

async function runAction(root, state, action) {
    const feedback = root.querySelector('[data-caption-provider-feedback]');
    const buttons = root.querySelectorAll('[data-caption-provider-form] button[type="submit"], [data-caption-provider-form] [data-caption-test]');
    buttons.forEach((button) => { button.disabled = true; });
    show(feedback, action === 'save' ? '正在保存…' : '正在测试…', 'info');
    try {
        const draft = collectProviderDraft(root);
        const path = action === 'save' ? '/settings' : `/test/${action}`;
        const method = action === 'save' ? 'PUT' : 'POST';
        const payload = await captioningApi(path, jsonOptions(method, draft));
        if (action === 'save') {
            state.settings = payload;
            const keyInput = root.querySelector('[name="api_key"]');
            if (keyInput) keyInput.value = '';
            root.querySelector('[data-key-state]').textContent = payload.api_key_hint;
            root.querySelector('[data-key-state]').dataset.keyState = payload.api_key_configured ? 'ready' : 'missing';
            show(feedback, '配置已保存', 'success');
        } else if (action === 'ping') {
            const modelText = payload.model_available ? '所选模型可用' : '服务可达，但模型列表中未找到所选模型';
            show(feedback, `Ping ${payload.elapsed_ms}ms · ${modelText}`, payload.model_available ? 'success' : 'warning');
        } else {
            show(feedback, `实际调用 ${payload.elapsed_ms}ms · ${payload.response}`, 'success');
        }
    } catch (error) {
        show(feedback, error.message, 'error');
    } finally {
        buttons.forEach((button) => { button.disabled = false; });
    }
}

function show(node, message, tone) {
    if (!node) return;
    node.textContent = message;
    node.dataset.tone = tone;
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
}

function escapeAttribute(value) { return escapeHtml(value); }
