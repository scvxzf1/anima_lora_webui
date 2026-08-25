import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

const DETAIL_SECTIONS = Object.freeze([
    ['怎么设置', 'fill', 'setup'],
    ['收益', 'benefit', 'benefit'],
    ['代价', 'cost', 'cost'],
    ['风险', 'risk', 'risk'],
    ['推荐', 'recommend', 'recommend'],
    ['补充', 'ps', 'note'],
]);

export function configHelpSummary(help) {
    return help ? (help.summary || help['作用'] || '') : '';
}

export function resolveConfigFieldHelp(key, label, helpCatalog) {
    const catalogHelp = helpCatalog?.[key];
    if (catalogHelp) return catalogHelp;
    const fieldLabel = label || key || '该参数';
    return {
        summary: `${fieldLabel}（${key}）是当前训练配置暴露的高级参数。`,
        fill: '优先保留当前方法或导入配置提供的值；修改前核对对应方法文档和训练前检查结果。',
        benefit: ['允许导入配置和实验方法完整呈现其高级参数。'],
        cost: ['该参数通常需要结合具体训练方法理解，单独修改不一定产生收益。'],
        risk: ['未知或不兼容的值可能导致训练前检查失败、启动失败或行为偏离配方。'],
        recommend: '不确定时保持当前值；只有明确理解训练端语义时再修改。',
        ps: `当前字段尚无专项说明，原始配置键为 ${key}。`,
    };
}

export function renderConfigHelpButton(key, label) {
    return `<button class="dragon-field-help-btn" type="button" data-help-key="${escapeHtml(key)}"
        data-help-label="${escapeHtml(label)}" aria-haspopup="dialog" aria-controls="dragon-config-help-dialog"
        aria-label="查看${escapeHtml(label)}说明" title="查看说明">?</button>`;
}

export function bindConfigFieldHelpDialog(root, helpCatalog) {
    const dialog = ensureConfigHelpDialog(root);
    if (!dialog) return;
    root.querySelectorAll('.dragon-field-help-btn:not([data-help-dialog-bound])').forEach((button) => {
        button.dataset.helpDialogBound = 'true';
        button.addEventListener('click', () => {
            const key = button.dataset.helpKey || '';
            const label = button.dataset.helpLabel || key;
            const help = resolveConfigFieldHelp(key, label, helpCatalog);
            dialog.querySelector('[data-config-help-title]').textContent = label;
            dialog.querySelector('[data-config-help-key]').textContent = key;
            dialog.querySelector('[data-config-help-body]').innerHTML = renderConfigHelpBody(help);
            if (!dialog.open) dialog.showModal();
        });
    });
}

function ensureConfigHelpDialog(root) {
    let dialog = root.querySelector('[data-config-help-dialog]');
    if (dialog) return dialog;
    root.insertAdjacentHTML('beforeend', renderConfigHelpDialog());
    dialog = root.querySelector('[data-config-help-dialog]');
    if (!dialog) return null;
    const close = () => {
        if (dialog.open) dialog.close('cancel');
    };
    dialog.querySelectorAll('[data-config-help-close]').forEach((button) => button.addEventListener('click', close));
    dialog.addEventListener('click', (event) => {
        if (event.target === dialog) close();
    });
    return dialog;
}

function renderConfigHelpDialog() {
    return `<dialog class="dragon-config-help-dialog" id="dragon-config-help-dialog" data-config-help-dialog
        aria-labelledby="dragon-config-help-dialog-title">
        <div class="dragon-config-help-dialog-shell">
            <header class="dragon-config-help-dialog-header">
                <div>
                    <span class="dragon-eyebrow">参数说明</span>
                    <h2 id="dragon-config-help-dialog-title" data-config-help-title>参数说明</h2>
                    <code data-config-help-key></code>
                </div>
                <button class="dragon-icon-button" type="button" data-config-help-close aria-label="关闭参数说明" title="关闭">
                    ${renderIcon('x')}
                </button>
            </header>
            <div class="dragon-config-help-dialog-body" data-config-help-body></div>
            <footer class="dragon-config-help-dialog-footer">
                <button class="dragon-btn dragon-btn-primary dragon-btn-sm" type="button" data-config-help-close>关闭</button>
            </footer>
        </div>
    </dialog>`;
}

function renderConfigHelpBody(help) {
    const summary = configHelpSummary(help);
    const details = DETAIL_SECTIONS
        .map(([heading, property, tone]) => renderHelpSection(heading, help?.[property], tone))
        .filter(Boolean)
        .join('');
    const summarySection = summary ? renderHelpSection('作用', summary, 'summary') : '';
    return `${summarySection}${details}`;
}

function renderHelpSection(heading, value, tone) {
    const items = normalizeHelpItems(value);
    if (!items.length) return '';
    const body = items.length === 1
        ? `<p>${escapeHtml(items[0])}</p>`
        : `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
    return `<section class="dragon-config-field-help-section" data-help-tone="${tone}">
        <div class="dragon-config-field-help-heading">${heading}</div>${body}
    </section>`;
}

function normalizeHelpItems(value) {
    const values = Array.isArray(value) ? value : [value];
    return values.map((item) => String(item ?? '').trim()).filter(Boolean);
}
