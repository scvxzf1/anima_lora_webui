import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260902v36';

const DETAIL_SECTIONS = Object.freeze([
    ['为什么通常这样设', 'fill', 'setup'],
    ['改了可能有什么好处', 'benefit', 'benefit'],
    ['同时会带来什么', 'cost', 'cost'],
    ['可能遇到什么问题', 'risk', 'risk'],
    ['补充说明', 'ps', 'note'],
]);

const MODEL_FAMILY_LABELS = Object.freeze({
    anima: 'Anima',
    krea2: 'Krea-2 Raw',
    krea2_raw: 'Krea-2 Raw',
    z_image: 'Z-Image',
});

export function configHelpSummary(help) {
    return help ? (help.summary || help['作用'] || '') : '';
}

export function resolveConfigFieldHelp(key, label, helpCatalog) {
    const catalogHelp = helpCatalog?.[key];
    if (catalogHelp) return catalogHelp;
    const fieldLabel = label || key || '该参数';
    return {
        summary: `${fieldLabel} 是一个不常用的进阶设置。`,
        fill: '这一项暂时没有单独的新手说明，当前值通常是由你选择的训练方法或配置文件带入的。',
        benefit: ['有明确的调试目标时，它可以用来微调特定方法。'],
        cost: ['单独改它不一定会让训练更好，还会让问题更难排查。'],
        risk: ['填入不兼容的值时，训练前检查或启动过程可能会拒绝继续。'],
        recommend: '如果你刚开始用，先保持当前值。等有明确的对照目标时再改。',
        ps: `它在配置文件里的名字是 ${key}。`,
    };
}

export function renderConfigHelpButton(key, label, {
    unavailableReason = '',
    currentValue,
    defaultValue,
    modelFamily = '',
} = {}) {
    const unavailable = Boolean(unavailableReason);
    const buttonClass = unavailable ? 'dragon-field-help-btn dragon-field-help-btn-unavailable' : 'dragon-field-help-btn';
    const title = unavailable ? '查看不可用原因' : '查看说明';
    const ariaLabel = unavailable ? `查看${label}不可用原因` : `查看${label}说明`;
    const currentValueAttribute = ` data-help-current-value="${escapeHtml(serializeHelpValue(currentValue))}"`;
    const defaultValueAttribute = defaultValue === undefined
        ? ''
        : ` data-help-default-value="${escapeHtml(serializeHelpValue(defaultValue))}"`;
    const modelFamilyAttribute = modelFamily
        ? ` data-help-model-family="${escapeHtml(modelFamily)}"`
        : '';
    return `<button class="${buttonClass}" type="button" data-help-key="${escapeHtml(key)}"
        ${unavailable ? `data-help-unavailable-reason="${escapeHtml(unavailableReason)}"` : ''}
        ${currentValueAttribute}${defaultValueAttribute}${modelFamilyAttribute}
        data-help-label="${escapeHtml(label)}" aria-haspopup="dialog" aria-controls="dragon-config-help-dialog"
        aria-label="${escapeHtml(ariaLabel)}" title="${escapeHtml(title)}">${renderIcon('circleHelp')}</button>`;
}

export function bindConfigFieldHelpDialog(root, helpCatalogSource) {
    const dialog = ensureConfigHelpDialog(root);
    if (!dialog) return;
    root.querySelectorAll('.dragon-field-help-btn:not([data-help-dialog-bound])').forEach((button) => {
        button.dataset.helpDialogBound = 'true';
        button.addEventListener('click', async () => {
            const key = button.dataset.helpKey || '';
            const label = button.dataset.helpLabel || key;
            const helpCatalog = await resolveHelpCatalog(helpCatalogSource);
            if (!button.isConnected) return;
            const help = resolveConfigFieldHelp(key, label, helpCatalog);
            dialog.querySelector('[data-config-help-title]').textContent = label;
            dialog.querySelector('[data-config-help-key]').textContent = key;
            dialog.querySelector('[data-config-help-body]').innerHTML = renderConfigHelpBody(
                help,
                button.dataset.helpUnavailableReason || '',
                {
                    currentValue: resolveLiveHelpValue(button),
                    defaultValue: button.hasAttribute('data-help-default-value')
                        ? button.dataset.helpDefaultValue
                        : undefined,
                    modelFamily: button.dataset.helpModelFamily || '',
                },
            );
            if (!dialog.open) dialog.showModal();
        });
    });
}

function resolveLiveHelpValue(button) {
    const control = button.closest('[data-config-field-key]')?.querySelector('[data-key]');
    if (!control) return button.dataset.helpCurrentValue ?? '';
    if (control.classList.contains('dragon-toggle')) return control.dataset.checked || 'false';
    return control.value ?? button.dataset.helpCurrentValue ?? '';
}

async function resolveHelpCatalog(source) {
    try {
        return typeof source === 'function' ? await source() : source;
    } catch {
        return null;
    }
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

function renderConfigHelpBody(help, unavailableReason = '', context = {}) {
    const summary = configHelpSummary(help);
    const details = DETAIL_SECTIONS
        .map(([heading, property, tone]) => renderHelpSection(heading, help?.[property], tone))
        .filter(Boolean)
        .join('');
    const contextSection = renderHelpContext({ ...context, unavailable: Boolean(unavailableReason) });
    const recommendSection = renderHelpSection('新手建议', help?.recommend, 'recommend');
    const summarySection = summary ? renderHelpSection('它是做什么的', summary, 'summary') : '';
    const unavailableSection = unavailableReason
        ? renderHelpSection('为什么现在不能改', unavailableReason, 'unavailable')
        : '';
    return `${contextSection}${unavailableSection}${summarySection}${recommendSection}${details}`;
}

function renderHelpContext({ currentValue = '', defaultValue, modelFamily = '', unavailable = false } = {}) {
    const current = formatHelpValue(currentValue);
    const hasDefault = defaultValue !== undefined;
    const initial = hasDefault ? formatHelpValue(defaultValue) : '';
    const family = MODEL_FAMILY_LABELS[modelFamily] || modelFamily;
    const sameAsInitial = hasDefault && currentValue === defaultValue;
    const hint = unavailable
        ? '这一项现在不会参与训练，保持原样就好。下面会告诉你它为什么不能改。'
        : !hasDefault
        ? '这一项没有统一的页面起始值，通常由当前配置或训练方法决定。'
        : sameAsInitial
            ? '当前值和页面起始值一样，新手可以先把它当作稳妥起点。'
            : '当前配置或训练方法改过这一项。这不代表填错了，先看下面的建议再决定要不要改回去。';
    return `<section class="dragon-config-field-help-section dragon-config-help-context" data-help-tone="context">
        <div class="dragon-config-field-help-heading">你现在的设置</div>
        <div class="dragon-config-help-context-values">
            ${renderContextValue('当前值', current)}
            ${hasDefault ? renderContextValue('页面起始值', initial) : ''}
            ${family ? renderContextValue('当前模型', family) : ''}
        </div>
        <p class="dragon-config-help-context-hint">${hint}</p>
    </section>`;
}

function renderContextValue(label, value) {
    return `<span class="dragon-config-help-context-value"><small>${escapeHtml(label)}</small><strong>${escapeHtml(value)}</strong></span>`;
}

function serializeHelpValue(value) {
    if (value === undefined || value === null) return '';
    if (typeof value === 'object') {
        try {
            return JSON.stringify(value);
        } catch {
            return String(value);
        }
    }
    return String(value);
}

function formatHelpValue(value) {
    if (value === '') return '未填写';
    if (value === 'true') return '开启';
    if (value === 'false') return '关闭';
    return value;
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
