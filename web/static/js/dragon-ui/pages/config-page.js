/* Generic config page renderer.
 * Renders a config sub-page with fields, generous spacing,
 * help text, and save/restore actions.
 * Uses the existing config catalog (form-layout.js, labels-options.js, etc.)
 * Language: Chinese labels and descriptions, English config keys hidden by default.
 */

import { FIELD_LABEL_ZH, FIELD_OPTIONS } from '../../config/catalog/labels-options.js?v=dragon-ui-20260812v35';
import { FIELD_HELP_ZH } from '../../config/catalog/field-help.js?v=dragon-ui-20260812v35';
import {
    ALL_LORA_ADAPTER_SCOPED_FIELD_KEYS,
    FORM_UI_DEFAULTS,
    LOKR_SCOPED_FIELD_KEYS,
    METHOD_SCOPED_CONFIG_FORM_FIELDS,
    NETWORK_ARG_FIELD_MAP,
    RETIRED_CONFIG_FORM_FIELDS,
    VERA_SCOPED_FIELD_KEYS,
} from '../../config/catalog/defaults.js?v=dragon-ui-20260812v35';
import { VARIANT_METHOD_FAMILY } from '../../config/catalog/form-layout.js?v=dragon-ui-20260812v35';
import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { SECTION_GROUPS } from './section-groups.js?v=dragon-ui-20260812v35';
import { findCategory, isConfigCategory } from '../category-map.js?v=dragon-ui-20260812v35';
import { scanForReveal } from '../animations.js?v=dragon-ui-20260816v67';
import { keysForConfigSubItem } from './config-field-map.js?v=dragon-ui-20260812v35';
import { configValueForControl, displayConfigValue, prepareConfigPatch, serializeConfigValue } from './config-values.js?v=dragon-ui-20260812v35';
import { bindTrainingControls, isEditableConfigFile, loadTrainingContext, mergedConfigUrl, renderTrainingControls, selectTrainingConfigFile, selectTrainingPreset, commitTrainingContext } from './training-controls.js?v=dragon-ui-20260816v67';
import { bindTrainingPresetLibrary, renderTrainingPresetLibrary } from './training-preset-library.js?v=dragon-ui-20260816v76';
import {
    bindModelQuickPicker,
    MODEL_QUICK_PATH_KEYS,
    renderModelQuickPickerDialog,
    renderModelQuickPickerTrigger,
} from './model-quick-picker.js?v=dragon-ui-20260819v3';

const api = createApiClient();

function categoryLabel(sub) {
    if (!sub || !sub.categoryId) return "";
    const cat = findCategory(sub.categoryId);
    return cat ? cat.label : "";
}

function categoryDescription(categoryId) {
    const descriptions = {
        'training-config': '训练所需的基础模型、数据行为、适配器、步数和采样设置。',
        'memory-optimization': '显存、计算精度、编译、缓存和数据传输设置。',
        'advanced-methods': 'LoRA 扩展、条件注入、路由、损失加权和实验工具。',
    };
    return descriptions[categoryId] || '按功能组织的训练配置。';
}

function activeMethodFamily(trainingContext, values = {}) {
    const variant = String(trainingContext?.variant || '').trim().toLowerCase();
    if (trainingContext?.methodsSubdir === 'methods' && variant === 'spd') return 'spd';
    if (VARIANT_METHOD_FAMILY[variant]) return VARIANT_METHOD_FAMILY[variant];
    if (values.use_ip_adapter || String(values.network_module || '').includes('ip_adapter')) return 'ip_adapter';
    if (values.use_easycontrol || String(values.network_module || '').includes('easycontrol')) return 'easycontrol';
    if (String(values.network_module || '').includes('soft_tokens')) return 'soft_tokens';
    if (values.use_chimera_hydra) return 'chimera';
    return 'lora';
}

function activeAdapterKind(values = {}) {
    if (values.use_glora) return 'glora';
    if (values.use_vera) return 'vera';
    if (values.use_lokr) return 'lokr';
    if (values.use_loha) return 'loha';
    return 'lora';
}

function visibleConfigKeys(keys, trainingContext, values) {
    const method = activeMethodFamily(trainingContext, values);
    const adapter = activeAdapterKind(values);
    return keys.filter((key) => {
        if (RETIRED_CONFIG_FORM_FIELDS.has(key)) return false;
        const methodScope = METHOD_SCOPED_CONFIG_FORM_FIELDS.get(key);
        if (methodScope && !methodScope.has(method)) return false;
        const spec = NETWORK_ARG_FIELD_MAP.get(key);
        if (spec) {
            const familyMatches = {
                lokr: adapter === 'lokr' || method === 'lokr',
                vera: adapter === 'vera' || method === 'vera',
                soft_tokens: method === 'soft_tokens',
                ip_adapter: method === 'ip_adapter',
                easycontrol: method === 'easycontrol',
            };
            if (!familyMatches[spec.family]) return false;
        }
        if (LOKR_SCOPED_FIELD_KEYS.has(key) && adapter !== 'lokr') return false;
        if (VERA_SCOPED_FIELD_KEYS.has(key) && adapter !== 'vera') return false;
        if (key === 'dora_wd' && adapter !== 'lora') return false;
        return !ALL_LORA_ADAPTER_SCOPED_FIELD_KEYS.has(key)
            || (key === 'lora_adapter_kind' || (adapter === 'lokr' && LOKR_SCOPED_FIELD_KEYS.has(key))
                || (adapter === 'vera' && VERA_SCOPED_FIELD_KEYS.has(key)) || (adapter === 'lora' && key === 'dora_wd'));
    });
}

export async function loadConfigPage(context) {
    const requestedSub = context.sub;
    const category = findCategory(context.categoryId || requestedSub?.categoryId);
    const isCategoryPage = Boolean(category && isConfigCategory(category.id));
    const rawEntries = isCategoryPage
        ? category.groups.flatMap((group) => group.items)
            .filter((item) => !item.isPage)
            .map((item) => ({ sub: item, keys: keysForConfigSubItem(item) }))
            .filter((entry) => entry.keys.length > 0)
        : [];
    const requestedEntry = isCategoryPage
        ? rawEntries.find((entry) => entry.sub.id === context.subId) || rawEntries[0]
        : null;
    const sub = requestedEntry?.sub || requestedSub;

    if (!sub) return '<div class="dragon-empty-state"><p>未找到配置项</p></div>';

   const trainingContext = await loadTrainingContext();
   let currentValues = {};
   let configError = '';
   try {
       const res = await api(mergedConfigUrl(trainingContext));
       if (!res || res.ok === false) throw new Error(res?.error || '后端没有返回可用配置');
       currentValues = res.config || res;
   } catch (error) {
       configError = error.message || '读取训练配置失败';
   }

    const wrapper = document.createElement('div');
    wrapper.className = 'dragon-page';

    const pageCategory = category || findCategory(sub.categoryId);
    if (configError) {
        wrapper.innerHTML = renderConfigLoadError(pageCategory, sub, configError, trainingContext);
        return {
            html: wrapper.innerHTML,
            onMount: (root) => {
                root.querySelector('[data-config-action="retry"]')?.addEventListener('click', () => {
                    window.dispatchEvent(new CustomEvent('dragon-refresh-route'));
                });
            },
        };
    }

    const currentMethodFamily = activeMethodFamily(trainingContext, currentValues);
    const entries = isCategoryPage
        ? rawEntries.map((entry) => ({
            ...entry,
            keys: visibleConfigKeys(entry.keys, trainingContext, currentValues),
        })).filter((entry) => entry.keys.length > 0 && (entry.sub.id !== 'spd' || currentMethodFamily === 'spd'))
        : [];
    const activeEntry = isCategoryPage
        ? entries.find((entry) => entry.sub.id === context.subId) || entries[0]
        : null;
    const keys = visibleConfigKeys(activeEntry?.keys || keysForConfigSubItem(sub), trainingContext, currentValues);
    if (sub.id === 'spd' && currentMethodFamily !== 'spd') {
        return '<div class="dragon-empty-state"><p>SPD 是 CLI 实验配置，当前训练配置不会使用这些参数</p></div>';
    }
    if (!keys || keys.length === 0) {
        return '<div class="dragon-empty-state"><p>此分类暂无当前方法可生效的配置项</p></div>';
    }

    const pageState = { dirty: false, beforeUnload: null };
    if (isCategoryPage) {
        wrapper.innerHTML = renderCategorySubPage(pageCategory, entries, sub, keys, currentValues, trainingContext);
    } else {
        wrapper.innerHTML = renderSingleConfigPage(sub, keys, currentValues, trainingContext);
    }

    let routeUpdater = null;
    let disposeMountedPage = () => cleanupConfigPage(pageState);
    return {
        html: wrapper.innerHTML,
        onMount: (root) => {
            if (!isCategoryPage) {
                const saveChanges = wireConfigInteractions(root, keys, currentValues, trainingContext, pageState);
                const beforeContextChange = () => confirmConfigDiscard(pageState, '切换训练配置');
                bindTrainingControls(root, trainingContext, { saveChanges, beforeContextChange });
                return;
            }

            let committed = { context: trainingContext, sub, keys, values: currentValues };
            let requestedContext = trainingContext;
            let libraryController = null;
            let saveCurrentChanges = null;
            let transitionSequence = 0;
            const beforeContextChange = () => confirmConfigDiscard(pageState, '切换训练配置');
            disposeMountedPage = () => {
                transitionSequence += 1;
                libraryController?.destroy?.();
                cleanupConfigPage(pageState);
            };

            const bindEditablePane = () => {
                saveCurrentChanges = wireConfigInteractions(root, committed.keys, committed.values, committed.context, pageState);
                bindTrainingControls(root, committed.context, {
                    saveChanges: saveCurrentChanges,
                    beforeContextChange,
                    onConfigFileChange: (file) => {
                        const nextContext = selectTrainingConfigFile(committed.context, file, { notify: false, persist: false });
                        if (nextContext) transitionEditable({ context: nextContext });
                    },
                    onPresetChange: (preset) => {
                        transitionEditable({ context: selectTrainingPreset(committed.context, preset, { notify: false, persist: false }) });
                    },
                });
            };

            const transitionEditable = async ({ context = requestedContext, sub: nextSub = committed.sub } = {}) => {
                const target = entries.find((entry) => entry.sub.id === nextSub?.id) || entries[0];
                if (!target) return false;
                const targetSub = target.sub;
                const targetKeys = target.keys;
                const contextChanged = context.configFile !== committed.context.configFile
                    || context.preset !== committed.context.preset
                    || context.methodsSubdir !== committed.context.methodsSubdir
                    || context.variant !== committed.context.variant;
                if (!contextChanged && targetSub.id === committed.sub.id) return true;

                requestedContext = context;
                const sequence = ++transitionSequence;
                const pane = root.querySelector('[data-config-editable-pane]');
                pane?.setAttribute('aria-busy', 'true');
                try {
                    let values = committed.values;
                    if (contextChanged) {
                        const res = await api(mergedConfigUrl(context));
                        if (!res || res.ok === false) throw new Error(res?.error || '后端没有返回可用配置');
                        values = res.config || res;
                    }
                    if (sequence !== transitionSequence) return true;

                    pageState.dirty = false;
                    cleanupConfigPage(pageState);
                    committed = { context, sub: targetSub, keys: targetKeys, values };
                    requestedContext = context;
                    commitTrainingContext(context);
                    currentValues = values;
                    const currentPane = root.querySelector('[data-config-editable-pane]');
                    if (currentPane) currentPane.outerHTML = renderEditableConfigPane(pageCategory, entries, committed);
                    bindEditablePane();
                    scanForReveal();
                    libraryController?.updateContext(committed.context);
                    return true;
                } catch (error) {
                    if (sequence !== transitionSequence) return true;
                    requestedContext = committed.context;
                    libraryController?.updateContext(committed.context);
                    root.querySelector('[data-config-editable-pane]')?.removeAttribute('aria-busy');
                    window.alert(`切换训练配置失败：${error.message || error}`);
                    return false;
                }
            };

            bindEditablePane();
            libraryController = bindTrainingPresetLibrary(root, committed.context, {
                beforeContextChange,
                onSaveChanges: () => saveCurrentChanges?.() ?? false,
                onConfigFileChange: (_file, nextContext) => transitionEditable({ context: nextContext }),
            });

            routeUpdater = async ({ subId }) => {
                const target = entries.find((entry) => entry.sub.id === subId) || entries[0];
                if (!target) return false;
                const updated = await transitionEditable({ context: requestedContext, sub: target.sub });
                if (!updated) return false;
                const detail = root.querySelector(`[data-config-entry="${target.sub.id}"]`);
                const navHeight = Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--dragon-nav-height')) || 44;
                if (detail) window.scrollTo({ top: Math.max(0, window.scrollY + detail.getBoundingClientRect().top - navHeight - 16), behavior: 'smooth' });
                return true;
            };
        },
        onRouteUpdate: (route) => routeUpdater?.(route) ?? false,
        beforeLeave: () => confirmConfigDiscard(pageState, '离开页面'),
        onUnmount: () => disposeMountedPage(),
    };
}

function renderConfigLoadError(category, sub, error, trainingContext) {
    const title = category?.label || sub?.label || '训练配置';
    return `
        <div class="dragon-config-page">
            <div class="dragon-config-hero dragon-reveal">
                <span class="dragon-eyebrow">训练工作台</span>
                <h1>${escapeHtml(title)}</h1>
                <p>当前配置没有成功读取，因此不会展示默认值，也不会允许保存或启动训练。</p>
            </div>
            <section class="dragon-config-load-error dragon-reveal" role="alert">
                <span class="dragon-eyebrow">读取失败</span>
                <h2>无法安全打开训练配置</h2>
                <p>${escapeHtml(error)}</p>
                <p class="dragon-text-mono">${escapeHtml(trainingContext.configFile || '未选择配置文件')}</p>
                <button class="dragon-btn dragon-btn-primary" type="button" data-config-action="retry">重新读取配置</button>
            </section>
        </div>
    `;
}

function renderSingleConfigPage(sub, keys, currentValues, trainingContext) {
    return `
        <div class="dragon-config-page">
            <div class="dragon-config-hero dragon-reveal">
                <span class="dragon-eyebrow">${categoryLabel(sub)}</span>
                <h1>${sub.label}</h1>
                <p>${sub.desc || ''}</p>
            </div>
            ${renderTrainingControls(trainingContext)}
            <div class="dragon-reveal" data-stagger="1" id="dragon-config-fields">
                ${renderFields(sub.id, keys, currentValues)}
            </div>
            ${renderConfigActions()}
        </div>
    `;
}

function renderCategorySubPage(category, entries, sub, keys, currentValues, trainingContext) {
    const groupLabel = category.groups.find((group) =>
        group.items.some((item) => item.id === sub.id)
    )?.header || category.label;

    return `
        <div class="dragon-config-page dragon-config-category-page dragon-config-subpage"
             data-config-category="${category.id}" data-config-subpage="${sub.id}">
            <div class="dragon-config-hero dragon-reveal">
                <h1>${category.label}</h1>
                <p>${categoryDescription(category.id)}</p>
            </div>
            <div class="dragon-config-shell-layout">
                ${renderEditableConfigPane(category, entries, { context: trainingContext, sub, keys, values: currentValues })}
                ${renderTrainingPresetLibrary(trainingContext)}
            </div>
        </div>
    `;
}

function renderEditableConfigPane(category, entries, state) {
    return `<div class="dragon-config-editable-pane" data-config-editable-pane>
        ${renderTrainingControls(state.context)}
        ${renderEditableConfigWorkspace(category, entries, state.sub, state.keys, state.values)}
    </div>`;
}

function renderEditableConfigWorkspace(category, entries, sub, keys, currentValues) {
    const groupLabel = category.groups.find((group) =>
        group.items.some((item) => item.id === sub.id)
    )?.header || category.label;
    return `<div class="dragon-config-workspace" data-config-editable-workspace>
        ${renderConfigNavigation(category, entries, sub.id)}
        <section class="dragon-config-detail" data-config-entry="${sub.id}"
                 aria-labelledby="dragon-config-detail-title">
            <header class="dragon-config-detail-header">
                <div class="dragon-config-detail-header-copy">
                    <span class="dragon-eyebrow">${groupLabel}</span>
                    <h2 id="dragon-config-detail-title">${sub.label}</h2>
                    <p>${sub.desc || ''}</p>
                </div>
                ${sub.id === 'base-models' ? renderModelQuickPickerTrigger() : ''}
            </header>
            <div class="dragon-config-detail-fields dragon-reveal" data-stagger="1" id="dragon-config-fields">
                ${renderFields(sub.id, keys, currentValues)}
            </div>
            ${renderConfigActions()}
        </section>
        ${sub.id === 'base-models' ? renderModelQuickPickerDialog() : ''}
    </div>`;
}

function renderConfigNavigation(category, entries, activeId) {
    const availableIds = new Set(entries.map((entry) => entry.sub.id));
    const groupsHtml = category.groups.map((group) => {
        const links = group.items.filter((item) => availableIds.has(item.id)).map((item) => {
            const isActive = item.id === activeId;
            return `<a class="dragon-config-index-link" href="#config/${category.id}/${item.id}"
                       data-config-target="${item.id}" data-active="${isActive}"
                       ${isActive ? 'aria-current="page"' : ''} title="${escapeHtml(item.desc || item.label)}">
                        <span class="dragon-config-index-label">${escapeHtml(item.label)}</span>
                    </a>`;
        }).join('');
        if (!links) return '';
        return `<div class="dragon-config-index-group">
                    <div class="dragon-config-index-group-title">${escapeHtml(group.header)}</div>
                    <div class="dragon-config-index-links">${links}</div>
                </div>`;
    }).join('');

    return `<nav class="dragon-config-index dragon-reveal" aria-label="${escapeHtml(category.label)}导航">
                <div class="dragon-config-index-heading">配置分组</div>
                <div class="dragon-config-index-groups">${groupsHtml}</div>
            </nav>`;
}

function renderConfigActions() {
    return `
        <div class="dragon-config-actions dragon-config-actions-sticky">
            <button class="dragon-btn dragon-btn-secondary" type="button" id="dragon-config-reset">恢复默认</button>
            <button class="dragon-btn dragon-btn-primary" type="button" id="dragon-config-save">保存配置</button>
        </div>
        <div class="dragon-config-feedback" id="dragon-config-feedback" role="status" aria-live="polite"></div>
    `;
}

function renderFields(subId, keys, currentValues) {
    const groups = subId ? SECTION_GROUPS[subId] : null;

    if (groups && groups.length > 0) {
        const groupedKeys = new Set(groups.flatMap((section) => section.keys));
        const sectionsHtml = groups.map((section) => {
            const sectionKeys = section.keys.filter((k) => keys.includes(k));
            if (sectionKeys.length === 0) return '';
            const fieldsHtml = renderSectionFields(sectionKeys, currentValues);
            return `
                <div class="dragon-config-section">
                    <div class="dragon-config-section-header">
                        <span class="dragon-eyebrow">${section.eyebrow}</span>
                        <h2 class="dragon-config-section-title">${section.title}</h2>
                        <p class="dragon-config-section-desc">${section.desc}</p>
                    </div>
                    ${fieldsHtml}
                </div>
            `;
        }).join('');
        const remainingKeys = keys.filter((key) => !groupedKeys.has(key));
        if (!remainingKeys.length) return sectionsHtml;
        return `${sectionsHtml}
            <div class="dragon-config-section">
                <div class="dragon-config-section-header">
                    <span class="dragon-eyebrow">补充设置</span>
                    <h2 class="dragon-config-section-title">其他可用参数</h2>
                    <p class="dragon-config-section-desc">当前分类中尚未归入固定小节的参数。</p>
                </div>
                ${renderSectionFields(remainingKeys, currentValues)}
            </div>`;
    }

    return renderSectionFields(keys, currentValues);
}

/* Render fields within a section, using 2-col grid for compact inputs */
function renderSectionFields(keys, currentValues) {
    const toggleKeys = keys.filter((k) => {
            const v = displayConfigValue(k, currentValues);
        const options = FIELD_OPTIONS[k];
        return typeof v === 'boolean' && !options;
    });
    const inputKeys = keys.filter((k) => !toggleKeys.includes(k));

    let html = '';

   /* Keep related switches together so long configuration sections remain scannable. */
    if (toggleKeys.length > 0) {
        const toggleHtml = toggleKeys.map((key) =>
            renderField(key, displayConfigValue(key, currentValues))
        ).join('');
        html += toggleKeys.length > 1
            ? `<div class="dragon-toggle-grid">${toggleHtml}</div>`
            : toggleHtml;
    }

   /* Input/select fields: 2-column grid for compact layout */
   if (inputKeys.length > 0) {
       const gridHtml = inputKeys.map((key) =>
           renderField(key, displayConfigValue(key, currentValues))
       ).join('');
       if (inputKeys.length >= 2) {
           html += `<div class="dragon-field-grid-2">${gridHtml}</div>`;
       } else {
           html += gridHtml;
       }
   }

    return html;
}

function renderField(key, value) {
    const label = FIELD_LABEL_ZH[key] || key;
    const help = FIELD_HELP_ZH[key];
    const options = FIELD_OPTIONS[key];
    const helpSummary = help ? (help.summary || help['\u4f5c\u7528'] || '') : '';
    const fieldToken = configFieldToken(key);
    const fieldId = `dragon-config-field-${fieldToken}`;
    const name = `config_${String(key).replace(/[^A-Za-z0-9_-]+/g, '_')}`;
    const placeholder = `例如：${label}…`;

    let control;
    if (options) {
        control = `<select class="dragon-select" id="${fieldId}" name="${name}" autocomplete="off" data-key="${key}">
            ${options.map((opt) => `<option value="${escapeHtml(opt)}" ${String(value) === String(opt) ? 'selected' : ''}>${escapeHtml(opt)}</option>`).join('')}
        </select>`;
    } else if (typeof value === 'boolean') {
        control = `<div class="dragon-toggle-row">
            <div class="dragon-toggle" id="${fieldId}" data-key="${key}" data-checked="${value}" role="switch" tabindex="0" aria-checked="${value}" aria-label="${escapeHtml(label)}"></div>
            <div>
                <div class="dragon-toggle-label">${escapeHtml(label)}</div>
                ${helpSummary ? `<div class="dragon-toggle-desc">${escapeHtml(helpSummary)}</div>` : ''}
            </div>
        </div>`;
        return `<div class="dragon-field">${control}${help ? renderHelp(key, help, fieldToken) : ''}</div>`;
    } else if (key.includes('prompt') || key === 'optimizer_args' || key === 'network_args') {
        control = `<textarea class="dragon-textarea" id="${fieldId}" name="${name}" autocomplete="off" spellcheck="false" data-key="${key}" placeholder="${escapeHtml(placeholder)}">${escapeHtml(configValueForControl(value) || '')}</textarea>`;
    } else {
        const inputType = typeof value === 'number' ? 'number' : 'text';
        const inputMode = inputType === 'number' ? ' inputmode="decimal"' : '';
        control = `<input class="dragon-input" id="${fieldId}" name="${name}" type="${inputType}"${inputMode} autocomplete="off" spellcheck="false" data-key="${key}" value="${escapeHtml(value ?? '')}" placeholder="${escapeHtml(placeholder)}">`;
    }

    return `
        <div class="dragon-field">
            <div class="dragon-field-label">
               <label class="dragon-field-label-text" for="${fieldId}">${escapeHtml(label)}</label>
               ${help ? `<button class="dragon-field-help-btn" type="button" data-help-key="${key}" aria-expanded="false" aria-controls="dragon-config-help-${fieldToken}" aria-label="查看${escapeHtml(label)}说明" title="查看说明">?</button>` : ''}
            </div>
            ${control}
            ${help ? renderHelp(key, help, fieldToken) : ''}
        </div>
    `;
}

function renderHelp(key, help, fieldToken) {
    const summary = help.summary || help['\u4f5c\u7528'] || '';
    const why = help.why || help['\u4e3a\u4ec0\u4e48'] || '';
    const helpId = `dragon-config-help-${fieldToken}`;
    return `
        <div class="dragon-field-help" id="${helpId}" data-help-key="${key}">
            ${summary ? `<div class="dragon-field-help-section"><div class="dragon-field-help-heading">\u4f5c\u7528</div><p>${escapeHtml(summary)}</p></div>` : ''}
            ${why ? `<div class="dragon-field-help-section"><div class="dragon-field-help-heading">\u4e3a\u4ec0\u4e48</div><p>${escapeHtml(why)}</p></div>` : ''}
        </div>
    `;
}

function configFieldToken(key) {
    const scope = String(key).replace(/[^A-Za-z0-9_-]+/g, '-');
    return `${scope}-${configFieldToken.counter++}`;
}
configFieldToken.counter = 0;

function wireConfigInteractions(wrapper, keys, originalValues, trainingContext, state) {
    const syncDirty = () => {
        state.dirty = Object.keys(prepareConfigPatch(
            collectChangedValues(wrapper, keys, originalValues),
            originalValues,
        )).length > 0;
    };
    state.beforeUnload = (event) => {
        if (!state.dirty) return;
        event.preventDefault();
        event.returnValue = '';
    };
    window.addEventListener('beforeunload', state.beforeUnload);

    wrapper.querySelectorAll('.dragon-toggle').forEach((toggle) => {
        toggle.addEventListener('click', () => {
            const checked = toggle.dataset.checked === 'true';
            toggle.dataset.checked = String(!checked);
            toggle.setAttribute('aria-checked', String(!checked));
            syncDirty();
        });
        toggle.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggle.click();
            }
        });
    });

    wrapper.querySelectorAll('.dragon-field-help-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            const helpEl = wrapper.querySelector(`#${btn.getAttribute('aria-controls')}`);
            if (helpEl) {
                const isOpen = helpEl.dataset.open === 'true';
                helpEl.dataset.open = String(!isOpen);
                btn.setAttribute('aria-expanded', String(!isOpen));
            }
        });
    });

    const saveBtn = wrapper.querySelector('#dragon-config-save');
    const resetBtn = wrapper.querySelector('#dragon-config-reset');
    const feedbackEl = wrapper.querySelector('#dragon-config-feedback');
    wrapper.querySelectorAll('#dragon-config-fields input, #dragon-config-fields select, #dragon-config-fields textarea').forEach((field) => {
        field.addEventListener('input', syncDirty);
        field.addEventListener('change', syncDirty);
    });

    const saveChanges = async ({ quiet = false } = {}) => {
        const changedValues = prepareConfigPatch(
            collectChangedValues(wrapper, keys, originalValues),
            originalValues,
        );
        if (Object.keys(changedValues).length === 0) {
            if (!quiet) showFeedback(feedbackEl, '没有修改', 'info');
            return true;
        }

        if (!isEditableConfigFile(trainingContext)) {
            showFeedback(feedbackEl, '当前训练配置为系统只读，无法保存修改；请先在左侧预设库选择可编辑配置', 'error');
            return false;
        }

        saveBtn.disabled = true;
        saveBtn.textContent = '保存中…';

        try {
            if (!trainingContext.configFile) {
                showFeedback(feedbackEl, '未找到可保存的配置文件', 'error');
                return false;
            }

            const res = await api('/api/config/raw', {
                method: 'PATCH',
                body: JSON.stringify({ file: trainingContext.configFile, values: changedValues }),
            });

            if (res.ok !== false) {
                showFeedback(feedbackEl, '配置已保存', 'success');
                Object.assign(originalValues, changedValues);
                state.dirty = false;
                return true;
            } else {
                showFeedback(feedbackEl, res.error || '保存失败', 'error');
                return false;
            }
        } catch (err) {
            showFeedback(feedbackEl, '保存失败: ' + err.message, 'error');
            return false;
        } finally {
            saveBtn.disabled = false;
            saveBtn.textContent = '保存配置';
        }
    };
    saveBtn?.addEventListener('click', () => saveChanges());

    resetBtn?.addEventListener('click', () => {
        keys.forEach((key) => {
            const defaultValue = FORM_UI_DEFAULTS[key];
            const input = wrapper.querySelector(`[data-key="${key}"]`);
            if (!input) return;

            if (input.classList.contains('dragon-toggle')) {
                input.dataset.checked = String(Boolean(defaultValue));
                input.setAttribute('aria-checked', String(Boolean(defaultValue)));
            } else if (input.tagName === 'SELECT') {
                input.value = defaultValue || '';
            } else if (input.tagName === 'TEXTAREA') {
                input.value = defaultValue || '';
            } else {
                input.value = defaultValue ?? '';
            }
        });
        syncDirty();
        showFeedback(feedbackEl, '已恢复默认值（需点击保存才会生效）', 'info');
    });
    bindModelQuickPicker(wrapper, {
        onApply: (item) => {
            MODEL_QUICK_PATH_KEYS.forEach((key) => {
                const input = wrapper.querySelector(`[data-key="${key}"]`);
                if (!input) return;
                input.value = item[key] || '';
                input.dispatchEvent(new Event('input', { bubbles: true }));
            });
            showFeedback(feedbackEl, `已应用“${item.name || '未命名配置'}”（需点击保存才会生效）`, 'success');
        },
    });
    return () => saveChanges({ quiet: true });
}

function confirmConfigDiscard(state, action) {
    if (!state.dirty) return true;
    return window.confirm(`当前训练配置有未保存修改。${action}会丢弃这些修改，是否继续？`);
}

function cleanupConfigPage(state) {
    if (state.beforeUnload) window.removeEventListener('beforeunload', state.beforeUnload);
}

function collectChangedValues(wrapper, keys, originalValues) {
    const changed = {};
    for (const key of keys) {
        const input = wrapper.querySelector(`[data-key="${key}"]`);
        if (!input) continue;

        const original = displayConfigValue(key, originalValues);
        const currentValue = serializeConfigValue(input, original);
        if (String(currentValue) !== String(original ?? '')) {
            changed[key] = currentValue;
        }
    }
    return changed;
}

function showFeedback(el, message, tone) {
    if (!el) return;
    el.textContent = message;
    el.dataset.tone = tone;
    el.classList.add('dragon-config-feedback-visible');
    setTimeout(() => el.classList.remove('dragon-config-feedback-visible'), 3000);
}
