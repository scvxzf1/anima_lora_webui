/* Generic config page renderer.
 * Renders a config sub-page with fields, generous spacing,
 * help text, and save/restore actions.
 * Uses the existing config catalog (form-layout.js, labels-options.js, etc.)
 * Language: Chinese labels and descriptions, English config keys hidden by default.
 */

import { FIELD_LABEL_ZH, FIELD_OPTIONS } from '../../config/catalog/labels-options.js?v=dragon-ui-20260812v35';
import { FIELD_HELP_ZH } from '../../config/catalog/field-help.js?v=dragon-ui-20260812v35';
import { FORM_UI_DEFAULTS } from '../../config/catalog/defaults.js?v=dragon-ui-20260812v35';
import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import { SECTION_GROUPS } from './section-groups.js?v=dragon-ui-20260812v35';
import { findCategory, isConfigCategory } from '../category-map.js?v=dragon-ui-20260812v35';
import { keysForConfigSubItem } from './config-field-map.js?v=dragon-ui-20260812v35';
import { configValueForControl, displayConfigValue, prepareConfigPatch, serializeConfigValue } from './config-values.js?v=dragon-ui-20260812v35';
import { bindTrainingControls, loadTrainingContext, mergedConfigUrl, renderTrainingControls } from './training-controls.js?v=dragon-ui-20260812v35';

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

export async function loadConfigPage(context) {
    const sub = context.sub;
    const category = findCategory(context.categoryId || sub?.categoryId);
    const isCategoryPage = Boolean(category && isConfigCategory(category.id));
    const entries = isCategoryPage
        ? category.groups.flatMap((group) => group.items)
            .filter((item) => !item.isPage)
            .map((item) => ({ sub: item, keys: keysForConfigSubItem(item) }))
            .filter((entry) => entry.keys.length > 0)
        : [];

    if (!isCategoryPage && !sub) return '<div class="dragon-empty-state"><p>未找到配置项</p></div>';

    const keys = isCategoryPage
        ? [...new Set(entries.flatMap((entry) => entry.keys))]
        : keysForConfigSubItem(sub);
    if (!keys || keys.length === 0) {
        return '<div class="dragon-empty-state"><p>此分类暂无配置项映射</p></div>';
    }

   const trainingContext = await loadTrainingContext();
   let currentValues = {};
   try {
       const res = await api(mergedConfigUrl(trainingContext));
       if (res && res.ok !== false) currentValues = res.config || res;
   } catch { /* use defaults */ }

    const wrapper = document.createElement('div');
    wrapper.className = 'dragon-page';

    const pageCategory = category || findCategory(sub.categoryId);
    if (isCategoryPage) {
        wrapper.innerHTML = renderCategoryPage(pageCategory, entries, keys, currentValues, trainingContext);
    } else {
        wrapper.innerHTML = renderSingleConfigPage(sub, keys, currentValues, trainingContext);
    }

    return {
        html: wrapper.innerHTML,
        onMount: (root) => {
            const saveChanges = wireConfigInteractions(root, keys, currentValues, trainingContext);
            bindTrainingControls(root, trainingContext, { saveChanges });
            if (isCategoryPage) {
                bindConfigIndex(root, pageCategory.id);
                scrollToConfigEntry(root, context.subId);
            }
        },
    };
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

function renderCategoryPage(category, entries, keys, currentValues, trainingContext) {
    const indexHtml = entries.map(({ sub }) => `
        <a class="dragon-config-index-link" href="#dragon-config-anchor-${sub.id}" data-config-target="${sub.id}">
            <span class="dragon-config-index-label">${sub.label}</span>
            <span class="dragon-config-index-desc">${sub.desc || ''}</span>
        </a>
    `).join('');
    const entriesHtml = entries.map(({ sub, keys: entryKeys }) => `
        <section class="dragon-config-entry" id="dragon-config-anchor-${sub.id}" data-config-entry="${sub.id}">
            <header class="dragon-config-entry-header">
                <span class="dragon-eyebrow">${category.label}</span>
                <h2>${sub.label}</h2>
                <p>${sub.desc || ''}</p>
            </header>
            <div class="dragon-config-entry-fields">
                ${renderFields(sub.id, entryKeys, currentValues)}
            </div>
        </section>
    `).join('');

    return `
        <div class="dragon-config-page dragon-config-category-page" data-config-category="${category.id}">
            <div class="dragon-config-hero dragon-reveal">
                <span class="dragon-eyebrow">训练工作台</span>
                <h1>${category.label}</h1>
                <p>${categoryDescription(category.id)}</p>
            </div>
            ${renderTrainingControls(trainingContext)}
            <nav class="dragon-config-index dragon-reveal" aria-label="${category.label}目录">
                <div class="dragon-config-index-heading">本页目录</div>
                <div class="dragon-config-index-grid">${indexHtml}</div>
            </nav>
            <div class="dragon-config-entry-list" id="dragon-config-fields">
                ${entriesHtml}
            </div>
            ${renderConfigActions()}
        </div>
    `;
}

function renderConfigActions() {
    return `
        <div class="dragon-config-actions dragon-config-actions-sticky">
            <button class="dragon-btn dragon-btn-secondary" type="button" id="dragon-config-reset">恢复默认</button>
            <button class="dragon-btn dragon-btn-primary" type="button" id="dragon-config-save">保存配置</button>
        </div>
        <div class="dragon-config-feedback" id="dragon-config-feedback"></div>
    `;
}

function bindConfigIndex(root, categoryId) {
    root.querySelectorAll('.dragon-config-index-link').forEach((link) => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            const targetId = link.dataset.configTarget;
            const target = root.querySelector(`[data-config-entry="${targetId}"]`);
            if (!target) return;
            scrollToElement(target, 'smooth');
            history.replaceState(null, '', `#config/${categoryId}/${targetId}`);
            setActiveConfigIndex(root, targetId);
        });
    });
}

function scrollToConfigEntry(root, subId) {
    if (!subId) return;
    // Wait for the page-enter motion to settle before measuring the anchor.
    // Otherwise the wrapper's translateY shifts the selected heading under the nav.
    window.setTimeout(() => {
        const target = root.querySelector(`[data-config-entry="${subId}"]`);
        if (target) {
            scrollToElement(target, 'auto');
            setActiveConfigIndex(root, subId);
        }
    }, 760);
}

function scrollToElement(element, behavior) {
    const navHeight = Number.parseFloat(
        getComputedStyle(document.documentElement).getPropertyValue('--dragon-nav-height')
    ) || 44;
    const topOffset = navHeight + 16;
    const top = Math.max(0, window.scrollY + element.getBoundingClientRect().top - topOffset);
    window.scrollTo({ top, behavior });
}

function setActiveConfigIndex(root, subId) {
    root.querySelectorAll('.dragon-config-index-link').forEach((link) => {
        const isActive = link.dataset.configTarget === subId;
        link.dataset.active = String(isActive);
        if (isActive) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
    });
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

    let control;
    if (options) {
        control = `<select class="dragon-select" data-key="${key}">
            ${options.map((opt) => `<option value="${opt}" ${String(value) === String(opt) ? 'selected' : ''}>${opt}</option>`).join('')}
        </select>`;
    } else if (typeof value === 'boolean') {
        control = `<div class="dragon-toggle-row">
            <div class="dragon-toggle" data-key="${key}" data-checked="${value}" role="switch" tabindex="0"></div>
            <div>
                <div class="dragon-toggle-label">${label}</div>
                ${helpSummary ? `<div class="dragon-toggle-desc">${helpSummary}</div>` : ''}
            </div>
        </div>`;
        return `<div class="dragon-field">${control}${help ? renderHelp(key, help) : ''}</div>`;
    } else if (key.includes('prompt') || key === 'optimizer_args' || key === 'network_args') {
        control = `<textarea class="dragon-textarea" data-key="${key}" placeholder="${label}">${configValueForControl(value) || ''}</textarea>`;
    } else {
        const inputType = typeof value === 'number' ? 'number' : 'text';
        control = `<input class="dragon-input" type="${inputType}" data-key="${key}" value="${value ?? ''}" placeholder="${label}">`;
    }

    return `
        <div class="dragon-field">
            <div class="dragon-field-label">
               <span class="dragon-field-label-text">${label}</span>
               ${help ? `<button class="dragon-field-help-btn" type="button" data-help-key="${key}" title="查看说明">?</button>` : ''}
            </div>
            ${control}
            ${help ? renderHelp(key, help) : ''}
        </div>
    `;
}

function renderHelp(key, help) {
    const summary = help.summary || help['\u4f5c\u7528'] || '';
    const why = help.why || help['\u4e3a\u4ec0\u4e48'] || '';
    return `
        <div class="dragon-field-help" data-help-key="${key}">
            ${summary ? `<div class="dragon-field-help-section"><div class="dragon-field-help-heading">\u4f5c\u7528</div><p>${summary}</p></div>` : ''}
            ${why ? `<div class="dragon-field-help-section"><div class="dragon-field-help-heading">\u4e3a\u4ec0\u4e48</div><p>${why}</p></div>` : ''}
        </div>
    `;
}

function wireConfigInteractions(wrapper, keys, originalValues, trainingContext) {
    wrapper.querySelectorAll('.dragon-toggle').forEach((toggle) => {
        toggle.addEventListener('click', () => {
            const checked = toggle.dataset.checked === 'true';
            toggle.dataset.checked = String(!checked);
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
            const key = btn.dataset.helpKey;
            const helpEl = wrapper.querySelector(`.dragon-field-help[data-help-key="${key}"]`);
            if (helpEl) {
                const isOpen = helpEl.dataset.open === 'true';
                helpEl.dataset.open = String(!isOpen);
            }
        });
    });

    const saveBtn = wrapper.querySelector('#dragon-config-save');
    const resetBtn = wrapper.querySelector('#dragon-config-reset');
    const feedbackEl = wrapper.querySelector('#dragon-config-feedback');

    const saveChanges = async ({ quiet = false } = {}) => {
        const changedValues = prepareConfigPatch(
            collectChangedValues(wrapper, keys, originalValues),
            originalValues,
        );
        if (Object.keys(changedValues).length === 0) {
            if (!quiet) showFeedback(feedbackEl, '没有修改', 'info');
            return true;
        }

        saveBtn.disabled = true;
        saveBtn.textContent = '保存中...';

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
            } else if (input.tagName === 'SELECT') {
                input.value = defaultValue || '';
            } else if (input.tagName === 'TEXTAREA') {
                input.value = defaultValue || '';
            } else {
                input.value = defaultValue ?? '';
            }
        });
        showFeedback(feedbackEl, '已恢复默认值（需点击保存才会生效）', 'info');
    });
    return () => saveChanges({ quiet: true });
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
