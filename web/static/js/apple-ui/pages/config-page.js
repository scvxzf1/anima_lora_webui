/* Generic config page renderer.
 * Renders a config sub-page with fields, Apple-style spacing,
 * help text, and save/restore actions.
 * Uses the existing config catalog (form-layout.js, labels-options.js, etc.)
 * Language: Chinese labels and descriptions, English config keys hidden by default.
 */

import { FIELD_LABEL_ZH, FIELD_OPTIONS } from '../../config/catalog/labels-options.js?v=apple-ui-20260812v33';
import { FIELD_HELP_ZH } from '../../config/catalog/field-help.js?v=apple-ui-20260812v33';
import { FORM_UI_DEFAULTS } from '../../config/catalog/defaults.js?v=apple-ui-20260812v33';
import { createApiClient } from '../../shared/api.js?v=apple-ui-20260812v33';
import { SECTION_GROUPS } from './section-groups.js?v=apple-ui-20260812v33';
import { findCategory, isConfigCategory } from '../category-map.js?v=apple-ui-20260812v33';
import { keysForConfigSubItem } from './config-field-map.js?v=apple-ui-20260812v33';
import { configValueForControl, displayConfigValue, prepareConfigPatch, serializeConfigValue } from './config-values.js?v=apple-ui-20260812v33';
import { bindTrainingControls, loadTrainingContext, mergedConfigUrl, renderTrainingControls } from './training-controls.js?v=apple-ui-20260812v33';

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

    if (!isCategoryPage && !sub) return '<div class="apple-empty-state"><p>未找到配置项</p></div>';

    const keys = isCategoryPage
        ? [...new Set(entries.flatMap((entry) => entry.keys))]
        : keysForConfigSubItem(sub);
    if (!keys || keys.length === 0) {
        return '<div class="apple-empty-state"><p>此分类暂无配置项映射</p></div>';
    }

   const trainingContext = await loadTrainingContext();
   let currentValues = {};
   try {
       const res = await api(mergedConfigUrl(trainingContext));
       if (res && res.ok !== false) currentValues = res.config || res;
   } catch { /* use defaults */ }

    const wrapper = document.createElement('div');
    wrapper.className = 'apple-page';

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
        <div class="apple-config-page">
            <div class="apple-config-hero apple-reveal">
                <span class="apple-eyebrow">${categoryLabel(sub)}</span>
                <h1>${sub.label}</h1>
                <p>${sub.desc || ''}</p>
            </div>
            ${renderTrainingControls(trainingContext)}
            <div class="apple-reveal" data-stagger="1" id="apple-config-fields">
                ${renderFields(sub.id, keys, currentValues)}
            </div>
            ${renderConfigActions()}
        </div>
    `;
}

function renderCategoryPage(category, entries, keys, currentValues, trainingContext) {
    const indexHtml = entries.map(({ sub }) => `
        <a class="apple-config-index-link" href="#apple-config-anchor-${sub.id}" data-config-target="${sub.id}">
            <span class="apple-config-index-label">${sub.label}</span>
            <span class="apple-config-index-desc">${sub.desc || ''}</span>
        </a>
    `).join('');
    const entriesHtml = entries.map(({ sub, keys: entryKeys }) => `
        <section class="apple-config-entry" id="apple-config-anchor-${sub.id}" data-config-entry="${sub.id}">
            <header class="apple-config-entry-header">
                <span class="apple-eyebrow">${category.label}</span>
                <h2>${sub.label}</h2>
                <p>${sub.desc || ''}</p>
            </header>
            <div class="apple-config-entry-fields">
                ${renderFields(sub.id, entryKeys, currentValues)}
            </div>
        </section>
    `).join('');

    return `
        <div class="apple-config-page apple-config-category-page" data-config-category="${category.id}">
            <div class="apple-config-hero apple-reveal">
                <span class="apple-eyebrow">训练工作台</span>
                <h1>${category.label}</h1>
                <p>${categoryDescription(category.id)}</p>
            </div>
            ${renderTrainingControls(trainingContext)}
            <nav class="apple-config-index apple-reveal" aria-label="${category.label}目录">
                <div class="apple-config-index-heading">本页目录</div>
                <div class="apple-config-index-grid">${indexHtml}</div>
            </nav>
            <div class="apple-config-entry-list" id="apple-config-fields">
                ${entriesHtml}
            </div>
            ${renderConfigActions()}
        </div>
    `;
}

function renderConfigActions() {
    return `
        <div class="apple-config-actions apple-config-actions-sticky">
            <button class="apple-btn apple-btn-secondary" type="button" id="apple-config-reset">恢复默认</button>
            <button class="apple-btn apple-btn-primary" type="button" id="apple-config-save">保存配置</button>
        </div>
        <div class="apple-config-feedback" id="apple-config-feedback"></div>
    `;
}

function bindConfigIndex(root, categoryId) {
    root.querySelectorAll('.apple-config-index-link').forEach((link) => {
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
        getComputedStyle(document.documentElement).getPropertyValue('--apple-nav-height')
    ) || 44;
    const topOffset = navHeight + 16;
    const top = Math.max(0, window.scrollY + element.getBoundingClientRect().top - topOffset);
    window.scrollTo({ top, behavior });
}

function setActiveConfigIndex(root, subId) {
    root.querySelectorAll('.apple-config-index-link').forEach((link) => {
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
                <div class="apple-config-section">
                    <div class="apple-config-section-header">
                        <span class="apple-eyebrow">${section.eyebrow}</span>
                        <h2 class="apple-config-section-title">${section.title}</h2>
                        <p class="apple-config-section-desc">${section.desc}</p>
                    </div>
                    ${fieldsHtml}
                </div>
            `;
        }).join('');
        const remainingKeys = keys.filter((key) => !groupedKeys.has(key));
        if (!remainingKeys.length) return sectionsHtml;
        return `${sectionsHtml}
            <div class="apple-config-section">
                <div class="apple-config-section-header">
                    <span class="apple-eyebrow">补充设置</span>
                    <h2 class="apple-config-section-title">其他可用参数</h2>
                    <p class="apple-config-section-desc">当前分类中尚未归入固定小节的参数。</p>
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
            ? `<div class="apple-toggle-grid">${toggleHtml}</div>`
            : toggleHtml;
    }

   /* Input/select fields: 2-column grid for compact layout */
   if (inputKeys.length > 0) {
       const gridHtml = inputKeys.map((key) =>
           renderField(key, displayConfigValue(key, currentValues))
       ).join('');
       if (inputKeys.length >= 2) {
           html += `<div class="apple-field-grid-2">${gridHtml}</div>`;
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
        control = `<select class="apple-select" data-key="${key}">
            ${options.map((opt) => `<option value="${opt}" ${String(value) === String(opt) ? 'selected' : ''}>${opt}</option>`).join('')}
        </select>`;
    } else if (typeof value === 'boolean') {
        control = `<div class="apple-toggle-row">
            <div class="apple-toggle" data-key="${key}" data-checked="${value}" role="switch" tabindex="0"></div>
            <div>
                <div class="apple-toggle-label">${label}</div>
                ${helpSummary ? `<div class="apple-toggle-desc">${helpSummary}</div>` : ''}
            </div>
        </div>`;
        return `<div class="apple-field">${control}${help ? renderHelp(key, help) : ''}</div>`;
    } else if (key.includes('prompt') || key === 'optimizer_args' || key === 'network_args') {
        control = `<textarea class="apple-textarea" data-key="${key}" placeholder="${label}">${configValueForControl(value) || ''}</textarea>`;
    } else {
        const inputType = typeof value === 'number' ? 'number' : 'text';
        control = `<input class="apple-input" type="${inputType}" data-key="${key}" value="${value ?? ''}" placeholder="${label}">`;
    }

    return `
        <div class="apple-field">
            <div class="apple-field-label">
               <span class="apple-field-label-text">${label}</span>
               ${help ? `<button class="apple-field-help-btn" type="button" data-help-key="${key}" title="查看说明">?</button>` : ''}
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
        <div class="apple-field-help" data-help-key="${key}">
            ${summary ? `<div class="apple-field-help-section"><div class="apple-field-help-heading">\u4f5c\u7528</div><p>${summary}</p></div>` : ''}
            ${why ? `<div class="apple-field-help-section"><div class="apple-field-help-heading">\u4e3a\u4ec0\u4e48</div><p>${why}</p></div>` : ''}
        </div>
    `;
}

function wireConfigInteractions(wrapper, keys, originalValues, trainingContext) {
    wrapper.querySelectorAll('.apple-toggle').forEach((toggle) => {
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

    wrapper.querySelectorAll('.apple-field-help-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            const key = btn.dataset.helpKey;
            const helpEl = wrapper.querySelector(`.apple-field-help[data-help-key="${key}"]`);
            if (helpEl) {
                const isOpen = helpEl.dataset.open === 'true';
                helpEl.dataset.open = String(!isOpen);
            }
        });
    });

    const saveBtn = wrapper.querySelector('#apple-config-save');
    const resetBtn = wrapper.querySelector('#apple-config-reset');
    const feedbackEl = wrapper.querySelector('#apple-config-feedback');

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

            if (input.classList.contains('apple-toggle')) {
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
    el.classList.add('apple-config-feedback-visible');
    setTimeout(() => el.classList.remove('apple-config-feedback-visible'), 3000);
}
