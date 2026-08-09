/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
import {
    createNoDatasetRegularizationAdvancedFields,
    createNoDatasetRegularizationModePanel,
} from './no-dataset-regularization.js?v=module-bootstrap-20260809-nf4-v2';
import {
    ADVANCED_CATEGORY_DEFAULT_OPEN_GROUPS,
    FORM_CATEGORY_DEFS,
    FORM_CATEGORY_SECTION_MAP,
    STICKY_CONFIG_CATEGORY_IDS,
} from '../../config/catalog.js?v=module-bootstrap-20260809-nf4-v2';
import { reloadCurrentConfig, renderConfigForm, syncConfigDraftFromForm } from '../anima-app/helpers/app-shell-startup-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { getConfigState } from '../anima-app/helpers/config-state-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { formatFieldName } from '../anima-app/helpers/config-field-display.js?v=module-bootstrap-20260809-nf4-v2';
import { updateChangedFieldMarks } from '../anima-app/helpers/toml-selection-bridge.js?v=module-bootstrap-20260809-nf4-v2';
import { createStepEstimatePanel, scheduleStepEstimatePanelRefresh } from './step-estimate.js?v=module-bootstrap-20260809-nf4-v2';
import {
    createFillGlobalModelPathsButton,
    createNoDatasetRegularizationQuickPresetPanel,
    createNoDatasetRegularizationQuickPresetsButton,
    createResourceQuickPresetPanel,
    createResourceQuickPresetsButton,
} from './stage-resolution.js?v=module-bootstrap-20260809-nf4-v2';
import { appendFieldRows } from './field-rows.js?v=module-bootstrap-20260809-nf4-v2';
import { createConfigDatasetPicker } from './dataset-picker.js?v=module-bootstrap-20260809-nf4-v2';
import { debounce } from '../../shared/debounce.js?v=module-bootstrap-20260809-nf4-v2';

const configState = getConfigState();
const configFormState = configState.configFormState;
const stageResolutionState = configState.stageResolutionState;

const applyConfigSearch = debounce((rawValue) => {
    syncConfigDraftFromForm();
    configFormState.search = rawValue || '';
    renderConfigForm(currentConfigState());
    requestAnimationFrame(() => {
        const next = document.getElementById('config-search-input');
        if (next) {
            next.focus();
            const length = next.value.length;
            next.setSelectionRange(length, length);
        }
    });
}, 150);

function currentConfigState() {
    return configState.currentConfig || {};
}

export function createConfigGroupEntry(name, fields, extraClass = '', description = '', defaultOpen = undefined, notice = '') {
    const categoryId = FORM_CATEGORY_SECTION_MAP.get(name) || 'advanced';
    return {
        name,
        fields,
        extraClass,
        description,
        categoryId,
        defaultOpen,
        notice,
    };
}

export function appendConfigGroupsByCategory(container, groups) {
    if (!groups.length) {
        container.appendChild(createConfigFormEmpty('当前配置没有可编辑字段。'));
        return;
    }
    const buckets = new Map(FORM_CATEGORY_DEFS.map((category) => [category.id, []]));
    for (const group of groups) {
        const categoryId = group.categoryId || FORM_CATEGORY_SECTION_MAP.get(group.name) || 'advanced';
        const bucket = buckets.get(categoryId) || buckets.get('advanced');
        bucket.push(group);
    }

    const searchText = normalizeConfigSearch(configFormState.search);
    const categories = FORM_CATEGORY_DEFS.filter((category) => {
        const categoryGroups = buckets.get(category.id) || [];
        return categoryGroups.length && configCategoryVisible(category, searchText);
    });
    const activeCategory = normalizeConfigActiveCategory(categories);
    updateConfigStickyDirectory(categories, buckets, activeCategory, searchText);
    updateConfigStickyPlacement();
    const renderedGroups = [];
    const sourceCategories = categories.filter((category) => category.id === activeCategory);
    const scopedGroups = sourceCategories.flatMap((category) => buckets.get(category.id) || []);
    for (const category of FORM_CATEGORY_DEFS) {
        if (!sourceCategories.some((item) => item.id === category.id)) continue;
        for (const group of buckets.get(category.id) || []) {
            const filtered = filterConfigGroupEntry(group, searchText);
            if (filtered) renderedGroups.push(filtered);
        }
    }

    const shell = document.createElement('div');
    shell.className = ['config-form-shell', searchText ? 'searching' : '', configFormState.showAdvanced ? 'advanced-visible' : 'basic-only'].filter(Boolean).join(' ');

    const main = document.createElement('div');
    main.className = 'config-form-main';
    main.appendChild(createConfigFormControls(scopedGroups, renderedGroups, searchText));
    const groupList = document.createElement('div');
    groupList.className = 'config-form-group-list';
    if (!renderedGroups.length) {
        groupList.appendChild(createConfigFormEmpty(searchText ? '没有匹配的配置项。' : '这个分类暂无可编辑项。'));
    } else {
        for (const group of renderedGroups) {
            groupList.appendChild(createGroup(
                group.name,
                group.fields,
                group.extraClass,
                group.description,
                searchText,
                group.defaultOpen,
                group.notice
            ));
        }
    }
    main.appendChild(groupList);
    shell.appendChild(main);
    container.appendChild(shell);
    updateChangedFieldMarks();
    requestAnimationFrame(updateConfigStickyPlacement);
}

    function normalizeConfigSearch(value) {
        return String(value || '').trim().toLowerCase();
    }

    function configCategoryVisible(category, searchText = '') {
        return Boolean(searchText) || configFormState.showAdvanced || !category.advanced;
    }

    function normalizeConfigActiveCategory(categories) {
        if (!categories.length) return '';
        const ids = new Set(categories.map((category) => category.id));
        if (!ids.has(configFormState.activeCategory)) {
            configFormState.activeCategory = categories[0].id;
        }
        return configFormState.activeCategory;
    }

export function selectConfigCategory(categoryId, options = {}) {
    if (!categoryId) return;
    const category = FORM_CATEGORY_DEFS.find((item) => item.id === categoryId);
    syncConfigDraftFromForm();
    if (category?.advanced) {
        configFormState.showAdvanced = true;
    }
    configFormState.activeCategory = categoryId;
    renderConfigForm(currentConfigState());
    if (options.scrollToForm) {
        requestAnimationFrame(() => scrollConfigFormContentToTop('smooth'));
    }
}

    function scrollConfigFormContentToTop(behavior = 'auto') {
        const scroller = document.querySelector('#tab-config .config-left');
        if (!scroller) return;
        scroller.scrollTo({ top: 0, behavior });
    }

    function updateConfigStickyDirectory(categories, buckets, activeCategory, searchText) {
        const visibleCategories = new Set(categories.map((category) => category.id));
        document.querySelectorAll('[data-sticky-config-category]').forEach((btn) => {
            const categoryId = btn.dataset.stickyConfigCategory || '';
            const category = FORM_CATEGORY_DEFS.find((item) => item.id === categoryId);
            const categoryGroups = buckets.get(categoryId) || [];
            const hasFields = categoryGroups.length > 0;
            const enabled = Boolean(category && STICKY_CONFIG_CATEGORY_IDS.has(categoryId) && (visibleCategories.has(categoryId) || (category.advanced && hasFields)));
            const fieldCount = categoryGroups.reduce((sum, group) => sum + group.fields.length, 0);
            btn.hidden = !category || !STICKY_CONFIG_CATEGORY_IDS.has(categoryId);
            btn.disabled = !enabled;
            btn.classList.toggle('active', enabled && categoryId === activeCategory);
            btn.setAttribute('aria-current', enabled && categoryId === activeCategory ? 'true' : 'false');
            btn.title = enabled ? `切换到${category.title}配置` : '当前配置没有这个分类';
            const count = btn.querySelector('em');
            if (count) count.textContent = `${fieldCount} 项`;
        });
    }

export function updateConfigStickyPlacement() {
    const bar = document.getElementById('config-sticky-actions');
    const workspace = document.getElementById('config-form-workspace');
    if (!bar || !workspace || workspace.hidden) return;
    const rect = workspace.getBoundingClientRect();
    if (rect.width <= 0) return;
    const sidePadding = Math.min(16, Math.max(0, rect.width / 18));
    const maxWidth = Math.max(0, rect.width - sidePadding * 2);
    const width = Math.min(1040, maxWidth);
    const left = rect.left + Math.max(sidePadding, (rect.width - width) / 2);
    bar.style.setProperty('--config-sticky-left', `${Math.round(left)}px`);
    bar.style.setProperty('--config-sticky-width', `${Math.round(width)}px`);

    const scroller = workspace.querySelector('.config-left');
    if (!scroller) return;
    const barRect = bar.getBoundingClientRect();
    const barStyle = window.getComputedStyle(bar);
    const bottomOffset = Number.parseFloat(barStyle.bottom) || 20;
    // 滚动区底边停在悬浮栏上方；底部内容垫高按栏高 * 1.3，避免最后一项被盖住。
    const barHeight = Math.max(Math.ceil(barRect.height), 48);
    const barReserve = Math.ceil((barHeight + bottomOffset + 20) * 1.3);
    const safeSpace = Math.ceil((barHeight + bottomOffset + 56) * 1.3);
    const scrollerRect = scroller.getBoundingClientRect();
    const availableHeight = Math.max(180, Math.floor(window.innerHeight - scrollerRect.top - barReserve));
    workspace.style.setProperty('--config-sticky-safe-space', `${safeSpace}px`);
    workspace.style.setProperty('--config-left-max-height', `${availableHeight}px`);
}

    function createConfigFormControls(allGroups, renderedGroups, searchText) {
        const controls = document.createElement('div');
        controls.className = 'config-form-controls';
        controls.appendChild(createConfigScopeStatus(allGroups, renderedGroups, searchText));

        const searchLabel = document.createElement('label');
        searchLabel.className = 'config-search-box';
        const searchCaption = document.createElement('span');
        searchCaption.textContent = '搜索配置项';
        const search = document.createElement('input');
        search.id = 'config-search-input';
        search.type = 'search';
        search.autocomplete = 'off';
        search.spellcheck = false;
        search.setAttribute('aria-label', '搜索配置项');
        search.placeholder = '输入学习率、caption、network_dim 或中文名称';
        search.value = configFormState.search;
        search.addEventListener('input', (event) => {
            if (event.isComposing) return;
            applyConfigSearch(event.target.value || '');
        });
        search.addEventListener('compositionstart', () => {
            applyConfigSearch.cancel();
        });
        search.addEventListener('compositionend', (event) => {
            applyConfigSearch(event.target.value || '');
        });
        search.addEventListener('keydown', (event) => {
            if (event.isComposing || event.keyCode === 229) return;
            if (event.key !== 'Escape') return;
            if (!search.value) {
                search.blur();
                return;
            }
            event.preventDefault();
            applyConfigSearch.cancel();
            syncConfigDraftFromForm();
            configFormState.search = '';
            renderConfigForm(currentConfigState());
            requestAnimationFrame(() => {
                document.getElementById('config-search-input')?.focus();
            });
        });
        searchLabel.append(searchCaption, search);
        controls.appendChild(searchLabel);

        const advanced = document.createElement('label');
        advanced.className = 'config-advanced-toggle';
        const advancedInput = document.createElement('input');
        advancedInput.id = 'config-advanced-toggle';
        advancedInput.type = 'checkbox';
        advancedInput.checked = configFormState.showAdvanced;
        advancedInput.addEventListener('change', (event) => {
            syncConfigDraftFromForm();
            configFormState.showAdvanced = event.target.checked;
            renderConfigForm(currentConfigState());
        });
        const advancedText = document.createElement('span');
        advancedText.textContent = '显示高级配置';
        advanced.append(advancedInput, advancedText);
        controls.appendChild(advanced);

        const resetBtn = document.createElement('button');
        resetBtn.id = 'btn-reset-config-changes';
        resetBtn.type = 'button';
        resetBtn.className = 'btn btn-small';
        resetBtn.textContent = '重置当前改动';
        resetBtn.title = '重新读取当前配置文件，丢弃尚未保存的表单修改。';
        resetBtn.addEventListener('click', reloadCurrentConfig);
        controls.appendChild(resetBtn);

        const summary = document.createElement('div');
        summary.className = 'config-form-summary';
        const total = allGroups.reduce((sum, group) => sum + group.fields.length, 0);
        const rendered = renderedGroups.reduce((sum, group) => sum + group.fields.length, 0);
        summary.innerHTML = [
            `<span>${searchText ? '匹配' : '显示'} <strong>${rendered}</strong> / ${total} 项</span>`,
            '<span>已修改 <strong id="config-modified-count">0</strong> 项</span>',
        ].join('');
        controls.appendChild(summary);
        return controls;
    }

    function createConfigScopeStatus(allGroups, renderedGroups, searchText) {
        const active = FORM_CATEGORY_DEFS.find((category) => category.id === configFormState.activeCategory) || FORM_CATEGORY_DEFS[0];
        const total = allGroups.reduce((sum, group) => sum + group.fields.length, 0);
        const visible = renderedGroups.reduce((sum, group) => sum + group.fields.length, 0);
        const scope = document.createElement('div');
        scope.className = 'config-form-scope';

        const caption = document.createElement('span');
        caption.textContent = '当前目录';
        const title = document.createElement('strong');
        title.textContent = active?.title || '配置';
        const meta = document.createElement('em');
        meta.textContent = searchText ? `${visible} / ${total} 项匹配` : active?.description || '';
        scope.append(caption, title, meta);
        return scope;
    }

    function filterConfigGroupEntry(group, searchText) {
        if (!searchText) return group;
        const fields = group.fields.filter(([key, value]) => configFieldMatchesSearch(key, value, searchText));
        if (!fields.length) return null;
        return { ...group, fields };
    }

    function configFieldMatchesSearch(key, value, searchText) {
        return configTextMatches([
            key,
            formatFieldName(key),
            value,
            configState.fieldHelp[key] ? JSON.stringify(configState.fieldHelp[key]) : '',
        ], searchText);
    }

    function configTextMatches(parts, searchText) {
        return parts.some((part) => String(part ?? '').toLowerCase().includes(searchText));
    }

    function createConfigFormEmpty(text) {
        const empty = document.createElement('div');
        empty.className = 'config-form-empty';
        empty.textContent = text;
        return empty;
    }

    function configCategoryIsAdvanced(categoryId) {
        return Boolean(FORM_CATEGORY_DEFS.find((category) => category.id === categoryId)?.advanced);
    }

    function configGroupIsCollapsed(name, searchText = '', defaultOpen = undefined) {
        if (searchText) return false;
        if (configFormState.collapsedGroups.has(name)) return true;
        if (configFormState.expandedGroups.has(name)) return false;
        if (configFormState.activeCategory === 'advanced' && ADVANCED_CATEGORY_DEFAULT_OPEN_GROUPS.has(name)) return false;
        if (defaultOpen === true) return false;
        if (defaultOpen === false) return true;
        return configCategoryIsAdvanced(FORM_CATEGORY_SECTION_MAP.get(name) || 'advanced');
    }

    function createGroup(name, fields, extraClass = '', description = '', searchText = '', defaultOpen = undefined, notice = '') {
        const filtering = Boolean(searchText);
        const section = document.createElement('section');
        section.className = ['config-group', extraClass].filter(Boolean).join(' ');
        section.dataset.groupName = name;
        section.dataset.categoryId = FORM_CATEGORY_SECTION_MAP.get(name) || 'advanced';

        const header = document.createElement('div');
        header.className = 'config-group-title';
        const heading = document.createElement('span');
        heading.className = 'config-group-heading';
        const title = document.createElement('strong');
        title.className = 'config-group-name';
        title.textContent = name;
        const count = document.createElement('em');
        count.className = 'config-group-count';
        count.textContent = `${fields.length} 项`;
        heading.append(title, count);
        header.appendChild(heading);
        if (extraClass === 'config-group-experimental') {
            const badge = document.createElement('span');
            badge.className = 'config-group-badge config-group-badge-experimental';
            badge.textContent = '实验项 / 默认关闭';
            header.appendChild(badge);
        }

        const content = document.createElement('div');
        content.className = 'config-group-body';
        const collapsed = configGroupIsCollapsed(name, searchText, defaultOpen);
        content.hidden = collapsed;
        const collapseBtn = document.createElement('button');
        collapseBtn.type = 'button';
        collapseBtn.className = 'config-group-collapse';
        collapseBtn.textContent = collapsed ? '展开' : '收起';
        collapseBtn.setAttribute('aria-expanded', String(!collapsed));
        collapseBtn.title = collapsed ? '展开这个配置区' : '收起这个配置区';
        collapseBtn.addEventListener('click', () => {
            const nextCollapsed = !content.hidden;
            content.hidden = nextCollapsed;
            collapseBtn.textContent = nextCollapsed ? '展开' : '收起';
            collapseBtn.setAttribute('aria-expanded', String(!nextCollapsed));
            collapseBtn.title = nextCollapsed ? '展开这个配置区' : '收起这个配置区';
            if (nextCollapsed) {
                configFormState.collapsedGroups.add(name);
                configFormState.expandedGroups.delete(name);
            } else {
                configFormState.expandedGroups.add(name);
                configFormState.collapsedGroups.delete(name);
            }
        });
        let hint = null;
        if (description) {
            const hintId = `config-group-hint-${++configState.configGroupHintSeq}`;
            const btn = document.createElement('button');
            btn.className = 'info-toggle config-group-info-toggle';
            btn.textContent = '?';
            btn.type = 'button';
            btn.title = '展开分组说明';
            btn.setAttribute('aria-label', `${name} 说明`);
            btn.setAttribute('aria-controls', hintId);
            btn.setAttribute('aria-expanded', 'false');
            header.appendChild(btn);

            hint = document.createElement('p');
            hint.className = 'config-group-hint';
            hint.id = hintId;
            hint.hidden = true;
            hint.textContent = description;
            btn.addEventListener('click', () => {
                const nextVisible = hint.hidden;
                hint.hidden = !nextVisible;
                btn.classList.toggle('active', nextVisible);
                btn.setAttribute('aria-expanded', String(nextVisible));
                btn.title = nextVisible ? '收起分组说明' : '展开分组说明';
            });
            content.appendChild(hint);
        }
        if (notice) {
            const noticeEl = document.createElement('p');
            noticeEl.className = 'config-group-notice';
            noticeEl.textContent = notice;
            content.appendChild(noticeEl);
        }
        const titleActions = document.createElement('div');
        titleActions.className = 'config-group-title-actions';
        if (!filtering && extraClass === 'config-group-model') {
            titleActions.appendChild(createFillGlobalModelPathsButton());
        }
        if (!filtering && extraClass === 'config-group-resource') {
            titleActions.appendChild(createResourceQuickPresetsButton(content, collapseBtn));
        }
        if (!filtering && extraClass === 'config-group-no-dataset-regularization') {
            titleActions.appendChild(createNoDatasetRegularizationQuickPresetsButton(content, collapseBtn));
        }
        titleActions.appendChild(collapseBtn);
        header.appendChild(titleActions);
        section.appendChild(header);
        if (!filtering && extraClass === 'config-group-data') {
            content.appendChild(createConfigDatasetPicker());
            // 分阶段调度入口只保留在数据集页顶栏，配置页不再显示课表摘要卡片。
        }
        if (!filtering && extraClass === 'config-group-resource') {
            content.appendChild(createResourceQuickPresetPanel());
        }
        if (!filtering && extraClass === 'config-group-no-dataset-regularization') {
            content.appendChild(createNoDatasetRegularizationQuickPresetPanel());
            content.appendChild(createNoDatasetRegularizationModePanel());
            content.appendChild(createNoDatasetRegularizationAdvancedFields(fields, extraClass));
        } else {
            appendFieldRows(content, fields, extraClass);
        }
        if (!filtering && extraClass === 'config-group-steps') {
            content.appendChild(createStepEstimatePanel());
            scheduleStepEstimatePanelRefresh();
        }
        section.appendChild(content);
        return section;
    }
