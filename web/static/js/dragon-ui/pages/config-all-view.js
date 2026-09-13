import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260902v36';
import { bindConfigWorkspaceScroll } from './config-workspace-scroll.js?v=dragon-ui-20260906-workspace-v1';
import { CONFIG_VISIBILITY_OPTIONS, configVisibilityLabel, normalizeConfigVisibilityLevel } from './config-field-tiers.js?v=auto-block-swap-20260908-v3';

export const ALL_CONFIG_VIEW_ID = 'all';
const NEUTRAL_SECTION_ACCENT = '#8e8e93';

const ALL_CONFIG_SUB = Object.freeze({
    id: ALL_CONFIG_VIEW_ID,
    label: '全部参数',
    desc: '按训练流程连续编辑当前方法适用的全部参数。',
    categoryId: 'training-config',
});

export function isAllConfigView(category, subId) {
    return category?.id === 'training-config' && subId === ALL_CONFIG_VIEW_ID;
}

export function uniqueConfigEntries(entries) {
    const seen = new Set();
    return entries.map((entry) => ({
        ...entry,
        keys: entry.keys.filter((key) => {
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        }),
    })).filter((entry) => entry.keys.length > 0);
}

export function resolveConfigView(entries, requestedSubId, category) {
    if (isAllConfigView(category, requestedSubId)) {
        const uniqueEntries = uniqueConfigEntries(entries);
        return {
            sub: ALL_CONFIG_SUB,
            keys: uniqueEntries.flatMap((entry) => entry.keys),
            entries: uniqueEntries,
            isAll: true,
        };
    }
    const entry = entries.find((item) => item.sub.id === requestedSubId) || entries[0];
    return {
        sub: entry?.sub,
        keys: entry?.keys || [],
        entries,
        isAll: false,
    };
}

export function renderConfigViewSwitch(activeId) {
    const allActive = activeId === ALL_CONFIG_VIEW_ID;
    const groupHref = allActive ? '#config/training-config/common' : `#config/training-config/${activeId}`;
    return `<div class="dragon-config-view-switch" role="group" aria-label="配置查看方式">
        <a href="${groupHref}" data-config-view-mode="grouped" data-active="${!allActive}" ${!allActive ? 'aria-current="page"' : ''}>
            ${renderIcon('list')}<span>分组视图</span>
        </a>
        <a href="#config/training-config/all" data-config-view-mode="all" data-active="${allActive}" ${allActive ? 'aria-current="page"' : ''}>
            ${renderIcon('panels')}<span>参数工作台</span>
        </a>
    </div>`;
}

export function renderAllConfigWorkspace({
    blocks,
    chapters,
    bilingual = false,
    visibilityLevel = 'all',
    hideUnavailable = false,
    showAllCandidates = false,
    renderBlock,
    renderActions,
    renderClusterFooter = () => '',
    renderModelPickerTrigger,
    renderModelPickerDialog,
    renderDatasetDialog = () => '',
}) {
    const normalizedVisibilityLevel = normalizeConfigVisibilityLevel(visibilityLevel);
    const displayChapters = chapters;
    const total = blocks.length;
    const applicable = blocks.filter((block) => block.presentation?.visible !== false).length;
    const bilingualAction = bilingual ? '关闭双语渲染' : '开启双语渲染';
    const capsules = displayChapters.map((stage, index) => `
        <button class="dragon-config-tag-filter" type="button" data-config-tag-filter="${escapeHtml(stage.id)}"
                data-color="${escapeHtml(stage.color || 'neutral')}" data-active="${index === 0}"
                ${index === 0 ? 'aria-current="location"' : ''}
                style="--dragon-config-section-accent: ${sectionAccent(stage.accent)}">
            <span>${escapeHtml(stage.label)}</span>
        </button>`).join('');
    const sections = displayChapters.map((chapter) => `
        <section class="dragon-config-flow-section" data-config-filter-group data-config-section="${escapeHtml(chapter.id)}"
                 data-color="${escapeHtml(chapter.color || 'neutral')}"
                 style="--dragon-config-section-accent: ${sectionAccent(chapter.accent)}"
                 aria-labelledby="section-${escapeHtml(chapter.id)}-title">
            ${renderSectionDivider(chapter, chapter.blocks.filter((block) => block.presentation?.visible !== false).length)}
            <div class="dragon-config-stage-clusters">
                ${chapter.clusters.map((cluster) => renderConfigCluster(chapter, cluster, renderBlock, renderClusterFooter, renderModelPickerTrigger)).join('')}
            </div>
            <div class="dragon-config-stage-empty" data-config-stage-empty ${chapter.count ? 'hidden' : ''}>当前上下文在此阶段暂无可显示项</div>
        </section>`).join('');

    return `<div class="dragon-config-workspace dragon-config-all-workspace" data-config-editable-workspace>
        <section class="dragon-config-detail dragon-config-all-detail" data-config-entry="all"
                 aria-labelledby="dragon-config-detail-title">
            <div class="dragon-config-all-toolbar">
                <div class="dragon-config-all-toolbar-main">
                    <h2 id="dragon-config-detail-title" class="dragon-sr-only">训练参数</h2>
                    <nav class="dragon-config-tag-filters" aria-label="配置阶段导航">${capsules}</nav>
                    <label class="dragon-config-all-search">
                        ${renderIcon('search')}
                        <input class="dragon-input" type="search" autocomplete="off" data-config-field-search
                               aria-label="搜索全部适用参数" placeholder="搜索参数名、配置键或值">
                    </label>
                    <details class="dragon-config-display-menu" data-config-display-menu>
                        <summary class="dragon-icon-button" aria-label="参数显示设置" title="参数显示设置">${renderIcon('settings')}</summary>
                        <div class="dragon-config-filter-actions">
                        ${renderConfigVisibilityControls(normalizedVisibilityLevel, hideUnavailable, showAllCandidates)}
                        <button class="dragon-config-bilingual-toggle" type="button"
                                data-config-bilingual-toggle data-active="${Boolean(bilingual)}"
                                aria-pressed="${Boolean(bilingual)}" aria-label="${bilingualAction}" title="${bilingualAction}">
                            ${renderIcon('tags', 'dragon-btn-icon')}<span>双语渲染</span>
                        </button>
                        <a href="#config/training-config/common" class="dragon-config-legacy-link">旧分组视图</a>
                        </div>
                    </details>
                    <button class="dragon-icon-button" type="button" data-config-preset-toggle
                            aria-expanded="true" aria-label="切换配置侧栏" title="切换配置侧栏">${renderIcon('panels')}</button>
                </div>
            </div>
            <div class="dragon-config-detail-fields dragon-config-block-grid" id="dragon-config-fields">
                ${sections}
                <div class="dragon-config-filter-empty" data-config-filter-empty hidden role="status">当前筛选条件下没有可显示的配置项</div>
            </div>
            <footer class="dragon-config-all-footer">
                <output class="dragon-config-field-filter-count" data-config-field-filter-count aria-live="polite"
                        aria-label="当前适用 ${applicable} 项，全部候选 ${total} 项">适用 ${applicable} / 候选 ${total}</output>
                ${renderActions()}
            </footer>
        </section>
        ${renderModelPickerDialog()}
        ${renderDatasetDialog()}
    </div>`;
}

function renderConfigCluster(chapter, cluster, renderBlock, renderClusterFooter, renderModelPickerTrigger) {
    const titleId = `cluster-${chapter.id}-${cluster.id}-title`;
    return `<section class="dragon-config-cluster" data-config-cluster-group="${escapeHtml(cluster.id)}"
                     aria-labelledby="${escapeHtml(titleId)}">
        <div class="dragon-config-cluster-heading">
            <h4 id="${escapeHtml(titleId)}">${escapeHtml(cluster.label)}</h4>
            <span aria-hidden="true"></span>
            ${chapter.id === 'input' && cluster.id === 'models' ? renderModelPickerTrigger() : ''}
        </div>
        <div class="dragon-config-section-grid">${cluster.blocks.map(renderBlock).join('')}</div>
        ${renderClusterFooter(chapter, cluster)}
    </section>`;
}

function renderConfigVisibilityControls(visibilityLevel, hideUnavailable, showAllCandidates) {
    const currentLabel = configVisibilityLabel(visibilityLevel);
    const options = CONFIG_VISIBILITY_OPTIONS.map((option) => `
        <button type="button" role="menuitemradio" data-config-visibility-level="${option.id}"
                aria-checked="${option.id === visibilityLevel}">
            <span>${escapeHtml(option.label)}</span>${renderIcon('check', 'dragon-config-visibility-check')}
        </button>`).join('');
    return `
        <button class="dragon-config-toolbar-control" type="button" data-config-show-all-candidates
                data-active="${Boolean(showAllCandidates)}" aria-pressed="${Boolean(showAllCandidates)}"
                aria-label="${showAllCandidates ? '返回当前适用配置项' : '查看全部候选配置项'}">
            ${renderIcon('eye', 'dragon-btn-icon')}<span data-config-candidate-toggle-label>${showAllCandidates ? '返回当前适用' : '查看全部候选'}</span>
        </button>
        <details class="dragon-config-visibility-menu" data-config-visibility-menu>
            <summary class="dragon-config-toolbar-control" data-config-visibility-trigger
                     data-active="${visibilityLevel !== 'all'}">
                ${renderIcon('layers', 'dragon-btn-icon')}<span>显示范围</span>
                <strong class="dragon-config-visibility-label" data-config-visibility-label>${escapeHtml(currentLabel)}</strong>
                ${renderIcon('chevronDown', 'dragon-config-visibility-chevron')}
            </summary>
            <div class="dragon-config-visibility-options" role="menu" aria-label="配置显示范围">${options}</div>
        </details>
        <button class="dragon-config-toolbar-control" type="button" data-config-hide-unavailable
                data-active="${Boolean(hideUnavailable)}" aria-pressed="${Boolean(hideUnavailable)}">
            ${renderIcon('eye', 'dragon-btn-icon')}<span>隐藏不可用配置项</span>
        </button>`;
}

function sectionAccent(value) {
    const accent = String(value || '');
    return /^#[0-9a-f]{6}$/i.test(accent) ? accent : NEUTRAL_SECTION_ACCENT;
}

export function renderSectionDivider(chapter, visibleCount = chapter.count) {
    return `<div class="dragon-config-section-divider" id="section-${escapeHtml(chapter.id)}"
                 data-config-section-divider="${escapeHtml(chapter.id)}" data-color="${escapeHtml(chapter.color)}">
        <span class="dragon-config-section-dot" aria-hidden="true"></span>
        <h3 id="section-${escapeHtml(chapter.id)}-title">${escapeHtml(chapter.label)} <span class="dragon-config-section-count" data-config-stage-count>(${visibleCount})</span></h3>
        <span class="dragon-config-section-line" aria-hidden="true"></span>
    </div>`;
}

export function bindConfigPresetLibrary(root, { defaultPresetCollapsed = false, onPresetCollapseChange } = {}) {
    const shell = root.querySelector('.dragon-config-shell-layout');
    const setPresetCollapsed = (collapsed) => {
        if (!shell) return;
        shell.dataset.presetCollapsed = String(collapsed);
        onPresetCollapseChange?.(collapsed);
        root.querySelectorAll('[data-config-preset-toggle]').forEach((toggle) => {
            toggle.setAttribute('aria-expanded', String(!collapsed));
            const label = toggle.querySelector('span');
            if (label) label.textContent = collapsed ? '展开预设库' : '收起预设库';
        });
    };
    const onToggle = (event) => {
        if (!event.target.closest('[data-config-preset-toggle]')) return;
        setPresetCollapsed(shell?.dataset.presetCollapsed !== 'true');
        if (shell?.dataset.presetCollapsed === 'true') root.querySelector('[data-config-preset-toggle]')?.focus();
    };
    const onEscape = (event) => {
        if (event.key !== 'Escape' || !window.matchMedia('(max-width: 734px)').matches) return;
        if (shell?.dataset.presetCollapsed !== 'false' || root.querySelector('dialog[open]')) return;
        setPresetCollapsed(true);
        root.querySelector('[data-config-preset-toggle]')?.focus();
    };
    setPresetCollapsed(defaultPresetCollapsed);
    root.addEventListener('click', onToggle);
    root.addEventListener('keydown', onEscape);
    return () => {
        root.removeEventListener('click', onToggle);
        root.removeEventListener('keydown', onEscape);
    };
}

export function bindAllConfigWorkspace(root, { defaultPresetCollapsed = false, onPresetCollapseChange, onRadarChange } = {}) {
    const presetCleanup = bindConfigPresetLibrary(root, { defaultPresetCollapsed, onPresetCollapseChange });

    const search = root.querySelector('[data-config-field-search]');
    const focusSearch = (event) => {
        if (!(event.ctrlKey || event.metaKey) || event.altKey || event.key.toLocaleLowerCase() !== 'f') return;
        event.preventDefault();
        search?.focus();
        search?.select();
    };
    window.addEventListener('keydown', focusSearch);

    const scrollCleanup = bindConfigWorkspaceScroll(root, onRadarChange);
    return () => {
        presetCleanup?.();
        scrollCleanup();
        window.removeEventListener('keydown', focusSearch);
    };
}
