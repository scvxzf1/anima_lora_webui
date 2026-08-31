import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';

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
            ${renderIcon('panels')}<span>积木流 Beta</span>
        </a>
    </div>`;
}

export function renderAllConfigWorkspace({
    blocks,
    chapters,
    bilingual = false,
    renderBlock,
    renderActions,
    renderChapterLead = () => '',
    renderChapterFooter = () => '',
    renderModelPickerTrigger,
    renderModelPickerDialog,
    renderDatasetDialog = () => '',
}) {
    const total = blocks.length;
    const bilingualAction = bilingual ? '关闭双语渲染' : '开启双语渲染';
    const capsules = [{ id: 'all', label: '全部', count: total, color: 'neutral', accent: NEUTRAL_SECTION_ACCENT }, ...chapters].map((tag) => `
        <button class="dragon-config-tag-filter" type="button" data-config-tag-filter="${escapeHtml(tag.id)}"
                data-color="${escapeHtml(tag.color || 'neutral')}" data-active="${tag.id === 'all'}"
                style="--dragon-config-section-accent: ${sectionAccent(tag.accent)}">
            <span>${escapeHtml(tag.label)}</span><strong>${tag.count}</strong>
        </button>`).join('');
    const sections = chapters.map((chapter) => `
        <section class="dragon-config-flow-section" data-config-filter-group data-config-section="${escapeHtml(chapter.id)}"
                 data-color="${escapeHtml(chapter.color || 'neutral')}"
                 style="--dragon-config-section-accent: ${sectionAccent(chapter.accent)}"
                 aria-labelledby="section-${escapeHtml(chapter.id)}-title">
            ${renderSectionDivider(chapter)}
            <div class="dragon-config-section-grid">
                ${renderChapterLead(chapter)}
                ${chapter.blocks.map(renderBlock).join('')}
            </div>
            ${renderChapterFooter(chapter)}
        </section>`).join('');

    return `<div class="dragon-config-workspace dragon-config-all-workspace" data-config-editable-workspace>
        <section class="dragon-config-detail dragon-config-all-detail" data-config-entry="all"
                 aria-labelledby="dragon-config-detail-title">
            <header class="dragon-config-detail-header dragon-config-all-header">
                <div class="dragon-config-detail-header-copy">
                    <span class="dragon-eyebrow">流式积木画布 · Beta</span>
                    <h2 id="dragon-config-detail-title">全部参数</h2>
                    <p>当前方法的参数已扁平化到同一密排画布，分类仅用于筛选。</p>
                </div>
                <div class="dragon-config-all-header-actions">
                    ${renderModelPickerTrigger()}
                    <button class="dragon-btn dragon-btn-secondary dragon-btn-sm" type="button"
                            data-config-preset-toggle aria-expanded="true">
                        ${renderIcon('panels', 'dragon-btn-icon')}<span>收起预设库</span>
                    </button>
                </div>
            </header>
            <div class="dragon-config-all-toolbar">
                <div class="dragon-config-all-toolbar-main">
                    ${renderConfigViewSwitch(ALL_CONFIG_VIEW_ID)}
                    <label class="dragon-config-all-search">
                        ${renderIcon('search')}
                        <input class="dragon-input" type="search" autocomplete="off" data-config-field-search
                               aria-label="搜索全部适用参数" placeholder="搜索参数名、配置键或值">
                    </label>
                    <output class="dragon-config-field-filter-count" data-config-field-filter-count>${total} 项</output>
                </div>
                <div class="dragon-config-capsule-row">
                    <div class="dragon-config-tag-filters" role="toolbar" aria-label="参数章节导航">${capsules}</div>
                    <button class="dragon-config-bilingual-toggle" type="button"
                            data-config-bilingual-toggle data-active="${Boolean(bilingual)}"
                            aria-pressed="${Boolean(bilingual)}" aria-label="${bilingualAction}" title="${bilingualAction}">
                        ${renderIcon('tags', 'dragon-btn-icon')}<span>双语渲染</span>
                    </button>
                    <div class="dragon-config-capsule-mode" role="group" aria-label="胶囊点击行为">
                        <button type="button" data-config-capsule-mode="jump" data-active="true" aria-pressed="true">定位</button>
                        <button type="button" data-config-capsule-mode="filter" data-active="false" aria-pressed="false">过滤</button>
                    </div>
                </div>
            </div>
            <div class="dragon-config-detail-fields dragon-config-block-grid" id="dragon-config-fields">
                ${sections}
            </div>
            <footer class="dragon-config-all-footer">
                ${renderActions()}
            </footer>
        </section>
        ${renderModelPickerDialog()}
        ${renderDatasetDialog()}
    </div>`;
}

function sectionAccent(value) {
    const accent = String(value || '');
    return /^#[0-9a-f]{6}$/i.test(accent) ? accent : NEUTRAL_SECTION_ACCENT;
}

export function renderSectionDivider(chapter) {
    return `<div class="dragon-config-section-divider" id="section-${escapeHtml(chapter.id)}"
                 data-config-section-divider="${escapeHtml(chapter.id)}" data-color="${escapeHtml(chapter.color)}">
        <span class="dragon-config-section-dot" aria-hidden="true"></span>
        <h3 id="section-${escapeHtml(chapter.id)}-title">${escapeHtml(chapter.label)} <span class="dragon-config-section-count">(${chapter.count})</span></h3>
        <span class="dragon-config-section-line" aria-hidden="true"></span>
    </div>`;
}

export function bindAllConfigWorkspace(root, { defaultPresetCollapsed = false, onPresetCollapseChange, onRadarChange } = {}) {
    const shell = root.querySelector('.dragon-config-shell-layout');
    const toggle = root.querySelector('[data-config-preset-toggle]');
    const setPresetCollapsed = (collapsed) => {
        if (!shell || !toggle) return;
        shell.dataset.presetCollapsed = String(collapsed);
        onPresetCollapseChange?.(collapsed);
        toggle.setAttribute('aria-expanded', String(!collapsed));
        const label = toggle.querySelector('span');
        if (label) label.textContent = collapsed ? '展开预设库' : '收起预设库';
    };
    setPresetCollapsed(defaultPresetCollapsed);
    toggle?.addEventListener('click', () => setPresetCollapsed(shell?.dataset.presetCollapsed !== 'true'));

    const search = root.querySelector('[data-config-field-search]');
    const focusSearch = (event) => {
        if (!(event.ctrlKey || event.metaKey) || event.altKey || event.key.toLocaleLowerCase() !== 'f') return;
        event.preventDefault();
        search?.focus();
        search?.select();
    };
    window.addEventListener('keydown', focusSearch);

    const canvas = root.querySelector('#dragon-config-fields');
    const dividers = [...root.querySelectorAll('[data-config-section-divider]')];
    let radarFrame = 0;
    let activeRadar = '';
    const syncRadar = () => {
        radarFrame = 0;
        const threshold = Math.ceil(canvas?.getBoundingClientRect().top || 0) + configCanvasPadding(canvas);
        let active = dividers[0]?.dataset.configSectionDivider || 'all';
        for (const divider of dividers) {
            if (divider.getBoundingClientRect().top > threshold) break;
            active = divider.dataset.configSectionDivider;
        }
        if (active === activeRadar) return;
        activeRadar = active;
        onRadarChange?.(active);
    };
    const scheduleRadar = () => {
        if (!radarFrame) radarFrame = window.requestAnimationFrame(syncRadar);
    };
    const observer = typeof IntersectionObserver === 'function'
        ? new IntersectionObserver(scheduleRadar, {
            root: canvas,
            rootMargin: `-${configCanvasPadding(canvas)}px 0px -70% 0px`,
            threshold: [0, 1],
        })
        : null;
    dividers.forEach((divider) => observer?.observe(divider));
    const keepFocusVisible = (event) => keepConfigFocusVisible(canvas, event.target);
    if (!observer) canvas?.addEventListener('scroll', scheduleRadar, { passive: true });
    canvas?.addEventListener('focusin', keepFocusVisible);
    syncRadar();
    return () => {
        window.removeEventListener('keydown', focusSearch);
        if (!observer) canvas?.removeEventListener('scroll', scheduleRadar);
        canvas?.removeEventListener('focusin', keepFocusVisible);
        observer?.disconnect();
        if (radarFrame) window.cancelAnimationFrame(radarFrame);
    };
}

export function scrollConfigCanvasTo(canvas, target, behavior = 'smooth') {
    if (!canvas || !target) return;
    const canvasRect = canvas.getBoundingClientRect();
    const targetRect = target.getBoundingClientRect();
    canvas.scrollTo({
        top: Math.max(0, canvas.scrollTop + targetRect.top - canvasRect.top - configCanvasPadding(canvas)),
        behavior,
    });
}

function configCanvasPadding(canvas) {
    if (!canvas) return 12;
    return Number.parseFloat(getComputedStyle(canvas).scrollPaddingTop) || 12;
}

function keepConfigFocusVisible(canvas, target) {
    const block = target?.closest?.('.dragon-config-block');
    if (!canvas || !block) return;
    const canvasRect = canvas.getBoundingClientRect();
    const blockRect = block.getBoundingClientRect();
    const padding = configCanvasPadding(canvas);
    let delta = 0;
    if (blockRect.top < canvasRect.top + padding) delta = blockRect.top - canvasRect.top - padding;
    else if (blockRect.bottom > canvasRect.bottom - padding) delta = blockRect.bottom - canvasRect.bottom + padding;
    if (delta) canvas.scrollBy({ top: delta, behavior: 'smooth' });
}
