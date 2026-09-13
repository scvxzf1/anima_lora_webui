import { configFieldVisibleAtLevel } from "./config-field-tiers.js?v=auto-block-swap-20260908-v3";
import { configScrollContainer, configScrollPosition, restoreConfigScroll, scrollConfigCanvasTo } from './config-workspace-scroll.js?v=dragon-ui-20260906-workspace-v1';
import { renderConfigDirtyState } from './config-dirty-state.js?v=dragon-ui-20260826v1';
import { dragonScrollBehavior } from "../motion.js?v=dragon-ui-20260824v1";

export function bindConfigFieldFilter(root, state) {
    const input = root.querySelector('[data-config-field-search]');
    const output = root.querySelector('[data-config-field-filter-count]');
    const fieldsRoot = root.querySelector('#dragon-config-fields');
    if (!input || !output || !fieldsRoot) return null;
    const emptyState = fieldsRoot.querySelector('[data-config-filter-empty]');
    const fields = [...fieldsRoot.querySelectorAll('.dragon-field, .dragon-config-dataset-card')];
    const fieldRecords = fields.map((field) => {
        const key = field.querySelector('[data-key]')?.dataset.key || '';
        return {
            field,
            key,
            searchText: field.dataset.searchText || `${key} ${field.textContent || ''}`.toLocaleLowerCase(),
            visibilityLevel: field.dataset.configVisibilityLevel || 'advanced',
        };
    });
    const blockFlow = fieldsRoot.classList.contains('dragon-config-block-grid');
    const tagButtons = [...root.querySelectorAll('[data-config-tag-filter]')];
    const details = [...fieldsRoot.querySelectorAll('details.dragon-config-section')];
    const sectionRecords = [...fieldsRoot.querySelectorAll('.dragon-config-section')]
        .map((section) => ({ section, fields: [...section.querySelectorAll('.dragon-field, .dragon-config-dataset-card')] }));
    const stageRecords = [...fieldsRoot.querySelectorAll('[data-config-filter-group]')]
        .map((group) => ({
            group,
            fields: [...group.querySelectorAll('.dragon-field, .dragon-config-dataset-card')],
            count: group.querySelector('[data-config-stage-count]'),
        }));
    const clusterRecords = [...fieldsRoot.querySelectorAll('[data-config-cluster-group]')]
        .map((cluster) => ({ cluster, fields: [...cluster.querySelectorAll('.dragon-field, .dragon-config-dataset-card')] }));
    const openStates = new Map(details.map((detail) => [detail, detail.open]));
    let previousQuery = '';
    let browseRadarTag = '';
    let browseScrollTop = 0;
    let searchTimer = null;
    input.value = state?.searchQuery || '';

    const syncStageNavigation = () => {
        const active = state?.radarTag || tagButtons[0]?.dataset.configTagFilter || '';
        tagButtons.forEach((button) => {
            const selected = active === button.dataset.configTagFilter;
            button.dataset.active = String(selected);
            if (selected) button.setAttribute('aria-current', 'location');
            else button.removeAttribute('aria-current');
        });
    };

    const update = () => {
        const query = String(input.value || '').trim().toLocaleLowerCase();
        if (query && !previousQuery) {
            browseScrollTop = configScrollPosition(fieldsRoot);
            browseRadarTag = state?.radarTag || '';
        }
        fieldsRoot.dataset.searching = String(Boolean(query || state?.showChangedOnly));
        if (state) state.searchQuery = input.value;
        let visible = 0;
        let applicable = 0;
        let hiddenMatches = 0;
        fieldRecords.forEach(({ field, key, searchText, visibilityLevel }) => {
            const control = field.querySelector('[data-key]');
            const currentText = `${searchText} ${control?.value ?? control?.dataset.checked ?? ''}`.toLocaleLowerCase();
            const matchesQuery = !query || currentText.includes(query);
            const presentationVisible = field.dataset.configPresentationVisible !== 'false';
            if (presentationVisible) applicable += 1;
            const revealHidden = Boolean(state?.showAllCandidates || state?.showChangedOnly);
            const matchesPresentation = revealHidden || presentationVisible;
            const matchesVisibility = revealHidden || Boolean(query) || !blockFlow
                || configFieldVisibleAtLevel(visibilityLevel, state?.configVisibilityLevel);
            const matchesAvailability = revealHidden || !state?.hideUnavailable
                || field.dataset.configAvailability !== 'unavailable';
            const matchesChanged = !state?.showChangedOnly || state.dirtyKeys?.has(key);
            const participates = matchesPresentation && matchesVisibility && matchesAvailability && matchesChanged;
            field.hidden = !(participates && matchesQuery);
            field.dataset.searchMuted = 'false';
            field.dataset.searchMatch = String(Boolean(blockFlow && query && matchesQuery));
            if (participates && matchesQuery) visible += 1;
            if (query && matchesQuery && !presentationVisible) hiddenMatches += 1;
        });
        sectionRecords.forEach(({ section, fields: sectionFields }) => {
            const hasVisibleField = sectionFields.some((field) => !field.hidden);
            section.hidden = !hasVisibleField;
            if (query && hasVisibleField && section.tagName === 'DETAILS') section.open = true;
        });
        clusterRecords.forEach(({ cluster, fields: clusterFields }) => {
            cluster.hidden = !clusterFields.some((field) => !field.hidden);
        });
        stageRecords.forEach(({ group, fields: stageFields, count }) => {
            const visibleFields = stageFields.filter((field) => !field.hidden);
            const hasVisibleField = visibleFields.length > 0;
            const stageEmpty = group.querySelector('[data-config-stage-empty]');
            group.hidden = !hasVisibleField && Boolean(query || state?.showChangedOnly);
            if (count) count.textContent = `(${visibleFields.length})`;
            if (stageEmpty) stageEmpty.hidden = hasVisibleField;
        });
        if (previousQuery && !query) details.forEach((detail) => { detail.open = openStates.get(detail); });
        if (query !== previousQuery) {
            if (!query) restoreConfigScroll(fieldsRoot, browseScrollTop);
            else if (configScrollContainer(fieldsRoot) !== window) restoreConfigScroll(fieldsRoot, 0);
            else scrollConfigCanvasTo(fieldsRoot, fieldsRoot, 'instant');
        }
        if (state && query) state.radarTag = stageRecords.find(({ group }) => !group.hidden)?.group.dataset.configSection || '';
        else if (state && previousQuery) state.radarTag = browseRadarTag;
        previousQuery = query;
        syncStageNavigation();
        const secondaryFilter = query
            || state?.showChangedOnly
            || (blockFlow && state?.configVisibilityLevel !== 'all')
            || state?.hideUnavailable;
        if (!blockFlow) {
            const hiddenSuffix = query && !state?.showAllCandidates && hiddenMatches
                ? ` · 隐藏匹配 ${hiddenMatches}`
                : '';
            output.textContent = query || state?.showChangedOnly
                ? `匹配 ${visible} / 候选 ${fields.length}${hiddenSuffix}`
                : (secondaryFilter || applicable !== fields.length
                    ? `适用 ${applicable} / 候选 ${fields.length}`
                    : `${fields.length} 项`);
        } else if (query || state?.showChangedOnly) {
            const hiddenSuffix = !state?.showAllCandidates && hiddenMatches
                ? ` · 隐藏匹配 ${hiddenMatches}`
                : '';
            output.textContent = `匹配 ${visible} / 候选 ${fields.length}${hiddenSuffix}`;
        } else if (secondaryFilter) {
            output.textContent = `显示 ${visible} / 适用 ${applicable} · 候选 ${fields.length}`;
        } else {
            output.textContent = state?.showAllCandidates
                ? `候选 ${fields.length} · 适用 ${applicable}`
                : `适用 ${applicable} / 候选 ${fields.length}`;
        }
        output.setAttribute('aria-label', `当前显示 ${visible} 项，当前适用 ${applicable} 项，全部候选 ${fields.length} 项`);
        if (emptyState) emptyState.hidden = visible > 0 || fields.length === 0;
    };

    const scheduleSearchUpdate = () => {
        if (state) state.searchQuery = input.value;
        if (searchTimer) window.clearTimeout(searchTimer);
        searchTimer = window.setTimeout(() => {
            searchTimer = null;
            update();
        }, 100);
    };
    input.addEventListener('input', scheduleSearchUpdate);
    const navigationHandlers = tagButtons.map((button) => {
        const handler = () => {
            if (input.value || state?.showChangedOnly) {
                input.value = '';
                if (state) state.showChangedOnly = false;
                if (state?.dirtyBindings) renderConfigDirtyState(state.dirtyBindings, state);
                update();
            }
            const tag = button.dataset.configTagFilter || 'all';
            if (state) state.radarTag = tag;
            syncStageNavigation();
            const target = fieldsRoot.querySelector(`[data-config-section="${tag}"]`);
            scrollConfigCanvasTo(fieldsRoot, target, dragonScrollBehavior());
        };
        button.addEventListener('click', handler);
        return [button, handler];
    });
    if (state) {
        state.filterUpdate = update;
        state.radarUpdate = syncStageNavigation;
    }
    update();
    return () => {
        input.removeEventListener('input', scheduleSearchUpdate);
        navigationHandlers.forEach(([button, handler]) => button.removeEventListener('click', handler));
        if (searchTimer) window.clearTimeout(searchTimer);
        searchTimer = null;
        if (state?.filterUpdate === update) state.filterUpdate = null;
        if (state?.radarUpdate === syncStageNavigation) state.radarUpdate = null;
    };
}
