export function bindConfigLibraryNavigation(library, state) {
    const search = library.querySelector('[data-training-preset-search]');
    const empty = library.querySelector('[data-training-preset-no-results]');
    const groups = [...library.querySelectorAll('[data-training-preset-group]')];
    state.libraryQuery ??= '';
    state.collapsedGroups ??= new Set();
    if (search) search.value = state.libraryQuery;
    const update = () => {
        const query = state.libraryQuery.trim().toLocaleLowerCase();
        let count = 0;
        groups.forEach((group) => {
            const id = group.dataset.trainingPresetGroup;
            const button = group.querySelector('[data-training-group-toggle]');
            const groupMatch = Boolean(query && button?.textContent.toLocaleLowerCase().includes(query));
            const rows = [...group.querySelectorAll('[data-training-preset-row]')];
            rows.forEach((row) => {
                row.hidden = Boolean(query && !groupMatch && !row.textContent.toLocaleLowerCase().includes(query));
                if (!row.hidden) count += 1;
            });
            group.hidden = Boolean(query && !rows.some((row) => !row.hidden));
            const collapsed = !query && state.collapsedGroups.has(id);
            const list = group.querySelector('.dragon-training-preset-list');
            if (list) list.hidden = collapsed;
            button?.setAttribute('aria-expanded', String(!collapsed));
        });
        if (empty) empty.hidden = !query || count > 0;
    };
    search?.addEventListener('input', () => {
        state.libraryQuery = search.value;
        update();
    });
    groups.forEach((group) => {
        group.querySelector('[data-training-group-toggle]')?.addEventListener('click', () => {
            const id = group.dataset.trainingPresetGroup;
            if (state.collapsedGroups.has(id)) state.collapsedGroups.delete(id);
            else state.collapsedGroups.add(id);
            if (state.libraryQuery) {
                state.libraryQuery = '';
                if (search) search.value = '';
            }
            update();
        });
    });
    library.querySelectorAll('.dragon-training-preset-toolbar-actions .dragon-btn').forEach((button) => {
        button.title = button.textContent.trim();
        button.setAttribute('aria-label', button.title);
    });
    update();
}
