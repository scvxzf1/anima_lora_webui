/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.createConfigGroupEntry = function createConfigGroupEntry(name, fields, extraClass = '', description = '', defaultOpen = undefined, notice = '') {
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

    globalThis.appendConfigGroupsByCategory = function appendConfigGroupsByCategory(container, groups) {
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
        const sourceCategories = searchText
            ? categories
            : categories.filter((category) => category.id === activeCategory);
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
        main.appendChild(createConfigFormControls(groups, renderedGroups, searchText));
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

    globalThis.normalizeConfigSearch = function normalizeConfigSearch(value) {
        return String(value || '').trim().toLowerCase();
    }

    globalThis.configCategoryVisible = function configCategoryVisible(category, searchText = '') {
        return Boolean(searchText) || configFormState.showAdvanced || !category.advanced;
    }

    globalThis.normalizeConfigActiveCategory = function normalizeConfigActiveCategory(categories) {
        if (!categories.length) return '';
        const ids = new Set(categories.map((category) => category.id));
        if (!ids.has(configFormState.activeCategory)) {
            configFormState.activeCategory = categories[0].id;
        }
        return configFormState.activeCategory;
    }

    globalThis.selectConfigCategory = function selectConfigCategory(categoryId, options = {}) {
        if (!categoryId) return;
        const category = FORM_CATEGORY_DEFS.find((item) => item.id === categoryId);
        syncConfigDraftFromForm();
        if (category?.advanced) {
            configFormState.showAdvanced = true;
        }
        configFormState.activeCategory = categoryId;
        configFormState.search = '';
        renderConfigForm(currentConfig);
        if (options.scrollToForm) {
            requestAnimationFrame(() => scrollConfigFormContentToTop('smooth'));
        }
    }

    globalThis.scrollConfigFormContentToTop = function scrollConfigFormContentToTop(behavior = 'auto') {
        const scroller = document.querySelector('#tab-config .config-left');
        if (!scroller) return;
        scroller.scrollTo({ top: 0, behavior });
    }

    globalThis.updateConfigStickyDirectory = function updateConfigStickyDirectory(categories, buckets, activeCategory, searchText) {
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
            btn.classList.toggle('active', enabled && categoryId === activeCategory && !searchText);
            btn.setAttribute('aria-current', enabled && categoryId === activeCategory && !searchText ? 'true' : 'false');
            btn.title = enabled ? `切换到${category.title}配置` : '当前配置没有这个分类';
            const count = btn.querySelector('em');
            if (count) count.textContent = `${fieldCount} 项`;
        });
    }

    globalThis.updateConfigStickyPlacement = function updateConfigStickyPlacement() {
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
        const safeSpace = Math.ceil(barRect.height + bottomOffset + 18);
        const scrollerRect = scroller.getBoundingClientRect();
        const availableHeight = Math.max(180, window.innerHeight - scrollerRect.top - 16);
        workspace.style.setProperty('--config-sticky-safe-space', `${safeSpace}px`);
        workspace.style.setProperty('--config-left-max-height', `${Math.round(availableHeight)}px`);
    }

    globalThis.createConfigFormControls = function createConfigFormControls(allGroups, renderedGroups, searchText) {
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
            syncConfigDraftFromForm();
            configFormState.search = event.target.value || '';
            renderConfigForm(currentConfig);
            requestAnimationFrame(() => {
                const next = document.getElementById('config-search-input');
                if (next) {
                    next.focus();
                    const length = next.value.length;
                    next.setSelectionRange(length, length);
                }
            });
        });
        search.addEventListener('keydown', (event) => {
            if (event.key !== 'Escape') return;
            if (!search.value) {
                search.blur();
                return;
            }
            event.preventDefault();
            syncConfigDraftFromForm();
            configFormState.search = '';
            renderConfigForm(currentConfig);
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
            renderConfigForm(currentConfig);
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

    globalThis.createConfigScopeStatus = function createConfigScopeStatus(allGroups, renderedGroups, searchText) {
        const active = FORM_CATEGORY_DEFS.find((category) => category.id === configFormState.activeCategory) || FORM_CATEGORY_DEFS[0];
        const total = allGroups.reduce((sum, group) => sum + group.fields.length, 0);
        const visible = renderedGroups.reduce((sum, group) => sum + group.fields.length, 0);
        const scope = document.createElement('div');
        scope.className = 'config-form-scope';

        const caption = document.createElement('span');
        caption.textContent = searchText ? '搜索结果' : '当前目录';
        const title = document.createElement('strong');
        title.textContent = searchText ? '筛选中' : active?.title || '配置';
        const meta = document.createElement('em');
        meta.textContent = searchText ? `${visible} / ${total} 项匹配` : active?.description || '';
        scope.append(caption, title, meta);
        return scope;
    }

    globalThis.filterConfigGroupEntry = function filterConfigGroupEntry(group, searchText) {
        if (!searchText) return group;
        const groupMatched = configTextMatches([group.name, group.description, group.categoryId], searchText);
        const fields = groupMatched
            ? group.fields
            : group.fields.filter(([key, value]) => configFieldMatchesSearch(key, value, searchText));
        if (!fields.length) return null;
        return { ...group, fields };
    }

    globalThis.configFieldMatchesSearch = function configFieldMatchesSearch(key, value, searchText) {
        return configTextMatches([
            key,
            formatFieldName(key),
            value,
            fieldHelp[key] ? JSON.stringify(fieldHelp[key]) : '',
        ], searchText);
    }

    globalThis.configTextMatches = function configTextMatches(parts, searchText) {
        return parts.some((part) => String(part ?? '').toLowerCase().includes(searchText));
    }

    globalThis.createConfigFormEmpty = function createConfigFormEmpty(text) {
        const empty = document.createElement('div');
        empty.className = 'config-form-empty';
        empty.textContent = text;
        return empty;
    }

    globalThis.configCategoryIsAdvanced = function configCategoryIsAdvanced(categoryId) {
        return Boolean(FORM_CATEGORY_DEFS.find((category) => category.id === categoryId)?.advanced);
    }

    globalThis.configGroupIsCollapsed = function configGroupIsCollapsed(name, searchText = '', defaultOpen = undefined) {
        if (searchText) return false;
        if (configFormState.collapsedGroups.has(name)) return true;
        if (configFormState.expandedGroups.has(name)) return false;
        if (configFormState.activeCategory === 'advanced' && ADVANCED_CATEGORY_DEFAULT_OPEN_GROUPS.has(name)) return false;
        if (defaultOpen === true) return false;
        if (defaultOpen === false) return true;
        return configCategoryIsAdvanced(FORM_CATEGORY_SECTION_MAP.get(name) || 'advanced');
    }

    globalThis.createGroup = function createGroup(name, fields, extraClass = '', description = '', searchText = '', defaultOpen = undefined, notice = '') {
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
            const hintId = `config-group-hint-${++configGroupHintSeq}`;
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
        if (extraClass === 'config-group-model') {
            titleActions.appendChild(createFillGlobalModelPathsButton());
        }
        if (extraClass === 'config-group-resource') {
            titleActions.appendChild(createResourceQuickPresetsButton(content, collapseBtn));
        }
        if (extraClass === 'config-group-no-dataset-regularization') {
            titleActions.appendChild(createNoDatasetRegularizationQuickPresetsButton(content, collapseBtn));
        }
        titleActions.appendChild(collapseBtn);
        header.appendChild(titleActions);
        section.appendChild(header);
        if (extraClass === 'config-group-data') {
            content.appendChild(createConfigDatasetPicker());
        }
        if (extraClass === 'config-group-resource') {
            content.appendChild(createResourceQuickPresetPanel());
        }
        if (extraClass === 'config-group-no-dataset-regularization') {
            content.appendChild(createNoDatasetRegularizationQuickPresetPanel());
            content.appendChild(createNoDatasetRegularizationModePanel());
            content.appendChild(createNoDatasetRegularizationAdvancedFields(fields, extraClass));
        } else {
            appendFieldRows(content, fields, extraClass);
        }
        if (extraClass === 'config-group-steps') {
            content.appendChild(createStepEstimatePanel());
            scheduleStepEstimatePanelRefresh();
        }
        section.appendChild(content);
        return section;
    }

    globalThis.createOpenStageResolutionDialogButton = function createOpenStageResolutionDialogButton() {
        const btn = document.createElement('button');
        btn.id = 'btn-open-stage-resolution-dialog';
        btn.type = 'button';
        btn.className = 'btn btn-small config-group-title-action';
        btn.textContent = '阶段调度';
        btn.title = '打开阶段分辨率调度面板';
        btn.addEventListener('click', openStageResolutionDialog);
        return btn;
    }

    globalThis.openStageResolutionDialog = function openStageResolutionDialog() {
        const dialog = document.getElementById('stage-resolution-dialog');
        if (!dialog) return;
        renderStageResolutionDialog();
        if (dialog.showModal && !dialog.open) {
            dialog.showModal();
        } else if (!dialog.open) {
            dialog.setAttribute('open', 'open');
        }
        requestAnimationFrame(drawStageResolutionChart);
    }

    globalThis.normalizedStageResolutionStages = function normalizedStageResolutionStages() {
        if (!Array.isArray(stageResolutionState.stages) || !stageResolutionState.stages.length) {
            stageResolutionState.stages = [
                { name: 'EP1', epochs: 1, maxSide: 1024, downRange: 256, manualRepeats: false, repeats: 1 },
            ];
        }
        stageResolutionState.stages = stageResolutionState.stages.map((stage, index) => ({
            name: String(stage.name || `EP${index + 1}`).trim() || `EP${index + 1}`,
            epochs: Number(stage.epochs) || 0,
            maxSide: Number(stage.maxSide) || 0,
            downRange: Number(stage.downRange) || 0,
            manualRepeats: Boolean(stage.manualRepeats),
            repeats: Math.max(1, Math.round(Number(stage.repeats) || 1)),
        }));
        stageResolutionState.selectedIndex = Math.max(
            0,
            Math.min(stageResolutionState.selectedIndex || 0, stageResolutionState.stages.length - 1)
        );
        return stageResolutionState.stages;
    }

    globalThis.stageResolutionMetrics = function stageResolutionMetrics() {
        stageResolutionState.enabled = Boolean(stageResolutionState.enabled);
        const stages = normalizedStageResolutionStages();
        let cursorStep = 0;
        const ranges = stages.map((stage, index) => {
            const epochs = Number(stage.epochs);
            const maxSide = Number(stage.maxSide);
            const downRange = Number(stage.downRange);
            const minSide = maxSide - downRange;
            const startStep = cursorStep;
            const steps = Math.max(0, epochs) * STAGE_RESOLUTION_STEPS_PER_EPOCH;
            cursorStep += steps;
            const problems = [];
            const warnings = [];
            if (!Number.isFinite(epochs) || epochs <= 0) problems.push('epochs 必须大于 0');
            if (!Number.isFinite(maxSide) || maxSide <= 0) problems.push('单边最大值无效');
            if (!Number.isFinite(downRange) || downRange <= 0) problems.push('向下波动必须大于 0');
            if (Number.isFinite(minSide) && minSide <= 0) problems.push('单边最小值无效');
            if (Number.isFinite(minSide) && Number.isFinite(maxSide) && minSide >= maxSide) problems.push('范围为空');
            return {
                ...stage,
                index,
                startStep,
                endStep: cursorStep,
                steps,
                minSide,
                imageCount: null,
                autoRepeats: stage.manualRepeats ? stage.repeats : 1,
                problems,
                warnings,
            };
        });

        for (let i = 0; i < ranges.length; i += 1) {
            for (let j = i + 1; j < ranges.length; j += 1) {
                const a = ranges[i];
                const b = ranges[j];
                if (a.problems.length || b.problems.length) continue;
                const overlaps = Math.max(a.minSide, b.minSide) < Math.min(a.maxSide, b.maxSide);
                if (overlaps) {
                    a.warnings.push('范围重叠');
                    b.warnings.push('范围重叠');
                }
            }
        }
        const sorted = ranges
            .filter((item) => !item.problems.length)
            .slice()
            .sort((a, b) => a.minSide - b.minSide);
        for (let i = 1; i < sorted.length; i += 1) {
            if (sorted[i].minSide > sorted[i - 1].maxSide) {
                sorted[i - 1].warnings.push('存在断档');
                sorted[i].warnings.push('存在断档');
            }
        }

        const problemCount = ranges.filter((item) => item.problems.length).length;
        const warningCount = ranges.filter((item) => item.warnings.length).length;
        return {
            enabled: stageResolutionState.enabled,
            stages: ranges,
            totalSteps: cursorStep,
            problemCount,
            warningCount,
            selected: ranges[stageResolutionState.selectedIndex] || ranges[0],
        };
    }

    globalThis.stageResolutionStatus = function stageResolutionStatus(stage) {
        if (stage.problems.length) return { tone: 'error', text: stage.problems[0] };
        if (stage.warnings.length) return { tone: 'warning', text: stage.warnings[0] };
        return { tone: 'ok', text: '就绪' };
    }

    globalThis.renderStageResolutionDialog = function renderStageResolutionDialog() {
        const body = document.getElementById('stage-resolution-dialog-body');
        if (!body) return;
        const metrics = stageResolutionMetrics();
        body.innerHTML = '';
        body.appendChild(createStageResolutionSummary(metrics));

        const workspace = document.createElement('div');
        workspace.className = 'stage-resolution-workspace';
        workspace.appendChild(createStageResolutionChartPanel());
        workspace.appendChild(createStageResolutionEditor(metrics.selected));
        body.appendChild(workspace);

        body.appendChild(createStageResolutionTable(metrics.stages));
        requestAnimationFrame(drawStageResolutionChart);
    }
