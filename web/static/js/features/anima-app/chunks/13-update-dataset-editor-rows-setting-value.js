/**
 * Mechanical split from the former monolithic app closure.
 * Keep this module focused; move newly edited behavior into domain modules.
 */
const ctx = globalThis.ctx;

    globalThis.updateDatasetEditorRowsSettingValue = function updateDatasetEditorRowsSettingValue(indices, key, value, options = {}) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        const targets = datasetValidTargetIndices(indices, rows.length);
        if (!targets.length) return;
        for (const targetIndex of targets) {
            const settings = normalizeDatasetDefaults(rows[targetIndex].settings || state.defaults || {});
            settings[key] = value;
            rows[targetIndex].settings = settings;
        }
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
        markDatasetEditorDirty();
        if (options.render) {
            renderDatasetEditor();
        }
    }

    globalThis.updateDatasetEditorRowNlTagMix = function updateDatasetEditorRowNlTagMix(index, nextMix) {
        updateDatasetEditorRowsNlTagMix([index], nextMix);
    }

    globalThis.updateDatasetEditorRowsNlTagMix = function updateDatasetEditorRowsNlTagMix(indices, nextMix) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        const targets = datasetValidTargetIndices(indices, rows.length);
        if (!targets.length) return;
        const mix = normalizeNlTagMix(nextMix);
        for (const targetIndex of targets) {
            rows[targetIndex].nl_tag_mix = mix;
        }
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
        markDatasetEditorDirty();
        renderDatasetEditor();
    }

    globalThis.updateDatasetEditorRowTriggerClone = function updateDatasetEditorRowTriggerClone(index, nextClone, options = {}) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (!rows[index]) return;
        rows[index].trigger_clone = normalizeTriggerClone({
            ...rows[index].trigger_clone,
            ...nextClone,
        });
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
        } else {
            datasetEditorState.datasets = rows;
        }
        markDatasetEditorDirty();
        if (options.render) {
            renderDatasetEditor();
        }
    }

    globalThis.datasetExperimentalScopeKey = function datasetExperimentalScopeKey(index) {
        return `${isDatasetTabActive() ? 'dataset-preset' : 'config-dataset'}:${index}`;
    }

    globalThis.datasetExperimentalScopeIndices = function datasetExperimentalScopeIndices(index, total = null) {
        const state = datasetEditorStateForActivePanel();
        const count = total ?? normalizeDatasetEditorRows(state.datasets).length;
        const key = datasetExperimentalScopeKey(index);
        const raw = datasetExperimentalScopeSelections.get(key) || [index];
        const selected = datasetValidTargetIndices(raw, count);
        if (!selected.length && index >= 0 && index < count) {
            selected.push(index);
        }
        datasetExperimentalScopeSelections.set(key, selected);
        return selected;
    }

    globalThis.setDatasetExperimentalScopeIndices = function setDatasetExperimentalScopeIndices(index, indices) {
        const state = datasetEditorStateForActivePanel();
        const count = normalizeDatasetEditorRows(state.datasets).length;
        const selected = datasetValidTargetIndices(indices, count);
        if (!selected.length && index >= 0 && index < count) {
            selected.push(index);
        }
        datasetExperimentalScopeSelections.set(datasetExperimentalScopeKey(index), selected);
    }

	    globalThis.datasetValidTargetIndices = function datasetValidTargetIndices(indices, count) {
	        return [...new Set((indices || [])
	            .map((value) => Number.parseInt(value, 10))
	            .filter((value) => Number.isInteger(value) && value >= 0 && value < count))]
	            .sort((left, right) => left - right);
	    }

	    globalThis.setDatasetEditorRowsAfterSort = function setDatasetEditorRowsAfterSort(rows) {
	        datasetExperimentalScopeSelections.clear();
	        if (isDatasetTabActive()) {
	            datasetPresetState.datasets = rows;
	        } else {
	            datasetEditorState.datasets = rows;
	        }
	        markDatasetEditorDirty();
	        renderDatasetEditor();
	    }

	    globalThis.moveDatasetEditorRow = function moveDatasetEditorRow(sourceIndex, targetIndex, placeAfter = false) {
	        const rows = normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets);
	        if (rows.length <= 1) return false;
	        if (sourceIndex < 0 || sourceIndex >= rows.length || targetIndex < 0 || targetIndex >= rows.length) return false;
	        if (sourceIndex === targetIndex) return false;
	        let insertIndex = targetIndex + (placeAfter ? 1 : 0);
	        if (sourceIndex < insertIndex) insertIndex -= 1;
	        insertIndex = Math.max(0, Math.min(rows.length - 1, insertIndex));
	        if (insertIndex === sourceIndex) return false;
	        const [moved] = rows.splice(sourceIndex, 1);
	        rows.splice(insertIndex, 0, moved);
	        setDatasetEditorRowsAfterSort(rows);
	        return true;
	    }

	    globalThis.moveDatasetEditorRowToIndex = function moveDatasetEditorRowToIndex(sourceIndex, targetIndex) {
	        const rows = normalizeDatasetEditorRows(datasetEditorStateForActivePanel().datasets);
	        const clamped = Math.max(0, Math.min(rows.length - 1, targetIndex));
	        if (clamped === sourceIndex) return false;
	        return moveDatasetEditorRow(sourceIndex, clamped, clamped > sourceIndex);
	    }

	    globalThis.markDatasetEditorDirty = function markDatasetEditorDirty() {
        if (isDatasetTabActive()) {
            datasetPresetState.dirty = true;
            datasetPresetState.status = '有未保存的数据集修改';
            renderDatasetPresetHeader();
        } else {
            datasetEditorState.dirty = true;
            updateTomlDirtyState();
            updateStepEstimatePanel();
        }
        const dirty = document.querySelector('#dataset-editor .dataset-editor-dirty');
        if (dirty) {
            dirty.classList.add('active');
            dirty.textContent = '有未保存的数据集修改';
        }
    }

    globalThis.addDatasetEditorRow = function addDatasetEditorRow() {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        rows.push({
            source_dir: '',
            image_dir: '',
            cache_dir: '',
            num_repeats: 1,
            trigger_clone: normalizeTriggerClone(DEFAULT_TRIGGER_CLONE),
            settings: normalizeDatasetDefaults(state.defaults || {}),
        });
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
            datasetPresetState.dirty = true;
        } else {
            datasetEditorState.datasets = rows;
            datasetEditorState.dirty = true;
        }
        renderDatasetEditor();
        if (!isDatasetTabActive()) {
            updateTomlDirtyState();
            updateStepEstimatePanel();
        }
    }

    globalThis.removeDatasetEditorRow = function removeDatasetEditorRow(index) {
        const state = datasetEditorStateForActivePanel();
        const rows = normalizeDatasetEditorRows(state.datasets);
        if (rows.length <= 1) return;
        rows.splice(index, 1);
        if (isDatasetTabActive()) {
            datasetPresetState.datasets = rows;
            datasetPresetState.dirty = true;
        } else {
            datasetEditorState.datasets = rows;
            datasetEditorState.dirty = true;
        }
        renderDatasetEditor();
        if (!isDatasetTabActive()) {
            updateTomlDirtyState();
            updateStepEstimatePanel();
        }
    }

    globalThis.syncDatasetEditorToCompatFields = function syncDatasetEditorToCompatFields() {
        const rows = normalizeDatasetEditorRows(datasetEditorState.datasets);
        const first = rows[0];
        if (!first) return;
        setFieldInputValue('source_image_dir', first.source_dir);
        setFieldInputValue('resized_image_dir', first.image_dir);
        setFieldInputValue('lora_cache_dir', first.cache_dir);
        if (datasetEditorState.dataset_config) {
            setFieldInputValue('dataset_config', datasetEditorState.dataset_config);
        }
    }

    globalThis.setFieldInputValue = function setFieldInputValue(key, value) {
        const input = document.querySelector(`#config-form .field-input[data-key="${CSS.escape(key)}"]`);
        if (!input) {
            if (NETWORK_ARG_FIELD_MAP.has(key)) {
                configFormState.draftValues.set(key, value);
            }
            return;
        }
        if (input.type === 'checkbox') {
            input.checked = Boolean(value);
        } else {
            input.value = value ?? '';
        }
        updateConfigDraftFromInput(input);
    }

    globalThis.escapeHtml = function escapeHtml(value) {
        return ctx.format.escapeHtml(value);
    }

    globalThis.setCurrentTrainingSourceFromVariant = function setCurrentTrainingSourceFromVariant(variant) {
        if (!variant) {
            clearCurrentTrainingSource();
            return;
        }
        if (val('method-select') === 'spd' || variant === 'spd') {
            currentTrainingSource = {
                method: 'spd',
                methods_subdir: 'methods',
                file: 'configs/methods/spd.toml',
            };
            return;
        }
        currentTrainingSource = {
            method: variant,
            methods_subdir: 'gui-methods',
            file: `configs/gui-methods/${variant}.toml`,
        };
    }

    globalThis.clearCurrentTrainingSource = function clearCurrentTrainingSource() {
        currentTrainingSource = {
            method: '',
            methods_subdir: '',
            file: '',
        };
    }

    globalThis.outputRunRuntimeFile = function outputRunRuntimeFile(run = selectedOutputRun()) {
        const runtime = (run?.files || []).find((item) => item.kind === 'runtime');
        return runtime?.file || '';
    }

    globalThis.rememberSelectionSnapshot = function rememberSelectionSnapshot() {
        selectionSnapshot.method = val('method-select');
        selectionSnapshot.variant = val('variant-select');
        selectionSnapshot.preset = val('preset-select');
    }

    globalThis.restoreSelectionSnapshot = function restoreSelectionSnapshot() {
        const methodSelect = document.getElementById('method-select');
        const variantSelect = document.getElementById('variant-select');
        const presetSelect = document.getElementById('preset-select');
        if (methodSelect && selectionSnapshot.method && [...methodSelect.options].some((opt) => opt.value === selectionSnapshot.method)) {
            methodSelect.value = selectionSnapshot.method;
        }
        if (variantSelect && selectionSnapshot.variant && [...variantSelect.options].some((opt) => opt.value === selectionSnapshot.variant)) {
            variantSelect.value = selectionSnapshot.variant;
        }
        if (presetSelect && selectionSnapshot.preset && [...presetSelect.options].some((opt) => opt.value === selectionSnapshot.preset)) {
            presetSelect.value = selectionSnapshot.preset;
        }
        setCurrentTrainingSourceFromVariant(val('variant-select'));
        updateChoiceGuide();
    }

    globalThis.confirmBeforeConfigSelectionChange = async function confirmBeforeConfigSelectionChange(message) {
        const ok = await handlePendingConfigSwitch({
            targetLabel: '新的配置选择',
        });
        if (!ok) restoreSelectionSnapshot();
        return ok;
    }

    globalThis.updateChoiceGuide = function updateChoiceGuide(config = currentConfig) {
        const container = document.getElementById('choice-guide');
        if (!container) return;
        container.innerHTML = '';
        const methodKey = activeMethodKey(config);
        container.appendChild(createChoiceCard('方法', methodKey, METHOD_GUIDE_ZH, defaultMethodGuide(), methodGuideFromConfig(methodKey, config)));
        const sourceKey = currentTrainingSource.method || val('variant-select');
        container.appendChild(createChoiceCard('配置', sourceKey, VARIANT_GUIDE_ZH, defaultVariantGuide(), configGuideFromCurrentSource(sourceKey, config)));
        const presetKey = val('preset-select');
        container.appendChild(createChoiceCard('预设', presetKey, PRESET_GUIDE_ZH, defaultPresetGuide(), presetGuideFromConfig(presetKey, config)));
    }

    globalThis.createChoiceCard = function createChoiceCard(kind, key, guideMap, fallback, overrideGuide = null) {
        const guide = overrideGuide || guideMap[key] || fallback;
        const helpId = `choice-guide-hint-${++choiceGuideHintSeq}`;
        const card = document.createElement('article');
        card.className = 'choice-card';

        const heading = document.createElement('div');
        heading.className = 'choice-card-heading';
        const title = document.createElement('strong');
        title.textContent = `${kind}: ${key || '-'}`;
        const name = document.createElement('span');
        name.textContent = guide.title;
        heading.appendChild(title);
        heading.appendChild(name);
        const toggle = document.createElement('button');
        toggle.type = 'button';
        toggle.className = 'info-toggle choice-info-toggle';
        toggle.textContent = '?';
        toggle.title = `展开${kind}说明`;
        toggle.setAttribute('aria-label', `${kind}说明`);
        toggle.setAttribute('aria-expanded', 'false');
        toggle.setAttribute('aria-controls', helpId);
        heading.appendChild(toggle);
        card.appendChild(heading);

        const body = document.createElement('div');
        body.id = helpId;
        body.className = 'choice-card-body';
        body.hidden = true;
        body.appendChild(choiceLine('说明', guide.summary));
        body.appendChild(choiceLine('取舍', guide.tradeoff));
        body.appendChild(choiceLine('推荐', guide.recommend, 'choice-recommend'));
        if (Array.isArray(guide.details) && guide.details.length) {
            const details = document.createElement('ul');
            details.className = 'choice-details';
            for (const detail of guide.details) {
                const item = document.createElement('li');
                item.textContent = detail;
                details.appendChild(item);
            }
            body.appendChild(details);
        }
        toggle.addEventListener('click', () => {
            const nextOpen = body.hidden;
            body.hidden = !nextOpen;
            toggle.classList.toggle('active', nextOpen);
            toggle.setAttribute('aria-expanded', String(nextOpen));
            toggle.title = nextOpen ? `收起${kind}说明` : `展开${kind}说明`;
        });
        card.appendChild(body);
        return card;
    }

    globalThis.choiceLine = function choiceLine(label, text, extraClass = '') {
        const line = document.createElement('p');
        line.className = extraClass;
        const strong = document.createElement('strong');
        strong.textContent = `${label}: `;
        line.appendChild(strong);
        line.appendChild(document.createTextNode(text));
        return line;
    }

    globalThis.defaultMethodGuide = function defaultMethodGuide() {
        return choiceHelp(
            '自定义方法',
            '当前方法没有专门说明，通常表示它来自后端方法列表。',
            '请结合变体 TOML 判断实际训练行为。',
            '不确定时使用 lora。'
        );
    }

    globalThis.defaultVariantGuide = function defaultVariantGuide() {
        return choiceHelp(
            '自定义变体',
            '当前变体对应一个 gui-methods TOML 文件，里面才是实际训练参数。',
            '自定义变体灵活，但需要自行确认字段组合是否合理。',
            '不确定时从内置 lora 变体复制再改。'
        );
    }

    globalThis.defaultPresetGuide = function defaultPresetGuide() {
        return choiceHelp(
            '自定义预设',
            '当前预设来自 presets.toml 或自定义配置。',
            '它会覆盖部分硬件、采样或性能参数。',
            '不确定时使用 default。'
        );
    }

    globalThis.activeMethodKey = function activeMethodKey(config = currentConfig) {
        const inferred = inferMethodFromConfig(config);
        if (inferred) return inferred;
        if (currentTrainingSource.methods_subdir === 'methods' && currentTrainingSource.method === 'spd') {
            return 'spd';
        }
        if (currentTrainingSource.methods_subdir === 'gui-methods') {
            return VARIANT_METHOD_FAMILY[currentTrainingSource.method] || val('method-select') || 'lora';
        }
        return val('method-select') || 'lora';
    }

    globalThis.inferMethodFromConfig = function inferMethodFromConfig(config) {
        if (!config || typeof config !== 'object') return '';
        const moduleName = String(config.network_module || '');
        if (currentTrainingSource.methods_subdir === 'methods' && currentTrainingSource.method === 'spd') return 'spd';
        if ('dit_path' in config && 'iterations' in config && currentTrainingSource.method === 'spd') return 'spd';
        if (isTruthy(config.use_glora)) return 'glora';
        if (isTruthy(config.use_vera)) return 'vera';
        if (isTruthy(config.use_lokr)) return 'lokr';
        if (isTruthy(config.use_loha)) return 'loha';
        if (isTruthy(config.use_easycontrol) || moduleName.includes('easycontrol')) return 'easycontrol';
        if (isTruthy(config.use_ip_adapter) || moduleName.includes('ip_adapter')) return 'ip_adapter';
        if (moduleName.includes('soft_tokens')) return 'soft_tokens';
        if (isTruthy(config.add_reft) || ('reft_dim' in config && Number(config.reft_dim) > 0)) return 'reft';
        if (
            isTruthy(config.use_hydra) ||
            isTruthy(config.use_sigma_router) ||
            String(config.use_moe_style || 'false') !== 'false' ||
            moduleName.includes('chimera') ||
            moduleName.includes('hydra')
        ) {
            if (moduleName.includes('chimera') || 'content_router_source' in config) return 'chimera';
            return 'hydralora';
        }
        if (isTruthy(config.use_timestep_mask)) return 'tlora';
        if (isTruthy(config.use_ortho)) return 'ortholora';
        return '';
    }

    globalThis.methodGuideFromConfig = function methodGuideFromConfig(methodKey, config = currentConfig) {
        const base = METHOD_GUIDE_ZH[methodKey] || defaultMethodGuide();
        const details = compactList([
            flagDetail('use_glora', 'GLoRA', config.use_glora),
            flagDetail('use_vera', 'VeRA', config.use_vera),
            flagDetail('use_lokr', 'LoKr', config.use_lokr),
            flagDetail('use_loha', 'LoHa', config.use_loha),
            flagDetail('dora_wd', 'DoRA', config.dora_wd),
            isTruthy(config.use_lokr) ? valueDetail('lokr_factor', config.lokr_factor) : '',
            isTruthy(config.use_vera) ? valueDetail('vera_projection_prng_key', config.vera_projection_prng_key) : '',
            isTruthy(config.use_vera) ? valueDetail('vera_d_initial', config.vera_d_initial) : '',
            valueDetail('network_dim', config.network_dim),
            valueDetail('network_alpha', config.network_alpha),
            valueDetail('learning_rate', config.learning_rate),
            valueDetail('max_train_epochs', config.max_train_epochs),
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 当前表单已读取关键训练字段。`,
            details,
        };
    }

    globalThis.configGuideFromCurrentSource = function configGuideFromCurrentSource(sourceKey, config = currentConfig) {
        const isImported = currentTrainingSource.methods_subdir === 'imported';
        const base = isImported
            ? choiceHelp(
                '导入训练配置',
                `当前表单来自 ${currentTrainingSource.file || '导入配置'}。`,
                '它会按 base.toml → 当前预设 → 该 TOML 的顺序合并；不会强行加入变体下拉。',
                '适合把历史训练配置作为独立入口继续查看、预检测或训练。'
            )
            : (VARIANT_GUIDE_ZH[sourceKey] || defaultVariantGuide());
        const details = compactList([
            currentTrainingSource.file ? `文件: ${currentTrainingSource.file}` : '',
            config.dataset_config ? `数据集配置: ${config.dataset_config}` : '',
            config.output_name ? `输出名称: ${config.output_name}` : '',
            globalSettings?.output_root ? `Web 输出根目录: ${globalSettings.output_root}` : '',
            config.source_image_dir ? `原始数据集: ${config.source_image_dir}` : '',
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 已读取当前 TOML 的路径和输出信息。`,
            details,
        };
    }

    globalThis.presetGuideFromConfig = function presetGuideFromConfig(presetKey, config = currentConfig) {
        const base = PRESET_GUIDE_ZH[presetKey] || defaultPresetGuide();
        const details = compactList([
            valueDetail('mixed_precision', config.mixed_precision),
            valueDetail('optimizer_type', config.optimizer_type),
            valueDetail('lr_scheduler', config.lr_scheduler),
            valueDetail('train_batch_size', config.train_batch_size),
            valueDetail('gradient_accumulation_steps', config.gradient_accumulation_steps),
            valueDetail('sample_ratio', config.sample_ratio),
        ]);
        if (!details.length) return base;
        return {
            ...base,
            summary: `${base.summary} 当前已合并后的预设/配置值如下。`,
            details,
        };
    }

    globalThis.isTruthy = function isTruthy(value) {
        return value === true || value === 1 || value === '1' || String(value).toLowerCase() === 'true';
    }

    globalThis.normalizeLoraAdapterKind = function normalizeLoraAdapterKind(value) {
        const text = String(value ?? '').trim().toLowerCase();
        if (text === 'loha' || text === 'lokr' || text === 'glora' || text === 'vera') return text;
        return 'lora';
    }
