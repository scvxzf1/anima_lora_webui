/* Generic config page renderer.
 * Renders a config sub-page with fields, generous spacing,
 * help text, and save/restore actions.
 * Uses the existing config catalog (form-layout.js, labels-options.js, etc.)
 * Language: Chinese labels and descriptions, English config keys hidden by default.
 */

import { FIELD_LABEL_ZH, FIELD_OPTIONS } from '../../config/catalog/labels-options.js?v=dragon-ui-20260902-krea2-pp-v1';
import { FIELD_HELP_SUMMARY_ZH } from '../../config/catalog/field-help-summary.js?v=dragon-ui-20260902-krea2-pp-v1';
import { configFieldPlaceholder } from '../../config/catalog/field-placeholders.js?v=dragon-ui-20260830v2';
import {
    ALL_LORA_ADAPTER_SCOPED_FIELD_KEYS,
    CONFIG_FORM_INTERNAL_KEYS,
    CONFIG_FORM_MERGED_FIELDS,
    DATASET_BLUEPRINT_FIELDS,
    DEPRECATED_CONFIG_FORM_FIELDS,
    FORM_UI_DEFAULTS,
    LOKR_SCOPED_FIELD_KEYS,
    RETIRED_CONFIG_FORM_FIELDS,
    VERA_SCOPED_FIELD_KEYS,
} from '../../config/catalog/defaults.js?v=dragon-ui-20260902-krea2-pp-v1';
import { VARIANT_METHOD_FAMILY } from '../../config/catalog/form-layout.js?v=dragon-ui-20260902-krea2-pp-v1';
import { createApiClient } from '../../shared/api.js?v=dragon-ui-20260812v35';
import {
    alertDragonDialog,
    confirmDragonDialog,
} from '../../shared/dialog.js?v=module-bootstrap-20260901-dialog-v1';
import { escapeHtml } from '../../shared/format.js?v=dragon-ui-20260812v35';
import { SECTION_GROUPS } from './section-groups.js?v=dragon-ui-20260902-krea2-pp-v1';
import { findCategory, isConfigCategory } from '../category-map.js?v=dragon-ui-20260826v45';
import { renderIcon } from '../icons.js?v=dragon-ui-20260812v35';
import { scanForReveal } from '../animations.js?v=dragon-ui-20260824v69';
import { dragonScrollBehavior } from '../motion.js?v=dragon-ui-20260824v1';
import { keysForConfigSubItem } from './config-field-map.js?v=dragon-ui-20260902-krea2-pp-v1';
import { configValueForControl, displayConfigValue, prepareConfigPatch, serializeConfigValue } from './config-values.js?v=dragon-ui-20260902-lokr-backend-v4';
import {
    createConfigDirtyBindings,
    renderConfigDirtyState,
    replaceConfigDirtyKeys,
    updateConfigDirtyKey,
} from './config-dirty-state.js?v=dragon-ui-20260826v1';
import {
    bindAllConfigWorkspace,
    isAllConfigView,
    renderAllConfigWorkspace,
    renderConfigViewSwitch,
    resolveConfigView,
    scrollConfigCanvasTo,
    uniqueConfigEntries,
} from './config-all-view.js?v=dragon-ui-20260825v15';
import { buildConfigBlocks } from './config-block-metadata.js?v=dragon-ui-20260902-krea2-pp-v1';
import {
    configFieldAvailability,
    resolveConfigAdapterKind,
} from './config-field-availability.js?v=dragon-ui-20260902-krea2-pp-v1';
import {
    bindConfigFieldHelpDialog,
    configHelpSummary,
    renderConfigHelpButton,
    resolveConfigFieldHelp,
} from './config-field-help.js?v=dragon-ui-20260902-lokr-availability-v1';
import {
    bindConfigViewPreference,
    persistConfigBilingual,
    persistConfigCapsuleMode,
    persistConfigViewMode,
    persistPresetLibraryCollapsed,
    preferredConfigBilingual,
    preferredConfigCapsuleMode,
    preferredConfigSubId,
    presetLibraryCollapsed,
} from './config-ui-preferences.js?v=dragon-ui-20260831v3';
import {
    booleanDefaultForKey,
    BOOLEAN_CONFIG_DEFAULTS,
    isBooleanConfigField,
    normalizeBooleanConfigValue,
} from './config-field-types.js?v=dragon-ui-20260902-lokr-backend-v4';
import { bindTrainingControls, isEditableConfigFile, loadTrainingContext, mergedConfigUrl, renderTrainingControls, selectTrainingConfigFile, selectTrainingPreset, commitTrainingContext } from './training-controls.js?v=dragon-ui-20260901v115';
import { bindTrainingPresetLibrary, renderTrainingPresetLibrary } from './training-preset-library.js?v=dragon-ui-20260901v116';
import {
    bindLazyModelQuickPicker,
    MODEL_QUICK_PATH_KEYS,
    renderModelQuickPickerDialog,
    renderModelQuickPickerTrigger,
} from './model-quick-picker-shell.js?v=dragon-ui-20260826v1';
import {
    bindTrainingDataTools,
    renderDatasetConfigField,
    renderDatasetPickerDialog,
    renderStepEstimatePanel,
} from './config-training-data.js?v=dragon-ui-20260826v7';
import {
    bindSamplePromptsDialog,
    renderSamplePromptsDialog,
    renderSamplePromptsFieldControl,
} from './sample-prompts-dialog.js?v=dragon-ui-20260902-sample-prompts-v4';

const api = createApiClient();
let fieldHelpCatalogPromise = null;

function loadFieldHelpCatalog() {
    if (!fieldHelpCatalogPromise) {
        fieldHelpCatalogPromise = import('../../config/catalog/field-help.js?v=dragon-ui-20260902-krea2-pp-v1')
            .then((module) => module.FIELD_HELP_ZH)
            .catch((error) => {
                fieldHelpCatalogPromise = null;
                throw error;
            });
    }
    return fieldHelpCatalogPromise;
}

const HIDDEN_CONFIG_KEYS = new Set([
    'output_dir',
    'general',
    'datasets',
    'stage_schedule',
    'stage_schedule_enabled',
    'mixed_precision',
    'full_fp16',
    'full_bf16',
]);

function categoryLabel(sub) {
    if (!sub || !sub.categoryId) return "";
    const cat = findCategory(sub.categoryId);
    return cat ? cat.label : "";
}

function categoryDescription(categoryId) {
    const descriptions = {
        'memory-optimization': '显存、计算精度、编译、缓存和数据传输设置。',
        'advanced-methods': 'LoRA 扩展、条件注入、路由、损失加权和实验工具。',
    };
    return descriptions[categoryId] || '';
}

function activeMethodFamily(trainingContext, values = {}) {
    const variant = String(trainingContext?.variant || '').trim().toLowerCase();
    if (trainingContext?.methodsSubdir === 'methods' && variant === 'spd') return 'spd';
    if (VARIANT_METHOD_FAMILY[variant]) return VARIANT_METHOD_FAMILY[variant];
    if (normalizeBooleanConfigValue('use_ip_adapter', values.use_ip_adapter)
        || String(values.network_module || '').includes('ip_adapter')) return 'ip_adapter';
    if (normalizeBooleanConfigValue('use_easycontrol', values.use_easycontrol)
        || String(values.network_module || '').includes('easycontrol')) return 'easycontrol';
    if (String(values.network_module || '').includes('soft_tokens')) return 'soft_tokens';
    if (normalizeBooleanConfigValue('use_chimera_hydra', values.use_chimera_hydra)) return 'chimera';
    return 'lora';
}

function activeAdapterKind(values = {}) {
    return resolveConfigAdapterKind(values);
}

function configAvailabilityContext(trainingContext, values = {}) {
    return {
        method: activeMethodFamily(trainingContext, values),
        adapter: activeAdapterKind(values),
        baseCompute: String(values.base_compute || 'bf16').trim().toLowerCase(),
        modelFamily: String(values.model_family || 'anima').trim().toLowerCase(),
        pipelineParallel: normalizeBooleanConfigValue('pipeline_parallel', values.pipeline_parallel),
    };
}

function visibleConfigKeys(keys, trainingContext, values) {
    return keys.filter((key) => {
        if (HIDDEN_CONFIG_KEYS.has(key)
            || CONFIG_FORM_INTERNAL_KEYS.has(key)
            || CONFIG_FORM_MERGED_FIELDS.has(key)
            || DATASET_BLUEPRINT_FIELDS.has(key)
            || DEPRECATED_CONFIG_FORM_FIELDS.has(key)
            || RETIRED_CONFIG_FORM_FIELDS.has(key)) return false;
        return !ALL_LORA_ADAPTER_SCOPED_FIELD_KEYS.has(key)
            || key === 'lora_adapter_kind'
            || LOKR_SCOPED_FIELD_KEYS.has(key)
            || VERA_SCOPED_FIELD_KEYS.has(key)
            || key === 'dora_wd';
    });
}

function unclassifiedConfigKeys(values, knownKeys) {
    return Object.entries(values).filter(([key, value]) => {
        if (knownKeys.has(key) || HIDDEN_CONFIG_KEYS.has(key)) return false;
        if (value !== null && typeof value === 'object' && !Array.isArray(value)) return false;
        return true;
    }).map(([key]) => key);
}

function buildCategoryEntries(category, rawEntries, trainingContext, values) {
    const knownKeys = new Set(rawEntries.flatMap((entry) => entry.keys));
    const sourceEntries = category?.id === 'training-config'
        ? rawEntries.map((entry) => entry.sub.id === 'advanced'
            ? { ...entry, keys: [...new Set([...entry.keys, ...unclassifiedConfigKeys(values, knownKeys)])] }
            : entry)
        : rawEntries;
    return sourceEntries.map((entry) => ({
        ...entry,
        keys: visibleConfigKeys(entry.keys, trainingContext, values),
    })).filter((entry) => entry.keys.length > 0);
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

    let entries = isCategoryPage
        ? buildCategoryEntries(category, rawEntries, trainingContext, currentValues)
        : [];
    const activeView = isCategoryPage
        ? resolveConfigView(entries, preferredConfigSubId(context.subId, category), category)
        : { sub, keys: visibleConfigKeys(keysForConfigSubItem(sub), trainingContext, currentValues), entries: [], isAll: false };
    const activeSub = activeView.sub || sub;
    const keys = activeView.keys;
    if (!keys || keys.length === 0) {
        return '<div class="dragon-empty-state"><p>此分类暂无当前方法可生效的配置项</p></div>';
    }

    const pageState = {
        dirty: false,
        beforeUnload: null,
        leavePrompt: null,
        showChangedOnly: false,
        searchQuery: '',
        capsuleMode: preferredConfigCapsuleMode(),
        bilingual: preferredConfigBilingual(),
        activeTag: 'all',
        radarTag: null,
        availabilityContext: configAvailabilityContext(trainingContext, currentValues),
    };
    resetConfigFormState(pageState, currentValues, isCategoryPage ? entries : [{ sub: activeSub, keys }]);
    if (isCategoryPage) {
        wrapper.innerHTML = renderCategorySubPage(
            pageCategory,
            entries,
            activeView,
            pageState.draftValues,
            trainingContext,
            pageState.bilingual,
            pageState.availabilityContext,
        );
    } else {
        wrapper.innerHTML = renderSingleConfigPage(
            activeSub,
            keys,
            currentValues,
            trainingContext,
            pageState.draftValues,
            pageState.bilingual,
            pageState.availabilityContext,
        );
    }

    let routeUpdater = null;
    let disposeMountedPage = () => cleanupConfigPage(pageState);
    return {
        html: wrapper.innerHTML,
        onMount: (root) => {
            if (!isCategoryPage) {
                const saveChanges = wireConfigInteractions(root, keys, trainingContext, pageState);
                const beforeContextChange = () => confirmConfigDiscard(pageState, '切换训练配置');
                bindTrainingControls(root, trainingContext, { saveChanges, beforeContextChange });
                return;
            }

            let committed = { context: trainingContext, ...activeView };
            let requestedContext = trainingContext;
            let libraryController = null;
            let allConfigCleanup = null;
            let filterCleanup = null;
            let bilingualCleanup = null;
            let viewPreferenceCleanup = null;
            let saveCurrentChanges = null;
            let transitionSequence = 0;
            pageState.onDirtyChange = (dirty) => libraryController?.updateDirty(dirty);
            const beforeContextChange = () => confirmConfigDiscard(pageState, '切换训练配置');
            disposeMountedPage = () => {
                transitionSequence += 1;
                allConfigCleanup?.();
                filterCleanup?.();
                bilingualCleanup?.();
                viewPreferenceCleanup?.();
                libraryController?.destroy?.();
                cleanupConfigPage(pageState);
            };

            const bindEditablePane = () => {
                allConfigCleanup?.();
                filterCleanup?.();
                bilingualCleanup?.();
                viewPreferenceCleanup?.();
                allConfigCleanup = null;
                filterCleanup = null;
                bilingualCleanup = null;
                viewPreferenceCleanup = bindConfigViewPreference(root);
                saveCurrentChanges = wireConfigInteractions(root, committed.keys, committed.context, pageState, { allView: committed.isAll });
                filterCleanup = bindConfigFieldFilter(root, pageState);
                bilingualCleanup = bindConfigBilingualToggle(root, pageState);
                if (committed.isAll) {
                    allConfigCleanup = bindAllConfigWorkspace(root, {
                        defaultPresetCollapsed: pageState.presetCollapsed
                            ?? presetLibraryCollapsed(window.matchMedia?.('(max-width: 1440px)').matches ?? false),
                        onPresetCollapseChange: (collapsed) => {
                            pageState.presetCollapsed = collapsed;
                            persistPresetLibraryCollapsed(collapsed);
                        },
                        onRadarChange: (tag) => {
                            pageState.radarTag = tag;
                            pageState.radarUpdate?.();
                        },
                    });
                } else {
                    const shell = root.querySelector('.dragon-config-shell-layout');
                    if (shell) shell.dataset.presetCollapsed = 'false';
                }
                bindTrainingControls(root, committed.context, {
                    saveChanges: saveCurrentChanges,
                    beforeContextChange,
                    onConfigFileChange: (file) => {
                        const nextContext = selectTrainingConfigFile(committed.context, file, { notify: false, persist: false });
                        return nextContext ? transitionEditable({ context: nextContext }) : false;
                    },
                    onPresetChange: (preset) => {
                        return transitionEditable({ context: selectTrainingPreset(committed.context, preset, { notify: false, persist: false }) });
                    },
                });
            };

            const transitionEditable = async ({ context = requestedContext, sub: nextSub = committed.sub } = {}) => {
                const requestedSubId = nextSub?.id;
                const contextChanged = context.configFile !== committed.context.configFile
                    || context.preset !== committed.context.preset
                    || context.methodsSubdir !== committed.context.methodsSubdir
                    || context.variant !== committed.context.variant;
                if (!contextChanged && requestedSubId === committed.sub.id) return true;

                requestedContext = context;
                const sequence = ++transitionSequence;
                const pane = root.querySelector('[data-config-editable-pane]');
                pane?.setAttribute('aria-busy', 'true');
                try {
                    let values = pageState.baselineValues;
                    if (contextChanged) {
                        const res = await api(mergedConfigUrl(context));
                        if (!res || res.ok === false) throw new Error(res?.error || '后端没有返回可用配置');
                        values = res.config || res;
                    }
                    if (sequence !== transitionSequence) return true;
                    const nextEntries = contextChanged
                        ? buildCategoryEntries(category, rawEntries, context, values)
                        : entries;
                    const nextView = resolveConfigView(nextEntries, requestedSubId, category);
                    if (!nextView.sub) return false;

                    cleanupConfigPage(pageState);
                    if (contextChanged) resetConfigFormState(pageState, values, nextEntries);
                    entries = nextEntries;
                    committed = { context, ...nextView };
                    persistConfigViewMode(committed.isAll);
                    requestedContext = context;
                    commitTrainingContext(context);
                    currentValues = values;
                    pageState.availabilityContext = configAvailabilityContext(context, values);
                    const currentPane = root.querySelector('[data-config-editable-pane]');
                    if (currentPane) currentPane.outerHTML = renderEditableConfigPane(pageCategory, entries, {
                        ...committed,
                        availabilityContext: pageState.availabilityContext,
                        values: pageState.draftValues,
                        bilingual: pageState.bilingual,
                    });
                    bindEditablePane();
                    scanForReveal();
                    libraryController?.updateContext(committed.context);
                    return true;
                } catch (error) {
                    if (sequence !== transitionSequence) return true;
                    requestedContext = committed.context;
                    libraryController?.updateContext(committed.context);
                    root.querySelector('[data-config-editable-pane]')?.removeAttribute('aria-busy');
                    await alertDragonDialog({
                        eyebrow: '训练配置',
                        title: '切换配置失败',
                        message: error.message || String(error),
                        description: '当前配置和未保存修改均已保留，请检查后重试。',
                        tone: 'danger',
                        icon: 'x',
                        confirmText: '关闭',
                    });
                    return false;
                }
            };

            bindEditablePane();
            libraryController = bindTrainingPresetLibrary(root, committed.context, {
                beforeContextChange,
                onSaveChanges: () => saveCurrentChanges?.() ?? false,
                onConfigFileChange: (_file, nextContext) => transitionEditable({ context: nextContext }),
            });
            libraryController?.updateDirty(pageState.dirty);

            routeUpdater = async ({ subId }) => {
                const target = resolveConfigView(entries, preferredConfigSubId(subId, category), category);
                if (!target.sub) return false;
                const updated = await transitionEditable({ context: requestedContext, sub: target.sub });
                if (!updated) return false;
                const detail = root.querySelector(`[data-config-entry="${target.sub.id}"]`);
                const navHeight = Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--dragon-nav-height')) || 44;
                if (detail) window.scrollTo({ top: Math.max(0, window.scrollY + detail.getBoundingClientRect().top - navHeight - 16), behavior: dragonScrollBehavior() });
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

function renderSingleConfigPage(
    sub,
    keys,
    currentValues,
    trainingContext,
    draftValues = null,
    bilingual = false,
    availabilityContext = null,
) {
    return `
        <div class="dragon-config-page" data-config-bilingual="${Boolean(bilingual)}">
            <div class="dragon-config-hero dragon-reveal">
                <span class="dragon-eyebrow">${categoryLabel(sub)}</span>
                <h1>${sub.label}</h1>
                <p>${sub.desc || ''}</p>
            </div>
            ${renderTrainingControls(trainingContext)}
            <div class="dragon-reveal" data-stagger="1" id="dragon-config-fields">
                ${renderFields(sub.id, keys, currentValues, { draftValues, availabilityContext })}
            </div>
            ${renderConfigActions()}
            ${renderSamplePromptsDialog()}
        </div>
    `;
}

function renderCategorySubPage(
    category,
    entries,
    view,
    draftValues,
    trainingContext,
    bilingual = false,
    availabilityContext = null,
) {
    const description = categoryDescription(category.id);

    return `
        <div class="dragon-config-page dragon-config-category-page dragon-config-subpage"
             data-config-bilingual="${Boolean(bilingual)}"
             data-config-category="${category.id}" data-config-subpage="${view.sub.id}">
            <div class="dragon-config-hero dragon-reveal">
                <h1>${category.label}</h1>
                ${description ? `<p>${description}</p>` : ''}
            </div>
            <div class="dragon-config-shell-layout">
                ${renderEditableConfigPane(category, entries, {
                    context: trainingContext,
                    availabilityContext,
                    ...view,
                    values: draftValues,
                    bilingual,
                })}
                ${renderTrainingPresetLibrary(trainingContext)}
            </div>
        </div>
    `;
}

function renderEditableConfigPane(category, entries, state) {
    const workspaceEntries = state.isAll ? state.entries : entries;
    return `<div class="dragon-config-editable-pane" data-config-editable-pane>
        ${renderTrainingControls(state.context)}
        ${renderEditableConfigWorkspace(
            category,
            workspaceEntries,
            state.sub,
            state.keys,
            state.values,
            state.bilingual,
            state.availabilityContext,
        )}
        ${renderSamplePromptsDialog()}
    </div>`;
}

function renderEditableConfigWorkspace(
    category,
    entries,
    sub,
    keys,
    currentValues,
    bilingual = false,
    availabilityContext = null,
) {
    if (isAllConfigView(category, sub.id)) {
        const { blocks, chapters } = buildConfigBlocks(
            entries,
            currentValues,
            FIELD_OPTIONS,
            FORM_UI_DEFAULTS,
            availabilityContext,
        );
        return renderAllConfigWorkspace({
            blocks,
            chapters,
            bilingual,
            renderBlock: (block) => renderField(block.key, fieldDisplayValue(block.key, currentValues, currentValues), block),
            renderActions: () => renderConfigActions({ allView: true }),
            renderChapterLead: (chapter) => chapter.id === 'foundation'
                ? renderDatasetConfigField(currentValues.dataset_config || '', { chapterId: 'foundation', tone: 'required', span: 2, required: true })
                : '',
            renderChapterFooter: (chapter) => chapter.id === 'training' ? renderStepEstimatePanel() : '',
            renderModelPickerTrigger: renderModelQuickPickerTrigger,
            renderModelPickerDialog: renderModelQuickPickerDialog,
            renderDatasetDialog: renderDatasetPickerDialog,
        });
    }
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
                ${sub.id === 'required' ? renderModelQuickPickerTrigger() : ''}
            </header>
            ${renderConfigFieldFilter(keys.length)}
            <div class="dragon-config-detail-fields dragon-reveal" data-stagger="1" id="dragon-config-fields">
                ${sub.id === 'required' ? renderDatasetConfigField(currentValues.dataset_config || '') : ''}
        ${renderFields(sub.id, keys, currentValues, {
            draftValues: currentValues,
            availabilityContext,
        })}
            </div>
            ${sub.id === 'common' ? renderStepEstimatePanel() : ''}
            ${renderConfigActions()}
        </section>
        ${sub.id === 'required' ? renderModelQuickPickerDialog() : ''}
        ${sub.id === 'required' ? renderDatasetPickerDialog() : ''}
    </div>`;
}

function renderConfigFieldFilter(total) {
    return `<div class="dragon-config-field-filter">
        <input class="dragon-input" type="search" autocomplete="off" data-config-field-search
               aria-label="搜索当前配置分组" placeholder="搜索当前分组">
        <output class="dragon-config-field-filter-count" data-config-field-filter-count>${total} 项</output>
    </div>`;
}

function bindConfigBilingualToggle(root, state) {
    const page = root.querySelector('.dragon-config-page') || root;
    const toggle = root.querySelector('[data-config-bilingual-toggle]');
    if (!page && !toggle) return null;

    const sync = () => {
        const enabled = state ? Boolean(state.bilingual) : page.dataset.configBilingual === 'true';
        page.dataset.configBilingual = String(enabled);
        if (!toggle) return;
        const action = enabled ? '关闭双语渲染' : '开启双语渲染';
        toggle.dataset.active = String(enabled);
        toggle.setAttribute('aria-pressed', String(enabled));
        toggle.setAttribute('aria-label', action);
        toggle.title = action;
    };
    const onClick = () => {
        if (state) {
            state.bilingual = !Boolean(state.bilingual);
            persistConfigBilingual(state.bilingual);
        }
        sync();
    };

    toggle?.addEventListener('click', onClick);
    sync();
    return () => toggle?.removeEventListener('click', onClick);
}

function bindConfigFieldFilter(root, state) {
    const input = root.querySelector('[data-config-field-search]');
    const output = root.querySelector('[data-config-field-filter-count]');
    const fieldsRoot = root.querySelector('#dragon-config-fields');
    if (!input || !output || !fieldsRoot) return null;
    const fields = [...fieldsRoot.querySelectorAll('.dragon-field')];
    const fieldRecords = fields.map((field) => {
        const key = field.querySelector('[data-key]')?.dataset.key || '';
        return {
            field,
            key,
            searchText: field.dataset.searchText || `${key} ${field.textContent || ''}`.toLocaleLowerCase(),
        };
    });
    const blockFlow = fieldsRoot.classList.contains('dragon-config-block-grid');
    const tagButtons = [...root.querySelectorAll('[data-config-tag-filter]')];
    const modeButtons = [...root.querySelectorAll('[data-config-capsule-mode]')];
    const details = [...fieldsRoot.querySelectorAll('details.dragon-config-section')];
    const sectionRecords = [...fieldsRoot.querySelectorAll('.dragon-config-section')]
        .map((section) => ({ section, fields: [...section.querySelectorAll('.dragon-field')] }));
    const groupRecords = [...fieldsRoot.querySelectorAll('[data-config-filter-group]')]
        .map((group) => ({ group, fields: [...group.querySelectorAll('.dragon-field')] }));
    const openStates = new Map(details.map((detail) => [detail, detail.open]));
    let previousQuery = '';
    let searchTimer = null;
    input.value = state?.searchQuery || '';
    if (state && !state.activeTag) state.activeTag = 'all';
    if (state && !state.capsuleMode) state.capsuleMode = 'jump';

    const syncCapsuleUI = () => {
        const active = state?.capsuleMode === 'filter'
            ? (state.activeTag || 'all')
            : (state?.radarTag || 'all');
        tagButtons.forEach((button) => {
            button.dataset.active = String(active === button.dataset.configTagFilter);
        });
        modeButtons.forEach((button) => {
            const selected = (state?.capsuleMode || 'jump') === button.dataset.configCapsuleMode;
            button.dataset.active = String(selected);
            button.setAttribute('aria-pressed', String(selected));
        });
    };

    const update = () => {
        const query = String(input.value || '').trim().toLocaleLowerCase();
        if (state) state.searchQuery = input.value;
        const filterMode = state?.capsuleMode === 'filter';
        const filterTag = filterMode ? (state.activeTag || 'all') : 'all';
        let visible = 0;
        fieldRecords.forEach(({ field, key, searchText }) => {
            const matchesQuery = !query || searchText.includes(query);
            const matchesChanged = !state?.showChangedOnly || state.dirtyKeys?.has(key);
            const matchesTag = !blockFlow || filterTag === 'all' || field.dataset.configTag === filterTag;
            const participates = matchesChanged && matchesTag;
            const hideSearchMismatch = blockFlow && filterMode && !matchesQuery;
            field.hidden = blockFlow ? (!participates || hideSearchMismatch) : !(participates && matchesQuery);
            field.dataset.searchMuted = String(Boolean(blockFlow && !filterMode && query && !matchesQuery));
            field.dataset.searchMatch = String(Boolean(blockFlow && query && matchesQuery));
            if (participates && matchesQuery) visible += 1;
        });
        sectionRecords.forEach(({ section, fields: sectionFields }) => {
            const hasVisibleField = sectionFields.some((field) => !field.hidden);
            section.hidden = !hasVisibleField;
            if (query && hasVisibleField && section.tagName === 'DETAILS') section.open = true;
        });
        groupRecords.forEach(({ group, fields: groupFields }) => {
            group.hidden = !groupFields.some((field) => !field.hidden);
        });
        if (previousQuery && !query) details.forEach((detail) => { detail.open = openStates.get(detail); });
        previousQuery = query;
        syncCapsuleUI();
        const filtered = query || state?.showChangedOnly || filterTag !== 'all';
        output.textContent = filtered ? `匹配 ${visible} / ${fields.length} 项` : `${fields.length} 项`;
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
    tagButtons.forEach((button) => button.addEventListener('click', () => {
        const tag = button.dataset.configTagFilter || 'all';
        if (state?.capsuleMode === 'filter') {
            state.activeTag = tag;
            update();
            return;
        }
        if (state) state.radarTag = tag;
        syncCapsuleUI();
        const target = tag === 'all'
            ? fieldsRoot.querySelector('[data-config-section-divider]')
            : fieldsRoot.querySelector(`[data-config-section="${tag}"]`);
        scrollConfigCanvasTo(fieldsRoot, target, dragonScrollBehavior());
    }));
    modeButtons.forEach((button) => button.addEventListener('click', () => {
        if (state) {
            state.capsuleMode = button.dataset.configCapsuleMode || 'jump';
            persistConfigCapsuleMode(state.capsuleMode);
        }
        update();
    }));
    if (state) {
        state.filterUpdate = update;
        state.radarUpdate = syncCapsuleUI;
    }
    update();
    return () => {
        input.removeEventListener('input', scheduleSearchUpdate);
        if (searchTimer) window.clearTimeout(searchTimer);
        searchTimer = null;
        if (state?.filterUpdate === update) state.filterUpdate = null;
        if (state?.radarUpdate === syncCapsuleUI) state.radarUpdate = null;
    };
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
                ${category.id === 'training-config' ? renderConfigViewSwitch(activeId) : ''}
                <div class="dragon-config-index-heading">配置分组</div>
                <div class="dragon-config-index-groups">${groupsHtml}</div>
            </nav>`;
}

function renderConfigActions({ allView = false } = {}) {
    return `
        <div class="dragon-config-actions dragon-config-actions-sticky">
            <div class="dragon-config-change-summary">
                <output data-config-dirty-count aria-live="polite">未修改</output>
                <button class="dragon-btn dragon-btn-ghost dragon-btn-sm" type="button" data-config-changed-only disabled>
                    ${renderIcon('eye', 'dragon-btn-icon')}<span>查看修改</span>
                </button>
            </div>
            <button class="dragon-btn dragon-btn-secondary" type="button" id="dragon-config-reset">
                ${renderIcon('refresh', 'dragon-btn-icon')}<span>${allView ? '恢复全部默认' : '恢复默认'}</span>
            </button>
            <button class="dragon-btn dragon-btn-primary" type="button" id="dragon-config-save">
                ${renderIcon('check', 'dragon-btn-icon')}<span>保存配置</span>
            </button>
        </div>
        <div class="dragon-config-feedback" id="dragon-config-feedback" role="status" aria-live="polite"></div>
    `;
}

function renderFields(
    subId,
    keys,
    currentValues,
    { draftValues = null, forceOpen = false, availabilityContext = null } = {},
) {
    const groups = subId ? SECTION_GROUPS[subId] : null;

    if (groups && groups.length > 0) {
        const groupedKeys = new Set(groups.flatMap((section) => section.keys));
        const sectionsHtml = groups.map((section) => {
            const sectionKeys = section.keys.filter((k) => keys.includes(k));
            if (sectionKeys.length === 0) return '';
            return renderConfigSection(section, sectionKeys, currentValues, {
                draftValues,
                forceOpen,
                availabilityContext,
            });
        }).join('');
        const remainingKeys = keys.filter((key) => !groupedKeys.has(key));
        if (!remainingKeys.length) return sectionsHtml;
        return `${sectionsHtml}${renderConfigSection({
            eyebrow: '补充设置',
            title: '其他可用参数',
            desc: '当前配置文件中尚未归入共享目录的参数。',
            collapsible: true,
            open: false,
        }, remainingKeys, currentValues, { draftValues, forceOpen, availabilityContext })}`;
    }

    return renderSectionFields(keys, currentValues, draftValues, availabilityContext);
}

function renderConfigSection(
    section,
    keys,
    currentValues,
    { draftValues = null, forceOpen = false, availabilityContext = null } = {},
) {
    const header = `
        <div class="dragon-config-section-header">
            <span class="dragon-eyebrow">${escapeHtml(section.eyebrow || '')}</span>
            <h2 class="dragon-config-section-title">${escapeHtml(section.title || '')}</h2>
            <p class="dragon-config-section-desc">${escapeHtml(section.desc || '')}</p>
        </div>`;
    const fields = renderSectionFields(keys, currentValues, draftValues, availabilityContext);
    if (!section.collapsible) {
        return `<div class="dragon-config-section">${header}${fields}</div>`;
    }
    return `
        <details class="dragon-config-section dragon-config-section-collapsible" ${section.open || forceOpen ? 'open' : ''}>
            <summary class="dragon-config-section-summary">
                ${header}
                <span class="dragon-config-section-count">${keys.length} 项</span>
            </summary>
            <div class="dragon-config-section-body">${fields}</div>
        </details>`;
}

/* Render fields within a section, using 2-col grid for compact inputs */
function renderSectionFields(keys, currentValues, draftValues = null, availabilityContext = null) {
    const toggleKeys = keys.filter((k) => {
        const v = fieldDisplayValue(k, currentValues, draftValues);
        const options = FIELD_OPTIONS[k];
        return isBooleanConfigField(k, v, options);
    });
    const inputKeys = keys.filter((k) => !toggleKeys.includes(k));

    let html = '';

   /* Keep related switches together so long configuration sections remain scannable. */
    if (toggleKeys.length > 0) {
        const toggleHtml = toggleKeys.map((key) =>
            renderField(key, fieldDisplayValue(key, currentValues, draftValues), null, availabilityContext)
        ).join('');
        html += toggleKeys.length > 1
            ? `<div class="dragon-toggle-grid">${toggleHtml}</div>`
            : toggleHtml;
    }

   /* Input/select fields: 2-column grid for compact layout */
   if (inputKeys.length > 0) {
       const gridHtml = inputKeys.map((key) =>
           renderField(key, fieldDisplayValue(key, currentValues, draftValues), null, availabilityContext)
       ).join('');
       if (inputKeys.length >= 2) {
           html += `<div class="dragon-field-grid-2">${gridHtml}</div>`;
       } else {
           html += gridHtml;
       }
   }

    return html;
}

function fieldDisplayValue(key, currentValues, draftValues) {
    if (draftValues && Object.prototype.hasOwnProperty.call(draftValues, key)) return draftValues[key];
    return displayConfigValue(key, currentValues);
}

function renderConfigSemanticMarkers(block) {
    if (!block) return '';
    const required = block.required
        ? '<span class="dragon-config-required-dot" title="必填参数" aria-label="必填"></span>'
        : '';
    const experimental = block.experimental
        ? '<span class="dragon-config-exp-badge" title="实验性参数">EXP</span>'
        : '';
    return required + experimental;
}

function renderField(key, value, block = null, availabilityContext = null) {
    if (key === 'dataset_config') return renderDatasetConfigField(value, block);
    const label = FIELD_LABEL_ZH[key] || key;
    const options = FIELD_OPTIONS[key];
    const availability = block?.availability
        || (availabilityContext ? configFieldAvailability(key, availabilityContext) : null);
    const unavailableReason = availability?.enabled === false ? availability.reason : '';
    const unavailable = Boolean(unavailableReason);
    const booleanField = isBooleanConfigField(key, value, options);
    const controlValue = booleanField ? normalizeBooleanConfigValue(key, value) : value;
    const helpSummary = FIELD_HELP_SUMMARY_ZH[key]
        || configHelpSummary(resolveConfigFieldHelp(key, label, null));
    const fieldToken = configFieldToken(key);
    const fieldId = `dragon-config-field-${fieldToken}`;
    const name = `config_${String(key).replace(/[^A-Za-z0-9_-]+/g, '_')}`;
    const placeholder = configFieldPlaceholder(key, label);
    const fieldSize = block ? (block.span === 2 ? 'wide' : 'compact') : configFieldSize(key, controlValue, options);
    const blockClass = block ? ' dragon-config-block' : '';
    const availabilityAttributes = ` data-config-availability="${unavailable ? 'unavailable' : 'available'}"${unavailable ? ` data-config-unavailable-reason="${escapeHtml(unavailableReason)}"` : ''}`;
    const blockAttributes = block
        ? ` data-field-span="${block.span}" data-config-tag="${escapeHtml(block.chapterId)}" data-config-tone="${escapeHtml(block.tone)}" data-control-kind="${escapeHtml(block.control)}" data-required="${block.required}" data-experimental="${block.experimental}" data-path-field="${block.pathField}" data-search-text="${escapeHtml(`${key} ${label} ${helpSummary} ${block.tagLabel} ${block.chapterLabel} ${value ?? ''}`.toLocaleLowerCase())}"`
        : '';
    const tagBadge = block ? `<span class="dragon-config-block-tag" data-tone="${escapeHtml(block.tone)}">${escapeHtml(block.tagLabel)}</span>` : '';
    const semanticMarkers = renderConfigSemanticMarkers(block);
    const unavailableBadge = unavailable ? renderConfigUnavailableBadge() : '';
    const helpButton = renderConfigHelpButton(key, label, unavailable ? { unavailableReason } : undefined);
    const pathTooltip = block?.pathField && typeof value === 'string' && value
        ? `<span class="dragon-config-path-tooltip" role="tooltip">${escapeHtml(value)}</span>`
        : '';
    const resetButton = `<button class="dragon-field-reset" type="button" data-config-reset-field="${escapeHtml(key)}" hidden
        aria-label="撤销${escapeHtml(label)}的修改" title="撤销此项修改">${renderIcon('refresh')}</button>`;

    let control;
    if (key === 'sample_prompts') {
        control = renderSamplePromptsFieldControl({
            fieldId,
            name,
            value: configValueForControl(controlValue),
            disabled: unavailable,
        });
    } else if (options && !booleanField) {
        control = `<select class="dragon-select" id="${fieldId}" name="${name}" autocomplete="off" data-key="${key}"${unavailable ? ' disabled' : ''}>
            ${options.map((opt) => `<option value="${escapeHtml(opt)}" ${String(controlValue) === String(opt) ? 'selected' : ''}>${escapeHtml(opt)}</option>`).join('')}
        </select>`;
    } else if (booleanField) {
        const checked = normalizeBooleanConfigValue(key, controlValue);
        control = `<div class="dragon-toggle-row">
            <div class="dragon-toggle" id="${fieldId}" data-key="${key}" data-checked="${checked}" data-config-disabled="${unavailable}" role="switch" tabindex="${unavailable ? '-1' : '0'}" aria-disabled="${unavailable}" aria-checked="${checked}" aria-label="${escapeHtml(label)}"></div>
            <div>
                <div class="dragon-toggle-label">
                    <span class="dragon-config-label-primary">${escapeHtml(label)}</span>
                    <span class="dragon-config-label-key"> | ${escapeHtml(key)}</span>
                </div>
                ${helpSummary ? `<div class="dragon-toggle-desc">${escapeHtml(helpSummary)}</div>` : ''}
            </div>
        </div>`;
        return `<div class="dragon-field${blockClass}" data-field-size="${fieldSize}" data-config-field-key="${escapeHtml(key)}" data-dirty="false"${availabilityAttributes}${blockAttributes}>
            ${block ? `<div class="dragon-config-block-head"><span class="dragon-config-semantic-markers">${semanticMarkers}</span><span class="dragon-config-block-actions">${unavailableBadge}${tagBadge}${helpButton}${resetButton}</span></div>` : `<div class="dragon-field-floating-actions">${unavailableBadge}${helpButton}${resetButton}</div>`}
            ${control}</div>`;
    } else if (key.includes('prompt') || key === 'optimizer_args' || key === 'network_args') {
        control = `<textarea class="dragon-textarea" id="${fieldId}" name="${name}" autocomplete="off" spellcheck="false" data-key="${key}"${unavailable ? ' disabled' : ''} placeholder="${escapeHtml(placeholder)}">${escapeHtml(configValueForControl(value) || '')}</textarea>`;
    } else {
        const inputType = typeof controlValue === 'number' ? 'number' : 'text';
        const inputMode = inputType === 'number' ? ' inputmode="decimal"' : '';
        control = `<input class="dragon-input" id="${fieldId}" name="${name}" type="${inputType}"${inputMode} autocomplete="off" spellcheck="false" data-key="${key}"${unavailable ? ' disabled' : ''} value="${escapeHtml(controlValue ?? '')}" placeholder="${escapeHtml(placeholder)}">`;
    }

    return `
        <div class="dragon-field${blockClass}" data-field-size="${fieldSize}" data-config-field-key="${escapeHtml(key)}" data-dirty="false"${availabilityAttributes}${blockAttributes}>
            <div class="dragon-field-label">
               <label class="dragon-field-label-text" for="${fieldId}">
                   <span class="dragon-config-label-primary">${escapeHtml(label)}</span>
                   <span class="dragon-config-label-key"> | ${escapeHtml(key)}</span>
                   ${semanticMarkers}
               </label>
               <span class="dragon-field-label-actions">
                   ${unavailableBadge}
                   ${tagBadge}
                   ${helpButton}
                   ${resetButton}
               </span>
            </div>
            ${control}
            ${pathTooltip}
        </div>
    `;
}

function renderConfigUnavailableBadge() {
    return '<span class="dragon-config-unavailable-badge" title="当前条件下不可编辑">不可用</span>';
}

function syncConfigFieldAvailability(wrapper, state, trainingContext) {
    if (!wrapper) return;
    const values = {
        ...(state?.baselineValues || {}),
        ...(state?.draftValues || {}),
    };
    const availabilityContext = configAvailabilityContext(trainingContext, values);
    if (state) state.availabilityContext = availabilityContext;
    wrapper.querySelectorAll('[data-config-field-key]').forEach((field) => {
        const key = field.dataset.configFieldKey || '';
        if (!key) return;
        const availability = configFieldAvailability(key, availabilityContext);
        const unavailable = availability.enabled === false;
        field.dataset.configAvailability = unavailable ? 'unavailable' : 'available';
        if (unavailable) field.dataset.configUnavailableReason = availability.reason;
        else delete field.dataset.configUnavailableReason;

        const control = field.querySelector('[data-key]');
        if (control?.classList.contains('dragon-toggle')) {
            control.dataset.configDisabled = String(unavailable);
            control.tabIndex = unavailable ? -1 : 0;
            control.setAttribute('aria-disabled', String(unavailable));
        } else if (control) {
            control.disabled = unavailable;
        }

        const actions = field.querySelector('.dragon-field-label-actions, .dragon-config-block-actions, .dragon-field-floating-actions');
        const badge = actions?.querySelector('.dragon-config-unavailable-badge');
        if (unavailable && actions && !badge) actions.insertAdjacentHTML('afterbegin', renderConfigUnavailableBadge());
        if (!unavailable) badge?.remove();

        const helpButton = field.querySelector('.dragon-field-help-btn');
        if (!helpButton) return;
        if (unavailable) {
            helpButton.dataset.helpUnavailableReason = availability.reason;
            helpButton.classList.add('dragon-field-help-btn-unavailable');
            helpButton.title = '查看不可用原因';
            helpButton.setAttribute('aria-label', `查看${helpButton.dataset.helpLabel || key}不可用原因`);
        } else {
            delete helpButton.dataset.helpUnavailableReason;
            helpButton.classList.remove('dragon-field-help-btn-unavailable');
            helpButton.title = '查看说明';
            helpButton.setAttribute('aria-label', `查看${helpButton.dataset.helpLabel || key}说明`);
        }
    });
}

function configFieldSize(key, value, options) {
    if (typeof value === 'boolean' || typeof value === 'number' || options) return 'compact';
    if (key.includes('prompt') || key === 'optimizer_args' || key === 'network_args') return 'full';
    if (/(?:^|_)(?:path|dir|file|jsonl)$/.test(key) || key.endsWith('_path')) return 'wide';
    return 'normal';
}

function configFieldToken(key) {
    const scope = String(key).replace(/[^A-Za-z0-9_-]+/g, '-');
    return `${scope}-${configFieldToken.counter++}`;
}
configFieldToken.counter = 0;

function resetConfigFormState(state, originalValues, entries) {
    const uniqueEntries = uniqueConfigEntries(entries);
    state.baselineValues = { ...originalValues };
    state.scopeKeys = uniqueEntries.flatMap((entry) => entry.keys);
    if (!state.scopeKeys.includes('dataset_config')) {
        state.scopeKeys.unshift('dataset_config');
    }
    state.draftValues = Object.fromEntries(state.scopeKeys.map((key) => [key, displayConfigValue(key, originalValues)]));
    state.dirtyKeys = new Set();
    state.dirtyBindings = null;
    state.dirty = false;
    state.showChangedOnly = false;
    state.activeTag = 'all';
    state.radarTag = null;
}

function captureDraftValue(input, state) {
    const key = input?.dataset?.key;
    if (!key) return null;
    state.draftValues[key] = serializeConfigValue(input, state.draftValues[key]);
    return key;
}

function collectDraftChanges(state) {
    const changed = {};
    for (const key of state.scopeKeys) {
        if (state.availabilityContext && !configFieldAvailability(key, state.availabilityContext).enabled) continue;
        const original = displayConfigValue(key, state.baselineValues);
        const current = state.draftValues[key];
        if (JSON.stringify(current ?? '') !== JSON.stringify(original ?? '')) changed[key] = current;
    }
    return changed;
}

function syncConfigDirtyUI(wrapper, state, changedKey = null) {
    const previousDirty = state.dirty;
    const previousChangedOnly = state.showChangedOnly;
    let patch = null;
    if (changedKey) {
        updateConfigDirtyKey(state, changedKey, displayConfigValue(changedKey, state.baselineValues));
        state.dirty = state.dirtyKeys.size > 0;
    } else {
        const rawChanges = collectDraftChanges(state);
        patch = prepareConfigPatch(rawChanges, state.baselineValues);
        replaceConfigDirtyKeys(state, Object.keys(rawChanges));
        state.dirty = Object.keys(patch).length > 0;
    }
    if (!state.dirty) state.showChangedOnly = false;
    renderConfigDirtyState(state.dirtyBindings, state, changedKey);
    if (previousDirty !== state.dirty) state.onDirtyChange?.(state.dirty);
    if (!changedKey || state.showChangedOnly || previousChangedOnly !== state.showChangedOnly) state.filterUpdate?.();
    return patch;
}

function wireConfigInteractions(wrapper, keys, trainingContext, state, { allView = false } = {}) {
    bindConfigFieldHelpDialog(wrapper, loadFieldHelpCatalog);
    syncConfigFieldAvailability(wrapper, state, trainingContext);
    state.modelQuickCleanup?.();
    state.modelQuickCleanup = null;
    state.trainingDataCleanup?.();
    state.trainingDataCleanup = null;
    state.samplePromptsCleanup?.();
    state.samplePromptsCleanup = null;
    state.dirtyBindings = createConfigDirtyBindings(wrapper);
    const syncDirty = (changedKey = null) => syncConfigDirtyUI(wrapper, state, changedKey);
    state.beforeUnload = (event) => {
        if (!state.dirty) return;
        event.preventDefault();
        event.returnValue = '';
    };
    window.addEventListener('beforeunload', state.beforeUnload);

    wrapper.querySelectorAll('.dragon-toggle').forEach((toggle) => {
        toggle.addEventListener('click', () => {
            if (toggle.dataset.configDisabled === 'true') return;
            const checked = toggle.dataset.checked === 'true';
            toggle.dataset.checked = String(!checked);
            toggle.setAttribute('aria-checked', String(!checked));
            syncDirty(captureDraftValue(toggle, state));
        });
        toggle.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggle.click();
            }
        });
        toggle.closest('.dragon-toggle-row')?.addEventListener('click', (event) => {
            if (event.target.closest('.dragon-toggle')) return;
            if (toggle.dataset.configDisabled === 'true') return;
            toggle.click();
        });
    });

    if (allView) {
        wrapper.querySelectorAll('.dragon-config-block').forEach((block) => {
            block.addEventListener('click', (event) => {
                if (event.target.closest('input, select, textarea, button, a, label, [role="switch"]')) return;
                const control = block.querySelector('input:not([type="hidden"]), select, textarea, [role="switch"]');
                if (control?.disabled || control?.dataset.configDisabled === 'true') return;
                control?.focus({ preventScroll: true });
            });
        });
    }

    const saveBtn = wrapper.querySelector('#dragon-config-save');
    const resetBtn = wrapper.querySelector('#dragon-config-reset');
    const feedbackEl = wrapper.querySelector('#dragon-config-feedback');
    const changedOnlyBtn = wrapper.querySelector('[data-config-changed-only]');
    wrapper.querySelectorAll('[data-config-reset-field]').forEach((button) => button.addEventListener('click', () => {
        const key = button.dataset.configResetField;
        const input = wrapper.querySelector(`[data-key="${key}"]`);
        if (!input) return;
        setConfigControlValue(input, displayConfigValue(key, state.baselineValues));
        const changedKey = captureDraftValue(input, state);
        syncConfigFieldAvailability(wrapper, state, trainingContext);
        syncDirty(changedKey);
        input.focus();
    }));
    wrapper.querySelectorAll('#dragon-config-fields input, #dragon-config-fields select, #dragon-config-fields textarea').forEach((field) => {
        const updateDraft = () => {
            const changedKey = captureDraftValue(field, state);
            syncConfigFieldAvailability(wrapper, state, trainingContext);
            syncDirty(changedKey);
        };
        field.addEventListener('input', updateDraft);
        field.addEventListener('change', updateDraft);
    });

    const saveChanges = async ({ quiet = false } = {}) => {
        const changedValues = syncDirty();
        if (Object.keys(changedValues).length === 0) {
            if (!quiet) showFeedback(feedbackEl, '没有修改', 'info');
            return true;
        }

        if (!isEditableConfigFile(trainingContext)) {
            showFeedback(feedbackEl, '当前训练配置为系统只读，无法保存修改；请先在左侧预设库选择可编辑配置', 'error');
            return false;
        }

        saveBtn.disabled = true;
        setButtonLabel(saveBtn, '保存中…');

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
                Object.assign(state.baselineValues, changedValues);
                state.scopeKeys.forEach((key) => {
                    state.draftValues[key] = displayConfigValue(key, state.baselineValues);
                });
                syncConfigDirtyUI(wrapper, state);
                wrapper.dispatchEvent(new CustomEvent('dragon-config-saved'));
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
            setButtonLabel(saveBtn, '保存配置');
        }
    };
    saveBtn?.addEventListener('click', () => saveChanges());

    resetBtn?.addEventListener('click', async () => {
        const resetKeys = allView
            ? keys.filter((key) => Object.prototype.hasOwnProperty.call(FORM_UI_DEFAULTS, key)
                || Object.prototype.hasOwnProperty.call(BOOLEAN_CONFIG_DEFAULTS, key))
            : keys;
        if (allView) {
            const confirmed = await confirmDragonDialog({
                eyebrow: '恢复默认参数',
                title: `恢复 ${resetKeys.length} 个参数？`,
                message: `将把当前方法中有界面默认值的 ${resetKeys.length} 个参数恢复默认。`,
                description: '恢复后仍需点击“保存配置”才会生效。',
                tone: 'warning',
                icon: 'refresh',
                confirmText: '恢复默认',
            });
            if (!confirmed || !wrapper.isConnected || !resetBtn.isConnected) return;
        }
        resetKeys.forEach((key) => {
            const defaultValue = Object.prototype.hasOwnProperty.call(FORM_UI_DEFAULTS, key)
                ? FORM_UI_DEFAULTS[key]
                : booleanDefaultForKey(key);
            const input = wrapper.querySelector(`[data-key="${key}"]`);
            if (!input) return;

            setConfigControlValue(input, defaultValue);
            state.draftValues[key] = serializeConfigValue(input, state.draftValues[key]);
        });
        syncConfigFieldAvailability(wrapper, state, trainingContext);
        syncDirty();
        showFeedback(feedbackEl, allView ? `已恢复 ${resetKeys.length} 个参数的默认值（需点击保存才会生效）` : '已恢复默认值（需点击保存才会生效）', 'info');
    });
    changedOnlyBtn?.addEventListener('click', () => {
        state.showChangedOnly = !state.showChangedOnly;
        syncConfigDirtyUI(wrapper, state);
    });
    state.modelQuickCleanup = bindLazyModelQuickPicker(wrapper, {
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
    state.trainingDataCleanup = bindTrainingDataTools(wrapper, {
        trainingContext,
        getDraftValue: (key) => state.draftValues[key],
    });
    state.samplePromptsCleanup = bindSamplePromptsDialog(wrapper, { trainingContext });
    syncConfigDirtyUI(wrapper, state);
    return () => saveChanges({ quiet: true });
}

function setConfigControlValue(input, value) {
    if (input.classList?.contains?.('dragon-toggle')) {
        const checked = normalizeBooleanConfigValue(input.dataset?.key, value);
        input.dataset.checked = String(checked);
        input.setAttribute('aria-checked', String(checked));
    } else {
        input.value = value ?? '';
    }
}

function setButtonLabel(button, text) {
    const label = button?.querySelector('span');
    if (label) label.textContent = text;
    else if (button) button.textContent = text;
}

async function confirmConfigDiscard(state, action) {
    if (!state.dirty) return true;
    if (state.leavePrompt) return state.leavePrompt;
    state.leavePrompt = confirmDragonDialog({
        eyebrow: '未保存修改',
        title: '确认继续？',
        message: `当前训练配置有未保存修改。${action}会丢弃这些修改，是否继续？`,
        description: '选择取消可返回当前编辑内容并继续保存。',
        tone: 'warning',
        icon: 'edit',
        confirmText: '继续操作',
    }).finally(() => { state.leavePrompt = null; });
    return state.leavePrompt;
}

function cleanupConfigPage(state) {
    if (state.beforeUnload) window.removeEventListener('beforeunload', state.beforeUnload);
    state.modelQuickCleanup?.();
    state.dirtyBindings = null;
    state.modelQuickCleanup = null;
    state.trainingDataCleanup?.();
    state.trainingDataCleanup = null;
    state.samplePromptsCleanup?.();
    state.samplePromptsCleanup = null;
    state.beforeUnload = null;
    state.filterUpdate = null;
}

function showFeedback(el, message, tone) {
    if (!el) return;
    el.textContent = message;
    el.dataset.tone = tone;
    el.classList.add('dragon-config-feedback-visible');
    setTimeout(() => el.classList.remove('dragon-config-feedback-visible'), 3000);
}
