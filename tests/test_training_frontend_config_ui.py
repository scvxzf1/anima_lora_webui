# Split from test_training_frontend_state.py (config_ui)

from __future__ import annotations

from tests.frontend_test_support import *  # noqa: F403


import tests.frontend_test_support as _frontend_support
for _k, _v in vars(_frontend_support).items():
    if not _k.startswith("__"):
        globals()[_k] = _v

APP_SHELL_STARTUP_REL = "js/features/app-shell/startup.js"
CONFIG_FORM_REL = "js/features/config-form/index.js"
CHUNK02_REL = "js/features/anima-app/chunks/02-ensure-history-detail-feature.js"
CHUNK15_REL = "js/features/anima-app/chunks/15-append-sample-prompt-row.js"
SAMPLE_PROMPT_ROW_UI_REL = "js/features/sample-prompts/row-ui.js"
TOML_MANAGER_MODE_REL = "js/features/toml-manager/mode.js"

def _chunk15_compat_text() -> str:
    return _frontend_feature_text(CHUNK15_REL, SAMPLE_PROMPT_ROW_UI_REL, TOML_MANAGER_MODE_REL)

CHUNK31_REL = "js/features/anima-app/chunks/31-create-history-collection-workbench-card.js"
HISTORY_WORKBENCH_CARDS_REL = "js/features/history-list/workbench-cards.js"

def _chunk31_compat_text() -> str:
    return _frontend_feature_text(CHUNK31_REL, HISTORY_WORKBENCH_CARDS_REL)

CHUNK11_REL = "js/features/anima-app/chunks/11-create-dataset-editor-row.js"
DATASET_EDITOR_ROW_REL = "js/features/dataset-editor/row.js"
DATASET_EDITOR_ROW_SETTINGS_REL = "js/features/dataset-editor/row-settings.js"
DATASET_EDITOR_ROW_SETTINGS_BASIC_REL = "js/features/dataset-editor/row-settings-basic.js"
DATASET_EDITOR_ROW_SETTINGS_EXPERIMENTAL_REL = "js/features/dataset-editor/row-settings-experimental.js"

def _chunk11_compat_text() -> str:
    return _frontend_feature_text(CHUNK11_REL, DATASET_EDITOR_ROW_REL, DATASET_EDITOR_ROW_SETTINGS_REL, DATASET_EDITOR_ROW_SETTINGS_BASIC_REL, DATASET_EDITOR_ROW_SETTINGS_EXPERIMENTAL_REL)

CHUNK23_REL = "js/features/anima-app/chunks/23-move-current-toml-to-group.js"
TOML_ACTIONS_REL = "js/features/toml-manager/actions.js"
TRAINING_LAUNCH_REL = "js/features/training-launch/index.js"

def _chunk23_compat_text() -> str:
    return _frontend_feature_text(CHUNK23_REL, TOML_ACTIONS_REL, TRAINING_LAUNCH_REL)

CHUNK20_REL = "js/features/anima-app/chunks/20-can-drop-toml-file-to-group.js"
TOML_DRAG_REL = "js/features/toml-manager/drag.js"

def _chunk20_compat_text() -> str:
    return _frontend_feature_text(
        CHUNK20_REL,
        TOML_DRAG_REL,
        "js/features/toml-manager/drag-core.js",
        "js/features/toml-manager/drag-actions.js",
        "js/features/toml-manager/drag-render.js",
    )

def _chunk02_compat_text() -> str:
    """Shim + domain truth, matching the post-split chunk02 barrel."""
    return _frontend_feature_text(CHUNK02_REL, APP_SHELL_STARTUP_REL, CONFIG_FORM_REL)


def test_config_form_bridge_reaches_split_form_chunks() -> None:
    bridge_source = _frontend_module_text("js/features/anima-app/helpers/config-form-bridge.js")
    ensure_history_source = _chunk02_compat_text()
    bridge_names = (
        "syncConfigDraftFromForm",
        "updateConfigDraftFromInput",
        "originalConfigFieldValue",
        "displayConfigFieldValue",
        "configDraftValueChanged",
        "isActiveNetworkArgFieldKey",
    )

    for name in bridge_names:
        assert f"export function {name}(...args)" in bridge_source
        assert f"requireConfigFormHandler('{name}')" in bridge_source
        assert f"    {name}," in ensure_history_source

    assert "configureConfigFormBridge({" in ensure_history_source
    form_source = _frontend_module_text(CONFIG_FORM_REL)
    _assert_imports_from(
        form_source,
        "../anima-app/helpers/config-form-bridge.js",
        ("configureConfigFormBridge",),
    )

    required_imports = {
        "js/features/config-form/no-dataset-regularization.js": (
            "originalConfigFieldValue",
        ),
        "js/features/config-form/resource-values.js": (
            "originalConfigFieldValue",
            "readFieldInputValue",
        ),
        "js/features/config-form/field-input.js": (
            "configDraftValueChanged",
            "originalConfigFieldValue",
            "updateConfigDraftFromInput",
        ),
        "js/features/config-form/form-fields-ui.js": (
            "displayConfigFieldValue",
            "isActiveNetworkArgFieldKey",
            "originalConfigFieldValue",
            "syncConfigDraftFromForm",
        ),
        "js/features/anima-app/chunks/18-delete-dataset-preset-group.js": (
            "configDraftValueChanged",
            "displayConfigFieldValue",
            "isActiveNetworkArgFieldKey",
            "originalConfigFieldValue",
            "syncConfigDraftFromForm",
        ),
        "js/features/anima-app/chunks/21-update-toml-selection-ui.js": (
            "configDraftValueChanged",
            "isActiveNetworkArgFieldKey",
            "originalConfigFieldValue",
        ),
    }
    for relative_path, names in required_imports.items():
        bridge_from = (
            "../anima-app/helpers/config-form-bridge.js"
            if relative_path.startswith("js/features/config-form/")
            else "../helpers/config-form-bridge.js"
        )
        _assert_imports_from(
            _frontend_module_text(relative_path),
            bridge_from,
            names,
        )


def test_state_bucket_bridges_reach_hotspot_chunks() -> None:
    index_source = _frontend_module_text("js/features/anima-app/index.js")
    scope_source = _frontend_module_text("js/features/anima-app/chunks/01-scope-state.js")
    ensure_history_source = _chunk02_compat_text()
    dataset_runtime_source = _frontend_feature_text("js/features/anima-app/chunks/03-parse-network-arg-entry.js", "js/features/config-form/step-estimate.js", "js/features/dataset-editor/load.js", "js/features/live-training/dashboard-ui.js")
    config_groups_source = _frontend_feature_text("js/features/anima-app/chunks/04-create-config-group-entry.js", "js/features/config-form/group-entry.js")
    dataset_picker_source = _frontend_feature_text("js/features/anima-app/chunks/06-stronger-selective-checkpoint-value.js", "js/features/config-form/resource-values.js", "js/features/config-form/field-rows.js", "js/features/config-form/dataset-picker.js", "js/features/training-source/continue-lora.js")
    config_dataset_dialog_source = _frontend_feature_text("js/features/anima-app/chunks/07-render-config-dataset-picker-dialog.js", "js/features/config-form/dataset-picker-dialog.js", "js/features/dataset-editor/preset-page.js", "js/features/toml-manager/file-group-drag.js", "js/features/toml-manager/file-group-drag-core.js", "js/features/toml-manager/file-group-drag-targets.js")
    file_group_drag_source = _frontend_feature_text("js/features/anima-app/chunks/08-origin-closest.js", "js/features/toml-manager/file-group-drag.js", "js/features/toml-manager/file-group-drag-core.js", "js/features/toml-manager/file-group-drag-targets.js")
    dataset_group_source = _frontend_feature_text("js/features/anima-app/chunks/09-setup-config-group-drop-target.js", "js/features/toml-manager/config-group-drop.js", "js/features/toml-manager/config-group-drop-target.js", "js/features/dataset-editor/dataset-render.js")
    dataset_inline_help_source = _frontend_feature_text("js/features/anima-app/chunks/10a-dataset-inline-help.js", "js/features/dataset-editor/inline-help.js")
    dataset_row_source = _chunk11_compat_text()
    dataset_caption_source = _frontend_feature_text("js/features/anima-app/chunks/12-create-dataset-row-caption-source-mode-editor.js", "js/features/dataset-editor/row-fields.js", "js/features/dataset-editor/preview.js")
    dataset_guide_source = _frontend_feature_text("js/features/anima-app/chunks/13-update-dataset-editor-rows-setting-value.js", "js/features/config-form/choice-guide-ui.js", "js/features/dataset-editor/mutations.js", "js/features/training-source/source-state.js", "js/features/config-form/field-input.js", "js/features/config-form/method-key.js")
    form_fields_source = _frontend_feature_text("js/features/anima-app/chunks/14-lora-adapter-kind-from-config.js", "js/features/config-form/form-fields.js", "js/features/config-form/form-fields-adapters.js", "js/features/config-form/form-fields-sample.js", "js/features/config-form/form-fields-ui.js")
    dataset_apply_source = _frontend_module_text("js/features/anima-app/chunks/17-apply-selected-dataset-preset-to-current-config.js")
    config_form_patch_source = _frontend_module_text("js/features/anima-app/chunks/18-delete-dataset-preset-group.js")
    toml_manager_source = _chunk15_compat_text()
    output_run_source = _frontend_module_text("js/features/anima-app/chunks/16-load-output-run-config.js")
    sample_prompts_source = _frontend_module_text("js/features/anima-app/chunks/19-current-sample-prompt-text.js")
    preflight_source = _frontend_feature_text("js/features/preflight-dialog/index.js", "js/features/live-log/index.js")
    progress_source = _frontend_module_text("js/features/anima-app/chunks/25-update-progress.js")
    toml_drag_source = _chunk20_compat_text()
    toml_selection_source = _frontend_module_text("js/features/anima-app/chunks/21-update-toml-selection-ui.js")
    toml_action_state_source = _frontend_module_text("js/features/anima-app/chunks/22-update-toml-action-state.js")
    toml_actions_source = _chunk23_compat_text()
    settings_source = _frontend_module_text("js/features/global-settings/settings.js")
    history_list_source = _frontend_module_text("js/features/history-list/list.js")
    queue_view_source = _frontend_module_text("js/features/queue/view-mode.js")
    history_workbench_source = _frontend_module_text("js/features/anima-app/chunks/27-render-history-collections-workbench.js")
    collection_search_source = _frontend_module_text("js/features/anima-app/chunks/28-history-collection-search-text.js")
    config_group_drag_source = _frontend_module_text("js/features/anima-app/chunks/29-start-history-config-group-pointer-drag.js")
    collection_drag_source = _frontend_module_text("js/features/anima-app/chunks/30-start-history-collection-pointer-drag.js")
    collection_card_source = _chunk31_compat_text()
    collection_state_source = _frontend_module_text("js/features/anima-app/chunks/32-history-task-collection-label.js")
    history_item_source = _frontend_module_text("js/features/anima-app/chunks/33-create-history-task-item.js")
    history_collection_dialog_source = _frontend_module_text("js/features/anima-app/chunks/34-show-history-collection-select-dialog.js")
    history_timeline_source = _frontend_module_text("js/features/anima-app/chunks/35-render-config-group-timeline.js")
    listeners_source = _frontend_feature_text("js/features/app-shell/event-listeners.js", "js/features/app-shell/event-listeners-contract.js", "js/features/app-shell/event-listeners-setup.js", "js/features/app-shell/beginner-tooltips.js")
    training_source = _frontend_module_text("js/features/training-source/index.js")

    _assert_imports_from(
        toml_drag_source,
        "../anima-app/helpers/output-run-bridge.js",
        ("selectAndApplyTomlFile",),
    )
    _assert_imports_from(
        toml_manager_source,
        "../anima-app/helpers/output-run-bridge.js",
        ("loadTomlFile",),
    )
    _assert_imports_from(
        toml_selection_source,
        "../helpers/output-run-bridge.js",
        ("loadTomlFile", "saveTomlFile"),
    )
    _assert_imports_from(
        toml_action_state_source,
        "../helpers/toml-selection-bridge.js",
        ("updateTomlDirtyState",),
    )
    _assert_imports_from(
        listeners_source,
        "../anima-app/helpers/output-run-bridge.js",
        ("loadTomlFile", "selectAndApplyTomlFile"),
    )
    _assert_imports_from(
        toml_drag_source,
        "../anima-app/helpers/toml-io-bridge.js",
        ("getSortableTomlGroups", "isTrainingTomlGroup"),
    )
    _assert_imports_from(
        toml_drag_source,
        "../anima-app/helpers/toml-action-state-bridge.js",
        ("createTomlGroup",),
    )

    for snippet in (
        "configureAppShellStateBridge(runtime.state.appShell);",
        "configureConfigStateBridge(runtime.state.config);",
        "configureDatasetStateBridge(runtime.state.dataset);",
        "configureHistoryStateBridge(runtime.state.history);",
        "configureTomlStateBridge(runtime.state.toml);",
        "createStatusPollingBridge(runtime.state.training)",
    ):
        assert snippet in index_source

    assert "const appShellState = getAppShellState();" in scope_source
    assert "configureQueueFeatureEnsurer(ctx, appShellState, {" in scope_source
    assert "configurePreviewFeatureEnsurer(ctx, appShellState, {" in scope_source
    assert "getTrainingViewMode: () => trainingState.trainingViewMode" in scope_source

    for snippet in (
        "const configState = getConfigState();",
        "const datasetState = getDatasetState();",
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "const configFormState = configState.configFormState;",
        "function currentConfigState() {",
        "function currentTrainingSourceState() {",
        "function currentContinueTrainingSource() {",
        "getLossChart: () => trainingState.lossChart,",
        "trainingState.lossChart = new MetricsChart(document.getElementById('loss-chart'), {",
        "showLr: trainingState.liveChartState.showLr,",
        "configState.fieldHelp = help;",
        "if (!tomlState.currentTomlFile) {",
        "const requestSeq = ++configState.configLoadSeq;",
        "const currentTrainingSource = currentTrainingSourceState();",
        "configState.currentConfig = data;",
        "datasetState.selectedConfigDatasetFile = data.dataset_config || '';",
        "if (currentContinueTrainingSource()?.abs_path) {",
        "if (configState.samplePromptsMode === 'editor-file') {",
        "configState.samplePromptsLoadSeq += 1;",
        "updateTomlActionState(tomlState.currentTomlFile);",
        "if (tomlState.tomlFiles.includes(tomlFile) && tomlState.currentTomlFile !== tomlFile) {",
        "configState.configFormState.draftValues.clear();",
        "if (key === 'sample_prompts' && configState.samplePromptsMode !== 'path') {",
        "return configState.samplePromptsContent || '';",
        "export function shouldRenderConfigSection(section, config = currentConfigState()) {",
        "export function activeNetworkArgSpecs(config = currentConfigState()) {",
    ):
        assert snippet in ensure_history_source

    for snippet in (
        "const configState = getConfigState();",
        "const configFormState = configState.configFormState;",
        "const stageResolutionState = configState.stageResolutionState;",
        "function currentConfigState() {",
        "renderConfigForm(currentConfigState());",
        "configState.fieldHelp[key] ? JSON.stringify(configState.fieldHelp[key]) : ''",
        "const hintId = `config-group-hint-${++configState.configGroupHintSeq}`;",
    ):
        assert snippet in config_groups_source

    for snippet in (
        "const configState = getConfigState();",
        "const datasetState = getDatasetState();",
        "const trainingState = getTrainingState();",
        "function currentTrainingSourceState() {",
        "export async function loadStepEstimate(parentSeq = configState.configLoadSeq) {",
        "const requestSeq = ++configState.stepEstimateSeq;",
        "const requestSeq = ++datasetState.datasetLoadSeq;",
        "datasetState.datasetEditorState = {",
        "const requestSeq = ++datasetState.datasetPresetLoadSeq;",
        "datasetState.selectedConfigDatasetSummary = datasetPresetSummaryByFile(datasetState.selectedConfigDatasetFile);",
        "const trainingRuntime = trainingState.trainingRuntime;",
    ):
        assert snippet in dataset_runtime_source

    for snippet in (
        "const appShellState = getAppShellState();",
        "const configState = getConfigState();",
        "const datasetState = getDatasetState();",
        "const historyState = getHistoryState();",
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "function currentTrainingSourceState() {",
        "if (!appShellState.globalSettings && location.protocol !== 'file:') {",
        "openBtn.textContent = datasetState.selectedConfigDatasetFile ? '更换预设' : '选择预设';",
        "const continueTrainingSource = currentContinueTrainingSource();",
        "updateTomlActionState(tomlState.currentTomlFile);",
        "trainingState.continueTrainingSource = payload;",
    ):
        assert snippet in dataset_picker_source

    for snippet in (
        "const configState = getConfigState();",
        "const datasetState = getDatasetState();",
        "function currentConfigState() {",
        "function currentDatasetPresetState() {",
        "function currentConfigDatasetPreviewState() {",
        "search.value = datasetState.configDatasetPickerSearch;",
        "const active = file === datasetState.selectedConfigDatasetFile;",
        "const summary = datasetState.selectedConfigDatasetSummary || preset?.summary || {};",
        "if (datasetState.selectedConfigDatasetFile !== (currentConfigState().dataset_config || '')) {",
        "datasetState.selectedConfigDatasetFile = file || '';",
        "datasetState.configDatasetPreviewState = {",
        "const requestSeq = ++datasetState.configDatasetPreviewRequestSeq;",
        "datasetState.fileGroupDragState = payload;",
        "datasetState.fileGroupDropTargets.set(node, resolve);",
        "datasetState.fileGroupDropTargetNodes.add(node);",
    ):
        assert snippet in config_dataset_dialog_source

    for snippet in (
        "const datasetState = getDatasetState();",
        "function currentFileGroupDragState() {",
        "function currentFileGroupPointerDrag() {",
        "function currentFileGroupDropPreviewElement() {",
        "function currentFileGroupDropTargets() {",
        "function currentFileGroupDropTargetNodes() {",
        "function currentFileGroupActiveDropTargetNode() {",
        "function currentFileGroupActiveDropPosition() {",
        "const payload = currentFileGroupDragState();",
        "const drag = currentFileGroupPointerDrag();",
        "datasetState.fileGroupDropPreviewElement = preview;",
        "datasetState.fileGroupPointerDrag = drag;",
        "datasetState.fileGroupActiveDropTargetNode = node;",
        "datasetState.fileGroupActiveDropPosition = normalizedPosition;",
        "datasetState.fileGroupDragState = null;",
    ):
        assert snippet in file_group_drag_source

    for snippet in (
        "const configState = getConfigState();",
        "const datasetState = getDatasetState();",
        "function currentConfigState() {",
        "function currentDatasetPresetState() {",
        "function currentDatasetEditorState() {",
        "function currentFileGroupDragState() {",
        "const payload = currentFileGroupDragState();",
        "const datasetPresetState = currentDatasetPresetState();",
        "source_dir: currentConfigState().source_image_dir || '',",
        "return isDatasetTabActive() ? currentDatasetPresetState() : currentDatasetEditorState();",
        "datasetState.datasetPresetState.datasets = rows;",
        "datasetState.datasetEditorState.datasets = rows;",
        "return currentDatasetEditorState().dataset_config || currentConfigState().dataset_config || '保存后自动生成 configs/datasets/<当前配置>.toml';",
    ):
        assert snippet in dataset_group_source

    for snippet in (
        "const datasetState = getDatasetState();",
        "function currentDatasetPresetState() {",
        "function currentDatasetEditorState() {",
        "`preset:${currentDatasetPresetState().selectedFile || 'new'}`",
        "`config:${currentDatasetEditorState().dataset_config || (typeof currentTrainingConfigFile === 'function' ? currentTrainingConfigFile() : '') || 'current'}`",
        "datasetState.datasetExperimentalOpenStates.set(datasetExperimentalOpenKey(index), Boolean(open));",
        "return datasetState.datasetExperimentalOpenStates.has(key)",
        "? datasetState.datasetExperimentalOpenStates.get(key)",
    ):
        assert snippet in dataset_inline_help_source

    for snippet in (
        "const datasetState = getDatasetState();",
        "function currentDatasetPresetState() {",
        "const presetState = currentDatasetPresetState();",
        "previewBtn.disabled = !presetState.selectedFile || presetState.dirty;",
    ):
        assert snippet in dataset_row_source

    for snippet in (
        "const datasetState = getDatasetState();",
        "function currentDatasetPresetState() {",
        "function currentDatasetPreviewState() {",
        "const helpId = `dataset-caption-source-notes-${++datasetState.datasetCaptionSourceHelpSeq}`;",
        "if (!currentDatasetPresetState().selectedFile) {",
        "const previewState = currentDatasetPreviewState();",
        "const requestSeq = ++datasetState.datasetPreviewLoadSeq;",
        "if (requestSeq !== datasetState.datasetPreviewLoadSeq) return;",
        "currentDatasetPreviewState().payload = {",
        "['数据集文件', payload.file || currentDatasetPresetState().selectedFile || '-'],",
        "datasetState.datasetPresetState.defaults = defaults;",
        "datasetState.datasetEditorState.defaults = defaults;",
        "datasetState.datasetPresetState.datasets = rows;",
        "datasetState.datasetEditorState.datasets = rows;",
    ):
        assert snippet in dataset_caption_source

    for snippet in (
        "const appShellState = getAppShellState();",
        "const configState = getConfigState();",
        "const datasetState = getDatasetState();",
        "const trainingState = getTrainingState();",
        "function currentConfigState() {",
        "function currentTrainingSourceState() {",
        "function datasetExperimentalScopeSelectionsState() {",
        "datasetState.datasetPresetState.datasets = rows;",
        "datasetState.datasetEditorState.datasets = rows;",
        "const selectionSnapshot = configState.selectionSnapshot;",
        "trainingState.currentTrainingSource = {",
        "const helpId = `choice-guide-hint-${++configState.choiceGuideHintSeq}`;",
        "const globalSettings = appShellState.globalSettings;",
    ):
        assert snippet in dataset_guide_source

    for snippet in (
        "const configState = getConfigState();",
        "function currentConfigState() {",
        "const configFormState = configState.configFormState;",
        "const currentConfig = currentConfigState();",
        "if (key === 'sample_prompts' && configState.samplePromptsMode !== 'path') {",
        "if (configState.samplePromptsMode === 'path') {",
        "const rawNetworkArgsChanged = configFormState.draftValues.has('network_args');",
    ):
        assert snippet in form_fields_source

    for snippet in (
        "const configState = getConfigState();",
        "const datasetState = getDatasetState();",
        "const tomlState = getTomlState();",
        "const currentConfig = currentConfigState();",
        "const nextDataset = datasetState.selectedConfigDatasetFile || '';",
        "if (typeof res.content === 'string' && file === (tomlState.currentTomlFile || val('toml-file-select'))) {",
        "tomlState.tomlSavedContent = res.content;",
        "datasetState.datasetPresetState = {",
    ):
        assert snippet in dataset_apply_source

    for snippet in (
        "const configState = getConfigState();",
        "const datasetState = getDatasetState();",
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "function currentTrainingSourceState() {",
        "const configFormState = configState.configFormState;",
        "const targetFile = options.trainFile || currentTrainingSource.file || tomlState.currentTomlFile || '';",
        "tomlState.tomlSavedContent = res.train_content;",
        "datasetState.datasetEditorState = {",
        "if (configState.samplePromptsMode === 'path') {",
        "nextValues.sample_prompts = saved.file || configState.samplePromptsPath;",
    ):
        assert snippet in config_form_patch_source

    for snippet in (
        "const configState = getConfigState();",
        "const datasetState = getDatasetState();",
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "const remote = configState.fieldHelp[key];",
        "export function updateConfigPageSummary(mode = tomlState.tomlManagerMode) {",
        "datasetState.outputRunState = {",
        "const target = currentTrainingSourceState().file || `configs/${methodsSubdir}/${variant}.toml`;",
    ):
        assert snippet in toml_manager_source

    for snippet in (
        "const datasetState = getDatasetState();",
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "function currentOutputRunState() {",
        "armTomlSaveConfirm,",
        "isTomlLocked,",
        "tomlState.currentTomlFile = filePath;",
        "tomlState.tomlSavedContent = data.content || '';",
        "updateTomlActionState,",
        "return await saveRawTomlContent(file, document.getElementById('toml-editor').value, { reloadConfig: currentTrainingSourceState().file === file });",
    ):
        assert snippet in output_run_source

    for snippet in (
        "const configState = getConfigState();",
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "const previousMode = configState.samplePromptsMode;",
        "const requestSeq = ++configState.samplePromptsLoadSeq;",
        "train_config_file: currentTrainingSourceState().file || tomlState.currentTomlFile || '',",
        "trainingState.currentTrainingSource = {",
    ):
        assert snippet in sample_prompts_source

    for snippet in (
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "const trainingRuntime = trainingState.trainingRuntime;",
        "return currentTrainingSource.file || tomlState.currentTomlFile || val('toml-file-select') || '';",
        "trainingState.ws = new WebSocket(",
    ):
        assert snippet in preflight_source

    for snippet in (
        "const trainingState = getTrainingState();",
        "const trainingRuntime = trainingState.trainingRuntime;",
        "const step = msg.step || ++trainingState.stepCounter;",
        "trainingState.lossChart?.push(step, lossNumber, metadata);",
        "trainingState.lossChart?.updatePointMetadata?.(msg.step, { lr: lrNumber });",
    ):
        assert snippet in progress_source

    for snippet in (
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "return Boolean(item?.path && !item.locked && !hasPendingConfigChanges(tomlState.currentTomlFile));",
        "if (tomlState.tomlFiles.includes(prev)) {",
        "if (tomlState.tomlGroupActionBusy) return;",
        "tomlState.tomlGroupActionBusy = true;",
        "if (currentTrainingSourceState().file === item.path) tags.push('当前训练');",
    ):
        assert snippet in toml_drag_source

    for snippet in (
        "const configState = getConfigState();",
        "const datasetState = getDatasetState();",
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "return currentTrainingSourceState().file || '';",
        "const selectedFile = tomlState.currentTomlFile || val('toml-file-select') || '';",
        "setBadge('toml-current-badge', Boolean(filePath && currentTrainingSourceState().file === filePath), '当前训练');",
    ):
        assert snippet in toml_selection_source

    for snippet in (
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "const trainingRuntime = trainingState.trainingRuntime;",
        "const selectedFile = filePath || tomlState.currentTomlFile || val('toml-file-select') || '';",
        "const confirming = canDelete && tomlState.tomlDeleteConfirmFile === selectedFile;",
        "return Boolean(tomlState.tomlFileMeta[filePath]?.locked);",
        "const open = Boolean(panel && !panel.hidden && tomlState.tomlManagerMode === 'project');",
        "trainingState.currentTrainingSource = {",
    ):
        assert snippet in toml_action_state_source

    for snippet in (
        "const datasetState = getDatasetState();",
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "const trainingRuntime = trainingState.trainingRuntime;",
        "const file = tomlState.currentTomlFile || val('toml-file-select');",
        "delete tomlState.tomlFileMeta[file];",
        "tomlState.tomlFiles = tomlState.tomlFiles.filter((item) => item !== file);",
        "tomlState.currentTomlFile = '';",
        "if (tomlState.tomlManagerMode !== 'output' || !currentOutputRunState().file) {",
        "const variant = currentTrainingSourceState().method || val('variant-select');",
    ):
        assert snippet in toml_actions_source

    for snippet in (
        "const appShellState = getAppShellState();",
        "const tomlState = getTomlState();",
        "appShellState.globalSettings = data;",
    ):
        assert snippet in settings_source

    for snippet in (
        "const historyState = getHistoryState();",
        "const trainingState = getTrainingState();",
        "historyState.historyTasks = payload.tasks || [];",
    ):
        assert snippet in history_list_source

    for snippet in (
        "const trainingState = getTrainingState();",
        "trainingState.trainingViewMode = ['live', 'queue', 'history'].includes(mode) ? mode : 'live';",
    ):
        assert snippet in queue_view_source

    for snippet in (
        "const historyState = getHistoryState();",
        "historyState.historyCurrentVisibleTaskIds = historyTaskIds(currentVisibleTasks);",
        "if (historyState.viewingHistoryTaskId === task.id && isHistoryDetailDialogOpen())",
    ):
        assert snippet in history_workbench_source

    for snippet in (
        "const historyState = getHistoryState();",
        "historyState.historyCollectionSettings.collection_order",
        "historyCollectionsForWorkbench(historyState.historyTasks)",
        "Array.from(historyState.selectedHistoryTaskIds)",
        "historyState.historyDragState = {",
        "historyState.historyConfigGroupSortState = {",
        "historyState.selectedHistoryCollectionKey = clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY;",
    ):
        assert snippet in collection_search_source

    for snippet in (
        "const trainingState = getTrainingState();",
        "trainingState.lossChart?.setXLabel?.('step');",
        "trainingState.lossChart?.setData(chartPoints, { keepAll: true });",
    ):
        assert snippet in history_collection_dialog_source

    for snippet in (
        "const trainingState = getTrainingState();",
        "trainingState.lossChart?.setXLabel?.('step');",
        "trainingState.lossChart?.setData(lossPoints.map((item) => ({",
        "trainingState.lossChart?.clear();",
    ):
        assert snippet in history_timeline_source

    for snippet in (
        "const historyState = getHistoryState();",
        "historyState.historyConfigGroupSortState.pending = true;",
        "historyState.historyCollectionDragState = {",
    ):
        assert snippet in config_group_drag_source

    for snippet in (
        "const trainingState = getTrainingState();",
        "trainingState.liveChartState.showLr = Boolean(event.target.checked);",
        "trainingState.liveChartState.rangeMode = event.target.value || 'all';",
    ):
        assert snippet in listeners_source

    for snippet in (
        "const historyState = getHistoryState();",
        "const trainingState = getTrainingState();",
        "historyState.historyDragState.pending = true;",
        "historyState.selectedHistoryCollectionKey = clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY;",
        "if (trainingState.trainingViewMode === 'history') renderHistoryManager();",
    ):
        assert snippet in collection_drag_source

    for snippet in (
        "const historyState = getHistoryState();",
        "historyState.historyCollectionWorkbenchTarget =",
        "historyState.historyCollectionSettings.collection_order",
    ):
        assert snippet in collection_card_source

    for snippet in (
        "const historyState = getHistoryState();",
        "historyState.selectedHistoryTaskIds = new Set(",
        "historyState.historyCollectionSettings = normalizeHistoryCollectionSettings({",
        "historyTaskCollectionValue,",
    ):
        assert snippet in collection_state_source

    for snippet in (
        "const historyState = getHistoryState();",
        "if (task.id === historyState.viewingHistoryTaskId && isHistoryReviewMode())",
        "historyState.currentHistoryTaskForResume = payload.task || null;",
    ):
        assert snippet in history_item_source

    for snippet in (
        "const historyState = getHistoryState();",
        "const tomlState = getTomlState();",
        "const trainingState = getTrainingState();",
        "updateTomlActionState(tomlState.currentTomlFile);",
        "if (!historyState.historyTasks.length && typeof loadTrainingHistoryList === 'function') {",
        "return (historyState.historyTasks || [])",
    ):
        assert snippet in training_source


def test_runtime_dataset_preset_api_timeout_contract() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for anima-app runtime api checks")
    script = r"""
import { createRuntimeApi } from './web/static/js/features/anima-app/runtime/api.js';

const calls = [];
const delays = [];
const cleared = [];
let lastTimeout = null;

globalThis.window = {
    setTimeout(callback, delay) {
        delays.push(delay);
        lastTimeout = callback;
        return delays.length;
    },
    clearTimeout(id) {
        cleared.push(id);
    },
};

const api = createRuntimeApi({
    api(url, opts = {}) {
        calls.push({ url, opts: { ...opts } });
        return Promise.resolve({ ok: true, url });
    },
});

const first = await api.datasetPresetApi('/default', { method: 'GET' });
const second = await api.datasetPresetApi('/custom', { method: 'POST', timeoutMs: 25, body: 'x' });

let timeoutMessage = '';
const slowApi = createRuntimeApi({
    api(url, opts = {}) {
        calls.push({ url, opts: { ...opts } });
        return new Promise(() => {});
    },
});
const pending = slowApi.datasetPresetApi('/slow', { timeoutMs: 12, marker: 'keep' }).catch((error) => {
    timeoutMessage = error.message;
});
lastTimeout();
await pending;

console.log(JSON.stringify({
    first,
    second,
    calls,
    delays,
    cleared,
    timeoutMessage,
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "first": {"ok": True, "url": "/default"},
        "second": {"ok": True, "url": "/custom"},
        "calls": [
            {"url": "/default", "opts": {"method": "GET"}},
            {"url": "/custom", "opts": {"method": "POST", "body": "x"}},
            {"url": "/slow", "opts": {"marker": "keep"}},
        ],
        "delays": [15000, 25, 12],
        "cleared": [1, 2, 3],
        "timeoutMessage": "数据集预设请求超时，请查看终端日志或刷新预设列表",
    }


def test_network_arg_helpers_are_exported() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for anima-app network arg helper checks")
    script = r"""
import {
    coerceNetworkArgValue,
    formatNetworkArg,
    formatNetworkArgValue,
    parseBooleanNetworkArg,
    parseNetworkArgEntry,
    stripNetworkArgQuotes,
} from './web/static/js/features/anima-app/helpers/network-args.js';

const result = {
    parsed: parseNetworkArgEntry('rank="8"'),
    quoted: stripNetworkArgQuotes("'hello'"),
    trueValue: parseBooleanNetworkArg('yes'),
    falseValue: parseBooleanNetworkArg('0', true),
    integerValue: coerceNetworkArgValue('3.8', { valueType: 'integer', default: 1 }),
    numberFallback: coerceNetworkArgValue('nope', { valueType: 'number', default: 0.5 }),
    stringValue: coerceNetworkArgValue(null, { valueType: 'string', default: 'x' }),
    formattedBoolean: formatNetworkArg({ arg: 'enabled', valueType: 'boolean', default: false }, 'yes'),
    formattedNumber: formatNetworkArgValue({ valueType: 'number', default: 0.1 }, '2.5'),
};

console.log(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "parsed": {"arg": "rank", "value": "8", "raw": 'rank="8"'},
        "quoted": "hello",
        "trueValue": True,
        "falseValue": False,
        "integerValue": 3,
        "numberFallback": 0.5,
        "stringValue": "x",
        "formattedBoolean": "enabled=true",
        "formattedNumber": "2.5",
    }


def test_form_value_helpers_are_exported() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for anima-app form value helper checks")
    script = r"""
import {
    isBooleanLikeValue,
    isNumberLikeValue,
    normalizeBooleanLikeValue,
    normalizeMultilineText,
    parseArrayValue,
    parseNumberValue,
    valuesEqual,
} from './web/static/js/features/anima-app/helpers/form-values.js';

const result = {
    emptyNumber: parseNumberValue('', ''),
    fallbackNumber: parseNumberValue('', 7),
    invalidNumber: parseNumberValue('nope', 3),
    parsedJsonArray: parseArrayValue('["a", 2]'),
    parsedCsvArray: parseArrayValue('a, b,, c'),
    booleanEqual: valuesEqual(true, 'true'),
    numberEqual: valuesEqual('2.0', 2),
    booleanLike: isBooleanLikeValue('false'),
    normalizedBoolean: normalizeBooleanLikeValue('true'),
    numberLike: isNumberLikeValue('3.5'),
    multiline: normalizeMultilineText(` a

 b
 `),
};

console.log(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "emptyNumber": "",
        "fallbackNumber": 7,
        "invalidNumber": 3,
        "parsedJsonArray": ["a", 2],
        "parsedCsvArray": ["a", "b", "c"],
        "booleanEqual": True,
        "numberEqual": True,
        "booleanLike": True,
        "normalizedBoolean": True,
        "numberLike": True,
        "multiline": "a\nb",
    }


def test_config_value_helpers_are_exported() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for anima-app config value helper checks")
    script = r"""
import {
    isTruthy,
    loraAdapterFlagsForKind,
    loraAdapterFlagsMatchConfig,
    loraAdapterKindFromConfig,
    normalizeLoraAdapterKind,
    normalizePrecisionPreference,
    precisionPreferenceFromConfig,
    precisionPreferencePatch,
} from './web/static/js/features/anima-app/helpers/config-values.js';

const result = {
    trueBoolean: isTruthy(true),
    trueNumber: isTruthy(1),
    trueString: isTruthy('true'),
    falseString: isTruthy('false'),
    glora: normalizeLoraAdapterKind(' GLoRA '),
    fallback: normalizeLoraAdapterKind('unknown'),
    kindFromConfig: loraAdapterKindFromConfig({ use_lokr: true }),
    flags: loraAdapterFlagsForKind('vera'),
    flagsMatch: loraAdapterFlagsMatchConfig('glora', { use_glora: 'true' }),
    precision: normalizePrecisionPreference(' FP32 '),
    derivedPrecision: precisionPreferenceFromConfig({ mixed_precision: 'no' }),
    precisionPatch: precisionPreferencePatch('fp32', { full_fp16: true }),
};

console.log(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "trueBoolean": True,
        "trueNumber": True,
        "trueString": True,
        "falseString": False,
        "glora": "glora",
        "fallback": "lora",
        "kindFromConfig": "lokr",
        "flags": {"use_glora": False, "use_loha": False, "use_lokr": False, "use_vera": True},
        "flagsMatch": True,
        "precision": "fp32",
        "derivedPrecision": "fp32",
        "precisionPatch": {"mixed_precision": "no", "full_fp16": False},
    }


def test_optimizer_value_helpers_are_exported() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for anima-app optimizer value helper checks")
    script = r"""
import {
    normalizeCameOptimizerArgs,
    normalizeOptimizerArgArray,
    normalizeOptimizerType,
    optimizerArgEntryKey,
    optimizerArgEntryValue,
} from './web/static/js/features/anima-app/helpers/optimizer-values.js';

const result = {
    optimizer: normalizeOptimizerType(' CAME '),
    key: optimizerArgEntryKey('betas=0.9,0.999'),
    value: optimizerArgEntryValue('betas=0.9,0.999'),
    csv: normalizeOptimizerArgArray('a=1, b=2'),
    patched: normalizeCameOptimizerArgs(['betas=0.9,0.999', 'eps=1e-8']),
};

console.log(JSON.stringify(result));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "optimizer": "came",
        "key": "betas",
        "value": "0.9,0.999",
        "csv": ["a=1", "b=2"],
        "patched": ["betas=0.9,0.999,0.9999", "eps=1e-8"],
    }


def test_image_test_theme_variables_are_root_scoped_for_dialogs() -> None:
    css = (STATIC_DIR / "css" / "42-image-test.css").read_text(encoding="utf-8")

    assert re.search(r":root\s*\{[^}]*--image-test-page-bg:\s*#111827;", css, re.S)
    assert re.search(r':root\[data-theme="light"\]\s*\{[^}]*--image-test-page-bg:\s*#f6f7ff;', css, re.S)
    assert re.search(r"#tab-image-test\s*\{[^}]*background:\s*var\(--image-test-page-bg\);", css, re.S)


def test_global_ui_scale_override_controls_and_runtime_hooks_are_present() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    defaults_source = _frontend_module_text("js/config/catalog/defaults.js")
    settings_source = _frontend_module_text("js/features/global-settings/settings.js")
    listeners_source = _frontend_feature_text("js/features/app-shell/event-listeners.js", "js/features/app-shell/event-listeners-contract.js", "js/features/app-shell/event-listeners-setup.js", "js/features/app-shell/beginner-tooltips.js")
    ensure_history_source = _chunk02_compat_text()
    ui_scale_source = _frontend_module_text("js/features/app-shell/ui-scale.js")
    history_dialog_source = _frontend_module_text("js/features/history-detail/dialog.js")
    css = STYLE_CSS.read_text(encoding="utf-8")

    assert "GLOBAL_UI_TOP_LEVEL_OVERRIDE_FIELDS" in defaults_source
    assert "GLOBAL_UI_HISTORY_DETAIL_OVERRIDE_FIELDS" in defaults_source
    assert "GLOBAL_UI_OVERRIDE_FIELDS" in defaults_source
    assert "GLOBAL_UI_FIELDS" in defaults_source
    assert "followDefaultId" in defaults_source
    assert "detailTab: 'config_files'" in defaults_source
    assert "tab: 'weight-analysis'" in defaults_source
    assert "tab: 'image-test'" in defaults_source

    for snippet in (
        "global-ui-scale-pages-title",
        "global-ui-scale-history-title",
        "主页面独立比例",
        "历史详情独立比例",
        "global-ui-scale-config-follow-default",
        "global-ui-scale-datasets-follow-default",
        "global-ui-scale-training-follow-default",
        "global-ui-scale-weight-analysis-follow-default",
        "global-ui-scale-image-test-follow-default",
        "global-ui-scale-settings-follow-default",
        "global-ui-scale-environment-follow-default",
        "global-ui-scale-history-overview-follow-default",
        "global-ui-scale-history-analysis-follow-default",
        "global-ui-scale-history-preview-follow-default",
        "global-ui-scale-history-logs-follow-default",
        "global-ui-scale-history-config-files-follow-default",
    ):
        assert snippet in html

    for selector in (
        ".global-ui-scale-group",
        ".global-ui-scale-group-head",
        ".global-ui-scale-rows",
        ".global-ui-scale-row",
        ".global-ui-scale-copy",
        ".global-ui-scale-follow-default",
        ".global-ui-scale-row.is-follow-default > input[type=\"number\"]",
    ):
        assert selector in css

    for snippet in (
        "function resolveGlobalUIScaleDefaultValue",
        "function syncGlobalUIScaleOverrideField",
        "function syncAllGlobalUIScaleOverrideFields",
        "function applyGlobalUIScaleOverrideInputs",
        "function collectGlobalUIScaleOverridePayload",
        "applyGlobalUIScaleOverrideInputs(snapshot);",
        "return collectGlobalUIScaleOverridePayload(payload);",
        "GLOBAL_UI_OVERRIDE_FIELDS.map(({ key }) => [key, defaults[key] ?? ''])",
        "activeHistoryDetailTab: historyDetailFeature?.getActiveTab?.(),",
    ):
        assert snippet in settings_source

    for snippet in (
        "on('global-ui-scale', 'input'",
        "on('global-ui-scale', 'change'",
        "GLOBAL_UI_OVERRIDE_FIELDS.forEach((field) => {",
        "on(field.followDefaultId, 'change'",
        "syncGlobalUIScaleOverrideField(field);",
    ):
        assert snippet in listeners_source

    for snippet in (
        "topLevelFields: GLOBAL_UI_TOP_LEVEL_OVERRIDE_FIELDS",
        "historyDetailFields: GLOBAL_UI_HISTORY_DETAIL_OVERRIDE_FIELDS",
        "applyHistoryDetailUIScale: (detailTab) => {",
        "uiScaleController?.applyHistoryDetailScale?.(appShellState.globalSettings || {}, detailTab || 'overview');",
    ):
        assert snippet in ensure_history_source

    for snippet in (
        "function resolveBaseScale(settings)",
        "function resolveOverrideScale(settings, key, baseScale)",
        "function applyScopedZoom(element, effectiveScale, baseScale)",
        "function applyTopLevelScales(settings, baseScale = resolveBaseScale(settings))",
        "function applyHistoryDetailScale(settings, activeHistoryDetailTab = 'overview', baseScale = resolveBaseScale(settings))",
        "function applyScaleFromSettings(settings, { activeHistoryDetailTab = 'overview' } = {})",
        "document.getElementById(`tab-${field.tab}`)",
        "document.getElementById('history-detail-content')",
        "element.style.setProperty('zoom', String(zoom));",
    ):
        assert snippet in ui_scale_source

    for snippet in (
        "deps.applyHistoryDetailUIScale?.('overview');",
        "deps.applyHistoryDetailUIScale?.(state.detailTab);",
    ):
        assert snippet in history_dialog_source


def test_global_settings_cards_follow_requested_numbering_order() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")

    output_index = html.index('id="global-output-settings-title"')
    model_index = html.index('id="global-model-settings-title"')
    config_index = html.index('id="global-config-paths-title"')
    ui_index = html.index('id="global-ui-settings-title"')

    assert output_index < model_index < config_index < ui_index
    assert '<span class="global-settings-card-mark">01</span>' in html
    assert '<span class="global-settings-card-mark">02</span>' in html
    assert '<span class="global-settings-card-mark">03</span>' in html
    assert '<span class="global-settings-card-mark">04</span>' in html

    summary_section = _section(html, '<div class="global-settings-summary" aria-label="全局设置范围">', '<div class="global-settings-summary-note">')
    assert summary_section.index("输出根目录") < summary_section.index("基础模型路径")
    assert summary_section.index("基础模型路径") < summary_section.index("配置目录路径")
    assert summary_section.index("配置目录路径") < summary_section.index("界面设置")


def test_new_training_launch_enters_live_monitoring() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    config_values = _frontend_module_text("js/features/anima-app/helpers/config-values.js")
    tabs_source = _frontend_module_text("js/features/app-shell/tabs.js")
    helper = _section(source, "function enterLiveTrainingForNewRun()", "function showPreflightDialog")
    tab_setup = _section(tabs_source, "function setupTabs()", "return {")

    assert "returnToLiveTraining({ refresh: false });" in helper
    assert 'document.querySelector(\'[data-tab="training"]\')?.click();' in helper
    assert "recoverLiveTrainingState();" in helper
    assert "previousTab === 'training' && nextTab !== 'training'" in tab_setup
    assert "resetTrainingExpandedStateOnLeave();" in tab_setup

    start_path = _section(source, "async function startTrainingUnchecked", "function enterLiveTrainingForNewRun")
    preprocess_path = _section(source, "async function startPreprocessFromPreflight", "function currentTrainingConfigFile")

    assert "enterLiveTrainingForNewRun();" in start_path
    assert "enterLiveTrainingForNewRun();" in preprocess_path
    assert 'document.querySelector(\'[data-tab="training"]\').click();' not in start_path
    assert 'document.querySelector(\'[data-tab="training"]\').click();' not in preprocess_path


def test_start_training_resolves_pending_change_helpers() -> None:
    """Regression: startTraining must not throw ReferenceError on first click."""
    launch_source = _frontend_module_text(TRAINING_LAUNCH_REL)
    start_section = _section(launch_source, "export async function startTraining", "export async function queueCurrentTrainingFromConfig")

    assert "hasPendingConfigChanges" in start_section
    assert "updateTomlActionState" in start_section
    assert "currentOutputRunState()" in start_section
    assert "function currentOutputRunState()" in launch_source
    assert "const datasetState = getDatasetState();" in launch_source

    _assert_imports_from(
        launch_source,
        "../anima-app/helpers/toml-selection-bridge.js",
        ("hasPendingConfigChanges", "showAppConfirmDialog"),
    )
    _assert_imports_from(
        launch_source,
        "../anima-app/helpers/toml-action-state-bridge.js",
        ("setTomlStatus", "updateTomlActionState"),
    )
    _assert_imports_from(
        launch_source,
        "../anima-app/helpers/dataset-state-bridge.js",
        ("getDatasetState",),
    )


def test_status_poll_skips_log_and_metric_replay_for_idle_snapshot_recovery() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for live-training status snapshot checks")
    script = r"""
const calls = [];

globalThis.document = { hidden: false };
globalThis.window = { setTimeout: () => 0, clearTimeout: () => {} };
globalThis.location = { protocol: 'http:' };
globalThis.ws = { readyState: 0 };
globalThis.WebSocket = { OPEN: 1 };
globalThis.trainingStatusPollTimer = null;
globalThis.trainingStatusPollPromise = null;
globalThis.trainingStatusPollForceReplayMetrics = false;
globalThis.trainingStatusPollFailures = 0;
globalThis.historyTasks = [];
globalThis.trainingRuntime = { state: 'idle', lastLogId: 0 };
globalThis.isHistoryReviewMode = () => false;
globalThis.updateStatus = (payload) => {
    globalThis.trainingRuntime.state = payload.state;
};
globalThis.updateProgress = () => calls.push('progress');
globalThis.updateMetrics = () => calls.push('metric');
globalThis.updateSystem = () => calls.push('system');
globalThis.replayTrainingLogs = async () => calls.push('logs');
globalThis.replayMetricsHistory = async () => calls.push('metrics');
globalThis.appendLog = () => {};
globalThis.setLogStatus = () => {};
globalThis.setTrainingHealthNotice = () => {};
globalThis.updateLogStatusText = () => {};
globalThis.loadTrainingQueue = async () => {};
globalThis.loadTrainingHistoryList = async () => {};

const { configureRuntimeBridge } = await import('./web/static/js/features/anima-app/helpers/runtime-bridge.js?v=module-bootstrap-20260711-ir1');
const { configureHistoryDetailBridge } = await import('./web/static/js/features/anima-app/helpers/history-detail-bridge.js?v=module-bootstrap-20260711-ir1');
configureHistoryDetailBridge({
    isHistoryReviewMode: () => false,
});
configureRuntimeBridge({
    api: async () => ({
        ok: true,
        status: 'idle',
        variant: 'lora',
        preset: 'default',
        methods_subdir: 'gui-methods',
        job: '',
        output_dir: 'output/runs/finished/training_output',
        sample_dir: 'output/runs/finished/training_output/sample',
        sample_config: { prompt: 'demo' },
        run_dir: 'output/runs/finished',
        runtime_config_file: 'output/runs/finished/config.runtime.toml',
        original_config_file: 'output/runs/finished/config.original.toml',
        dataset_config_file: '',
        model_cache_dir: '',
        dataset_cache_dir: '',
        training_output_dir: 'output/runs/finished/training_output',
        logs_dir: '',
        task_id: '',
        last_output_at: 1,
        last_log_line: '训练完成',
        last_log_id: 9,
        latest_progress: { current: 9, total: 10 },
        latest_metric: { loss: 0.01 },
        latest_system: { gpu_util: 10 },
        error_hint: '',
        anomaly_message: '',
    }),
    dom: {
        val: () => '',
        populateSelect: () => {},
    },
});

const statusPollingModule = await import('./web/static/js/features/anima-app/chunks/26a-status-polling.js?idle-snapshot-fixture');
const bridge = statusPollingModule.createStatusPollingBridge(globalThis);
await bridge.pollStatus({ forceReplayMetrics: true });

console.log(JSON.stringify({
    calls,
    shouldReplayIdle: bridge.shouldReplayRecoveredLiveArtifacts({ status: 'idle' }),
    shouldReplayError: bridge.shouldReplayRecoveredLiveArtifacts({ status: 'error' }),
}));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {
        "calls": [],
        "shouldReplayIdle": False,
        "shouldReplayError": True,
    }


def test_status_poll_refreshes_training_sidebar_summaries() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    polling_source = _frontend_module_text("js/features/live-training/status-polling.js")
    poll_section = _section(source, "async function pollStatus", "function refreshTrainingSidebarSummariesFromPoll")
    refresh_section = _section(source, "function refreshTrainingSidebarSummariesFromPoll", "function applyStatusSnapshotFallbacks")

    assert "trainingSidebarSummaryLastRefreshAt" in polling_source
    assert "trainingSidebarSummaryLastTaskId" in polling_source
    assert "trainingSidebarSummaryLastStatus" in polling_source
    assert "trainingSidebarSummaryRefreshPromise" in polling_source
    assert "refreshTrainingSidebarSummariesFromPoll(status);" in poll_section
    assert poll_section.index("updateStatus({") < poll_section.index("refreshTrainingSidebarSummariesFromPoll(status);")
    assert "const historyTasks = readHistoryTasks();" in refresh_section
    assert "Array.isArray(historyTasks)" in refresh_section
    assert "&& historyTasks.some((task) => String(task.id || '') === taskId)" in refresh_section
    assert "now - trainingSidebarSummaryLastRefreshAt >= 15000" in refresh_section
    assert "loadTrainingQueue()" in refresh_section
    assert "loadTrainingHistoryList()" in refresh_section


def test_log_replay_keeps_tqdm_average_rate_out_of_live_metrics() -> None:
    source = _frontend_feature_text("js/features/preflight-dialog/index.js", "js/features/live-log/index.js")
    section = _section(source, "function replayMetricsFromLogRecord", "function setLogStatus")

    assert "const metrics = { ...parsed };" in section
    assert "delete metrics.rate;" in section
    assert "updateMetrics({ ...metrics, ts: record.ts });" in section
    assert "updateMetrics({ ...parsed, ts: record.ts });" not in section


def test_launch_readiness_panel_is_removed() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    listeners_source = _frontend_feature_text("js/features/app-shell/event-listeners.js", "js/features/app-shell/event-listeners-contract.js", "js/features/app-shell/event-listeners-setup.js", "js/features/app-shell/beginner-tooltips.js")
    action_source = _frontend_module_text("js/features/anima-app/chunks/22-update-toml-action-state.js")

    listener_section = _section(listeners_source, "function setupEventListeners", "function installBeginnerTooltips")
    action_state = _section(action_source, "function updateTomlActionState", "function isTomlLocked")

    assert "launch-readiness" not in html
    assert "launch-readiness" not in css
    assert "launchReadiness" not in source
    assert "启动准备" not in html
    assert "启动准备" not in source

    assert "btn-start-from-config" in html
    assert "btn-queue-from-config" in html
    assert "on('btn-start-from-config', 'click', startTraining)" in listener_section
    assert "on('btn-queue-from-config', 'click', queueCurrentTrainingFromConfig)" in listener_section
    assert "handleLaunchReadinessPrimaryAction" not in source
    assert "btn-start-from-config" in action_state
    assert "startBtn.textContent = sourceMode === 'full_resume'" in action_state
    assert "开始完整续训" in action_state
    assert "开始热启动训练" in action_state
    assert "开始训练" in action_state


def test_config_form_uses_navigation_search_and_progressive_disclosure() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    app_constants = _frontend_module_text("js/features/anima-app/helpers/app-constants.js")
    labels_options = (STATIC_DIR / "js" / "config" / "catalog" / "labels-options.js").read_text(encoding="utf-8")
    step_estimate_source = _frontend_feature_text(
        "js/features/anima-app/chunks/03-parse-network-arg-entry.js",
        "js/features/config-form/step-estimate.js",
        "js/features/dataset-editor/load.js",
    )
    field_rows_source = _frontend_module_text("js/features/config-form/field-rows.js")
    dataset_picker_source = _frontend_module_text("js/features/config-form/dataset-picker.js")
    field_input_source = _frontend_module_text("js/features/config-form/field-input.js")

    category_defs = _section(source, "const FORM_CATEGORY_DEFS = [", "const FORM_CATEGORY_SECTION_MAP")
    render_section = _section(source, "function renderConfigForm", "function shouldRenderConfigSection")
    order_section = _section(source, "function appendConfigGroupsByCategory", "function createGroup")
    collect_impl = _frontend_module_text("js/features/anima-app/chunks/18-delete-dataset-preset-group.js")
    collect_section = _section(collect_impl, "function collectChangedFormValues", "function networkArgInputChanged")
    load_steps = _section(step_estimate_source, "async function loadStepEstimate", "async function loadDatasetEditor")
    defaults = _section(source, "const FORM_UI_DEFAULTS = {", "const OPTIONAL_EMPTY_FIELDS")
    catalog_defaults = _frontend_module_text("js/config/catalog/defaults.js")
    catalog_form_layout = _frontend_module_text("js/config/catalog/form-layout.js")
    catalog_help_training = _frontend_module_text("js/config/catalog/field-help-training.js")
    persist_defaults = _section(catalog_defaults, "export const FORM_UI_PERSIST_DEFAULT_FIELDS = new Set([", "]);")
    options = _section(labels_options, "export const FIELD_OPTIONS = {", "\n};")

    assert category_defs.count("id: '") == 5
    assert category_defs.index("id: 'common'") < category_defs.index("id: 'preview'")
    assert category_defs.index("id: 'preview'") < category_defs.index("id: 'optimization'")
    assert category_defs.index("id: 'optimization'") < category_defs.index("id: 'advanced'")

    for title in [
        "基础模型路径",
        "常用训练设置",
        "步数与训练量",
        "数据集设置",
        "训练中预览图",
        "显存与速度优化",
        "LoKr 专用优化",
        "数据加载与 VAE 资源",
        "实验性功能",
        "缓存与预处理",
        "更多数据集配置",
        "SPD CLI 实验",
        "输出格式与训练范围",
        "方法内部与实验架构",
        "Soft Tokens 参数",
        "IP-Adapter 高级参数",
        "EasyControl 高级参数",
        "其他高级选项",
    ]:
        assert title in category_defs

    assert "sectionEntries.push(createConfigGroupEntry(" in render_section
    assert "appendConfigGroupsByCategory(container, sectionEntries);" in render_section
    assert "const buckets = new Map(FORM_CATEGORY_DEFS.map((category) => [category.id, []]));" in order_section
    assert "FORM_CATEGORY_SECTION_MAP.get(group.name) || 'advanced'" in order_section
    assert "createConfigFormControls(groups, renderedGroups, searchText)" in order_section
    assert "filterConfigGroupEntry(group, searchText)" in order_section
    assert "configFormState.showAdvanced || !category.advanced" in source
    assert "configFormState.search = event.target.value || ''" in source
    assert "search.addEventListener('keydown'" in source
    assert "if (event.key !== 'Escape') return;" in source
    assert "configFormState.search = '';" in source
    assert "configFormState.activeCategory = categoryId" in source
    assert "draftValues: new Map()" in source
    assert "syncConfigDraftFromForm();" in source
    assert "displayConfigFieldValue(key, value)" in source
    assert "section.open" in render_section
    assert "section.notice || ''" in render_section
    assert "configGroupIsCollapsed(name, searchText, defaultOpen)" in source
    assert "if (defaultOpen === false) return true;" in source
    assert "if (searchText) return false;" in source
    assert "function createConfigCategory" not in source
    assert "configFormState.draftValues.entries()" in collect_section
    assert "const rawNetworkArgsChanged = 'network_args' in values;" in collect_section
    assert "{ skipUnchangedInputs: rawNetworkArgsChanged }" in collect_section
    assert "applyLoraAdapterPatch(values)" in collect_section
    assert "const container = document.getElementById('choice-guide');" in source
    assert "if (!container) return;" in source
    assert "function updateChangedFieldMarks" in source
    assert "field-row-changed" in source
    assert "config-modified-count" in source
    assert "if (datasetState.selectedConfigDatasetFile !== (currentConfig.dataset_config || ''))" in source
    assert "const params = new URLSearchParams({" in load_steps
    assert "const configFile = currentTrainingConfigFile();" in load_steps
    assert "params.set('config_file', configFile);" in load_steps
    assert "const datasetConfigOverride = selectedDatasetConfigOverride();" in load_steps
    assert "if (datasetConfigOverride !== null) params.set('dataset_config', datasetConfigOverride);" in load_steps
    assert "const data = await api(`/api/config/steps?${params.toString()}`);" in load_steps
    assert "applied ? 'ok' : 'error'" in source and "setTomlStatus(" in source
    assert ".config-form-shell" in css
    assert "createConfigNav" not in source
    assert ".config-form-nav" not in css
    assert ".config-nav-tab" not in css
    assert ".config-search-box" in css
    assert ".config-advanced-toggle" in css
    assert ".field-row-changed" in css
    assert ".field-row:focus-within" in css
    assert ".field-row:focus-within::before" in css
    assert ".field-name:hover" in css
    assert ".config-group" in css
    assert "content.className = 'config-group-body';" in source
    assert "titleActions.className = 'config-group-title-actions';" in source
    assert "titleActions.appendChild(createFillGlobalModelPathsButton());" in source
    assert "titleActions.appendChild(createResourceQuickPresetsButton(content, collapseBtn));" in source
    assert "titleActions.appendChild(createNoDatasetRegularizationQuickPresetsButton(content, collapseBtn));" in source
    assert "content.appendChild(createResourceQuickPresetPanel());" in source
    assert "content.appendChild(createNoDatasetRegularizationQuickPresetPanel());" in source
    assert "titleActions.appendChild(collapseBtn);" in source
    assert ".config-category" not in css
    assert ".config-field-grid-4col" in css
    assert ".config-group-title-actions" in css
    assert ".config-group-badge-experimental" in css
    assert ".config-group-notice" in css
    assert ".field-state-hint" in css
    assert ".config-quick-presets" in css
    assert ".config-quick-preset-btn" in css
    assert ".config-resource-quick-presets" in css
    assert ".config-resource-preset-btn" in css
    assert ".no-dataset-regularization-panel" in css
    assert ".no-dataset-regularization-modes" in css
    assert ".no-dataset-regularization-advanced" in css
    assert '.no-dataset-regularization-panel[data-mode="dop"] .no-dataset-control-dop' in css
    assert '.no-dataset-regularization-panel[data-mode="conflict"] .no-dataset-control-mask-weight' in css
    resource_quick_css = _section(css, ".config-quick-presets,", ".config-quick-presets[hidden],")
    assert "grid-template-columns: repeat(auto-fit, minmax(96px, 1fr));" in resource_quick_css
    assert "grid-column: 1 / -1;" in _section(css, ".config-quick-label,", ".config-quick-preset-btn,")
    assert "function createConfigQuickPresetsButton" in source
    assert "function createConfigQuickPresetPanel" in source
    assert "RESOURCE_QUICK_PRESETS" in source
    for label in ["快捷·全 GPU", "快捷·Balanced 16G", "快捷·FP8 测试", "快捷·更省显存", "快捷·LoKr 16G", "快捷·OOM 兜底"]:
        assert label in source
    assert "merge: {" in source
    assert "blocks_to_swap: 'max'" in source
    assert "selective_checkpoint: 'checkpoint_strength_max'" in source
    quick_presets = _section(app_constants, "export const RESOURCE_QUICK_PRESETS = [", "export const NO_DATASET_REGULARIZATION_FIELD_KEYS")
    assert quick_presets.count("gradient_checkpointing: false") == 6
    assert "export const SELECTIVE_CHECKPOINT_STRENGTH = new Map([" in app_constants
    assert "['mlp_only', 4]" in source
    assert "['every_other', 5]" in source
    assert "function resourceQuickPresetValue(preset, key, value)" in source
    assert "function resourceQuickPresetPatch(preset)" in source
    assert "function strongerSelectiveCheckpointValue(current, fallback)" in source
    assert "Object.entries(resourceQuickPresetPatch(preset))" in source
    assert "patch.gradient_checkpointing = false;" in source
    assert "patch.cpu_offload_checkpointing = false;" in source
    assert "patch.unsloth_offload_checkpointing = false;" in source
    assert "return Math.max(current, next);" in source
    assert "return currentStrength >= fallbackStrength ? currentKey : fallbackKey;" in source
    assert "NO_DATASET_REGULARIZATION_QUICK_PRESETS" in source
    for label in ["先验基线", "DOP 角色", "遮罩保护", "关闭"]:
        assert label in source
    no_dataset_quick_presets = _section(app_constants, "export const NO_DATASET_REGULARIZATION_QUICK_PRESETS = [", "export const SELECTIVE_CHECKPOINT_STRENGTH")
    assert "prior_preservation_weight: 0.1" in no_dataset_quick_presets
    assert "blank_prompt_preservation: true" in no_dataset_quick_presets
    assert "diff_output_preservation_trigger: 'sks'" in no_dataset_quick_presets
    assert "diff_output_preservation_class: ''" in no_dataset_quick_presets
    assert "inverted_mask_prior_weight: 0.1" in no_dataset_quick_presets
    assert no_dataset_quick_presets.count("use_text_cache: true") == 3
    assert no_dataset_quick_presets.count("cache_llm_adapter_outputs: true") == 3
    assert "function applyNoDatasetRegularizationQuickPreset" in source
    assert "还需要填写泛化类别作为 DOP 类提示，例如 woman / character，并重新生成文本缓存" in source
    assert "NO_DATASET_REGULARIZATION_MODE_SPECS" in source
    for label in ["空提示先验", "DOP / class prompt", "反转遮罩保护"]:
        assert label in source
    no_dataset_mode_panel = _section(source, "function createNoDatasetRegularizationModePanel", "function createNoDatasetRegularizationAdvancedFields")
    no_dataset_advanced = _section(source, "function createNoDatasetRegularizationAdvancedFields", "function createNoDatasetRegularizationNumberControl")
    no_dataset_number_control = _section(source, "function createNoDatasetRegularizationNumberControl", "function createNoDatasetRegularizationTextControl")
    no_dataset_text_control = _section(source, "function createNoDatasetRegularizationTextControl", "function updateNoDatasetRegularizationFieldFromMirror")
    no_dataset_patch = _section(source, "function noDatasetRegularizationPatchForMode", "function updateNoDatasetRegularizationModePanel")
    no_dataset_update = _section(source, "function updateNoDatasetRegularizationModePanel", "function readNoDatasetRegularizationValues")
    no_dataset_infer = _section(source, "function inferNoDatasetRegularizationMode", "function noDatasetRegularizationStatusMessage")
    no_dataset_status = _section(source, "function noDatasetRegularizationStatusMessage", "function setNoDatasetRegularizationMirrorValue")
    assert "no-dataset-regularization-panel" in no_dataset_mode_panel
    assert "role', 'radiogroup'" in no_dataset_mode_panel
    assert "dataset.noDatasetRegularizationMode = spec.id" in no_dataset_mode_panel
    assert "input.dataset.noDatasetRegularizationMirror = options.key" in no_dataset_number_control
    assert "input.dataset.noDatasetRegularizationMirror = options.key" in no_dataset_text_control
    assert "填 caption 中代表训练目标的词" in no_dataset_mode_panel
    assert "人物/角色用 woman、man、character" in no_dataset_mode_panel
    assert "label.appendChild(hint);" in no_dataset_text_control
    assert "NO_DATASET_REGULARIZATION_ADVANCED_SUMMARY" in no_dataset_advanced
    assert "appendFieldRows(body, fields, groupClass);" in no_dataset_advanced
    assert "blank_prompt_preservation: true" in no_dataset_patch
    assert "blank_prompt_preservation: false" in no_dataset_patch
    assert "diff_output_preservation_class: dopClass" in no_dataset_patch
    assert "inverted_mask_prior_weight: maskWeight" in no_dataset_patch
    assert "NO_DATASET_REGULARIZATION_CACHE_PATCH" in no_dataset_patch
    assert "prior_preservation_weight: 0.0" in no_dataset_patch
    assert "active.length > 1 || orphanPrior || (blankEnabled && dopEnabled)" in no_dataset_infer
    assert "NO_DATASET_REGULARIZATION_CONFLICT_MESSAGE" in no_dataset_status
    assert "NO_DATASET_REGULARIZATION_DOP_CLASS_REQUIRED" in no_dataset_status
    assert "advanced.open = true;" in no_dataset_update
    assert "configureNoDatasetRegularizationModePanelUpdater(updateNoDatasetRegularizationModePanel);" in source
    assert "updateNoDatasetRegularizationModePanelCallback();" in source
    for value in [
        "blocks_to_swap: 12",
        "blocks_to_swap: 16",
        "blocks_to_swap: 23",
        "block_swap_transfer_dtype: 'bf16'",
        "block_swap_transfer_dtype: 'fp8_e4m3'",
        "block_swap_restore_mode: 'foreach'",
        "selective_checkpoint: 'mlp_only'",
        "block_swap_profile_jsonl: 'auto'",
        "memory_probe_jsonl: 'auto'",
        "memory_probe_max_steps: 2",
        "memory_probe_max_steps: 3",
        "lokr_factor_group_size: 8",
        "lokr_project_chunk_bytes: 4194304",
    ]:
        assert value in source
    balanced_preset = _section(source, "id: 'balanced_16g'", "id: 'fp8_swap_test'")
    fp8_preset = _section(source, "id: 'fp8_swap_test'", "id: 'vram_saver'")
    assert "block_swap_profile_jsonl: 'off'" in balanced_preset
    assert "block_swap_profile_jsonl: 'auto'" in fp8_preset
    set_field_section = _section(field_input_source, "function setFieldInputValue", "function escapeHtml")
    assert "configDraftValueChanged(key, value, original)" in set_field_section
    assert "configFormState.draftValues.delete(key);" in set_field_section
    assert "input.value = value ?? '';" in set_field_section
    compact_grid_section = field_rows_source
    compact_field_css = _section(css, ".config-field-grid-3col .field-main", ".field-label-stack")
    filler_css = _section(css, ".config-field-grid .field-row-filler", ".config-field-grid .field-row-filler::before")
    config_filler_css = _section(css, "#tab-config .config-field-grid .field-row-filler", "#tab-config .config-field-grid .field-row:focus-within")
    assert "appendCompactGridFillers(grid);" in compact_grid_section
    assert "querySelectorAll('.field-row-filler').forEach((node) => node.remove());" in compact_grid_section
    assert "grid.appendChild(createCompactGridFiller());" in compact_grid_section
    assert "filler.className = 'field-row field-row-compact field-row-filler';" in compact_grid_section
    assert "grid.classList.remove('config-field-grid-2col', 'config-field-grid-3col', 'config-field-grid-4col', 'config-field-grid-5col');" in compact_grid_section
    assert "grid-template-rows: auto auto;" in compact_field_css
    assert "row-gap: 0.24rem;" in compact_field_css
    assert "grid-row: 1;" in compact_field_css
    assert "grid-row: 2;" in compact_field_css
    assert "pointer-events: none;" in filler_css
    assert "color-mix(in srgb, var(--config-panel-bg) 72%, var(--config-panel-soft))" in config_filler_css
    data_section = _section(source, "title: '数据集设置'", "title: '训练中预览图'")
    data_compact = _section(source, "'config-group-data': [", "'config-group-sampling': [")
    inline_flag_css = _section(css, ".config-field-grid-inline-flags .field-main", ".field-label-stack")
    assert "'use_shuffled_caption_variants'," in data_section
    assert "'masked_loss'," in data_section
    assert "'caption_dropout_rate'," in data_section
    assert data_section.index("'masked_loss',") < data_section.index("'caption_dropout_rate',")
    assert "config-field-grid-2col config-field-grid-inline-flags" in data_compact
    assert "keys: ['use_shuffled_caption_variants', 'masked_loss']" in data_compact
    assert "grid-template-columns: minmax(0, 1fr) auto 18px;" in inline_flag_css
    assert "grid-template-rows: auto;" in inline_flag_css

    primary_section = _section(source, "title: '常用训练设置'", "title: '步数与训练量'")
    resource_section = _section(source, "title: '显存与速度优化'", "title: 'LoKr 专用优化'")
    optimization_section = _section(source, "title: '显存与速度优化'", "title: '缓存与预处理'")
    experimental_section = _section(source, "title: '实验性功能'", "title: '无数据集正则化'")
    no_dataset_reg_section = _section(source, "title: '无数据集正则化'", "title: '缓存与预处理'")
    resource_compact = _section(source, "'config-group-resource': [", "'config-group-data-resource': [")
    data_resource_compact = _section(source, "'config-group-data-resource': [", "const VARIANT_METHOD_FAMILY")
    assert "'gradient_checkpointing'," not in primary_section
    assert "'save_last_n_epochs'," in primary_section
    assert "'checkpointing_last_n_epochs'," in primary_section
    assert primary_section.index("'save_every_n_epochs',") < primary_section.index("'save_last_n_epochs',")
    assert primary_section.index("'save_last_n_epochs',") < primary_section.index("'checkpointing_epochs',")
    assert primary_section.index("'checkpointing_epochs',") < primary_section.index("'checkpointing_last_n_epochs',")
    assert "save_last_n_epochs: -1" in defaults
    assert "checkpointing_last_n_epochs: 1" in defaults
    assert "open: true," in resource_section
    assert "'gradient_checkpointing'," in optimization_section
    assert "'block_swap_transfer_dtype'," in optimization_section
    assert "'block_swap_restore_mode'," in optimization_section
    assert "'compile_block_scope'," in optimization_section
    assert "'memory_probe_jsonl'," in optimization_section
    assert "'memory_probe_max_steps'," in optimization_section
    assert "'peak_probe_jsonl'," in optimization_section
    assert "'peak_probe_max_steps'," in optimization_section
    assert "'peak_probe_level'," in optimization_section
    assert "'lr_warmup_steps'," in optimization_section
    assert "'lokr_factor_group_size'," in optimization_section
    assert "'lokr_project_chunk_bytes'," in optimization_section
    assert "sections: ['显存与速度优化', 'LoKr 专用优化', '数据加载与 VAE 资源', '实验性功能', '无数据集正则化']" in category_defs
    assert category_defs.index("数据加载与 VAE 资源") < category_defs.index("实验性功能")
    assert category_defs.index("实验性功能") < category_defs.index("无数据集正则化")
    assert "notice: '建议：正式训练保持默认。'" in experimental_section
    assert "不额外准备正则化图片时使用的先验保留方案" in no_dataset_reg_section
    assert "notice: '需要先开启文本缓存；DOP 和反转遮罩先验还需要缓存 LLM 适配器输出。'" in no_dataset_reg_section
    assert "className: 'config-group-no-dataset-regularization'" in no_dataset_reg_section
    assert "open: false," in no_dataset_reg_section
    assert "config-group-badge-experimental" in source
    assert "export const LOSS_WEIGHTING_DEPENDENT_FIELDS = new Map([" in app_constants
    assert "['min_snr_gamma', 'min_snr']" in source
    assert "['p2_gamma', 'p2']" in source
    assert "['p2_k', 'p2']" in source
    assert "function updateLossWeightingFieldState()" in source
    assert "仅 weighting_scheme = ${state.requiredScheme} 时生效" in source
    assert "updateLossWeightingFieldState();" in source
    for key in (
        "sigmoid_scale",
        "sigmoid_bias",
        "weighting_scheme",
        "min_snr_gamma",
        "p2_gamma",
        "p2_k",
        "velocity_direction_loss_weight",
    ):
        assert f"'{key}'," in experimental_section
        assert f"{key}:" in defaults
    for key in (
        "prior_preservation_weight",
        "blank_prompt_preservation",
        "diff_output_preservation_trigger",
        "diff_output_preservation_class",
        "inverted_mask_prior_weight",
    ):
        assert f"'{key}'," in no_dataset_reg_section
        assert f"'{key}'," not in experimental_section
        assert f"'{key}'" in catalog_form_layout
        assert f"{key}:" in catalog_defaults
    assert "prior_preservation_weight: '无数据集先验保留权重'" in labels_options
    assert "blank_prompt_preservation: '空提示先验保留'" in labels_options
    assert "diff_output_preservation_trigger: 'DOP 触发词'" in labels_options
    assert "diff_output_preservation_class: 'DOP 类提示'" in labels_options
    assert "inverted_mask_prior_weight: '反转遮罩先验权重'" in labels_options
    assert "无额外数据集的先验保留辅助损失权重" in catalog_help_training
    assert "使用空提示 T5" in catalog_help_training
    assert "作为先验保留条件" in catalog_help_training
    assert "prior_crossattn_emb" in catalog_help_training
    assert "不能和 blank_prompt_preservation 同时使用" in catalog_help_training
    assert "真正必填的是 DOP 类提示" in catalog_help_training
    assert "如果去掉专名，这批图大体属于什么类别" in catalog_help_training
    assert "class prompt 是 prior caption 的目标文本" in catalog_help_training
    assert "只在遮罩外区域做先验保留" in catalog_help_training
    assert "block_swap_transfer_dtype: '块交换传输精度'" in source
    assert "block_swap_transfer_dtype: ['bf16', 'fp8_e4m3']" in source
    assert "block_swap_restore_mode: '块交换恢复路径'" in source
    assert "block_swap_restore_mode: ['foreach', 'slab']" in source
    assert "compile_block_scope: '编译块范围'" in source
    assert "compile_block_scope: ['resident', 'all']" in source
    assert "compile_block_scope: 'resident'" in source
    assert "哪些 DiT block 参与 torch.compile" in catalog_help_training
    assert "block_swap_profile_jsonl: ['off', 'auto']" in source
    assert "这里不是显卡训练精度开关" in catalog_help_training
    assert "即使显卡本身不支持 bf16 训练，也可以继续使用这个默认值" in catalog_help_training
    assert "训练精度请看上面的“精度倾向”" in catalog_help_training
    assert "同一 slot 的多个小 weight 恢复合并成更少的大 H2D" in catalog_help_training
    assert "precision_preference: '精度倾向'" in source
    assert "precision_preference: ['bf16', 'fp16', 'fp32']" in source
    assert "memory_probe_jsonl: '显存探针'" in source
    assert "memory_probe_jsonl: ['off', 'auto']" in source
    assert "memory_probe_max_steps: [1, 2, 3, 5, 0]" in source
    assert "peak_probe_jsonl: '峰值探针'" in source
    assert "peak_probe_jsonl: ['off', 'auto']" in source
    assert "peak_probe_level: ['block', 'ops', 'lokr', 'full']" in source
    assert "lokr_factor_group_size: 'LoKr 分组'" in source
    assert "lokr_factor_group_size: [1, 2, 4, 8]" in source
    assert "lokr_project_chunk_bytes: 'LoKr 张量切块阈值'" in source
    assert "lokr_project_chunk_bytes: [1048576, 2097152, 4194304, 8388608, 16777216]" in source
    assert "keys: ['blocks_to_swap', 'block_swap_transfer_dtype', 'block_swap_restore_mode', 'selective_checkpoint', 'selective_checkpoint_blocks']" in resource_compact
    assert "keys: ['block_swap_profile_jsonl', 'memory_probe_jsonl', 'memory_probe_max_steps']" in resource_compact
    assert "keys: ['peak_probe_jsonl', 'peak_probe_max_steps', 'peak_probe_level']" in resource_compact
    assert "keys: ['preprocess_vae_cache_batch_size', 'preprocess_text_cache_batch_size', 'preprocess_memory_profile', 'reuse_dataset_cache_copy', 'reuse_vae_latents', 'reuse_text_encoder_cache', 'cache_fingerprint_mode', 'force_rebuild_preprocess_cache']" in resource_compact
    assert "keys: ['attn_mode', 'torch_compile', 'compile_block_scope', 'compile_inductor_mode']" in resource_compact
    assert "'preprocess_precision_preference'," in optimization_section
    assert "'precision_preference'," in optimization_section
    assert "'mixed_precision'," not in optimization_section
    assert "keys: ['gradient_checkpointing', 'precision_preference']" in resource_compact
    assert "keys: ['unsloth_offload_checkpointing', 'disable_block_swap_for_eval']" in resource_compact
    assert "keys: ['max_data_loader_n_workers', 'vae_chunk_size', 'vae_disable_cache']" in data_resource_compact
    assert "keys: ['dataloader_pin_memory', 'persistent_data_loader_workers']" in data_resource_compact
    assert "config-field-grid-2col config-field-grid-inline-flags" in resource_compact
    assert "config-field-grid-2col config-field-grid-inline-flags" in data_resource_compact
    assert "preprocess_memory_profile: 'auto'" in defaults
    assert "preprocess_vae_cache_batch_size: 'auto'" in defaults
    assert "preprocess_text_cache_batch_size: 'auto'" in defaults
    assert "preprocess_precision_preference: 'bf16'" in defaults
    assert "precision_preference: 'bf16'" in defaults
    assert "sample_sampler: 'euler'" in defaults
    assert "sample_sampler: ['euler', 'er_sde', 'lcm']" in options
    assert "训练时优先采用哪种数值精度方案" in catalog_help_training
    assert "fp16/32 混合精度" in catalog_help_training
    assert "全程使用 fp32" in catalog_help_training
    assert "preprocess_precision_preference: '预处理精度'" in source
    assert "preprocess_precision_preference: ['bf16', 'fp16', 'fp32']" in source
    assert "只影响 WebUI/任务链触发的 VAE latent cache 和文本缓存计算精度" in catalog_help_training
    assert "'precision_preference'," in persist_defaults
    assert "'preprocess_precision_preference'," in persist_defaults


def test_preprocess_memory_profile_updates_cache_batch_inputs() -> None:
    source = _frontend_feature_text("js/features/anima-app/chunks/14-lora-adapter-kind-from-config.js", "js/features/config-form/form-fields.js", "js/features/config-form/form-fields-adapters.js", "js/features/config-form/form-fields-sample.js", "js/features/config-form/form-fields-ui.js")

    assert "const PREPROCESS_MEMORY_PROFILE_VALUES = {" in source
    assert "auto: { preprocess_vae_cache_batch_size: 'auto', preprocess_text_cache_batch_size: 'auto' }" in source
    assert "low_vram: { preprocess_vae_cache_batch_size: 1, preprocess_text_cache_batch_size: 4 }" in source
    assert "balanced: { preprocess_vae_cache_batch_size: 2, preprocess_text_cache_batch_size: 8 }" in source
    assert "speed: { preprocess_vae_cache_batch_size: 4, preprocess_text_cache_batch_size: 16 }" in source
    assert "function applyPreprocessMemoryProfileSelection(event)" in source
    assert "target?.dataset?.key !== 'preprocess_memory_profile'" in source
    assert "setConfigFieldInputValue(key, value)" in source
    assert "applyPreprocessMemoryProfileSelection(event);" in source


def test_precision_preference_ui_maps_to_training_precision_fields() -> None:
    form_source = _chunk02_compat_text()
    helper_source = _frontend_module_text("js/features/anima-app/helpers/config-values.js")
    form_helper_source = _frontend_feature_text("js/features/anima-app/chunks/14-lora-adapter-kind-from-config.js", "js/features/config-form/form-fields.js", "js/features/config-form/form-fields-adapters.js", "js/features/config-form/form-fields-sample.js", "js/features/config-form/form-fields-ui.js")
    option_source = _chunk15_compat_text()
    patch_source = _frontend_module_text("js/features/anima-app/chunks/18-delete-dataset-preset-group.js")
    guide_source = _frontend_feature_text("js/features/anima-app/chunks/13-update-dataset-editor-rows-setting-value.js", "js/features/config-form/choice-guide-ui.js")

    assert "function normalizePrecisionPreference" in helper_source
    assert "function precisionPreferenceFromConfig" in helper_source
    assert "function precisionPreferencePatch" in helper_source
    assert "mixedPrecision === 'no'" in helper_source
    assert "patch.full_fp16 = false;" in helper_source
    assert "patch.full_bf16 = false;" in helper_source
    assert "if (key === 'precision_preference')" in form_source
    assert "return precisionPreferenceFromConfig(currentConfig);" in form_source
    assert "return configFormState.draftValues.has(key)" in form_source
    assert ": precisionPreferenceFromConfig(currentConfig);" in form_source
    assert "key === 'mixed_precision' || key === 'full_fp16' || key === 'full_bf16'" in form_source
    assert "Object.assign(liveConfig, precisionPreferencePatch(next, currentConfig));" in form_helper_source
    assert "混合精度 / fp16/32" in option_source
    assert "全程 fp32 / full fp32" in option_source
    assert "'precision_preference' in nextValues" in patch_source
    assert "delete nextValues.precision_preference;" in patch_source
    assert "Object.assign(nextValues, precisionPreferencePatch(nextValues.precision_preference, currentConfig));" in patch_source
    assert "const original = precisionPreferenceFromConfig(currentConfig);" in patch_source
    assert "const normalized = normalizePrecisionPreference(next);" in patch_source
    assert "if (!valuesEqual(normalized, original)) {" in patch_source
    assert "values[key] = normalized;" in patch_source
    assert "valueDetail('precision_preference', precisionPreferenceFromConfig(config))" in guide_source


def test_precision_preference_dirty_state_uses_derived_config_value() -> None:
    form_source = _chunk02_compat_text()

    assert "if (key === 'precision_preference')" in form_source
    assert "return normalizePrecisionPreference(next) !== precisionPreferenceFromConfig(currentConfig);" in form_source


def test_precision_preference_display_value_uses_derived_config_value() -> None:
    form_source = _chunk02_compat_text()

    assert "if (key === 'precision_preference')" in form_source
    assert "? normalizePrecisionPreference(configFormState.draftValues.get(key))" in form_source
    assert ": precisionPreferenceFromConfig(currentConfig);" in form_source


def test_precision_preference_persists_even_when_matching_ui_default() -> None:
    defaults_source = _frontend_module_text("js/config/catalog/defaults.js")
    persist_defaults = _section(defaults_source, "export const FORM_UI_PERSIST_DEFAULT_FIELDS = new Set([", "]);")

    assert "'precision_preference'," in persist_defaults


def test_preprocess_precision_preference_is_persisted_when_missing_from_current_config() -> None:
    source = _frontend_module_text("js/features/anima-app/chunks/18-delete-dataset-preset-group.js")
    collect_section = _section(source, "function collectChangedFormValues", "function networkArgInputChanged")

    assert "options.persistDefaultFields" in collect_section
    assert "!('preprocess_precision_preference' in values)" in collect_section
    assert "!Object.prototype.hasOwnProperty.call(currentConfig || {}, 'preprocess_precision_preference')" in collect_section
    assert "values.preprocess_precision_preference = normalizePrecisionPreference(" in collect_section


def test_soft_tokens_advanced_fields_match_training_defaults() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    labels_options = (STATIC_DIR / "js" / "config" / "catalog" / "labels-options.js").read_text(encoding="utf-8")
    defaults = _section(source, "const FORM_UI_DEFAULTS = {", "const OPTIONAL_EMPTY_FIELDS")
    network_arg_specs = _section(source, "const NETWORK_ARG_FIELD_SPECS = [", "const NETWORK_ARG_FIELD_MAP")
    layout = _section(source, "title: 'Soft Tokens 参数'", "title: 'IP-Adapter 高级参数'")
    options = _section(labels_options, "export const FIELD_OPTIONS = {", "\n};")
    option_labels = _section(source, "function optionLabel", "function generateDefaultHelp")
    extra_help = _section(source, "export const EXTRA_FIELD_HELP_ZH = {", "    encoder: help(")

    assert "agsm" not in network_arg_specs.lower()
    assert "agsm" not in options.lower()
    assert "AGSM" not in option_labels
    assert "AGSM" not in extra_help
    assert "contrastive_objective: 'infonce'" in defaults
    assert "contrastive_negative_mode: 'shuffled'" in defaults
    assert "contrastive_every_n: 1" in defaults
    assert "n_layers: 10" in defaults
    assert "n_t_buckets: 100" in defaults
    assert "splice_position: 'end_of_sequence'" in defaults
    assert "{ family: 'soft_tokens', key: 'contrastive_objective', arg: 'contrastive_objective', default: 'infonce', valueType: 'string' }" in network_arg_specs
    assert "{ family: 'soft_tokens', key: 'softrank_softness', arg: 'softrank_softness', default: 0.1, valueType: 'number' }" in network_arg_specs
    assert "{ family: 'soft_tokens', key: 'softrank_method', arg: 'softrank_method', default: 'neuralsort', valueType: 'string' }" in network_arg_specs
    assert "{ family: 'soft_tokens', key: 'dual_bank', arg: 'dual_bank', default: false, valueType: 'boolean' }" in network_arg_specs
    assert "contrastive_objective: ['infonce', 'softrank']" in options
    assert "softrank_method: ['neuralsort', 'softsort']" in options
    assert "dual_bank: [false, true]" in options
    for key in ("'softrank_softness'", "'softrank_method'", "'dual_bank'"):
        assert key in layout


def test_network_args_raw_editor_keeps_unmodified_split_controls_from_overwriting() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    collect_impl = _frontend_module_text("js/features/anima-app/chunks/18-delete-dataset-preset-group.js")
    form_fields = _frontend_feature_text(
        "js/features/anima-app/chunks/14-lora-adapter-kind-from-config.js",
        "js/features/config-form/form-fields.js", "js/features/config-form/form-fields-adapters.js", "js/features/config-form/form-fields-sample.js", "js/features/config-form/form-fields-ui.js",
    )
    collect_section = _section(collect_impl, "function collectChangedFormValues", "function networkArgInputChanged")
    live_section = _section(form_fields, "function liveConfigFromForm", "function createFieldInput")
    network_args_section = _section(collect_impl, "function collectNetworkArgsFromForm", "function prepareFormPatchValues")

    assert "const rawNetworkArgsChanged = 'network_args' in values;" in collect_section
    assert "{ skipUnchangedInputs: rawNetworkArgsChanged }" in collect_section
    assert "const rawNetworkArgsChanged = configFormState.draftValues.has('network_args');" in live_section
    assert "collectNetworkArgsFromForm(liveConfig, { skipUnchangedInputs: rawNetworkArgsChanged })" in live_section
    assert "function collectNetworkArgsFromForm(baseConfig = currentConfigState(), options = {})" in network_args_section
    assert "if (options.skipUnchangedInputs && !networkArgInputChanged(input)) continue;" in network_args_section


def test_config_form_keeps_dora_as_lora_addon_and_merges_exclusive_adapters() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    config_values = _frontend_module_text("js/features/anima-app/helpers/config-values.js")
    form_fields = _frontend_feature_text(
        "js/features/anima-app/chunks/14-lora-adapter-kind-from-config.js",
        "js/features/config-form/form-fields.js", "js/features/config-form/form-fields-adapters.js", "js/features/config-form/form-fields-sample.js", "js/features/config-form/form-fields-ui.js",
    )
    collect_impl = _frontend_module_text("js/features/anima-app/chunks/18-delete-dataset-preset-group.js")
    defaults = _section(source, "const FORM_UI_DEFAULTS = {", "const OPTIONAL_EMPTY_FIELDS")
    network_arg_specs = _section(source, "const NETWORK_ARG_FIELD_SPECS = [", "const NETWORK_ARG_FIELD_MAP")
    layout = _section(source, "const FORM_SECTION_DEFS = [", "const STICKY_CONFIG_CATEGORY_IDS")
    merged_fields = _section(source, "const CONFIG_FORM_MERGED_FIELDS = new Set([", "const DEPRECATED_CONFIG_FORM_FIELDS")
    render_section = _section(source, "function renderConfigForm", "function shouldRenderConfigSection")
    collect_section = _section(collect_impl, "function collectChangedFormValues", "function networkArgInputChanged")
    live_section = _section(form_fields, "function liveConfigFromForm", "function createFieldInput")
    state_section = _section(collect_impl, "function readLoKrEnabled", "function currentLossWeightingScheme")

    assert "lora_adapter_kind: 'lora'" in defaults
    assert "dora_wd: false" in defaults
    assert "use_glora: false" in defaults
    assert "use_vera: false" in defaults
    assert "vera_projection_prng_key: 0" in defaults
    assert "vera_d_initial: 0.1" in defaults
    assert "vera_save_projection: false" in defaults
    assert "lokr_use_einsum: true" in defaults
    assert "lokr_decompose_w2: false" in defaults
    assert "'lora_adapter_kind'" in layout
    assert "'dora_wd'" in layout
    assert "'use_loha'" not in layout
    assert "'use_lokr'" not in layout
    assert "'use_glora'" not in layout
    assert "'use_vera'" not in layout
    assert "keys: ['network_dim', 'network_alpha', 'lora_adapter_kind', 'dora_wd', 'lokr_factor', 'vera_projection_prng_key', 'vera_d_initial', 'vera_save_projection']" in source
    assert "{ family: 'lokr', key: 'lokr_factor_group_size', arg: 'lokr_factor_group_size', default: 8, valueType: 'integer' }" in network_arg_specs
    assert "{ family: 'lokr', key: 'lokr_project_chunk_bytes', arg: 'lokr_project_chunk_bytes', default: 4194304, valueType: 'integer' }" in network_arg_specs
    assert "{ family: 'lokr', key: 'lokr_use_einsum', arg: 'lokr_use_einsum', default: true, valueType: 'boolean' }" in network_arg_specs
    assert "{ family: 'lokr', key: 'lokr_decompose_w2', arg: 'lokr_decompose_w2', default: false, valueType: 'boolean' }" in network_arg_specs
    assert "'dora_wd'" not in merged_fields
    assert "'use_glora'" in merged_fields
    assert "'use_loha'" in merged_fields
    assert "'use_lokr'" in merged_fields
    assert "'use_vera'" in merged_fields
    assert "CONFIG_FORM_MERGED_FIELDS?.has?.(key)" in render_section
    assert "ALWAYS_VISIBLE_NETWORK_ARG_FIELDS = new Set([" in source
    assert "'lokr_use_einsum'," in source
    assert "'lokr_decompose_w2'," in source
    assert "if (NETWORK_ARG_FIELD_MAP.has(key)) return ALWAYS_VISIBLE_NETWORK_ARG_FIELDS.has(key);" in source
    assert "function loraAdapterFlagsForKind" in config_values
    assert "values.use_glora = flags.use_glora" in source
    assert "values.use_loha = flags.use_loha" in source
    assert "values.use_lokr = flags.use_lokr" in source
    assert "values.use_vera = flags.use_vera" in source
    assert "values.dora_wd = false" in source
    assert "if (key === 'lora_adapter_kind')" in collect_section
    assert "continue;" in collect_section
    assert "Object.assign(liveConfig, loraAdapterFlagsForKind(nextKind));" in live_section
    assert "if (nextKind !== 'lora') liveConfig.dora_wd = false;" in live_section
    assert "return readLiveLoraAdapterKind() === 'lokr';" in state_section
    assert "return readLiveLoraAdapterKind() === 'vera';" in source
    assert "return readLiveLoraAdapterKind() === 'lora';" in state_section
    assert "function updateDoRAFieldState" in state_section
    assert "input.checked = false" in state_section
    assert "DoRA 仅支持普通 LoRA；切到 LoHa/LoKr/GLoRA/VeRA 时会自动关闭" in source
    assert "function focusConfigFieldInput" in source
    assert "nameSpan.addEventListener('click', () => focusConfigFieldInput(input));" in source
    assert "target.focus();" in source
    assert "target.select();" in source


def test_sample_prompts_save_uses_current_training_config_context() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    body = _section(source, "async function saveSamplePrompts", "async function importTomlFile")
    prepare_body = _section(source, "async function prepareFormPatchValues", "function shouldSkipUiDefaultField")

    assert "train_config_file: currentTrainingSourceState().file || tomlState.currentTomlFile || ''" in body
    assert "await saveSamplePrompts('');" not in prepare_body


def test_config_form_save_reload_and_launch_share_training_config_file() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    preflight_source = _frontend_feature_text("js/features/preflight-dialog/index.js", "js/features/live-log/index.js")
    output_run_source = _frontend_module_text("js/features/anima-app/chunks/16-load-output-run-config.js")
    action_source = _frontend_module_text("js/features/anima-app/chunks/22-update-toml-action-state.js")
    step_estimate_source = _frontend_feature_text(
        "js/features/anima-app/chunks/03-parse-network-arg-entry.js",
        "js/features/config-form/step-estimate.js",
        "js/features/dataset-editor/load.js",
    )
    save_patch = _section(output_run_source, "async function saveFormPatchToToml", "configureOutputRunBridge")
    dataset_apply = _frontend_module_text("js/features/anima-app/chunks/17-apply-selected-dataset-preset-to-current-config.js")
    action_state = _section(action_source, "function updateTomlActionState", "function isTomlLocked")
    save_toml = _section(output_run_source, "async function saveTomlFile", "async function saveRawTomlContent")
    selection_source = _frontend_module_text("js/features/anima-app/chunks/21-update-toml-selection-ui.js")
    pending_changes = _section(selection_source, "function hasUnsavedFormChanges", "function collectPendingConfigChangeDetails")
    startup_source = _frontend_module_text("js/features/app-shell/startup.js")
    load_config = _section(startup_source, "async function loadConfig", "async function reloadCurrentConfig")
    load_steps = _section(step_estimate_source, "async function loadStepEstimate", "async function loadDatasetEditor")
    launch_source = _frontend_module_text("js/features/training-launch/index.js")
    run_preflight = _section(launch_source, "async function runPreflight", "function isCliOnlySpdSource")
    start_unchecked = _section(launch_source, "async function startTrainingUnchecked", "async function enqueueTrainingFromConfig")
    current_file = _section(preflight_source, "function currentTrainingConfigFile", "function preflightPlainText")

    assert "const content = currentTomlEditorContentForFile(file);" in save_patch
    assert "if (content !== undefined) payload.content = content;" in save_patch
    assert "body: JSON.stringify(payload)" in save_patch
    assert "file === (tomlState.currentTomlFile || val('toml-file-select'))" in save_patch
    assert "currentTomlEditorContentForFile(file)" in dataset_apply
    assert "currentTomlEditorContentForFile(targetFile)" in source
    assert "await loadConfig();" in save_patch
    assert "function currentFormConfigFile" in source
    assert "return currentTrainingSourceState().file || '';" in source
    assert "currentTomlEditorContentForFile" in pending_changes
    assert "const formFile = currentFormConfigFile();" in action_state
    assert "const saveFile = formDirty ? formFile : selectedFile;" in action_state
    assert "const saveLocked = formDirty ? isTomlLocked(saveFile) : Boolean(saveMeta?.locked);" in action_state
    assert "saveBtn.disabled = saveLocked || !saveFile || !dirty;" in action_state
    assert "保存更新当前表单配置" in action_state
    assert "const file = !directEditorSave && formDirty ? formFile : selectedFile;" in save_toml
    assert "editorDirty && formDirty && selectedFile !== formFile" in save_toml
    assert "configState.currentConfig = data;" in load_config
    assert "renderConfigForm(data);" in load_config
    assert "const params = new URLSearchParams({ variant, preset, methods_subdir: methodsSubdir });" in load_config
    assert "if (configFile) params.set('config_file', configFile);" in load_config
    assert "const data = await api(`/api/config/merged?${params.toString()}`);" in load_config
    assert "scheduleStepEstimatePanelRefresh();" in load_config
    assert "const tomlFile = currentTrainingSource.file || `configs/${methodsSubdir}/${variant}.toml`;" in load_config
    assert "const configFile = currentTrainingConfigFile();" in load_steps
    assert "params.set('config_file', configFile);" in load_steps
    assert "config_file: currentTrainingConfigFile()," in run_preflight
    assert "config_file: currentTrainingConfigFile()," in start_unchecked
    assert "return outputRunRuntimeFile();" in current_file
    assert "return currentTrainingSource.file || tomlState.currentTomlFile || val('toml-file-select') || '';" in current_file


def test_config_training_source_modes_are_audited_before_launch() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    legacy_source = _anima_app_container_text()
    config_source = _frontend_module_text("js/features/training-source/index.js")
    action_state = _frontend_module_text("js/features/anima-app/chunks/22-update-toml-action-state.js")
    launch_source = _chunk23_compat_text()
    training_state_source = _frontend_module_text("js/features/anima-app/state/training-state.js")
    queue_enqueue = _frontend_module_text("js/features/queue/enqueue.js")
    tabs_source = _frontend_module_text("js/features/app-shell/tabs.js")
    history_labels = _frontend_module_text("js/features/anima-app/chunks/32-history-task-collection-label.js")
    listener_section = _section(legacy_source, "function setupEventListeners", "function installBeginnerTooltips")

    for snippet in (
        'data-training-source-mode="fresh"',
        'data-training-source-mode="full_resume"',
        'data-training-source-mode="weight_hotstart"',
        'config-full-resume-task-select',
        'config-full-resume-checkpoint-select',
        'config-weight-hotstart-panel',
        'config-training-source-status',
        '选择 LoRA/LoHa/LoKr/GLoRA 权重热启动',
    ):
        assert snippet in html
    assert "选择 LoRA/LoKr 继续训练" not in html
    assert "权重热启动 ${kind}" in history_labels

    assert "auditConfigTrainingSourceOnEnter?.();" in tabs_source
    assert "setConfigTrainingSourceMode" in listener_section
    assert "handleConfigFullResumeTaskChange" in listener_section
    assert "handleConfigFullResumeCheckpointChange" in listener_section
    assert "btn-refresh-config-full-resume" in listener_section

    assert "mode: 'fresh'" in training_state_source

    for snippet in (
        "mode === 'full_resume'",
        "mode === 'weight_hotstart'",
        "auditConfigFullResumeSource",
        "auditConfigWeightHotstartSource",
        "resume_available === false",
        "remaining_steps",
        "/api/training/history/${encodeURIComponent(full.task_id)}/resume-options",
        "queueMode ? '/api/training/queue/resume' : '/api/training/resume'",
        "configFullResumeDurationOverrides()",
        "configFullResumeCanAppendCompletedCheckpoint",
        "configFullResumeCheckpointUsable",
        "duration_overrides: durationOverrides",
        "max_train_epochs",
        "max_train_steps",
        "configTrainingSourceMode() !== 'weight_hotstart'",
        "continue_from_weight_abs_path",
    ):
        assert snippet in config_source

    assert "configTrainingSourceMode() === 'full_resume'" in launch_source
    assert "startConfigFullResumeSource(false)" in launch_source
    assert "startConfigFullResumeSource(true)" in launch_source
    assert "ensureTrainingSourceReadyForLaunch()" in launch_source
    assert "trainingSourceLaunchBlockReason()" in launch_source
    assert "sourceReady.checking" in action_state
    assert "sourceReady.ready" in action_state
    assert "开始完整续训" in action_state
    assert "开始热启动训练" in action_state
    assert "getTrainingSourceMode?.() === 'full_resume'" in queue_enqueue
    assert "ensureTrainingSourceReadyForLaunch" in queue_enqueue


def test_weight_hotstart_audit_keeps_pending_state_until_refresh_finishes() -> None:
    source = _frontend_module_text("js/features/training-source/index.js")
    helper = _frontend_module_text("js/features/anima-app/helpers/training-source-ui.js")

    assert "syncWeightHotstartFieldsFromContinue();" in source
    assert "syncWeightHotstartAuditFromContinue();" not in source
    assert "weight.audit_status === 'checking' && ok == null && !reason" in source
    assert "state.audit_status = 'checking';" in source

    for snippet in (
        "export function textNode(",
        "export function optionNodeLocal(",
        "export function summaryLine(",
    ):
        assert snippet in helper


def test_optional_number_fields_can_be_cleared() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    form_values_source = _frontend_module_text("js/features/anima-app/chunks/18-delete-dataset-preset-group.js")
    optional_numbers = _section(source, "const OPTIONAL_EMPTY_NUMBER_FIELDS = new Set([", "const FORM_UI_PERSIST_DEFAULT_FIELDS")
    reader = _section(form_values_source, "function readFieldInputValue", "function readLoKrEnabled")

    for key in ("sample_every_n_epochs", "sample_every_n_steps", "max_train_epochs"):
        assert f"'{key}'" in optional_numbers
    assert "String(raw).trim() === '' && OPTIONAL_EMPTY_NUMBER_FIELDS.has(input.dataset.key)" in reader
    assert "return parseNumberValue(raw, originalValue);" in reader


def test_step_estimate_panel_shows_epoch_factor() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    create_body = _section(source, "function createStepEstimatePanel", "function updateStepEstimatePanel")
    update_body = _section(source, "function updateStepEstimatePanel", "function liveDatasetRowsForEstimate")
    group_body = _section(source, "if (extraClass === 'config-group-steps')", "section.appendChild(content);")

    assert "最大训练轮数" in create_body
    assert "step-max-train-epochs" in create_body
    assert "scheduleStepEstimatePanelRefresh();" in group_body
    assert "function scheduleStepEstimatePanelRefresh" in source
    assert "requestAnimationFrame(updateStepEstimatePanel)" in source
    assert "setText('step-max-train-epochs'" in update_body
    assert "${totalSteps} = ${stepsPerEpoch} x ${epochs}" in update_body
    assert "每轮步数 x max_train_epochs" in update_body


def test_output_scope_group_does_not_expose_unwired_stage_resolution_dialog() -> None:
    """输出范围分组不挂标题级主按钮；分阶段 dialog 由数据集顶栏 / 只读摘要打开。"""
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    group_entry = _frontend_feature_text(
        "js/features/anima-app/chunks/04-create-config-group-entry.js",
        "js/features/config-form/group-entry.js",
    )

    section = _section(source, "title: '输出格式与训练范围'", "title: '方法内部与实验架构'")

    assert "className: 'config-group-output-scope'" in section
    assert "header.appendChild(createOpenStageResolutionDialogButton());" not in group_entry
    assert "createOpenStageResolutionDialogButton" not in group_entry
    assert 'id="stage-resolution-dialog"' in html
    assert "btn-dataset-open-stage-schedule" in _frontend_module_text("js/features/dataset-editor/toolbar.js")


def test_config_form_hides_retired_and_unread_fields() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    resource_section = _section(source, "title: '显存与速度优化'", "title: '缓存与预处理'")
    method_section = _section(source, "title: '方法内部与实验架构'", "title: 'Soft Tokens 参数'")
    retired_fields = [
        "per_channel_scaling",
        "repa_layer",
        "repa_lr_scale",
        "repa_weight",
        "trim_crossattn_kv",
        "use_fei_router",
        "use_hydra",
        "use_repa",
        "use_sigma_router",
    ]
    for key in retired_fields:
        assert f"'{key}'," in source
    assert "'use_hydra'," not in method_section
    assert "'use_sigma_router'," not in method_section
    assert "'use_fei_router'," not in method_section
    assert "'use_repa'," not in method_section
    assert "'per_channel_scaling'," not in method_section
    assert "'trim_crossattn_kv'," not in resource_section
    assert "LoRA + REPA" not in source
    assert "const RETIRED_CONFIG_FORM_FIELDS = new Set([" in source
    assert "if (RETIRED_CONFIG_FORM_FIELDS.has(key)) return true;" in source
    assert "['weight_decay', new Set(['spd'])]" in source
    assert "const SOFT_TOKENS_UI_DEFAULT_FIELDS = new Set([]);" in source


def test_config_form_auto_fixes_came_optimizer_args_frontend_hooks_are_present() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    optimizer_helper = _frontend_module_text("js/features/anima-app/helpers/optimizer-values.js")
    form_fields = _frontend_feature_text(
        "js/features/anima-app/chunks/14-lora-adapter-kind-from-config.js",
        "js/features/config-form/form-fields.js", "js/features/config-form/form-fields-adapters.js", "js/features/config-form/form-fields-sample.js", "js/features/config-form/form-fields-ui.js",
    )
    config_form = _frontend_module_text("js/features/config-form/index.js")
    startup = _frontend_module_text("js/features/app-shell/startup.js")
    compatibility_section = _section(form_fields, "function applyOptimizerCompatibilityPatch", "function createFieldRow")
    load_config_section = _section(startup + "\n" + config_form, "async function loadConfig()", "function syncConfigDraftFromForm")
    prepare_section = _section(source, "async function prepareFormPatchValues", "function shouldSkipUiDefaultField")

    assert "export function normalizeOptimizerType(value)" in optimizer_helper
    assert "export function normalizeCameOptimizerArgs(args)" in optimizer_helper
    assert "function applyOptimizerCompatibilityPatch(values)" in compatibility_section
    assert "normalizeOptimizerType(optimizerType) !== 'came'" in compatibility_section
    assert "cameBetasNeedPatch(rawBetas)" in optimizer_helper
    assert "result[betasIndex] = 'betas=0.9,0.999,0.9999';" in optimizer_helper
    assert "const compatibilityPatch = applyConfigCompatibilityDrafts();" in load_config_section
    assert "function applyConfigCompatibilityDrafts()" in load_config_section
    assert "configFormState.draftValues.set(key, value);" in load_config_section
    assert "已自动修正 CAME optimizer_args 的 betas 格式" in load_config_section
    assert "const nextValues = applyOptimizerCompatibilityPatch(values);" in prepare_section


def test_variant_guides_match_gui_methods_or_legacy_aliases() -> None:
    """VARIANT_GUIDE_ZH 必须覆盖现网 gui-methods，ghost key 只能留在 legacy 别名里。"""
    from pathlib import Path
    import re

    root = Path(__file__).resolve().parents[1]
    gui = {p.stem for p in (root / "configs" / "gui-methods").glob("*.toml")}
    guides = (root / "web" / "static" / "js" / "config" / "catalog" / "guides.js").read_text(encoding="utf-8")

    assert "export const LEGACY_VARIANT_ALIASES" in guides

    variant_block = re.search(
        r"export const VARIANT_GUIDE_ZH = \{(.*?)^\};",
        guides,
        re.S | re.M,
    )
    assert variant_block is not None, "VARIANT_GUIDE_ZH block not found"
    variant_keys = {
        m.group(1) or m.group(2)
        for m in re.finditer(
            r"^\s{4}(?:'([^']+)'|([A-Za-z0-9_]+)):\s*choiceHelp\(",
            variant_block.group(1),
            re.M,
        )
    }
    assert variant_keys, "VARIANT_GUIDE_ZH keys empty"

    legacy_block = re.search(
        r"export const LEGACY_VARIANT_ALIASES = Object\.freeze\(\{(.*?)^\}\);",
        guides,
        re.S | re.M,
    )
    assert legacy_block is not None, "LEGACY_VARIANT_ALIASES block not found"
    legacy_keys = {
        m.group(1) or m.group(2)
        for m in re.finditer(
            r"^\s{4}(?:'([^']+)'|([A-Za-z0-9_]+)):\s*\{",
            legacy_block.group(1),
            re.M,
        )
    }
    assert legacy_keys, "LEGACY_VARIANT_ALIASES keys empty"

    unknown = sorted(variant_keys - gui - legacy_keys)
    assert unknown == [], f"variant guide keys not in gui-methods or LEGACY_VARIANT_ALIASES: {unknown}"

    missing_live = sorted(gui - variant_keys)
    assert missing_live == [], f"live gui-methods missing VARIANT_GUIDE_ZH entries: {missing_live}"

    # Ghost keys must stay resolvable in VARIANT_GUIDE_ZH so old imports do not hard-crash.
    for key in sorted(legacy_keys):
        assert key in variant_keys, f"legacy alias {key} missing VARIANT_GUIDE_ZH entry"
        assert re.search(rf"(?:'{re.escape(key)}'|{re.escape(key)}):\s*choiceHelp\(", variant_block.group(1)), key


def test_resource_naming_three_layers_are_distinguished() -> None:
    """硬件 preset / 方法变体 / 资源快捷按钮三层文案必须可区分。"""
    app_constants = _frontend_module_text("js/features/anima-app/helpers/app-constants.js")
    guides = _frontend_module_text("js/config/catalog/guides.js")

    quick_presets = _section(
        app_constants,
        "export const RESOURCE_QUICK_PRESETS = [",
        "export const NO_DATASET_REGULARIZATION_FIELD_KEYS",
    )
    for preset_id in (
        "gpu_full",
        "balanced_16g",
        "fp8_swap_test",
        "vram_saver",
        "lokr_16g_rescue",
        "oom_fallback",
    ):
        assert f"id: '{preset_id}'" in quick_presets

    note_lines = [line for line in quick_presets.splitlines() if "note:" in line]
    assert len(note_lines) >= 6
    for line in note_lines:
        assert (
            "快捷" in line
            or "一键资源" in line
            or "快捷资源" in line
        ), f"resource quick preset note missing layer word: {line}"

    lokr_quick = _section(quick_presets, "id: 'lokr_16g_rescue'", "id: 'oom_fallback'")
    assert (
        "仅 LoKr" in lokr_quick
        or "方法变体专用" in lokr_quick
        or "LoKr 专用" in lokr_quick
    )
    assert "快捷" in lokr_quick or "快捷资源" in lokr_quick or "一键资源" in lokr_quick

    preset_start = guides.index("export const PRESET_GUIDE_ZH = {")
    preset_guide = guides[preset_start:]
    low_vram_guide = _section(preset_guide, "low_vram: choiceHelp(", "low_vram_blockswap: choiceHelp(")
    balanced_guide = _section(preset_guide, "balanced_16g: choiceHelp(", "graft: choiceHelp(")
    assert "硬件预设" in low_vram_guide
    assert "硬件预设" in balanced_guide
    # balanced_16g 是硬件 preset，不能混入 LoKr 方法专用描述。
    assert "LoKr 专用" not in balanced_guide
    assert "lokr_factor_group_size" not in balanced_guide
    assert "仅 LoKr" not in balanced_guide

    variant_start = guides.index("export const VARIANT_GUIDE_ZH = {")
    variant_guide = guides[variant_start:preset_start]
    lora8_guide = _section(variant_guide, "'lora-8gb': choiceHelp(", "ortholora: choiceHelp(")
    assert "方法变体" in lora8_guide
    assert "硬件预设" not in lora8_guide
    assert "快捷资源" not in lora8_guide


def test_resource_quick_preset_diff_preview_and_method_guards_exist() -> None:
    """快捷资源按钮需提供 diff 预览与方法门禁接口。"""
    app_constants = _frontend_module_text("js/features/anima-app/helpers/app-constants.js")
    presets_source = _frontend_module_text("js/features/config-form/stage-resolution-presets.js")
    source = APP_JS.read_text(encoding="utf-8")

    assert "export function previewQuickPresetDiff" in presets_source
    assert "export function isQuickPresetApplicable" in presets_source
    assert "export function resolveQuickPresetMethodFamily" in presets_source
    assert "previewQuickPresetDiff(" in source
    assert "isQuickPresetApplicable(" in source

    quick_presets = _section(
        app_constants,
        "export const RESOURCE_QUICK_PRESETS = [",
        "export const NO_DATASET_REGULARIZATION_FIELD_KEYS",
    )
    lokr_quick = _section(quick_presets, "id: 'lokr_16g_rescue'", "id: 'oom_fallback'")
    assert "applicableMethods:" in lokr_quick
    assert "'lokr'" in lokr_quick
    assert "use_lokr" in lokr_quick

    fp8_quick = _section(quick_presets, "id: 'fp8_swap_test'", "id: 'vram_saver'")
    oom_quick = _section(quick_presets, "id: 'oom_fallback'", "];")
    assert "实验" in fp8_quick
    assert "兜底" in oom_quick or "OOM" in oom_quick

    apply_block = _section(
        presets_source,
        "function applyResourceQuickPreset",
        "function resourceQuickPresetPatch",
    )
    assert "previewQuickPresetDiff" in apply_block
    assert "isQuickPresetApplicable" in apply_block
    assert "setTomlStatus(" in apply_block
    assert "将修改" in apply_block or "将改动" in apply_block

    panel_block = _section(
        presets_source,
        "function createConfigQuickPresetPanel",
        "export function createResourceQuickPresetsButton",
    )
    assert "isQuickPresetApplicable" in panel_block or "applicableMethods" in panel_block
    assert "disabled" in panel_block or "aria-disabled" in panel_block


def test_balanced_16g_block_swap_fields_are_visible() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    form_layout = _frontend_module_text("js/config/catalog/form-layout.js")
    labels_options = _frontend_module_text("js/config/catalog/labels-options.js")
    guides = _frontend_module_text("js/config/catalog/guides.js")

    optimization_section = _section(form_layout, "title: '显存与速度优化'", "title: '缓存与预处理'")

    for key in (
        "blocks_to_swap",
        "compile_block_scope",
        "selective_checkpoint",
        "selective_checkpoint_blocks",
        "block_swap_profile_jsonl",
        "memory_probe_jsonl",
        "memory_probe_max_steps",
        "peak_probe_jsonl",
        "peak_probe_max_steps",
        "peak_probe_level",
        "preprocess_memory_profile",
        "preprocess_vae_cache_batch_size",
        "preprocess_text_cache_batch_size",
        "lr_warmup_steps",
        "lokr_factor_group_size",
        "lokr_project_chunk_bytes",
        "lokr_use_einsum",
        "lokr_decompose_w2",
        "disable_block_swap_for_eval",
    ):
        assert f"'{key}'," in optimization_section

    assert "block_swap_profile_jsonl: '块交换 Profile'" in labels_options
    assert "compile_block_scope: '编译块范围'" in labels_options
    assert "block_swap_restore_mode: '块交换恢复路径'" in labels_options
    assert "memory_probe_jsonl: '显存探针'" in labels_options
    assert "memory_probe_max_steps: '探针步数'" in labels_options
    assert "peak_probe_jsonl: '峰值探针'" in labels_options
    assert "peak_probe_max_steps: '峰值探针步数'" in labels_options
    assert "peak_probe_level: '峰值探针粒度'" in labels_options
    assert "preprocess_memory_profile: '预处理显存模式'" in labels_options
    assert "preprocess_vae_cache_batch_size: 'VAE 预处理批大小'" in labels_options
    assert "preprocess_text_cache_batch_size: '文本缓存批大小'" in labels_options
    assert "lr_warmup_steps: '预热步数'" in labels_options
    assert "lokr_factor_group_size: 'LoKr 分组'" in labels_options
    assert "lokr_project_chunk_bytes: 'LoKr 张量切块阈值'" in labels_options
    assert "lokr_use_einsum: 'LoKr 结构化 einsum'" in labels_options
    assert "lokr_decompose_w2: 'LoKr 轻量分解 W2'" in labels_options
    assert "use_glora: '启用 GLoRA'" in labels_options
    assert "use_vera: '启用 VeRA'" in labels_options
    assert "vera_projection_prng_key: 'VeRA 投影随机种子'" in labels_options
    assert "vera_d_initial: 'VeRA d 初始值'" in labels_options
    assert "vera_save_projection: '保存 VeRA 投影矩阵'" in labels_options
    assert "selective_checkpoint: '选择性重算'" in labels_options
    assert "selective_checkpoint_blocks: '定点重算块'" in labels_options
    assert "disable_block_swap_for_eval: '评估时暂停交换块'" in labels_options
    assert "block_swap_restore_mode: ['foreach', 'slab']" in labels_options
    assert "compile_block_scope: ['resident', 'all']" in labels_options
    assert "FIELD_STRICT_SELECT_OPTIONS = new Set([" in labels_options
    assert "'block_swap_profile_jsonl'," in labels_options
    assert "selective_checkpoint: ['off', 'adapter_aware', 'peak_blocks_adapter_aware', 'mlp_layer1_only', 'peak_blocks_mlp_layer1', 'peak_blocks_mlp', 'mlp_only', 'every_other']" in labels_options
    assert "memory_probe_jsonl: ['off', 'auto']" in labels_options
    assert "peak_probe_jsonl: ['off', 'auto']" in labels_options
    assert "peak_probe_level: ['block', 'ops', 'lokr', 'full']" in labels_options
    assert "preprocess_memory_profile: ['auto', 'low_vram', 'balanced', 'speed']" in labels_options
    assert "preprocess_vae_cache_batch_size: ['auto', 1, 2, 4, 8]" in labels_options
    assert "preprocess_text_cache_batch_size: ['auto', 1, 2, 4, 8, 16]" in labels_options
    assert "lokr_factor_group_size: [1, 2, 4, 8]" in labels_options
    assert "lokr_project_chunk_bytes: [1048576, 2097152, 4194304, 8388608, 16777216]" in labels_options
    assert "lora_adapter_kind: ['lora', 'loha', 'lokr', 'glora', 'vera']" in labels_options
    assert "dora_wd: [false, true]" not in labels_options
    assert "use_glora: [false, true]" in labels_options
    assert "use_vera: [false, true]" in labels_options
    assert "vera_projection_prng_key: [0, 1, 2, 3]" in labels_options
    assert "vera_d_initial: [0.01, 0.05, 0.1, 0.2]" in labels_options
    assert "vera_save_projection: [false, true]" in labels_options
    assert "network_dim: [" not in labels_options
    assert "network_alpha: [" not in labels_options
    field_ui_source = _frontend_module_text("js/features/sample-prompts/row-ui.js")
    numeric_field_section = _section(field_ui_source, "function isNumericField", "function isIntegerNumericField")
    integer_field_section = _section(field_ui_source, "function isIntegerNumericField", "function allowsNegativeNumberField")
    negative_field_section = _section(field_ui_source, "function allowsNegativeNumberField", "function createSelectInput")
    assert "'network_dim'," in numeric_field_section
    assert "'sample_every_n_steps'," in numeric_field_section
    assert "'blocks_to_swap'," in numeric_field_section
    assert "'save_last_n_epochs'," in numeric_field_section
    assert "'network_alpha'," in numeric_field_section
    assert "'network_dim'," in integer_field_section
    assert "'sample_every_n_steps'," in integer_field_section
    assert "'blocks_to_swap'," in integer_field_section
    assert "'save_last_n_epochs'," in integer_field_section
    assert "'save_last_n_epochs'" in negative_field_section
    assert "'network_alpha'," not in integer_field_section
    assert "'max-autotune-no-cudagraphs'" in labels_options
    assert "balanced_16g" in guides
    assert "预测式 DiT block swap" in guides


def test_block_swap_profile_uses_strict_select_options() -> None:
    defaults_source = _frontend_module_text("js/config/catalog/defaults.js")
    input_source = _frontend_feature_text("js/features/anima-app/chunks/14-lora-adapter-kind-from-config.js", "js/features/config-form/form-fields.js", "js/features/config-form/form-fields-adapters.js", "js/features/config-form/form-fields-sample.js", "js/features/config-form/form-fields-ui.js")
    display_helper = _frontend_module_text("js/features/anima-app/helpers/config-field-display.js")
    option_source = _chunk15_compat_text()
    labels_options = _frontend_module_text("js/config/catalog/labels-options.js")
    select_gate = display_helper
    input_factory = _frontend_module_text("js/features/config-form/form-fields-ui.js")

    assert "block_swap_profile_jsonl: 'off'" in defaults_source
    assert "block_swap_profile_jsonl: ['off', 'auto']" in labels_options
    assert "Boolean(FIELD_OPTIONS[key])" in select_gate
    assert "!Array.isArray(value)" in select_gate
    assert "if (shouldRenderSelectInput(key, value))" in input_factory
    assert "return createSelectInput(key, value, fieldOptions);" in input_factory
    assert "function selectUsesStrictOptions(key)" in option_source
    assert "FIELD_STRICT_SELECT_OPTIONS?.has?.(key)" in option_source
    assert "select.dataset.strictOptions = '1';" in option_source
    assert "opt.disabled = true;" in option_source
    assert "旧配置里的自定义值" in option_source
    assert "自定义路径（旧值）" in option_source
    assert "自动写入任务目录 / auto" in option_source


def test_config_form_options_cover_backend_choices() -> None:
    field_options = (STATIC_DIR / "js" / "config" / "catalog" / "labels-options.js").read_text(encoding="utf-8")

    for option in ["torch", "xformers", "flash", "sageattn", "flex", "sdpa"]:
        assert f"'{option}'" in field_options
    assert "'max-autotune-no-cudagraphs'" in field_options
    for option in ["tensorboard", "wandb", "all"]:
        assert f"'{option}'" in field_options
    for option in ["euler", "er_sde", "lcm"]:
        assert f"'{option}'" in field_options
    for legacy_sampler in ["ddim", "dpmsolver++", "k_dpm_2"]:
        assert f"'{legacy_sampler}'" not in field_options
    for option in ["ckpt", "pt", "safetensors"]:
        assert f"'{option}'" in field_options
    for option in ["sigma", "uniform", "sigmoid", "shift", "flux_shift"]:
        assert f"'{option}'" in field_options
    assert "'hard_backoff'" in field_options


def test_config_page_hides_unimplemented_dataset_placeholder() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")

    dataset_picker = _section(source, "function renderConfigDatasetPicker", "function createConfigDatasetCurrentSummary")
    assert "btn-open-unnamed-dataset-dialog" not in dataset_picker
    assert "待命名" not in dataset_picker
    assert 'id="unnamed-dataset-dialog"' not in html


def test_dataset_preset_page_has_group_manager_controls() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    list_body = _section(source, "function renderDatasetPresetList", "function renderDatasetPresetHeader")
    listener_section = _section(source, "function setupEventListeners", "function installBeginnerTooltips")

    assert "dataset-preset-manager" in html
    assert "dataset-preset-search" in html
    assert "btn-create-dataset-preset-group" in html
    assert "btn-refresh-dataset-presets" in html
    assert "dataset-page-summary" in html

    assert "createDatasetPresetGroupNode(group, stored)" in list_body
    assert "createDatasetPresetGroupDragHandle" in list_body
    assert "createDatasetPresetGroupActions" in list_body
    assert "createDatasetPresetGroupFileRow" in list_body
    assert "setupFileGroupRowDropTarget" in list_body
    assert "setupFileGroupListDropTarget" in list_body
    assert "setupFileGroupHeaderDropTarget(summary, group, datasetPresetDragOptions())" in list_body
    assert "setupConfigGroupDropTarget" in list_body
    assert "function setupFileGroupHeaderDropTarget" in source
    assert "function setFileGroupDragData" in source
    assert "function createFileGroupDragImage" in source
    assert "document.createElement('button')" in source
    assert "handle.textContent = '⋮⋮'" in source
    assert "transfer.setDragImage(image, 12, 12)" in source
    assert "row.addEventListener('dragenter', updateDropTarget)" in source
    assert "list.addEventListener('dragenter', updateDropTarget)" in source
    assert "node.addEventListener('dragenter', updateDropTarget)" in source
    assert "fileGroupContainsRelatedTarget" in source
    assert "if (!presets.length && !groups.length)" in list_body
    assert "dataset-preset-empty-state" in source
    assert "placeDatasetPresetFile" in list_body
    assert "placeDatasetPresetGroup" in list_body
    assert "/api/config/file-groups/place" in source
    assert "createDatasetPresetGroupOrderActions" not in source
    assert "reorderDatasetPresetInGroup" not in source
    assert "reorderDatasetPresetGroup" not in source
    assert "moveDatasetPresetToGroup" not in source
    assert "createDatasetPresetRowActionButton" not in source
    assert "dataset-preset-row-actions" not in source
    assert "DATASET_PRESET_GROUP_STATE_KEY" in source
    assert "anima_lora_dataset_preset_groups_v2" in source
    assert "kind: 'dataset'" in source
    assert "function isUnfiledDatasetGroup" in source
    assert "return sortDatasetPresetGroups(groups);" in source
    assert "orderDatasetPresetsForGroups(presets, sortedGroups)" in source
    assert "const defaultOpen = isUnfiledDatasetGroup(group);" in source
    assert "stored[group.id] ?? defaultOpen" in source
    assert "!isUnfiledDatasetGroup(group)" in source
    assert "JSON.stringify({ label: label.trim(), kind: 'dataset' })" in source
    assert "btn-create-dataset-preset-group" in listener_section
    assert "dataset-preset-search" in listener_section
    assert "btn-refresh-dataset-presets" in listener_section

    assert ".dataset-preset-manager" in css
    assert ".dataset-preset-group" in css
    assert ".dataset-preset-row-actions" not in css
    assert ".dataset-preset-row-action-btn" not in css
    assert ".file-group-drag-handle" in css
    assert ".file-group-drag-image" in css
    assert "-webkit-user-drag: element;" in css
    assert ".file-group-drop-before" in css
    assert ".dataset-preset-group.empty" in css
    assert ".dataset-preset-empty-state" in css
    assert ".dataset-page-summary" in css


def test_file_group_drag_has_pointer_fallback() -> None:
    source = _frontend_feature_text(
        "js/features/anima-app/chunks/08-origin-closest.js",
        "js/features/toml-manager/file-group-drag.js",
        "js/features/toml-manager/file-group-drag-core.js",
        "js/features/toml-manager/file-group-drag-targets.js",
        "js/features/toml-manager/config-group-drop.js", "js/features/toml-manager/config-group-drop-target.js", "js/features/dataset-editor/dataset-render.js",
        "js/features/anima-app/helpers/app-constants.js",
    )
    css = STYLE_CSS.read_text(encoding="utf-8")

    drag_helpers = _section(source, "function setFileGroupDragData", "function createFileGroupDragHandle")
    handle_body = _section(source, "function createFileGroupDragHandle", "function finishFileGroupDrag")
    drop_targets = _section(source, "function setupFileGroupRowDropTarget", "function createDatasetPresetGroupNode")

    assert "application/x-anima-file-group" in drag_helpers
    assert "function registerFileGroupDropTarget" in drag_helpers
    assert "function resolveFileGroupPointerDropTarget" in drag_helpers
    assert "function resolveNearestFileGroupDropTarget" in drag_helpers
    assert "function startFileGroupPointerDrag" in drag_helpers
    assert "function startFileGroupMouseDrag" in drag_helpers
    assert "function finishFileGroupPointerDrag" in drag_helpers
    assert "function ensureFileGroupDropPreview" in drag_helpers
    assert "function placeFileGroupDropPreview" in drag_helpers
    assert "function removeFileGroupDropPreview" in drag_helpers
    assert "function clearFileGroupDropTarget" in drag_helpers
    assert "释放后插入到这里" in drag_helpers
    assert "clearFileGroupDropIndicators({ keepPreview: true })" in drag_helpers
    assert "placeFileGroupDropPreview(node, position)" in drag_helpers
    assert "handle.addEventListener('pointerdown'" in handle_body
    assert "handle.addEventListener('mousedown'" in handle_body
    assert "if (currentFileGroupPointerDrag()) {" in handle_body
    assert "event.preventDefault();" in _section(handle_body, "handle.addEventListener('dragstart'", "if (!canBeginFileGroupDrag")
    assert "document.addEventListener('pointermove', drag.onMove, { passive: false })" in drag_helpers
    assert "document.addEventListener('pointerup', drag.onUp, { passive: false })" in drag_helpers
    assert "document.addEventListener('pointercancel', drag.onCancel, { passive: false })" in drag_helpers
    assert "const addMouseFallbackListeners = () =>" in drag_helpers
    assert "document.addEventListener('mousemove', drag.onMouseMove, { passive: false })" in drag_helpers
    assert "document.addEventListener('mouseup', drag.onMouseUp, { passive: false })" in drag_helpers
    assert "addMouseFallbackListeners();" in drag_helpers
    assert "document.addEventListener('keydown', drag.onKeydown)" in drag_helpers
    assert "fileGroupPointerDrag" in source
    assert "fileGroupActiveDropTargetNode" in source
    assert "fileGroupActiveDropPosition" in source
    assert "fileGroupDropTargetNodes" in source
    assert "data-file-group-drop-target" in source
    assert "autoScrollFileGroupPointerDrag" in drag_helpers
    assert "currentFileGroupActiveDropTargetNode() === node" in source
    assert "currentFileGroupActiveDropPosition() === normalizedPosition" in source

    assert len(re.findall(r"\bregisterFileGroupDropTarget\(", drop_targets)) == 4
    assert "position: 'inside'" in drop_targets
    assert "configFileDropIndex(group, targetFile, placeAfter, payload.file)" in drop_targets
    assert "configGroupDropIndex(options.getSortableGroups(), group.id, placeAfter, payload.groupId)" in drop_targets
    assert "clearFileGroupDropTarget(row);" in drop_targets
    assert "clearFileGroupDropTarget(list);" in drop_targets
    assert "clearFileGroupDropTarget(node);" in drop_targets

    assert ".file-group-pointer-drag-active" in css
    assert ".file-group-drag-image-pointer" in css
    assert ".file-group-drop-preview" in css
    assert "position: fixed;" in css


def test_dataset_preset_manager_is_isolated_from_config_page() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    tab_active = _section(source, "function isDatasetTabActive", "function setActiveDatasetRows")
    load_presets = _section(
        source,
        "async function loadDatasetPresets(options = {})",
        "async function loadDatasetPreset(file)",
    )
    import_presets = _section(
        source,
        "async function handleDatasetPresetImport(event)",
        "async function exportDatasetPreset()",
    )
    load_editor = _section(source, "async function loadDatasetEditor", "function renderDatasetEditor")
    api_helpers = _frontend_module_text("js/features/anima-app/runtime/api.js")
    listener_section = _section(source, "function setupEventListeners", "function installBeginnerTooltips")

    assert "classList.contains('active')" in tab_active
    assert "closest('#tab-datasets')" not in tab_active
    assert "DATASET_PRESET_REQUEST_TIMEOUT_MS" in source
    assert "api.datasetPresetApi = async function datasetPresetApi" in api_helpers
    assert "数据集预设请求超时" in api_helpers
    assert "ctx.api(url, opts)" in api_helpers
    assert "const managePresets = options.manage === true || (options.manage !== false && isDatasetTabActive());" in load_presets
    assert "if (!managePresets)" in load_presets
    assert "datasetPresetApi('/api/config/dataset-presets')" in load_presets
    assert "await loadDatasetPreset(datasetPresetState.selectedFile)" in load_presets
    assert "/api/config/dataset-presets/import" in import_presets
    assert "/api/config/raw/save-as" not in import_presets
    assert "datasetPresetState.loading = false;" in import_presets
    assert "loadDatasetPresets({ selectCurrent: false, manage: false })" in source
    assert "loadDatasetPresets({ manage: true })" in source
    assert "btn-refresh-dataset-presets" in listener_section
    assert "btn-config-dataset-dialog-refresh" in listener_section
    assert "on('btn-config-dataset-dialog-refresh', 'click', () => loadDatasetPresets({ selectCurrent: false, manage: false }))" in listener_section
    assert "const params = new URLSearchParams({" in load_editor
    assert "const configFile = currentTrainingConfigFile();" in load_editor
    assert "params.set('config_file', configFile);" in load_editor
    assert "function selectedDatasetConfigOverride" in source
    assert "const datasetConfigOverride = selectedDatasetConfigOverride();" in load_editor
    assert "if (datasetConfigOverride !== null) params.set('dataset_config', datasetConfigOverride);" in load_editor
    assert "params.set('dataset_config', selectedConfigDatasetFile);" not in load_editor
    assert "datasetConfig = selectedConfigDatasetFile || currentConfig.dataset_config" not in load_editor
    assert "api(`/api/config/datasets?${params.toString()}`)" in load_editor


def test_config_dataset_editor_save_syncs_picker_selection_and_summary() -> None:
    source = _frontend_module_text("js/features/anima-app/chunks/18-delete-dataset-preset-group.js")
    save_editor = _section(source, "export async function saveDatasetEditor(options = {})", "export function collectChangedFormValues")

    assert "const nextDatasetConfig = nextDatasetEditorState.dataset_config || '';" in save_editor
    assert "currentConfig.dataset_config = nextDatasetConfig;" in save_editor
    assert "datasetState.selectedConfigDatasetFile = nextDatasetConfig;" in save_editor
    assert "datasetState.selectedConfigDatasetSummary = nextDatasetConfig ? (res.summary || null) : null;" in save_editor
    assert "datasetState.configDatasetPreviewState = {" in save_editor
    assert "renderConfigDatasetPicker();" in save_editor
    assert "await loadDatasetPresets({ selectCurrent: false, manage: false });" in save_editor


def test_config_toml_manager_excludes_dataset_groups() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    toml_save_source = _frontend_module_text("js/features/anima-app/chunks/19-current-sample-prompt-text.js")
    toml_drag_source = _frontend_feature_text("js/features/toml-manager/drag-core.js", "js/features/toml-manager/drag-actions.js", "js/features/toml-manager/drag-render.js", "js/features/toml-manager/drag.js")
    toml_actions_source = _frontend_module_text("js/features/toml-manager/actions.js")
    action_state_source = _frontend_module_text("js/features/anima-app/chunks/22-update-toml-action-state.js")
    css = STYLE_CSS.read_text(encoding="utf-8")

    load_toml = _section(_frontend_module_text("js/features/toml-manager/mode.js"), "async function loadTomlFileList", "async function loadOutputRuns")
    toml_render = _section(toml_drag_source, "export function renderTomlFileGroups", "export function createTomlGroupActions")
    file_button = _section(toml_drag_source, "export function createTomlFileButton", "configureTomlDragBridge")
    save_as_groups = _section(toml_save_source, "export function saveAsTargetGroups", "export async function moveTomlFileToGroup")
    helper_section = _frontend_feature_text(
        "js/features/anima-app/chunks/19-current-sample-prompt-text.js",
        "js/features/toml-manager/drag-core.js", "js/features/toml-manager/drag-actions.js", "js/features/toml-manager/drag-render.js", "js/features/toml-manager/drag.js",
    )
    create_group = _section(action_state_source, "async function createTomlGroup", "async function renameTomlGroup")
    movable_groups = _section(toml_actions_source, "function getMovableTomlGroups", "function deleteTomlGroupButtonTitle")

    assert "/api/config/file-groups?kind=training" in load_toml
    assert "tomlFileGroups = filterTrainingTomlGroups(groups);" in load_toml
    assert "deferDefaultLoad" in load_toml
    assert "function loadDefaultTomlFile" in load_toml
    assert "function isTrainingTomlGroup" in helper_section
    assert "function filterTrainingTomlGroups" in helper_section
    assert "configs/datasets/" in helper_section
    assert "return isTrainingTomlGroup(group) && !isFixedSystemTomlGroup(group);" in helper_section
    assert "isTrainingTomlGroup(group) && (group.user_managed || group.lockable" in helper_section
    assert "isTrainingTomlGroup(group) && !isFixedSystemTomlGroup(group)" in helper_section
    assert "function isTomlGroupDraggable" in helper_section
    assert "function canDropTomlFileToGroup" in helper_section
    assert "function tomlFileDragOptions" in helper_section
    assert "function tomlGroupDragOptions" in helper_section
    assert "createTomlGroupDragHandle(group, details)" in toml_render
    assert "setupFileGroupListDropTarget(list, group, tomlFileDragOptions())" in toml_render
    assert "setupFileGroupHeaderDropTarget(summary, group, tomlFileDragOptions())" in toml_render
    assert "setupConfigGroupDropTarget(details, group, tomlGroupDragOptions())" in toml_render
    assert "function renderTomlFileGroupList" in toml_render
    assert "createFileGroupDragHandle" in file_button
    assert "placeTomlFile" in source
    assert "placeTomlGroup" in source
    assert "createTomlGroupOrderActions" not in source
    assert "createTomlFileOrderButton" not in source
    assert "toml-file-order-btn" not in css
    assert "const trainingGroups = filterTrainingTomlGroups(tomlState.tomlFileGroups);" in save_as_groups
    assert "const imported = trainingGroups.find((group) => group.id === 'imported');" in save_as_groups
    assert "JSON.stringify({ label: label.trim(), kind: 'training' })" in create_group
    assert "isTrainingTomlGroup(group) && group.movable" in movable_groups


def test_sample_prompts_editor_preserves_raw_text_when_needed() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    render_body = _section(source, "function renderSamplePromptRows", "function switchSamplePromptsEditorToTextMode")
    mode_button_body = _section(source, "function createSamplePromptTextModeButton", "function updateSamplePromptModeButtonState")
    table_mode_body = _section(source, "function switchSamplePromptsEditorToTableMode", "function appendSamplePromptRow")
    serialize_body = _section(source, "function serializeSamplePromptsEditor", "function samplePromptRowFromElement")
    css = STYLE_CSS.read_text(encoding="utf-8")

    assert "samplePromptsContentNeedsTextMode(content)" in render_body
    assert "sample-prompts-textarea" in render_body
    assert "return editor.querySelector('.sample-prompts-textarea')?.value || '';" in serialize_body
    assert "function createSamplePromptTextModeButton" in source
    assert "function switchSamplePromptsEditorToTableMode" in source
    assert "function updateSamplePromptModeButtonState" in source
    assert "btn.dataset.samplePromptsModeToggle = '1';" in mode_button_body
    assert "switchSamplePromptsEditorToTableMode(editor);" in mode_button_body
    assert "switchSamplePromptsEditorToTextMode(editor);" in mode_button_body
    assert "btn.textContent = textMode ? '表格模式' : '文本模式';" in source
    assert "btn.setAttribute('aria-pressed', String(textMode));" in source
    assert "const text = serializeSamplePromptsEditor(editor);" in table_mode_body
    assert "for (const row of parseSamplePromptRows(text))" in table_mode_body
    assert ".sample-prompts-mode-btn[aria-pressed=\"true\"]" in css


def test_dataset_json_caption_switch_ui_is_wired() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    defaults_editor = _section(source, "function createDatasetDefaultsEditor", "function createDatasetConfigInput")
    item_factory = _section(source, "function createDatasetEditorItem", "function createDatasetEditorRow")
    row_factory = _section(source, "function createDatasetEditorRow", "function createDatasetExperimentalFeaturesEditor")
    dataset_editor_row_source = _chunk11_compat_text()
    dataset_values_source = _frontend_module_text("js/features/anima-app/helpers/dataset-values.js")
    dataset_update_source = _frontend_feature_text("js/features/anima-app/chunks/12-create-dataset-row-caption-source-mode-editor.js", "js/features/dataset-editor/row-fields.js", "js/features/dataset-editor/preview.js")
    experimental_factory = _section(source, "function createDatasetExperimentalFeaturesEditor", "function createDatasetRowSettingsEditor")
    notice_factory = _section(source, "function createDatasetExperimentalNotice", "function createDatasetExperimentalAdvancedBody")
    advanced_body_factory = _section(source, "function createDatasetExperimentalAdvancedBody", "function datasetExperimentalOpenKey")
    inline_help_factory = _section(source, "function datasetExperimentalOpenKey", "function createDatasetIsRegEditor")
    is_reg_factory = _section(source, "function createDatasetIsRegEditor", "export {")
    caption_extension_factory = _section(source, "function createDatasetCaptionExtensionEditor", "function createDatasetNlTagMixEditor")
    mix_factory = _section(source, "function createDatasetNlTagMixEditor", "function createDatasetExperimentalScopePicker")
    help_specs = _section(source, "function datasetLocalHelpSpec", "function createDatasetHelpNode")
    caption_source_factory = _section(source, "function createDatasetRowCaptionSourceModeEditor", "function createDatasetRowSettingInput")
    normalize_factory = dataset_values_source
    payload_factory = _section(source, "function datasetRowsForPayload", "function normalizeDatasetRowSettings")
    row_update_factory = _section(
        dataset_update_source,
        "function updateDatasetEditorRowSetting",
        "function updateDatasetEditorRowSettingValue",
    )

    assert "通用标注设置" in defaults_editor
    assert "这里只保留 keep_tokens" in defaults_editor
    assert "文本标注扩展名等兼容项在每组数据集的高级区配置" in defaults_editor
    assert "['caption_extension', 'text']" not in defaults_editor
    assert "['keep_tokens', 'number']" in defaults_editor
    assert "['prefer_json_caption', 'switch', 'switch']" not in defaults_editor

    assert "createDatasetEditorItem(row, index)" in source
    assert "dataset-editor-item" in item_factory
    assert "createDatasetEditorRow(row, index, item)" in item_factory
    # 实验性编辑迁入弹窗后，主列表 item 不再内嵌折叠区
    assert "createDatasetExperimentalFeaturesEditor(row, index)" not in item_factory
    assert "createDatasetExperimentalFeaturesEditor(row, index)" in _frontend_module_text("js/features/dataset-editor/experimental-dialog.js")
    assert "createDatasetRowCaptionSourceModeEditor(settings, index)" in row_factory
    assert "createDatasetNlTagMixEditor(row, index)" in row_factory
    assert "实验性/高级/旧功能" in experimental_factory
    assert "dataset-experimental-features" in experimental_factory
    assert "createDatasetExperimentalAdvancedBody(row, index, overviewHelp, {" in experimental_factory
    assert "createDatasetExperimentalScopePicker(index)" in advanced_body_factory
    assert "createDatasetTriggerCloneEditor(row, index)" in advanced_body_factory
    assert "createDatasetCaptionExtensionEditor(row, index)" in advanced_body_factory
    assert "数据与路径规则" in advanced_body_factory
    assert "训练行为与策略" in advanced_body_factory
    assert "dataset-experimental-notice" in notice_factory
    assert "dataset-advanced-data-rules" in advanced_body_factory
    assert "dataset-advanced-training-rules" in advanced_body_factory
    assert "createDatasetNlTagMixEditor(row, index)" not in experimental_factory
    assert "createDatasetRowCaptionSourceModeEditor(settings, index)" not in experimental_factory
    assert "对应第 ${index + 1} 组数据集" in experimental_factory
    assert "dataset-experimental-overview-help" in experimental_factory
    assert "datasetLocalHelpSpec('experimental')" in experimental_factory
    assert "detailBtn" in experimental_factory
    assert "datasetExperimentalOpenState(index, defaultOpen)" in experimental_factory
    assert "bindDatasetExperimentalOpenState(panel, index)" in experimental_factory
    assert "captureDatasetExperimentalOpenStates(panel);" in source
    assert "datasetExperimentalOpenStates.set" in inline_help_factory
    assert "panel.addEventListener('toggle'" in inline_help_factory
    assert "收纳按单组数据集保存的高级兼容项" in help_specs
    assert "生效范围 / 对多数据集负责" in source
    assert "全选数据集" in source
    assert "datasetExperimentalScopeIndices" in source
    assert "setDatasetExperimentalScopeIndices" in source
    assert "datasetValidTargetIndices" in source
    assert "prior_loss_weight: Number.isFinite(priorLossWeight)" in normalize_factory

    assert ".dataset-editor-item" in css
    assert ".dataset-experimental-features" in css
    assert ".dataset-experimental-body" in css
    assert ".dataset-experimental-notice" in css
    assert ".dataset-advanced-section" in css
    assert ".dataset-advanced-grid" in css
    assert ".dataset-advanced-data-rules" in css
    assert ".dataset-advanced-training-rules" in css
    assert ".dataset-experimental-scope" in css
    assert ".dataset-scope-chip" in css
    assert ".dataset-caption-source" in css
    assert ".dataset-caption-source-options" in css
    assert ".dataset-caption-source-option.selected" in css
    assert ".dataset-caption-source-title-row" in css
    assert ".dataset-caption-source-help-toggle" in css
    assert ".dataset-caption-extension-advanced" in css
    assert ".dataset-caption-extension-input" in css
    assert ".dataset-caption-extension-help" in css
    assert ".dataset-trigger-clone" in css
    assert ".dataset-trigger-clone-summary" in css

    trigger_clone_factory = dataset_editor_row_source[
        dataset_editor_row_source.index("function createDatasetTriggerCloneEditor"):
    ]
    assert "触发提示词图像克隆" in trigger_clone_factory
    assert "触发提示词" in trigger_clone_factory
    assert "克隆循环次数" in trigger_clone_factory
    assert "datasetLocalHelpSpec('triggerClone')" in trigger_clone_factory
    assert "训练启动前在本次运行目录生成额外训练子集" in help_specs
    assert "原始数据集不会被修改" in help_specs
    assert "updateDatasetEditorRowTriggerClone(index" in trigger_clone_factory

    assert "正则化训练 / Regularization" in is_reg_factory
    assert "正则化损失权重" in is_reg_factory
    assert "Number.isFinite(nextWeight)" in is_reg_factory
    assert "parseFloat(weightInput.value) || 1.0" not in is_reg_factory
    assert "is_reg: row.is_reg" in payload_factory
    assert "settings: normalizeDatasetDefaults(row.settings || {})" in payload_factory
    assert "文本标注扩展名 / caption_extension" in caption_extension_factory
    assert "createDatasetInlineHelpButton(helpDiv, '查看文本标注扩展名说明')" in caption_extension_factory
    assert "updateDatasetEditorRowsSettingValue(" in caption_extension_factory
    assert "datasetExperimentalScopeIndices(index)" in caption_extension_factory
    assert "'caption_extension'" in caption_extension_factory
    assert "createHelpContent('caption_extension'" in caption_extension_factory

    assert "normalizeNlTagMix(row.nl_tag_mix)" in row_factory
    assert "nlTagMixSummary(mix)" in row_factory
    assert "const bucketText = settings.enable_bucket" in row_factory
    assert "const validationText = datasetPreviewValidationText(settings)" in row_factory
    assert "['桶', bucketText]" in row_factory
    assert "['验证', validationText]" in row_factory
    assert "createDatasetRepeatSettingField(row, index)" in source
    assert "panel.appendChild(createDatasetRepeatSettingField(row, index));" in source
    assert "bottomActions.append(remove);" in row_factory
    assert "num_repeats', input.value" not in row_factory
    assert "captions格式nl/tag权重调整" in mix_factory
    assert "自动识别 nl/tag" not in mix_factory
    assert "datasetLocalHelpSpec('nlTagMix')" in mix_factory
    assert "面向 DiffPipeForge captions.json 的多标注数据集" in help_specs
    assert "按 tag/nl 比例重建运行时 captions.json" in help_specs
    assert "重建后的 captions.json 和 results.json" in help_specs
    assert "从同一父目录下的 tag/ 与 nl/ 固定抽样。" not in mix_factory
    assert "tag 占比" in mix_factory
    assert "ratioInput.type = 'range';" in mix_factory
    assert "updateDatasetEditorRowNlTagMix(index" in mix_factory
    assert "datasetExperimentalScopeIndices(index)" not in mix_factory

    assert "caption_source_mode" in caption_source_factory
    assert "默认 auto 自动识别" in caption_source_factory
    assert "datasetCaptionSourceHelpSeq" in source
    assert "dataset-caption-source-help-toggle" in caption_source_factory
    assert "helpBtn.textContent = '?'" in caption_source_factory
    assert "helpBtn.setAttribute('aria-expanded', 'false')" in caption_source_factory
    assert "notes.hidden = true" in caption_source_factory
    assert "helpBtn.classList.toggle('active', nextVisible)" in caption_source_factory
    assert "sd-scripts" in caption_source_factory
    assert "AnimaLoraToolkit" in caption_source_factory
    assert "DiffPipeForge" in caption_source_factory
    assert '"1.png+1.txt"*n = sd-scripts格式标注' in caption_source_factory
    assert '"1.png+1.json"*n = AnimaLoraToolkit格式标注' in caption_source_factory
    assert '"png*n"+captions.json = DiffPipeForge格式标注' in caption_source_factory
    assert "caption_extension 仅影响 txt 来源或 auto 回退到文本标注" in caption_source_factory
    assert "json / captions.json 模式会忽略它" in caption_source_factory
    assert "updateDatasetEditorRowsSettingValue(" in caption_source_factory
    assert "[index]" in caption_source_factory
    assert "datasetExperimentalScopeIndices(index)" not in caption_source_factory
    assert "input.type === 'checkbox'" in row_update_factory
    assert "DEFAULT_NL_TAG_MIX" in normalize_factory
    assert "DEFAULT_TRIGGER_CLONE" in source
    assert "nl_tag_mix: normalizeNlTagMix(row.nl_tag_mix)" in normalize_factory
    assert "trigger_clone: normalizeTriggerClone(row.trigger_clone)" in normalize_factory
    assert "trigger_clone: normalizeTriggerClone(row.trigger_clone)" in source
    assert "function updateDatasetEditorRowTriggerClone" in source
    assert ".dataset-nl-tag-mix" in css
    assert ".dataset-nl-tag-summary" in css
    assert "grid-template-columns: minmax(210px, 0.72fr) minmax(320px, 1.65fr) 86px auto;" in css
    assert "grid-template-columns: repeat(auto-fit, minmax(154px, 1fr));" in css
    assert "grid-template-columns: repeat(4, minmax(118px, 1fr));" in css
    assert ".dataset-repeat-setting-field" in css
    assert "grid-column: auto;" in css
    assert "width: 100%;" in css
    assert ".dataset-repeat-input" in css
    assert "max-width: 112px;" in css
    assert "height: 38px;" in css
    assert "#tab-datasets .dataset-row-bottom-actions" in css
    assert "justify-content: flex-end;" in css


def test_dataset_editor_preserves_subset_filters_and_rederives_hidden_paths() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    experimental_factory = _section(source, "function createDatasetExperimentalFeaturesEditor", "function createDatasetRowSettingsEditor")
    advanced_body_factory = _section(source, "function createDatasetExperimentalAdvancedBody", "function datasetExperimentalOpenKey")
    filter_factory = _section(source, "function createDatasetPathFilterEditor", "function createDatasetRowSettingsEditor")
    dataset_values_source = _frontend_module_text("js/features/anima-app/helpers/dataset-values.js")
    normalize_factory = _section(dataset_values_source, "function normalizeDatasetEditorRows", "function normalizeDatasetRowSettings")
    defaults_factory = dataset_values_source[dataset_values_source.index("function normalizeDatasetDefaults"):]
    row_update_factory = _section(source, "function updateDatasetEditorRow(index", "function updateDatasetEditorRowSetting")

    assert "createDatasetExperimentalAdvancedBody(row, index, overviewHelp, {" in experimental_factory
    assert "createDatasetPathFilterEditor(row, index)" in advanced_body_factory
    assert "递归扫描子目录 / recursive" in filter_factory
    assert "路径筛选 / path_pattern" in filter_factory
    assert "recursive: row.recursive !== false && row.recursive !== 'false'" in normalize_factory
    assert "path_pattern: String(row.path_pattern || '*').trim() || '*'" in normalize_factory
    assert "recursive: row.recursive" in normalize_factory
    assert "path_pattern: row.path_pattern" in normalize_factory
    assert "rows[index].image_dir = '';" in row_update_factory
    assert "rows[index].cache_dir = '';" in row_update_factory
    assert "raw.keep_tokens ?? 3" in defaults_factory
    assert "raw.validation_seed ?? 42" in defaults_factory
    assert "raw.validation_split ?? 0" in defaults_factory
    assert ".dataset-path-filter-advanced" in css


def test_dataset_editor_drag_has_browser_fallbacks() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    drag_helpers = _section(source, "function datasetEditorEventPoint", "function setupDatasetEditorItemDropTarget")
    row_sort_helpers = _section(source, "function setDatasetEditorRowsAfterSort", "function removeDatasetEditorRow")
    row_factory = _section(source, "function createDatasetEditorRow", "function createDatasetExperimentalFeaturesEditor")
    item_factory = _section(source, "function createDatasetEditorItem", "function createDatasetEditorRow")

    assert "function datasetEditorEventPoint(event)" in drag_helpers
    assert "event.changedTouches?.[0]" in drag_helpers
    assert "function finishDatasetEditorPointerDrag(commit = false)" in drag_helpers
    assert "document.addEventListener('mousemove'" in drag_helpers
    assert "document.addEventListener('mouseup'" in drag_helpers
    assert "document.addEventListener('touchmove'" in drag_helpers
    assert "document.addEventListener('touchend'" in drag_helpers
    assert "document.addEventListener('touchcancel'" in drag_helpers
    assert "function startDatasetEditorMouseDrag(event, index, item, handle)" in drag_helpers
    assert "function startDatasetEditorTouchDrag(event, index, item, handle)" in drag_helpers
    assert "handle.addEventListener('mousedown'" in drag_helpers
    assert "handle.addEventListener('touchstart'" in drag_helpers
    assert "autoScrollFileGroupPointerDrag(" in drag_helpers
    assert "datasetEditorDropTargetFromPoint(" in drag_helpers
    assert "application/x-anima-dataset-row" in drag_helpers
    assert "Alt+方向键" in drag_helpers
    assert "moveDatasetEditorRowToIndex(index, targetIndex)" in drag_helpers
    assert "moveDatasetEditorRow(sourceIndex, clamped, clamped > sourceIndex)" in row_sort_helpers
    assert "if (insertIndex === sourceIndex) return false;" in row_sort_helpers
    assert "setupDatasetEditorItemDropTarget(item, index);" in item_factory
    assert "createDatasetEditorDragHandle(index, item)" in row_factory
    assert ".dataset-editor-drag-handle" in css
    assert ".dataset-editor-drag-image" in css
    assert ".dataset-editor-drop-before::before" in css
    assert ".dataset-editor-drop-after::after" in css
    assert ".dataset-editor-pointer-drag-active" in css



def test_dataset_page_toolbar_hosts_experimental_and_stage_entries() -> None:
    """数据集页顶栏承载实验性 + 分阶段入口；配置页不再当主编辑面。"""
    toolbar = _frontend_module_text("js/features/dataset-editor/toolbar.js")
    editor_chunk = _frontend_feature_text(
        "js/features/anima-app/chunks/09-setup-config-group-drop-target.js",
        "js/features/toml-manager/config-group-drop.js",
        "js/features/toml-manager/config-group-drop-target.js",
        "js/features/dataset-editor/dataset-render.js",
    )
    assert "btn-dataset-open-experimental" in toolbar
    assert "btn-dataset-open-stage-schedule" in toolbar
    assert "createDatasetEditorToolbarActions" in toolbar
    assert "createDatasetEditorToolbarActions" in editor_chunk
    assert "btn-dataset-open-stage-schedule" in editor_chunk or "createDatasetEditorToolbarActions" in editor_chunk


def test_stage_schedule_primary_entry_moves_to_dataset_page() -> None:
    """分阶段主入口迁到数据集页后，配置分组只读摘要，不再挂 createOpenStageResolutionDialogButton。"""
    group_entry = _frontend_feature_text(
        "js/features/anima-app/chunks/04-create-config-group-entry.js",
        "js/features/config-form/group-entry.js",
    )
    toolbar = _frontend_module_text("js/features/dataset-editor/toolbar.js")
    stage_ui = _frontend_feature_text("js/features/config-form/stage-resolution.js", "js/features/config-form/stage-resolution-model.js", "js/features/config-form/stage-resolution-ui.js", "js/features/config-form/stage-resolution-ui-widgets.js", "js/features/config-form/stage-resolution-ui-dialog.js")

    assert "createOpenStageResolutionDialogButton" not in group_entry
    assert "createStageScheduleInlineSummary" in group_entry
    assert "btn-dataset-open-stage-schedule" in toolbar
    assert "Always re-sync from draft || currentConfig" in stage_ui
    assert "key === 'stage_schedule' || key === 'stage_schedule_enabled'" in _frontend_module_text(
        "js/features/config-form/index.js"
    )


def test_stage_schedule_dialog_is_wired_from_dataset_group() -> None:
    """分阶段 dialog 由数据集顶栏打开，字段仍写训练配置 stage_schedule*。"""
    html = INDEX_HTML.read_text(encoding="utf-8")
    toolbar = _frontend_module_text("js/features/dataset-editor/toolbar.js")
    stage_ui = _frontend_feature_text("js/features/config-form/stage-resolution.js", "js/features/config-form/stage-resolution-model.js", "js/features/config-form/stage-resolution-ui.js", "js/features/config-form/stage-resolution-ui-widgets.js", "js/features/config-form/stage-resolution-ui-dialog.js")
    assert 'id="stage-resolution-dialog"' in html
    assert "btn-dataset-open-stage-schedule" in toolbar
    assert "stage_schedule_enabled" in stage_ui
    assert "subset_index" in stage_ui
    assert "start_pct" in stage_ui




def test_stage_schedule_subset_options_prefer_nonempty_dataset_rows() -> None:
    """课表下拉不能被空的 datasetEditorState.datasets 挡住多行预设。"""
    stage_ui = _frontend_feature_text("js/features/config-form/stage-resolution.js", "js/features/config-form/stage-resolution-model.js", "js/features/config-form/stage-resolution-ui.js", "js/features/config-form/stage-resolution-ui-widgets.js", "js/features/config-form/stage-resolution-ui-dialog.js")
    assert "function pickDatasetRows" in stage_ui
    assert "datasetPresetState?.datasets" in stage_ui
    assert "Array.isArray(rows) && rows.length" in stage_ui
    # Prefer source_dir for human labels when present.
    assert "source_dir || row?.image_dir" in stage_ui or "source_dir || row.image_dir" in stage_ui


def test_stage_schedule_ui_is_variable_n_not_hardcoded_three() -> None:
    """前端阶段模板支持可变 N，软上限 12，默认不是写死三段。"""
    stage_ui = _frontend_feature_text("js/features/config-form/stage-resolution.js", "js/features/config-form/stage-resolution-model.js", "js/features/config-form/stage-resolution-ui.js", "js/features/config-form/stage-resolution-ui-widgets.js", "js/features/config-form/stage-resolution-ui-dialog.js")
    assert "applyStageTemplate(2)" in stage_ui
    assert "Math.min(12" in stage_ui or "Math.min(12," in stage_ui
    assert "defaultStageScheduleStages" in stage_ui
    assert "阶段3" not in stage_ui.split("defaultStageScheduleStages")[1].split("export function")[0]
    assert "均分当前段" in stage_ui
    assert "applyStageTemplate(Math.max(1, stageResolutionState.stages.length || 2))" in stage_ui


def test_dataset_experimental_dialog_edits_selected_subset_only() -> None:
    """实验性控件进入弹窗，只编辑当前选中子集；主列表不再内嵌折叠区。"""
    html = INDEX_HTML.read_text(encoding="utf-8")
    dialog = _frontend_module_text("js/features/dataset-editor/experimental-dialog.js")
    row = _frontend_feature_text("js/features/dataset-editor/row.js", "js/features/dataset-editor/row-settings.js", "js/features/dataset-editor/row-settings-basic.js", "js/features/dataset-editor/row-settings-experimental.js")
    item = _frontend_feature_text("js/features/anima-app/chunks/10-create-dataset-config-input.js", "js/features/dataset-editor/config-input.js", "js/features/dataset-editor/item-drag.js")

    assert 'id="dataset-experimental-dialog"' in html
    assert "openDatasetExperimentalDialog" in dialog
    assert "selectedDatasetIndex" in row
    assert "createDatasetExperimentalFeaturesEditor(row, index)" in dialog
    # 主列表不再默认拼接卡内实验性折叠
    assert "createDatasetExperimentalFeaturesEditor(row, index)" not in item


def test_dataset_main_card_keeps_only_high_frequency_settings() -> None:
    """主卡只保留高频 settings；低频桶/验证细节进实验性弹窗。"""
    row = _frontend_feature_text("js/features/dataset-editor/row.js", "js/features/dataset-editor/row-settings.js", "js/features/dataset-editor/row-settings-basic.js", "js/features/dataset-editor/row-settings-experimental.js")
    dialog = _frontend_module_text("js/features/dataset-editor/experimental-dialog.js")
    settings_factory = _section(
        row,
        "function createDatasetRowSettingsEditor",
        "function createDatasetAdvancedSettingsEditor",
    )
    advanced_settings_factory = _section(
        row,
        "function createDatasetAdvancedSettingsEditor",
        "function createDatasetCaptionExtensionEditor",
    )
    experimental_factory = _section(
        row,
        "function createDatasetExperimentalFeaturesEditor",
        "function createDatasetPathFilterEditor",
    )

    assert "['resolution', 'number']" in settings_factory
    assert "['enable_bucket', 'select']" in settings_factory
    assert "['validation_split', 'number']" in settings_factory
    assert "createDatasetRepeatSettingField(row, index)" in settings_factory
    assert "['min_bucket_reso', 'number']" not in settings_factory
    assert "['max_bucket_reso', 'number']" not in settings_factory
    assert "['bucket_reso_steps', 'number']" not in settings_factory
    assert "['bucket_no_upscale', 'select']" not in settings_factory
    assert "['validation_split_num', 'number']" not in settings_factory
    assert "['validation_seed', 'number']" not in settings_factory

    for key, type_name in (
        ("min_bucket_reso", "number"),
        ("max_bucket_reso", "number"),
        ("bucket_reso_steps", "number"),
        ("bucket_no_upscale", "select"),
        ("validation_split_num", "number"),
        ("validation_seed", "number"),
    ):
        assert f"['{key}', '{type_name}']" in advanced_settings_factory
    assert "createDatasetRowSettingInput(index, key, type, settings)" in advanced_settings_factory
    assert "createDatasetAdvancedSettingsEditor(row, index)" in experimental_factory
    assert "createDatasetExperimentalFeaturesEditor(row, index)" in dialog

def test_field_presentation_provenance_and_presave_dirty_summary() -> None:
    """FieldPresentation helper + badge/pre-save dirty summary hooks must exist."""
    presentation = _frontend_module_text("js/features/config-form/field-presentation.js")
    field_row = _frontend_feature_text("js/features/anima-app/chunks/14-lora-adapter-kind-from-config.js", "js/features/config-form/form-fields.js", "js/features/config-form/form-fields-adapters.js", "js/features/config-form/form-fields-sample.js", "js/features/config-form/form-fields-ui.js")
    save_source = _frontend_module_text("js/features/anima-app/chunks/16-load-output-run-config.js")
    css = (STATIC_DIR / "css" / "13-shared-fields.css").read_text(encoding="utf-8")

    assert "export function buildFieldPresentation" in presentation
    assert "export function fieldSourceBadgeLabel" in presentation
    assert "export function summarizeDirtyDiff" in presentation
    assert "source: 'config'" in presentation or "source = 'config'" in presentation
    assert "ui_default" in presentation
    assert "isDirty" in presentation

    assert "buildFieldPresentation" in field_row
    assert "field-source-badge" in field_row
    assert "fieldSourceBadgeLabel" in field_row

    assert "summarizeDirtyDiff" in save_source
    assert "保存前将写入" in presentation
    assert "setTomlStatus('pending', summarizeDirtyDiff(changedValues)" in save_source
    assert ".field-source-badge" in css

def test_form_ui_defaults_and_help_align_with_base_facts() -> None:
    """Critical FORM_UI_DEFAULTS / help text must not contradict configs/base.toml facts.

    UI-only fallbacks that intentionally diverge must be listed in
    FORM_UI_ONLY_DEFAULT_KEYS instead of silently pretending to be merge defaults.
    """
    import tomllib

    base = tomllib.loads((REPO_ROOT / "configs" / "base.toml").read_text(encoding="utf-8"))
    assert base["lr_scheduler"] == "cosine"
    assert base["optimizer_type"] == "AdamW"
    assert base["lr_warmup_steps"] == 0.05
    assert base["gradient_checkpointing"] is False
    assert base["use_custom_down_autograd"] is False

    defaults_source = _frontend_module_text("js/config/catalog/defaults.js")
    help_source = _frontend_module_text("js/config/catalog/field-help-training.js")
    defaults_block = _section(
        defaults_source,
        "export const FORM_UI_DEFAULTS = {",
        "export const OPTIONAL_EMPTY_FIELDS",
    )

    # lr_scheduler is owned by the merge chain (base=cosine); do not ship a conflicting UI default.
    assert "lr_scheduler:" not in defaults_block

    lr_help = _section(help_source, "lr_scheduler: help(", "lr_warmup_steps: help(")
    assert "默认 constant" not in lr_help
    assert "cosine" in lr_help
    assert "base" in lr_help

    optimizer_help = _section(help_source, "optimizer_type: help(", "optimizer_args: help(")
    assert "默认 AdamW" in optimizer_help

    # gradient_checkpointing: FORM_UI true is a low-VRAM UI fallback, not base (false).
    assert "gradient_checkpointing: true" in defaults_block
    ui_only = _section(
        defaults_source,
        "export const FORM_UI_ONLY_DEFAULT_KEYS = new Set([",
        "]);",
    )
    assert "'gradient_checkpointing'" in ui_only

    custom_down_help = _section(
        help_source,
        "use_custom_down_autograd: help(",
        "log_every_n_steps: help(",
    )
    assert "默认 true" not in custom_down_help
    assert "false" in custom_down_help

def test_live_compat_warnings_mirror_key_conflict_codes() -> None:
    """Live compat helper surfaces key conflict codes without replacing preflight."""
    source = _frontend_module_text("js/features/config-form/live-compat.js")
    field_change = _frontend_feature_text("js/features/anima-app/chunks/14-lora-adapter-kind-from-config.js", "js/features/config-form/form-fields.js", "js/features/config-form/form-fields-adapters.js", "js/features/config-form/form-fields-sample.js", "js/features/config-form/form-fields-ui.js")

    assert "export function collectLiveCompatIssues" in source
    assert "export function formatLiveCompatStatus" in source
    assert "selective_full_gradient_checkpointing" in source
    assert "block_swap_soft_tokens" in source
    assert "不替代" in source or "Does NOT replace" in source or "preflight" in source

    assert "collectLiveCompatIssues" in field_change
    assert "updateLiveCompatWarningsFromForm" in field_change
    assert "setTomlStatus" in field_change
    assert "live 兼容" in field_change
    assert "wasLiveCompat" in field_change or "includes('live 兼容')" in field_change or 'includes("live 兼容")' in field_change

    if not shutil.which("node"):
        return

    script = r"""
import { collectLiveCompatIssues, formatLiveCompatStatus } from './web/static/js/features/config-form/live-compat.js';
const selective = collectLiveCompatIssues({
  selective_checkpoint: 'mlp_only',
  gradient_checkpointing: true,
});
const soft = collectLiveCompatIssues({
  blocks_to_swap: 8,
  network_module: 'networks.methods.soft_tokens',
});
const ok = collectLiveCompatIssues({
  selective_checkpoint: 'off',
  gradient_checkpointing: true,
  blocks_to_swap: 0,
});
console.log(JSON.stringify({
  selectiveCodes: selective.map((i) => i.code),
  softCodes: soft.map((i) => i.code),
  okCount: ok.length,
  formatted: formatLiveCompatStatus(selective),
}));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    payload = json.loads(proc.stdout)
    assert "selective_full_gradient_checkpointing" in payload["selectiveCodes"]
    assert "block_swap_soft_tokens" in payload["softCodes"]
    assert payload["okCount"] == 0
    assert "live 兼容" in payload["formatted"]

