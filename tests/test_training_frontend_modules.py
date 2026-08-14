# Split from test_training_frontend_state.py (modules)

from __future__ import annotations

from tests.frontend_test_support import *  # noqa: F403

import tests.frontend_test_support as _frontend_support
for _k, _v in vars(_frontend_support).items():
    if not _k.startswith("__"):
        globals()[_k] = _v


def test_frontend_module_graph_follows_production_entrypoint() -> None:
    graph = _frontend_module_graph()
    relative = [path.relative_to(STATIC_DIR).as_posix() for path in graph]

    assert relative[0] == "app.js"
    assert "chart.js" in relative
    assert "js/features/legacy-app.js" not in relative
    assert "js/features/anima-app/index.js" in relative
    assert "js/features/anima-app/imports.js" not in relative
    assert "js/features/anima-app/runtime.js" in relative
    assert "js/features/anima-app/runtime/api.js" in relative
    assert "js/features/anima-app/runtime/dom.js" in relative
    assert "js/features/anima-app/runtime/events.js" in relative
    assert "js/features/anima-app/runtime/feature-registry.js" in relative
    assert "js/features/anima-app/helpers/app-constants.js" in relative
    assert "js/features/anima-app/helpers/feature-ensurers.js" in relative
    assert "js/features/anima-app/helpers/history-artifacts.js" in relative
    assert "js/features/anima-app/legacy-globals.js" not in relative
    assert any(path.startswith("js/features/anima-app/chunks/") for path in relative)
    assert "js/features/preview/index.js" in relative
    assert "js/features/preview/state.js" in relative
    assert "js/features/preview/api.js" in relative
    assert "js/features/preview/workspace.js" in relative
    assert "js/features/preview/images.js" in relative
    assert "js/features/preview/weights.js" in relative
    assert "js/features/preview/dialog.js" in relative
    assert "js/features/queue/index.js" in relative
    assert "js/features/queue/state.js" in relative
    assert "js/features/queue/api.js" in relative
    assert "js/features/queue/render.js" in relative
    assert "js/features/queue/actions.js" in relative
    assert "js/features/queue/enqueue.js" in relative
    assert "js/features/weight-analysis/index.js" in relative
    assert "js/features/weight-analysis/state.js" in relative
    assert "js/features/weight-analysis/api.js" in relative
    assert "js/features/weight-analysis/render.js" in relative
    assert "js/features/environment-check/index.js" in relative
    assert "js/features/environment-check/state.js" in relative
    assert "js/features/environment-check/api.js" in relative
    assert "js/features/environment-check/render.js" in relative
    assert "js/features/image-test/index.js" in relative
    assert "js/features/image-test/state.js" in relative
    assert "js/features/image-test/api.js" in relative
    assert "js/features/image-test/gallery.js" in relative
    assert "js/features/image-test/render.js" in relative
    assert "js/features/image-test/selective-lora.js" in relative
    assert "js/features/app-shell/theme.js" in relative
    assert "js/features/app-shell/gpu-picker.js" in relative
    assert "js/features/app-shell/tabs.js" in relative
    assert "js/features/sample-prompts/model.js" in relative
    assert "js/features/toml-manager/group-state.js" in relative
    assert "js/features/history-detail/index.js" in relative
    assert "js/features/history-detail/state.js" in relative
    assert "js/features/history-detail/api.js" in relative
    assert "js/features/history-detail/dialog.js" in relative
    assert "js/features/history-detail/overview.js" in relative
    assert "js/features/history-detail/resume/index.js" in relative
    assert "js/features/history-detail/resume/state.js" in relative
    assert "js/features/history-detail/resume/panel.js" in relative
    assert "js/features/history-detail/resume/detail.js" in relative
    assert "js/features/history-detail/resume/actions.js" in relative
    assert "js/features/history-detail/analysis.js" in relative
    assert "js/features/history-detail/curve/index.js" in relative
    assert "js/features/history-detail/curve/data.js" in relative
    assert "js/features/history-detail/curve/toolbar.js" in relative
    assert "js/features/history-detail/curve/chart.js" in relative
    assert "js/features/history-detail/curve/hover.js" in relative
    assert "js/features/history-detail/system.js" in relative
    assert "js/features/history-detail/logs.js" in relative
    assert "js/features/history-detail/config-files.js" in relative
    assert "js/features/history-detail/workspace.js" in relative
    assert "js/features/history-detail/ui.js" in relative
    assert "js/config/catalog.js" in relative
    assert "js/config/catalog/labels-options.js" in relative
    assert "js/features/history-detail/resume.js" not in relative
    assert "js/features/history-detail/curve.js" not in relative
    assert "createHistoryResumeFeature } from './resume/index.js" in (
        STATIC_DIR / "js/features/history-detail/resume.js"
    ).read_text(encoding="utf-8")
    assert "createHistoryCurveRenderer } from './curve/index.js" in (
        STATIC_DIR / "js/features/history-detail/curve.js"
    ).read_text(encoding="utf-8")
    assert all(path.startswith(("app.js", "chart.js", "js/")) for path in relative)


def test_anima_app_global_this_writes_do_not_grow() -> None:
    graph = _frontend_module_graph()
    relative_paths = {path.relative_to(STATIC_DIR).as_posix(): path for path in graph}
    failures: list[str] = []

    for relative, baseline in ANIMA_APP_GLOBAL_THIS_BASELINE.items():
        path = relative_paths.get(relative)
        assert path is not None, f"{relative} is no longer reachable from app.js"
        actual = _global_this_write_counts(path)
        if actual[0] > baseline[0] or actual[1] > baseline[1]:
            failures.append(
                f"{relative}: globalThis writes grew from {baseline} to {actual}\n"
                + "\n".join(_global_this_write_lines(path))
            )

    for relative, path in sorted(relative_paths.items()):
        if not relative.startswith("js/features/anima-app/"):
            continue
        if relative in ANIMA_APP_GLOBAL_THIS_BASELINE:
            continue
        actual = _global_this_write_counts(path)
        if actual != (0, 0):
            failures.append(
                f"{relative}: new anima-app module writes globalThis without baseline: {actual}\n"
                + "\n".join(_global_this_write_lines(path))
            )

    assert not failures


def test_split_frontend_features_do_not_write_global_this() -> None:
    failures: list[str] = []
    for path in _frontend_module_graph():
        relative = path.relative_to(STATIC_DIR).as_posix()
        if not relative.startswith(GLOBAL_THIS_ZERO_WRITE_PREFIXES):
            continue
        actual = _global_this_write_counts(path)
        if actual != (0, 0):
            failures.append(
                f"{relative}: split frontend modules must not write globalThis: {actual}\n"
                + "\n".join(_global_this_write_lines(path))
            )

    assert not failures


def test_anima_app_production_path_has_no_state_global_bridge() -> None:
    graph = _frontend_module_graph()
    failures: list[str] = []

    for path in graph:
        relative = path.relative_to(STATIC_DIR).as_posix()
        actual = _global_this_write_counts(path)
        if actual == (0, 0):
            continue
        if relative in ANIMA_APP_GLOBAL_THIS_BASELINE:
            continue
        allowed = GLOBAL_THIS_ALLOWED_OUTSIDE_ANIMA_APP.get(relative)
        if allowed is not None and actual[0] <= allowed[0] and actual[1] <= allowed[1]:
            continue
        failures.append(
            f"{relative}: anima-app 生产路径不应再引入新的 globalThis bridge writes: {actual}\n"
            + "\n".join(_global_this_write_lines(path))
        )

    assert not failures


def test_legacy_globals_shim_is_deleted_and_unreachable() -> None:
    index_source = _frontend_module_text("js/features/anima-app/index.js")
    runtime_bridge_source = _frontend_module_text("js/features/anima-app/helpers/runtime-bridge.js")

    assert not LEGACY_GLOBALS_PATH.exists()
    assert not (STATIC_DIR / "js/features/anima-app/imports.js").exists()
    assert "legacy-globals.js" not in index_source
    assert "installLegacyImportGlobals" not in index_source
    assert "importsModule" not in index_source
    assert "installLegacyGlobals" not in index_source
    assert "installLegacyImageTestFeature" not in index_source
    assert "installLegacyStatusPolling" not in index_source
    assert "configureAppContextBridge(runtime.ctx);" in index_source
    assert "configureAppShellStateBridge(runtime.state.appShell);" in index_source
    assert "configureConfigStateBridge(runtime.state.config);" in index_source
    assert "configureDatasetStateBridge(runtime.state.dataset);" in index_source
    assert "configureHistoryStateBridge(runtime.state.history);" in index_source
    assert "configureRuntimeBridge(runtime);" in index_source
    assert "configureTomlStateBridge(runtime.state.toml);" in index_source
    assert "configureImageTestBridge(imageTestFeatureBridge.ensureImageTestFeature);" in index_source
    assert "configureStatusPollingBridge(statusPollingBridge);" in index_source
    assert "installLegacyStateGlobals(runtime);" not in index_source
    assert "import { installLegacyStateGlobals }" not in index_source
    assert "export function configureRuntimeBridge(runtime) {" in runtime_bridge_source
    assert "export function api(...args) {" in runtime_bridge_source
    assert "return requireRuntimeApi()(...args);" in runtime_bridge_source
    assert "return requireRuntimeApi().datasetPresetApi(...args);" in runtime_bridge_source
    assert "return requireRuntimeDom().val(...args);" in runtime_bridge_source
    assert "return requireRuntimeDom().populateSelect(...args);" in runtime_bridge_source


def test_legacy_globals_file_has_no_repo_source_consumers() -> None:
    failures: list[str] = []

    for path in _legacy_globals_repo_scan_paths():
        source = path.read_text(encoding="utf-8")
        matches = [
            needle
            for needle in ("legacy-globals.js", "installLegacyStateGlobals")
            if needle in source
        ]
        if not matches:
            continue
        failures.append(
            f"{path.relative_to(REPO_ROOT).as_posix()}: {', '.join(matches)}"
        )

    assert not failures


def test_runtime_bridge_helpers_do_not_require_legacy_globals_shim() -> None:
    if not shutil.which("node"):
        pytest.skip("node is required for anima-app runtime state bridge checks")
    script = r"""
import { createAnimaRuntime } from './web/static/js/features/anima-app/runtime.js';
import {
    api,
    configureRuntimeBridge,
    datasetPresetApi,
    populateSelect,
    val,
} from './web/static/js/features/anima-app/helpers/runtime-bridge.js';

const runtime = createAnimaRuntime({});
const apiCalls = [];
const populateCalls = [];
const bridgeRuntime = {
    api: Object.assign(
    (...args) => {
        apiCalls.push(['api', ...args]);
        return { kind: 'api', args };
    },
    {
        datasetPresetApi: (...args) => {
            apiCalls.push(['datasetPresetApi', ...args]);
            return { kind: 'datasetPresetApi', args };
        },
    },
    ),
    dom: {
        val: (...args) => `value:${args.join('|')}`,
        populateSelect: (...args) => {
            populateCalls.push(args);
            return { kind: 'populateSelect', args };
        },
    },
};
configureRuntimeBridge(bridgeRuntime);

globalThis.currentConfig = { name: 'shadow-config' };
globalThis.trainingStatusPollFailures = 2;
runtime.state.training.trainingRuntime.state = 'running';

const result = {
    currentConfigKeys: Object.keys(runtime.state.config.currentConfig).length,
    pollFailures: runtime.state.training.trainingStatusPollFailures,
    runtimeTrainingState: runtime.state.training.trainingRuntime.state,
    globalCurrentConfigName: globalThis.currentConfig.name,
    hasTrainingRuntimeOnGlobal: Object.prototype.hasOwnProperty.call(globalThis, 'trainingRuntime'),
    globalTrainingRuntimeType: typeof globalThis.trainingRuntime,
    apiKind: api('/api/test', { method: 'POST' }).kind,
    apiFirstPath: apiCalls[0]?.[1],
    apiFirstMethod: apiCalls[0]?.[2]?.method,
    datasetApiKind: datasetPresetApi('/api/datasets').kind,
    datasetApiPath: apiCalls[1]?.[1],
    valResult: val('variant-select'),
    populateKind: populateSelect('preset-select', ['default'], 'default').kind,
    populateTarget: populateCalls[0]?.[0],
    populateDefault: populateCalls[0]?.[2],
};

delete globalThis.currentConfig;
delete globalThis.trainingStatusPollFailures;

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
        "currentConfigKeys": 0,
        "pollFailures": 0,
        "runtimeTrainingState": "running",
        "globalCurrentConfigName": "shadow-config",
        "hasTrainingRuntimeOnGlobal": False,
        "globalTrainingRuntimeType": "undefined",
        "apiKind": "api",
        "apiFirstPath": "/api/test",
        "apiFirstMethod": "POST",
        "datasetApiKind": "datasetPresetApi",
        "datasetApiPath": "/api/datasets",
        "valResult": "value:variant-select",
        "populateKind": "populateSelect",
        "populateTarget": "preset-select",
        "populateDefault": "default",
    }


def test_frontend_module_cache_tokens_match_entrypoint() -> None:
    bootstrap = (STATIC_DIR / "js/ui-bootstrap.js").read_text(encoding="utf-8")
    match = re.search(r"const CLASSIC_ENTRY = '/static/app\.js\?v=([^']+)'", bootstrap)
    assert match, "missing versioned frontend module entrypoint"
    entry_token = match.group(1)
    assert entry_token.startswith("module-bootstrap-")

    graph_tokens: set[str] = set()
    for path in _frontend_module_graph():
        source = path.read_text(encoding="utf-8")
        for specifier in MODULE_IMPORT_RE.findall(source):
            child = _resolve_frontend_module(path, specifier)
            if child is None:
                continue
            token = _module_cache_token(specifier)
            if token:
                graph_tokens.add(token)

    assert len(graph_tokens) == 1, f"classic module graph uses mixed cache tokens: {sorted(graph_tokens)}"
    assert next(iter(graph_tokens)).startswith("module-bootstrap-")


def test_frontend_css_import_cache_tokens_match_entrypoint() -> None:
    bootstrap = (STATIC_DIR / "js/ui-bootstrap.js").read_text(encoding="utf-8")
    match = re.search(r"const CLASSIC_STYLESHEET = '/static/style\.css\?v=([^']+)'", bootstrap)
    assert match, "missing versioned frontend CSS entrypoint"
    entry_token = match.group(1)
    assert entry_token.startswith("frontend-chain-")

    mismatches: list[str] = []
    source = STYLE_CSS_PATH.read_text(encoding="utf-8")
    for specifier in CSS_IMPORT_RE.findall(source):
        token = _style_cache_token(specifier)
        if token != entry_token:
            mismatches.append(f"style.css: {specifier} uses {token!r}, expected {entry_token!r}")

    assert not mismatches


def test_style_import_order_puts_responsive_last() -> None:
    text = STYLE_CSS_PATH.read_text(encoding="utf-8")
    assert text.index('90-responsive.css') > text.index('42-image-test.css')


def test_anima_app_bootstrap_catches_startup_failures() -> None:
    source = APP_JS_PATH.read_text(encoding="utf-8")

    assert "createAnimaApp(ctx).catch((error) => {" in source
    assert "globalThis.__animaBootstrapError = error;" in source
    assert "console.error('[webui-bootstrap] failed to start Anima app', error);" in source
    assert "dom.optionalById('status-indicator')" in source
    assert "dom.optionalById('status-text')" in source


def test_anima_app_replaces_legacy_container_with_small_modules() -> None:
    anima_source = _anima_app_container_text()
    app_source = APP_JS_PATH.read_text(encoding="utf-8")
    graph = _frontend_module_graph()
    relative = {path.relative_to(STATIC_DIR).as_posix() for path in graph}
    feature_dirs = {
        path.parent.relative_to(STATIC_DIR).as_posix()
        for path in (STATIC_DIR / "js/features").glob("*/index.js")
    }
    app_modules = [
        path for path in graph
        if path.relative_to(STATIC_DIR).as_posix().startswith("js/features/anima-app/")
    ]
    oversized = [
        path.relative_to(STATIC_DIR).as_posix()
        for path in app_modules
        if len(path.read_text(encoding="utf-8").splitlines()) > 600
    ]
    known_oversized = {
        "js/features/anima-app/chunks/02-ensure-history-detail-feature.js": 10,
        "js/features/anima-app/chunks/05-create-stage-resolution-summary.js": 10,
        "js/features/anima-app/chunks/11-create-dataset-editor-row.js": 10,
        "js/features/anima-app/chunks/15-append-sample-prompt-row.js": 10,
        "js/features/anima-app/chunks/20-can-drop-toml-file-to-group.js": 10,
        "js/features/anima-app/chunks/23-move-current-toml-to-group.js": 10,
        "js/features/anima-app/chunks/24-show-preflight-pending-dialog.js": 10,
        "js/features/anima-app/chunks/26-load-global-settings.js": 40,
        "js/features/anima-app/chunks/31-create-history-collection-workbench-card.js": 10,
        "js/features/anima-app/chunks/36-setup-event-listeners.js": 10,
        "js/features/anima-app/chunks/37-config-training-source.js": 10,
    }
    unexpected_oversized = [path for path in oversized if path not in known_oversized]
    oversized_growth = [
        path
        for path, limit in known_oversized.items()
        if len((STATIC_DIR / path).read_text(encoding="utf-8").splitlines()) > limit
    ]

    assert not (STATIC_DIR / "js/features/legacy-app.js").exists()
    assert "createAnimaApp(ctx).catch" in app_source
    assert "createLegacyApp" not in app_source
    assert "return appShellModule.startAnimaApp();" in anima_source
    assert "js/features/live-training/index.js" in relative
    assert unexpected_oversized == []
    assert oversized_growth == []
    assert all(token not in app_source for token in ("fetch(", "addEventListener(", "getElementById("))
    assert feature_dirs
    assert feature_dirs <= {str(Path(item).parent) for item in relative}


def test_preview_feature_modules_are_loaded_from_production_entrypoint() -> None:
    legacy_source = _anima_app_container_text()
    feature_ensurers = _frontend_module_text("js/features/anima-app/helpers/feature-ensurers.js")
    preview_index = _frontend_module_text("js/features/preview/index.js")
    preview_state = _frontend_module_text("js/features/preview/state.js")
    preview_workspace = _frontend_module_text("js/features/preview/workspace.js")
    preview_images = _frontend_module_text("js/features/preview/images.js")
    css = STYLE_CSS.read_text(encoding="utf-8")

    assert "configurePreviewFeatureEnsurer(ctx, appShellState, {" in legacy_source
    assert "createPreviewFeature(ctx, deps)" in feature_ensurers
    assert "globalThis.ensurePreviewFeature" not in legacy_source
    for name in (
        "loadPreviewSettings",
        "savePreviewSettings",
        "resetPreviewSettings",
        "loadPreviewImages",
        "loadPreviewWeights",
        "setPreviewSource",
        "changePreviewTask",
        "openTrainingPreview",
        "openCurrentTrainingPreview",
        "openLiveSamplingPreview",
        "openHistoryConfigGroupPreview",
        "openPreviewPanel",
        "closePreviewPanel",
        "mountPreviewWorkspaceInHistoryDetail",
        "restorePreviewWorkspaceFromHistoryDetail",
        "activateHistoryDetailPreview",
        "updateRuntimeSampleState",
    ):
        assert name in preview_index
    assert "let previewSettings" not in legacy_source
    assert "let currentPreviewSource" not in legacy_source
    assert "mode: 'default'" in preview_state
    assert "state.selectedGroup = normalizePreviewGroup(options.group)" in preview_index
    assert "workspace.openPreviewPanel({ mode: 'sampling' });" in preview_index
    assert "function mountPreviewWorkspace(target)" in preview_workspace
    assert "function applyPreviewPanelMode" in preview_workspace
    assert "preview-panel-dialog-sampling" in preview_workspace
    assert "target.appendChild(workspace);" in preview_workspace
    assert "document.getElementById('preview-workspace')" in preview_workspace
    assert "export function normalizePreviewGroup(group)" in preview_state
    assert "grid.appendChild(createPreviewCard(image, index));" in preview_images
    assert "img.loading = previewImageLoadingMode(index);" in preview_images
    assert "const isHistorySelection = Boolean(state.selectedTaskId || state.selectedGroup);" in preview_images
    assert "HISTORY_PREVIEW_EAGER_IMAGE_LIMIT = 16" in preview_images
    assert "return isHistorySelection && index < HISTORY_PREVIEW_EAGER_IMAGE_LIMIT ? 'eager' : 'lazy';" in preview_images
    assert "preview-card-error-message" in preview_images
    assert "图片加载失败" in preview_images
    assert ".preview-card-error-message" in css
    assert ".preview-panel-dialog-sampling .preview-layout" in css
    assert ".preview-panel-dialog-sampling .preview-sidebar" in css


def test_weight_analysis_feature_modules_are_loaded_from_production_entrypoint() -> None:
    legacy_source = _anima_app_container_text()
    weight_index = _frontend_module_text("js/features/weight-analysis/index.js")
    weight_api = _frontend_module_text("js/features/weight-analysis/api.js")
    weight_render = _frontend_module_text("js/features/weight-analysis/render.js")
    weight_state = _frontend_module_text("js/features/weight-analysis/state.js")
    feature_ensurers = _frontend_module_text("js/features/anima-app/helpers/feature-ensurers.js")
    tabs_source = _frontend_module_text("js/features/app-shell/tabs.js")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    listener_section = _section(legacy_source, "export function setupEventListeners", "export function installBeginnerTooltips")
    tab_setup = _section(tabs_source, "function setupTabs()", "return {")
    tooltip_section = _section(legacy_source, "export function installBeginnerTooltips()", "// ── 工具函数 ──")

    assert "createWeightAnalysisFeature(ctx)" in feature_ensurers
    assert "ensureWeightAnalysisFeature(ctx, appShellState).bindWeightAnalysisEvents();" in listener_section
    assert "if (nextTab === 'weight-analysis')" in tab_setup
    assert "ensureWeightAnalysisFeature().loadAnalysisWeights();" in tab_setup
    assert "bindWeightAnalysisEvents" in weight_index
    assert "loadAnalysisWeights" in weight_index
    assert "runWeightAnalysis" in weight_index
    assert "createWeightAnalysisState" in weight_state
    assert "compareEnabled" in weight_state
    assert "candidateExpanded" in weight_state
    assert "fetchAnalysisWeights" in weight_api
    assert "inspectAnalysisWeight" in weight_api
    assert "inspectAnalysisWeightFile" in weight_api
    assert "'/api/analysis/weights'" in weight_api
    assert "'/api/analysis/inspect'" in weight_api
    assert "'/api/analysis/inspect-upload'" in weight_api
    assert "new FormData()" in weight_api
    assert "headers: {}" in weight_api
    assert "renderWeightOptions" in weight_render
    assert "renderHeatmap" in weight_render
    assert "renderComparison" in weight_render
    assert "renderBlockStructure" in weight_render
    assert "toggleCandidateExpanded" in weight_render
    assert "setActiveComponent" in weight_render
    assert "bindDropzoneEvents" in weight_index
    assert "handleWeightDrop" in weight_index
    assert "analyzeDroppedWeightFile" in weight_index
    assert "extractDroppedPath" in weight_index
    assert "toggleCompareMode" in weight_index
    assert "runWeightComparison" in weight_index
    assert "exportWeightAnalysisReport" in weight_index
    assert "exportWeightAnalysisJsonReport" in weight_index
    assert "buildWeightAnalysisJsonReport" in weight_index
    assert "ctx.download.downloadText" in weight_index
    assert "renderer.showCandidateKind" in weight_index
    assert "event.key === 'Enter'" in weight_index

    assert 'data-tab="weight-analysis"' in html
    assert 'id="tab-weight-analysis"' in html
    assert 'ΔW 权重结构分析' in html
    assert 'weight-analysis-select' in html
    assert 'weight-analysis-path' in html
    assert 'weight-analysis-dropzone' in html
    assert 'weight-analysis-file' in html
    assert 'weight-analysis-compare-path' in html
    assert 'weight-analysis-compare-dropzone' in html
    assert 'weight-analysis-compare-file' in html
    assert 'btn-toggle-weight-compare' in html
    assert 'btn-export-weight-analysis' in html
    assert 'btn-export-weight-analysis-json' in html
    assert 'btn-refresh-analysis-weights' in html
    assert 'btn-run-weight-analysis' in html
    assert 'weight-analysis-summary' in html
    assert 'weight-analysis-layer-list' in html
    assert 'weight-analysis-block-list' in html
    assert 'weight-analysis-style-top' in html
    assert 'weight-analysis-character-top' in html
    assert 'weight-analysis-block-structure' in html
    assert 'weight-analysis-compare-summary' in html
    assert 'btn-weight-analysis-toggle-candidates' in html
    assert 'weight-analysis-candidate-tab-style' in html
    assert 'weight-analysis-candidate-tab-character' in html
    assert 'weight-analysis-heatmap' in html
    assert '这里分析的是 safetensors 内的静态 ΔW 范数，不是跑图激活。' in html
    assert '早期 block 0–8 通常较弱，中后段 13–18、25–26 更值得关注' in html

    for selector in (
        "#tab-weight-analysis",
        ".weight-analysis-layout",
        ".weight-analysis-summary",
        ".weight-analysis-rank-list",
        ".weight-analysis-rank-bar",
        ".weight-analysis-candidate-list",
        ".weight-analysis-candidate-tabs",
        ".weight-analysis-block-structure",
        ".weight-analysis-compare-card",
        ".weight-analysis-dropzone",
        ".weight-analysis-heatmap-grid",
        ".weight-analysis-heatmap-cell",
    ):
        assert selector in css

    for tooltip_id in (
        "weight-analysis-select",
        "weight-analysis-path",
        "weight-analysis-dropzone",
        "weight-analysis-compare-path",
        "weight-analysis-compare-dropzone",
        "btn-toggle-weight-compare",
        "btn-export-weight-analysis",
        "btn-export-weight-analysis-json",
        "btn-refresh-analysis-weights",
        "btn-run-weight-analysis",
        "weight-analysis",
    ):
        assert tooltip_id in tooltip_section


def test_environment_check_feature_modules_are_loaded_from_production_entrypoint() -> None:
    legacy_source = _anima_app_container_text()
    environment_index = _frontend_module_text("js/features/environment-check/index.js")
    environment_api = _frontend_module_text("js/features/environment-check/api.js")
    environment_render = _frontend_module_text("js/features/environment-check/render.js")
    environment_state = _frontend_module_text("js/features/environment-check/state.js")
    feature_ensurers = _frontend_module_text("js/features/anima-app/helpers/feature-ensurers.js")
    tabs_source = _frontend_module_text("js/features/app-shell/tabs.js")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    listener_section = _section(legacy_source, "export function setupEventListeners", "export function installBeginnerTooltips")
    tab_setup = _section(tabs_source, "function setupTabs()", "return {")
    tooltip_section = _section(legacy_source, "export function installBeginnerTooltips()", "// ── 工具函数 ──")
    routes_source = (STATIC_DIR.parents[0] / "routes" / "__init__.py").read_text(encoding="utf-8")

    assert "createEnvironmentCheckFeature(ctx)" in feature_ensurers
    assert "ensureEnvironmentCheckFeature(ctx, appShellState).bindEnvironmentCheckEvents();" in listener_section
    assert "if (nextTab === 'environment')" in tab_setup
    assert "ensureEnvironmentCheckFeature?.().loadEnvironmentCheck();" in tab_setup
    assert "bindEnvironmentCheckEvents" in environment_index
    assert "loadEnvironmentCheck" in environment_index
    assert "copyReport" in environment_index
    assert "fetchEnvironmentCheck" in environment_api
    assert "'/api/environment/check'" in environment_api
    assert "createEnvironmentCheckRenderer" in environment_render
    assert "renderSummary" in environment_render
    assert "renderGroups" in environment_render
    assert "createEnvironmentCheckState" in environment_state
    assert "setup_environment_routes(app)" in routes_source

    assert 'data-tab="environment"' in html
    assert 'id="tab-environment"' in html
    assert '环境完整性检测' in html
    assert 'environment-check-summary-panel' in html
    assert 'environment-check-platform-meta' in html
    assert 'environment-check-status' in html
    assert 'environment-check-groups' in html
    assert 'btn-refresh-environment-check' in html
    assert 'btn-copy-environment-report' in html

    for selector in (
        "#tab-environment",
        ".environment-forge-layout",
        ".environment-check-editor",
        ".environment-check-sidebar",
        ".environment-check-toolbar",
        ".environment-check-platform-card",
        ".environment-check-summary-stat",
        ".environment-check-group",
        ".environment-check-item",
        ".environment-check-badge",
    ):
        assert selector in css

    for tooltip_id in (
        "btn-refresh-environment-check",
        "btn-copy-environment-report",
        "environment",
    ):
        assert tooltip_id in tooltip_section


def test_image_test_feature_modules_are_loaded_from_production_entrypoint() -> None:
    legacy_source = _anima_app_container_text()
    image_test_index = _frontend_module_text("js/features/image-test/index.js")
    image_test_api = _frontend_module_text("js/features/image-test/api.js")
    image_test_gallery = _frontend_feature_text(
        "js/features/image-test/gallery.js",
        *[
            path.relative_to(STATIC_DIR).as_posix()
            for path in sorted((STATIC_DIR / "js/features/image-test/gallery").glob("*.js"))
        ],
    )
    image_test_render = _frontend_module_text("js/features/image-test/render.js")
    image_test_state = _frontend_module_text("js/features/image-test/state.js")
    image_test_selective = _frontend_module_text("js/features/image-test/selective-lora.js")
    image_test_storage = _frontend_module_text("js/features/image-test/storage.js")
    tabs_source = _frontend_module_text("js/features/app-shell/tabs.js")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    listener_section = _section(legacy_source, "export function setupEventListeners", "export function installBeginnerTooltips")
    tab_setup = _section(tabs_source, "function setupTabs()", "return {")
    tooltip_section = _section(legacy_source, "export function installBeginnerTooltips()", "// ── 工具函数 ──")
    routes_source = (STATIC_DIR.parents[0] / "routes" / "__init__.py").read_text(encoding="utf-8")
    server_source = (STATIC_DIR.parents[0] / "server.py").read_text(encoding="utf-8")

    assert "createImageTestFeature(ctx, {" in legacy_source
    assert "ensureImageTestFeature().bindImageTestEvents();" in listener_section
    assert "if (nextTab === 'image-test')" in tab_setup
    assert "ensureImageTestFeature?.().loadImageTestPage();" in tab_setup
    assert "bindImageTestEvents" in image_test_index
    assert "loadImageTestPage" in image_test_index
    assert "syncFromCurrentConfig" in image_test_index
    assert "fetchImageTestStatus" in image_test_api
    assert "deleteImageTestImagesRequest" in image_test_api
    assert "resolveImageTestWeightPathRequest" in image_test_api
    assert "startImageTestRequest" in image_test_api
    assert "stopImageTestRequest" in image_test_api
    assert "fetchImageTestWeights" in image_test_api
    assert "fetchImageTestGpus" in image_test_api
    assert "fetchImageTestImages" in image_test_api
    assert "'/api/image-test/status'" in image_test_api
    assert "'/api/image-test/images'" in image_test_api
    assert "'/api/image-test/resolve-weight'" in image_test_api
    assert "'/api/image-test/start'" in image_test_api
    assert "'/api/image-test/stop'" in image_test_api
    assert "'/api/analysis/weights'" in image_test_api
    assert "'/api/training/gpus'" in image_test_api
    assert "/api/preview/images?" in image_test_api
    assert "createImageTestRenderer" in image_test_render
    assert "renderRuntime" in image_test_render
    assert "renderGpuOptions" in image_test_render
    assert "renderWeightOptions" in image_test_render
    assert "renderImages" in image_test_render
    assert "createImageTestState" in image_test_state
    assert "createImageTestGallery" in image_test_gallery
    assert "createImageTestSelectiveLoraController" in image_test_selective
    assert "createImageTestUiStorage" in image_test_storage
    assert "IMAGE_TEST_LAYER_DIALOG_STORAGE_KEY = 'anima.imageTest.layerDialog'" in image_test_selective
    assert "IMAGE_TEST_LAYER_DIALOG_STORAGE_VERSION = 1" in image_test_selective
    assert "IMAGE_TEST_UI_STORAGE_KEY = 'anima.imageTest.ui'" in image_test_storage
    assert "IMAGE_TEST_UI_STORAGE_VERSION = 1" in image_test_storage
    assert "IMAGE_TEST_DEFAULTS" in image_test_state
    assert "IMAGE_TEST_HISTORY_RANGE_OPTIONS" in image_test_state
    assert "IMAGE_TEST_SAMPLER_OPTIONS" in image_test_state
    assert "IMAGE_TEST_ATTN_MODE_OPTIONS" in image_test_state
    assert "IMAGE_TEST_RUNTIME_DTYPE_OPTIONS" in image_test_state
    assert "IMAGE_TEST_TEXT_ENCODER_DTYPE_OPTIONS" in image_test_state
    assert "normalizeImageTestHistoryRange" in image_test_state
    assert "daysForImageTestHistoryRange" in image_test_state
    assert "IMAGE_TEST_SELECTIVE_LORA_PRESET_OPTIONS" in image_test_state
    assert "IMAGE_TEST_SELECTIVE_LORA_GROUPS" in image_test_state
    assert "IMAGE_TEST_SELECTIVE_LORA_STRENGTH_STEP" in image_test_state
    assert "normalizeImageTestSelectiveLoraBlockStrengths" in image_test_state
    assert "enabledBlocksForImageTestSelectiveLoraStrengths" in image_test_state
    assert "image-test-runtime-dtype" in image_test_render
    assert "image-test-text-encoder-dtype" in image_test_render
    assert "image-test-gpu-index" in image_test_render
    assert "runtime_dtype: readValue('image-test-runtime-dtype')" in image_test_index
    assert "text_encoder_dtype: readValue('image-test-text-encoder-dtype')" in image_test_index
    assert "gpu_index: readValue('image-test-gpu-index')" in image_test_index
    assert "...selectiveLora.collectPayload()," in image_test_index
    assert "const selectiveError = selectiveLora.validate(payload);" in image_test_index
    assert "IMAGE_TEST_IMAGE_LIMIT = 500" in image_test_index
    assert "createImageTestGallery" in image_test_render
    assert "requestImageDelete" in image_test_render
    assert "gallery.render(payload);" in image_test_render
    assert "toggleImageSelection" in image_test_gallery
    assert "deleteImagesWithConfirmation" in image_test_gallery
    assert "visibleSelectionRange" in image_test_gallery
    assert "createMergedImageBlob" in image_test_gallery
    assert "GROUP_INITIAL_RENDER_COUNT = 24" in image_test_gallery
    assert "createLoadMoreFooter" in image_test_gallery
    assert "createZipDataBlob" in image_test_gallery
    assert "exportOriginalZipSelection" in image_test_gallery
    assert "syncFreshGroupCounts" in image_test_gallery
    assert "createFreshBadge" in image_test_gallery
    assert "normalizeZipEntryName" in image_test_gallery
    assert "virtualWindowByGroup" in image_test_gallery
    assert "scheduleVirtualWindowRefresh" in image_test_gallery
    assert "requestAnimationFrame" in image_test_gallery
    assert "renderGroupBody" in image_test_gallery
    assert "createVirtualSpacer" in image_test_gallery
    assert "GROUP_VIRTUALIZE_THRESHOLD = 48" in image_test_gallery
    assert "const startIndex = windowState.virtualized ? windowState.startIndex : 0;" in image_test_gallery
    assert "const endIndex = windowState.virtualized ? windowState.endIndex : visibleCount;" in image_test_gallery
    assert "if (changed) {" in image_test_gallery
    assert "refreshVisibleOrderedKeys();" in image_test_gallery
    assert "additive: event.ctrlKey || event.metaKey" in image_test_gallery
    assert "Shift 连选仅覆盖当前已展开且当前可见的图片；Ctrl/⌘ 可增量点选。" in image_test_gallery
    assert "if (options.additive) {" in image_test_gallery
    assert "btn-image-test-delete-selected" in image_test_gallery
    assert "btn-image-test-export-merged" in image_test_gallery
    assert "btn-image-test-export-originals" in image_test_gallery
    assert "image-test-history-filter" in image_test_gallery
    assert "state.visibleOrderedKeys" in image_test_gallery
    assert "btn-open-image-test-layer-dialog" in image_test_selective
    assert "image-test-layer-dialog" in image_test_selective
    assert "storage = window.localStorage" in image_test_selective
    assert "storage = window.localStorage" in image_test_storage
    assert "restorePersistedDialogState" in image_test_selective
    assert "readStoredDialogState" in image_test_selective
    assert "persistDialogState" in image_test_selective
    assert "storage.getItem(storageKey)" in image_test_selective
    assert "storage.setItem(storageKey, JSON.stringify({" in image_test_selective
    assert "persistFromDom" in image_test_storage
    assert "restoreToDom" in image_test_storage
    assert "restoreDeferredField" in image_test_storage
    assert "storedHistoryRange" in image_test_storage
    assert "history_range" in image_test_storage
    assert "image-test-prompt" in image_test_storage
    assert "image-test-weight-select" in image_test_storage
    assert "image-test-weight-path" in image_test_storage
    assert "layout: currentLayoutMode()," in image_test_selective
    assert "io_text: currentIoText()," in image_test_selective
    assert "toggleLayoutMode" in image_test_selective
    assert "body.dataset.layout = normalized;" in image_test_selective
    assert "range.type = 'range';" in image_test_selective
    assert "number.type = 'number';" in image_test_selective
    assert re.search(
        r"restoring = true;.*restorePersistedDialogState\(\);.*restoring = false;.*persistDialogState\(\);",
        image_test_selective,
        re.S,
    )
    assert "setup_image_test_routes(app)" in routes_source
    assert 'app["image_test_service"] = None' in server_source
    assert "ImageTestService(app)" in server_source

    assert 'data-tab="image-test"' in html
    assert 'id="tab-image-test"' in html
    assert 'image-test-prompt' in html
    assert 'image-test-negative-prompt' in html
    assert 'image-test-width' in html
    assert 'image-test-height' in html
    assert 'image-test-infer-steps' in html
    assert 'image-test-guidance-scale' in html
    assert 'image-test-flow-shift' in html
    assert 'image-test-seed' in html
    assert 'image-test-sampler' in html
    assert 'image-test-attn-mode' in html
    assert 'image-test-runtime-dtype' in html
    assert 'image-test-text-encoder-dtype' in html
    assert 'image-test-gpu-index' in html
    assert 'image-test-weight-drop-target' in html
    assert 'image-test-weight-select' in html
    assert 'image-test-weight-path' in html
    assert 'image-test-lora-multiplier' in html
    assert 'btn-open-image-test-layer-dialog' in html
    assert 'image-test-layer-dialog' in html
    assert 'image-test-layer-enable' in html
    assert 'btn-image-test-layer-layout-toggle' in html
    assert 'image-test-layer-layout-label' in html
    assert 'image-test-layer-preset' in html
    assert 'image-test-layer-dialog-summary' in html
    assert 'image-test-layer-inline-summary' in html
    assert 'image-test-layer-selection' in html
    assert 'image-test-layer-count' in html
    assert 'image-test-layer-dialog-count' in html
    assert 'image-test-layer-io-text' in html
    assert 'image-test-layer-io-status' in html
    assert 'btn-image-test-layer-export' in html
    assert 'btn-image-test-layer-import' in html
    assert 'image-test-layer-blocks-main' in html
    assert 'image-test-layer-blocks-adapter' in html
    assert 'image-test-layer-blocks-special' in html
    assert 'btn-start-image-test' in html
    assert 'btn-stop-image-test' in html
    assert 'btn-refresh-image-test-status' in html
    assert 'image-test-run-badge' in html
    assert 'image-test-run-summary' in html
    assert 'image-test-log' in html
    assert 'image-test-command' in html
    assert 'image-test-history-filter' in html
    assert 'data-range="7"' in html
    assert 'data-range="14"' in html
    assert 'data-range="30"' in html
    assert 'data-range="all"' in html
    assert 'image-test-selection-toolbar' in html
    assert 'image-test-selection-summary' in html
    assert 'btn-image-test-delete-selected' in html
    assert 'btn-image-test-export-merged' in html
    assert 'btn-image-test-export-originals' in html
    assert 'btn-image-test-clear-selection' in html
    assert 'image-test-grid' in html
    assert 'image-test-empty' in html
    assert 'btn-preview-dialog-delete' in html
    assert 'preview-dialog-status' in html

    for selector in (
        "#tab-image-test",
        ".image-test-layout",
        ".image-test-summary-item",
        ".image-test-grid-2",
        ".image-test-weight-drop-target",
        ".image-test-main-head-side",
        ".image-test-history-filter",
        ".image-test-history-filter-btn",
        ".image-test-selection-toolbar",
        ".image-test-selection-actions",
        ".image-test-history-groups",
        ".image-test-history-group",
        ".image-test-history-group-fresh-badge",
        ".image-test-history-group-toggle",
        ".image-test-history-group-grid",
        ".image-test-history-virtual-spacer",
        ".image-test-history-card",
        ".image-test-history-card-selection",
        ".image-test-history-card-fresh",
        ".image-test-history-card-delete",
        ".image-test-history-load-more",
        ".image-test-history-load-more-btn",
        ".image-test-layer-launch",
        ".image-test-layer-dialog",
        ".image-test-layer-dialog-toolbar",
        ".image-test-layer-layout-toggle",
        ".image-test-layer-layout-icon",
        ".image-test-layer-dialog-summary-row",
        ".image-test-layer-io",
        ".image-test-layer-io-actions",
        ".image-test-layer-io-status",
        ".image-test-layer-count",
        ".image-test-layer-dialog-body",
        ".image-test-layer-group",
        ".image-test-layer-rows",
        ".image-test-layer-row",
        ".image-test-layer-row-slider",
        ".image-test-layer-row-number",
        ".image-test-run-badge",
        ".image-test-run-summary",
        ".image-test-request-list",
        ".image-test-log",
        ".image-test-command",
        ".image-test-gallery",
        ".image-test-empty",
        ".image-test-card",
        ".preview-dialog-header-actions",
        ".preview-dialog-status",
    ):
        assert selector in css

    assert "position: sticky;" in _section(css, ".image-test-selection-toolbar {", ".image-test-selection-toolbar[hidden] {")
    assert "backdrop-filter: blur(10px);" in _section(css, ".image-test-selection-toolbar {", ".image-test-selection-toolbar[hidden] {")
    assert ".image-test-weight-drop-target.dragover select" in css
    assert "#image-test-weight-path.dragover" in css

    assert '.image-test-layer-dialog-body[data-layout="double"] .image-test-layer-rows' in css
    assert '.image-test-layer-dialog-body[data-layout="double"] .image-test-layer-row' in css

    for tooltip_id in (
        "btn-refresh-image-test-status",
        "btn-refresh-image-test-weights",
        "image-test-weight-select",
        "image-test-weight-path",
        "image-test-runtime-dtype",
        "image-test-text-encoder-dtype",
        "image-test-gpu-index",
        "image-test-history-filter",
        "btn-image-test-delete-selected",
        "btn-image-test-export-merged",
        "btn-image-test-export-originals",
        "btn-image-test-clear-selection",
        "btn-open-image-test-layer-dialog",
        "image-test-layer-dialog",
        "image-test-layer-enable",
        "btn-image-test-layer-layout-toggle",
        "image-test-layer-preset",
        "image-test-layer-selection",
        "image-test-layer-io-text",
        "btn-image-test-layer-export",
        "btn-image-test-layer-import",
        "btn-start-image-test",
        "btn-stop-image-test",
        "image-test",
    ):
        assert tooltip_id in tooltip_section


def test_format_path_label_contract_and_call_sites() -> None:
    """Shared path labels support length/basename/parent-basename and keep full-path titles."""
    format_source = _frontend_module_text("js/shared/format.js")
    history_item = _frontend_module_text(
        "js/features/anima-app/chunks/33-create-history-task-item.js"
    )
    dataset_input = _frontend_feature_text(
        "js/features/anima-app/chunks/10-create-dataset-config-input.js",
        "js/features/dataset-editor/config-input.js",
        "js/features/dataset-editor/item-drag.js",
    )

    assert "export function formatPathLabel" in format_source
    assert "export function compactPathLabel" in format_source
    assert re.search(r"mode\s*[:=]\s*['\"]length['\"]", format_source)
    assert "basename" in format_source
    assert "parent-basename" in format_source
    assert re.search(
        r"function compactPathLabel\([^)]*\)\s*\{[^}]*formatPathLabel\(",
        format_source,
        re.S,
    )

    task_item = _section(
        history_item,
        "function createHistoryTaskItem",
        "function compactPathLabel",
    )
    assert "pathText.title" in task_item or "pathText.setAttribute('title'" in task_item
    assert "continueText.title" in task_item or "continueText.setAttribute('title'" in task_item
    assert (
        "formatPathLabel" in history_item
        or "compactPathLabel(pathValue)" in task_item
    )
    assert "formatPathLabel" in dataset_input or "compactPathLabel(path)" in dataset_input

    if not shutil.which("node"):
        pytest.skip("node is required for formatPathLabel behavior checks")

    script = r"""
import { formatPathLabel, compactPathLabel } from './web/static/js/shared/format.js';

const longPath = '/data/very/long/nested/project/datasets/my-subset-images-folder-name';
const result = {
  lengthCompat: compactPathLabel(longPath, 24),
  lengthExplicit: formatPathLabel(longPath, { mode: 'length', maxLength: 24 }),
  lengthDefault: formatPathLabel(longPath),
  basename: formatPathLabel(longPath, { mode: 'basename' }),
  parentBasename: formatPathLabel(longPath, { mode: 'parent-basename' }),
  short: formatPathLabel('short/path', { mode: 'length', maxLength: 64 }),
  emptyBasename: formatPathLabel('', { mode: 'basename' }),
  windows: formatPathLabel('C:\\Users\\me\\datasets\\cats', { mode: 'parent-basename' }),
};
console.log(JSON.stringify(result));
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
    assert payload["basename"] == "my-subset-images-folder-name"
    assert payload["parentBasename"] == "datasets/my-subset-images-folder-name"
    assert payload["lengthCompat"] == payload["lengthExplicit"]
    assert "…" in payload["lengthCompat"]
    assert payload["lengthCompat"] != '/data/very/long/nested/project/datasets/my-subset-images-folder-name'
    assert len(payload["lengthDefault"]) <= 64 or "…" in payload["lengthDefault"]
    assert payload["short"] == "short/path"
    assert payload["windows"] == "datasets/cats"

def test_anima_app_startup_import_groups_use_promise_all() -> None:
    """Independent chunk imports should load in Promise.all batches without reordering bridge configure."""
    index_source = _frontend_module_text("js/features/anima-app/index.js")

    assert "Promise.all(" in index_source
    assert index_source.count("Promise.all(") >= 2

    # State bridges must still configure before any chunk import work starts.
    first_chunk_import = min(
        index_source.index("chunks/01-scope-state.js"),
        index_source.index("chunks/01a-image-test-feature.js"),
    )
    assert index_source.index("configureAppContextBridge(runtime.ctx);") < first_chunk_import
    assert index_source.index("configureRuntimeBridge(runtime);") < first_chunk_import
    assert index_source.index("configureTrainingStateBridge(runtime.state.training);") < first_chunk_import

    # Image-test and app-shell remain special serial stages.
    assert index_source.index("chunks/01a-image-test-feature.js") < index_source.index(
        "configureImageTestBridge(imageTestFeatureBridge.ensureImageTestFeature);"
    )
    assert index_source.index("configureImageTestBridge(imageTestFeatureBridge.ensureImageTestFeature);") < index_source.index(
        "chunks/02-ensure-history-detail-feature.js"
    )

    # 26a-d modules must load before their configure*Bridge calls.
    assert index_source.index("chunks/26a-global-settings.js") < index_source.index(
        "configureGlobalSettingsBridge(globalSettingsModule);"
    )
    assert index_source.index("chunks/26b-preview-view.js") < index_source.index(
        "configurePreviewViewBridge(previewViewModule);"
    )
    assert index_source.index("chunks/26c-queue-view.js") < index_source.index(
        "configureQueueViewBridge({"
    )
    assert index_source.index("chunks/26d-history-list.js") < index_source.index(
        "configureHistoryListBridge(historyListModule);"
    )
    assert index_source.index("configureHistoryListBridge(historyListModule);") < index_source.index(
        "chunks/26a-status-polling.js"
    )
    assert index_source.index("chunks/26a-status-polling.js") < index_source.index(
        "configureStatusPollingBridge(statusPollingBridge);"
    )

    # History task-actions self-configure still keeps 33 before 34 in source order.
    assert index_source.index("chunks/33-create-history-task-item.js") < index_source.index(
        "chunks/34-show-history-collection-select-dialog.js"
    )

    # Mid-range and history groups should both be Promise.all targets.
    mid_marker = "chunks/03-parse-network-arg-entry.js"
    history_marker = "chunks/27-render-history-collections-workbench.js"
    assert mid_marker in index_source
    assert history_marker in index_source
    assert "Promise.all(" in index_source[index_source.index(mid_marker) - 200 : index_source.index(mid_marker) + 80]
    assert "Promise.all(" in index_source[index_source.index(history_marker) - 200 : index_source.index(history_marker) + 80]


def test_history_task_actions_bridge_fails_fast_when_unconfigured() -> None:
    """Unconfigured history-task-actions methods must throw instead of silent no-op."""
    bridge_source = _frontend_module_text(
        "js/features/anima-app/helpers/history-task-actions-bridge.js"
    )
    index_source = _frontend_module_text("js/features/anima-app/index.js")
    chunk33 = _frontend_module_text(
        "js/features/anima-app/chunks/33-create-history-task-item.js"
    )
    chunk34 = _frontend_module_text(
        "js/features/anima-app/chunks/34-show-history-collection-select-dialog.js"
    )

    assert "legacyRoot = globalThis" not in bridge_source
    assert "legacyRoot.createHistoryTaskItem" not in bridge_source
    assert "configureHistoryTaskActionsBridge" in bridge_source
    assert "history-task-actions" in bridge_source
    assert "bridge not configured" in bridge_source
    assert "configureHistoryTaskActionsBridge({" in chunk33
    assert "configureHistoryTaskActionsBridge({" in chunk34
    assert "chunks/33-create-history-task-item.js" in index_source
    assert "chunks/34-show-history-collection-select-dialog.js" in index_source
    assert index_source.index("chunks/33-create-history-task-item.js") < index_source.index(
        "chunks/34-show-history-collection-select-dialog.js"
    )

    if not shutil.which("node"):
        pytest.skip("node is required for history-task-actions bridge fail-fast checks")

    script = r"""
const {
  configureHistoryTaskActionsBridge,
  deleteHistoryTask,
  loadHistoryTask,
  showHistoryTaskConfirmDialog,
} = await import('./web/static/js/features/anima-app/helpers/history-task-actions-bridge.js');

const unconfigured = [];
for (const [name, fn] of [
  ['deleteHistoryTask', deleteHistoryTask],
  ['loadHistoryTask', loadHistoryTask],
  ['showHistoryTaskConfirmDialog', showHistoryTaskConfirmDialog],
]) {
  try {
    fn({ id: 'task-1' });
    unconfigured.push({ name, ok: true });
  } catch (error) {
    unconfigured.push({
      name,
      ok: false,
      message: String(error && error.message ? error.message : error),
    });
  }
}

let configuredMessage = '';
configureHistoryTaskActionsBridge({
  deleteHistoryTask: (task) => `deleted:${task.id}`,
});
try {
  configuredMessage = deleteHistoryTask({ id: 'task-9' });
} catch (error) {
  configuredMessage = `error:${error && error.message ? error.message : error}`;
}

console.log(JSON.stringify({ unconfigured, configuredMessage }));
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
    assert payload["configuredMessage"] == "deleted:task-9"
    assert len(payload["unconfigured"]) == 3
    for item in payload["unconfigured"]:
        assert item["ok"] is False, item
        assert "history-task-actions" in item["message"]
        assert "not configured" in item["message"]
