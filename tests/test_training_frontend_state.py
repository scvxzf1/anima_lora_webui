from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"
APP_JS_PATH = STATIC_DIR / "app.js"
CHART_JS = STATIC_DIR / "chart.js"
INDEX_HTML = STATIC_DIR / "index.html"
STYLE_CSS_PATH = STATIC_DIR / "style.css"
MODULE_IMPORT_RE = re.compile(
    r"""(?:(?:import|export)\s+(?:[^'"]*?\s+from\s+)?|import\(\s*)['"]([^'"]+\.js(?:\?[^'"]*)?)['"]"""
)
CSS_IMPORT_RE = re.compile(r"""@import\s+(?:url\()?['"]?([^'")]+\.css)['"]?\)?\s*;""")


def _resolve_frontend_module(parent: Path, specifier: str) -> Path | None:
    parsed = urlparse(specifier)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path)
    if not path.endswith(".js"):
        return None
    if path.startswith("/static/"):
        resolved = (STATIC_DIR / path.removeprefix("/static/")).resolve()
    elif path.startswith("./") or path.startswith("../"):
        resolved = (parent.parent / path).resolve()
    else:
        return None
    if resolved == STATIC_DIR.resolve() or STATIC_DIR.resolve() in resolved.parents:
        return resolved
    raise AssertionError(f"frontend module import escapes static dir: {parent} -> {specifier}")


def _frontend_module_graph(entry: Path = APP_JS_PATH) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        ordered.append(resolved)
        source = resolved.read_text(encoding="utf-8")
        for specifier in MODULE_IMPORT_RE.findall(source):
            child = _resolve_frontend_module(resolved, specifier)
            if child is not None:
                assert child.is_file(), f"missing frontend module import: {resolved} -> {specifier}"
                visit(child)

    visit(entry)
    return ordered


def _frontend_module_text(relative_path: str) -> str:
    path = (STATIC_DIR / relative_path).resolve()
    graph = _frontend_module_graph()
    assert path in graph, f"{relative_path} is not reachable from app.js"
    return path.read_text(encoding="utf-8")


def _frontend_feature_text(*relative_paths: str) -> str:
    return "\n".join(_frontend_module_text(relative_path) for relative_path in relative_paths)


def _anima_app_container_text() -> str:
    graph = _frontend_module_graph()
    paths = [
        STATIC_DIR / "js/features/anima-app/index.js",
        STATIC_DIR / "js/features/anima-app/imports.js",
        *sorted((STATIC_DIR / "js/features/anima-app/chunks").glob("*.js")),
    ]
    for path in paths:
        assert path.resolve() in graph, f"{path.relative_to(STATIC_DIR).as_posix()} is not reachable"
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _module_cache_token(specifier: str) -> str | None:
    return parse_qs(urlparse(specifier).query).get("v", [None])[0]


class _FrontendJsSource:
    def read_text(self, encoding: str = "utf-8") -> str:
        return "\n".join(path.read_text(encoding=encoding) for path in _frontend_module_graph())


APP_JS = _FrontendJsSource()


def _resolve_frontend_css(parent: Path, specifier: str) -> Path:
    parsed = urlparse(specifier)
    if parsed.scheme or parsed.netloc:
        raise AssertionError(f"external css import is not allowed: {parent} -> {specifier}")
    path = unquote(parsed.path)
    if path.startswith("/static/"):
        resolved = (STATIC_DIR / path.removeprefix("/static/")).resolve()
    else:
        resolved = (parent.parent / path).resolve()
    if resolved == STATIC_DIR.resolve() or STATIC_DIR.resolve() in resolved.parents:
        assert resolved.is_file(), f"missing css import: {parent} -> {specifier}"
        return resolved
    raise AssertionError(f"css import escapes static dir: {parent} -> {specifier}")


def _frontend_css_text(entry: Path = STYLE_CSS_PATH, encoding: str = "utf-8") -> str:
    seen: set[Path] = set()
    chunks: list[str] = []

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        source = resolved.read_text(encoding=encoding)
        chunks.append(source)
        for specifier in CSS_IMPORT_RE.findall(source):
            visit(_resolve_frontend_css(resolved, specifier))

    visit(entry)
    return "\n".join(chunks)


class _FrontendCssSource:
    def read_text(self, encoding: str = "utf-8") -> str:
        return _frontend_css_text(encoding=encoding)


STYLE_CSS = _FrontendCssSource()


def _section(source: str, start: str, end: str) -> str:
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index]


def test_frontend_module_graph_follows_production_entrypoint() -> None:
    graph = _frontend_module_graph()
    relative = [path.relative_to(STATIC_DIR).as_posix() for path in graph]

    assert relative[0] == "app.js"
    assert "chart.js" in relative
    assert "js/features/legacy-app.js" not in relative
    assert "js/features/anima-app/index.js" in relative
    assert "js/features/anima-app/imports.js" in relative
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


def test_frontend_module_cache_tokens_match_entrypoint() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    match = re.search(r'<script[^>]+src="/static/app\.js\?v=([^"]+)"', html)
    assert match, "missing versioned frontend module entrypoint"
    entry_token = match.group(1)
    assert entry_token.startswith("module-bootstrap-")

    mismatches: list[str] = []
    for path in _frontend_module_graph():
        source = path.read_text(encoding="utf-8")
        for specifier in MODULE_IMPORT_RE.findall(source):
            child = _resolve_frontend_module(path, specifier)
            if child is None:
                continue
            token = _module_cache_token(specifier)
            if token != entry_token:
                relative = path.relative_to(STATIC_DIR).as_posix()
                mismatches.append(f"{relative}: {specifier} uses {token!r}, expected {entry_token!r}")

    assert not mismatches


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

    assert not (STATIC_DIR / "js/features/legacy-app.js").exists()
    assert "createAnimaApp(ctx);" in app_source
    assert "createLegacyApp" not in app_source
    assert "globalThis.startAnimaApp" in anima_source
    assert oversized == ["js/features/anima-app/chunks/25-update-progress.js"]
    assert all(token not in app_source for token in ("fetch(", "addEventListener(", "getElementById("))
    assert feature_dirs
    assert feature_dirs <= {str(Path(item).parent) for item in relative}


def test_preview_feature_modules_are_loaded_from_production_entrypoint() -> None:
    legacy_source = _anima_app_container_text()
    preview_index = _frontend_module_text("js/features/preview/index.js")
    preview_state = _frontend_module_text("js/features/preview/state.js")
    preview_workspace = _frontend_module_text("js/features/preview/workspace.js")
    preview_images = _frontend_module_text("js/features/preview/images.js")
    css = STYLE_CSS.read_text(encoding="utf-8")

    assert "createPreviewFeature(ctx, {" in legacy_source
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
    assert "return isHistorySelection && index < 80 ? 'eager' : 'lazy';" in preview_images
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
    tabs_source = _frontend_module_text("js/features/app-shell/tabs.js")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    listener_section = _section(legacy_source, "function setupEventListeners", "function installBeginnerTooltips")
    tab_setup = _section(tabs_source, "function setupTabs()", "return {")
    tooltip_section = _section(legacy_source, "function installBeginnerTooltips()", "// ── 工具函数 ──")

    assert "createWeightAnalysisFeature(ctx)" in legacy_source
    assert "ensureWeightAnalysisFeature().bindWeightAnalysisEvents();" in listener_section
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
    tabs_source = _frontend_module_text("js/features/app-shell/tabs.js")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    listener_section = _section(legacy_source, "function setupEventListeners", "function installBeginnerTooltips")
    tab_setup = _section(tabs_source, "function setupTabs()", "return {")
    tooltip_section = _section(legacy_source, "function installBeginnerTooltips()", "// ── 工具函数 ──")
    routes_source = (STATIC_DIR.parents[0] / "routes" / "__init__.py").read_text(encoding="utf-8")

    assert "createEnvironmentCheckFeature(ctx)" in legacy_source
    assert "ensureEnvironmentCheckFeature().bindEnvironmentCheckEvents();" in listener_section
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


def test_new_training_launch_enters_live_monitoring() -> None:
    source = APP_JS.read_text(encoding="utf-8")
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


def test_return_to_live_training_clears_runtime_cursor() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    body = _section(source, "function returnToLiveTraining", "async function loadResumeOptionsForTask")

    for snippet in (
        "viewingHistoryTaskId = '';",
        "historyViewMode = 'live';",
        "trainingRuntime.lastLogId = 0;",
        "trainingRuntime.logLineCount = 0;",
        "stepCounter = 0;",
        "lossChart?.clear();",
        "recoverLiveTrainingState();",
    ):
        assert snippet in body


def test_live_training_rest_fallbacks_are_wired() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    poll_delay_section = _section(source, "function trainingStatusPollDelayMs", "async function pollStatus")
    poll_section = _section(source, "async function pollStatus", "function applyStatusSnapshotFallbacks")
    update_status = _section(source, "function updateStatus", "function resetLiveSystemPeaks")
    health_section = _section(source, "function refreshTrainingHealth", "function parseMetricsFromProgressLine")
    parse_metrics_section = _section(source, "function parseMetricsFromProgressLine", "function lastValue")
    recovery_section = _section(source, "async function recoverLiveTrainingState", "function updateProgress")
    ready_section = _section(source, "function startAnimaApp", "function chartTheme")

    assert "function isLiveRunningState" in source
    assert "function trainingStatusPollDelayMs" in source
    assert "function scheduleStatusPoll(options = {})" in source
    assert "trainingStatusPollTimer = window.setTimeout" in source
    assert "window.clearTimeout(trainingStatusPollTimer);" in source
    assert "if (!visible) return wsOpen ? (running ? 30000 : 120000) : 60000;" in poll_delay_section
    assert "if (!wsOpen) return running ? 5000 : 15000;" in poll_delay_section
    assert "return running ? 10000 : 60000;" in poll_delay_section
    assert "last_log_line: status.last_log_line" in poll_section
    assert "error_hint: status.error_hint" in poll_section
    assert "anomaly_message: status.anomaly_message || ''" in poll_section
    assert "if (options.forceReplayMetrics) {" in poll_section
    assert "trainingStatusPollForceReplayMetrics = true;" in poll_section
    assert "if (trainingStatusPollPromise) return trainingStatusPollPromise;" in poll_section
    assert "const forceReplayMetrics = trainingStatusPollForceReplayMetrics;" in poll_section
    assert "trainingStatusPollForceReplayMetrics = false;" in poll_section
    assert "trainingStatusPollPromise = null;" in poll_section
    assert "scheduleStatusPoll();" in poll_section
    assert "applyStatusSnapshotFallbacks(status);" in poll_section
    assert "forceReplayMetrics || isLiveRunningState()" in poll_section
    assert "forceReplayMetrics || isLiveRunningState() || hasStatusPayload(status.latest_metric)" not in poll_section
    assert "function applyStatusSnapshotFallbacks(status = {})" in source
    assert "updateProgress(status.latest_progress, { replay: true });" in source
    assert "updateMetrics(status.latest_metric, { replay: true });" in source
    assert "updateSystem(status.latest_system, { replay: true });" in source
    assert "function hasStatusPayload(value)" in source

    assert "const state = liveStatusState(msg);" in update_status
    assert "const terminalMessage = terminalStatusMessage(msg);" in update_status
    assert "trainingRuntime.lastTerminalMessage = state === 'error' ? terminalMessage : '';" in update_status
    assert "const canStop = isLiveRunningState(state);" in update_status
    assert "stopBtn.disabled = !canStop;" in update_status
    assert "Object.prototype.hasOwnProperty.call(msg, 'anomaly_message')" in update_status
    assert "trainingRuntime.lastAnomalyMessage = String(msg.anomaly_message || '').trim();" in update_status
    assert "state === 'running' || (state === 'idle' && !terminalMessage)" in update_status
    assert "function liveStatusState(msg = {})" in update_status
    assert "if (state === 'idle' && terminalStatusMessage(msg)) return 'error';" in update_status
    assert "function terminalStatusMessage(msg = {})" in update_status
    assert "const state = String(msg.state || '');" in update_status
    assert "const lineIsError = logLineTone(line) === 'error';" in update_status
    assert "if (state !== 'error' && !lineIsError) return '';" in update_status
    assert "return line.includes(hint) ? line : `${line}；${hint}`;" in update_status

    assert "trainingRuntime.lastAnomalyMessage" in health_section
    assert "el.title = trainingRuntime.lastAnomalyMessage;" in health_section
    assert "el.removeAttribute('title');" in health_section
    assert "trainingRuntime.state === 'error' && trainingRuntime.lastTerminalMessage" in health_section
    assert "最近任务异常" in health_section
    assert "const metricNumberToken = '([+\\\\-]?" in parse_metrics_section
    assert "if (lossMatch) out.loss = lossMatch[1];" in parse_metrics_section
    assert "if (out.loss !== undefined && !Number.isFinite(out.loss)) delete out.loss;" not in parse_metrics_section
    assert "pollStatus({ forceReplayMetrics: true });" in recovery_section
    assert "replayTrainingLogs({ includeMetrics: false });" in recovery_section
    assert "scheduleStatusPoll();" in ready_section
    assert "document.addEventListener('visibilitychange'" in ready_section
    assert "scheduleStatusPoll({ immediate: !document.hidden });" in ready_section
    assert "scheduleStatusPoll({ immediate: true });" in ready_section
    assert "window.addEventListener('online', () => {" in ready_section
    assert "recoverLiveTrainingState();" in ready_section


def test_training_queue_frontend_hooks_are_present() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    legacy_source = _anima_app_container_text()
    queue_index = _frontend_module_text("js/features/queue/index.js")
    queue_state = _frontend_module_text("js/features/queue/state.js")
    queue_api = _frontend_module_text("js/features/queue/api.js")
    queue_render = _frontend_module_text("js/features/queue/render.js")
    queue_actions = _frontend_module_text("js/features/queue/actions.js")
    queue_enqueue = _frontend_module_text("js/features/queue/enqueue.js")
    queue_feature_source = "\n".join([
        queue_index,
        queue_state,
        queue_api,
        queue_render,
        queue_actions,
        queue_enqueue,
    ])
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    queue_section = queue_feature_source
    listener_section = _section(legacy_source, "function setupEventListeners", "function installBeginnerTooltips")
    group_actions = _section(source, "function createTomlGroupActions", "function createTomlGroupActionButton")
    current_queue = _section(queue_enqueue, "async function queueCurrentTrainingFromConfig", "async function enqueueTrainingFromConfig")
    enqueue_section = _section(queue_enqueue, "async function enqueueTrainingFromConfig", "async function enqueueTrainingQueueRequest")
    view_section = _section(legacy_source, "function renderTrainingViewMode", "// ── 状态轮询 ──")
    stop_section = _section(legacy_source, "async function stopTraining()", "    // ── WebSocket ──")
    poll_section = _section(legacy_source, "async function pollStatus", "function applyStatusSnapshotFallbacks")
    assert "createQueueFeature(ctx, {" in legacy_source
    assert "ensureQueueFeature().bindQueueEvents();" in listener_section
    for name in (
        "loadTrainingQueue",
        "renderTrainingQueue",
        "updateTrainingQueueFromPayload",
        "updateRunningQueueProgress",
        "queueCurrentTrainingFromConfig",
        "enqueueTrainingFromConfig",
        "enqueueTrainingQueueRequest",
        "enqueueTrainingQueueBatchRequest",
        "queueResumeTrainingFromCheckpoint",
        "bindQueueEvents",
    ):
        assert name in queue_index
    assert "function renderTrainingQueue()" in queue_section
    assert "function renderTrainingQueueManager()" in queue_section
    assert "function queueManagerSections(state)" in queue_state
    assert "const isErrorOnly = payload.ok === false && !hasItems" in queue_state
    assert "status: payload.status === undefined" in queue_state
    assert "isErrorOnly ? (previous.items || []) : []" in queue_state
    assert "isErrorOnly ? (previous.summary || {}) : {}" in queue_state
    assert "function createTrainingQueueSection" in queue_section
    assert "function createTrainingQueueItem" in queue_section
    assert "function createTrainingQueueManagerItem" in queue_section
    assert "filter: 'actionable'" in queue_state
    assert "let trainingQueueState" not in legacy_source
    assert "let trainingQueueFilter" not in legacy_source
    assert "async function toggleTrainingQueuePause()" in queue_section
    assert "cancelAllQueueItems" in queue_section
    assert "removeQueueItemFromList" in queue_section
    assert "移除列表" in queue_section
    assert "只会将这条记录从队列界面移除" in queue_section
    assert "event.preventDefault()" in queue_section
    assert "event.stopPropagation()" in queue_section
    assert "HTMLDetailsElement" in queue_section
    assert "delete_runtime: true" not in queue_section
    assert "queueDeleteRuntimeMessage" not in queue_section
    assert "queueRuntimeDirLabel" not in queue_section
    assert "新任务已加入队列" in queue_section
    assert "移除原记录" in queue_section
    assert "feedback:" in queue_state
    assert "setQueueFeedback" in queue_state
    assert "beginQueueFeedback" in queue_actions
    assert "finishQueueFeedback" in queue_actions
    assert "queueMoveDirectionLabel" in queue_actions
    assert "正在刷新队列状态" in queue_actions
    assert "清理已完成记录" in queue_section
    assert "清理已取消记录" in queue_section
    assert "retryQueueItem" in queue_section
    assert "cancelWaitingQueueItems" in queue_section
    assert "clearCompletedQueueItems" in queue_section
    assert "clearCanceledQueueItems" in queue_section
    assert "focusQueueFilterAfterTerminalClear" in queue_section
    assert "已完成记录已保留" in queue_section
    assert "清理已取消不会影响这里" in queue_section
    assert "清理已完成不会影响这里" in queue_section
    assert "缓存或任何实际文件" in queue_section
    assert "btn-clear-completed-queue" in html
    assert "btn-clear-canceled-queue" in html
    assert ".training-queue-item-more[open]" in css
    assert "z-index: 130" in css
    assert "state.filter" in queue_section
    assert "renderQueueManagerOverview" in queue_section
    assert "queueFilterLabel" in queue_section
    assert "createQueueFactRow" in queue_section
    assert "queueShortId" in queue_section
    assert "updateQueueFilterButton" in queue_section
    assert "queueFilterCount" in queue_section
    assert "queueEmptyStateText" in queue_section
    assert "updateQueueActionHints" in queue_section
    assert "renderQueueFeedback" in queue_section
    assert "queueFeedbackBusyAction" in queue_section
    assert "queueFeedbackItemState" in queue_section
    assert "queue-action-busy" in queue_section
    assert "aria-pressed" in queue_section
    assert "aria-disabled" in queue_section
    assert "aria-busy" in queue_section
    assert "queueDetailsSummaryText" in queue_section
    assert ".training-queue-overview-item" in css
    assert ".training-queue-feedback" in css
    assert ".training-queue-manager-item.queue-feedback-pending" in css
    assert ".task-history-action.queue-action-busy" in css
    assert ".training-queue-facts" in css
    assert ".training-queue-filter b" in css
    assert "grid-template-columns: repeat(5, minmax(0, 1fr))" in css
    assert "/api/training/queue" in queue_section
    assert "/api/training/queue/settings" in queue_section
    assert "/api/training/queue/cancel-all" in queue_section
    assert "/api/training/queue/abort-after-current" in queue_section
    assert "/api/training/queue/force-abort" in queue_section
    assert "/api/training/queue/cancel-waiting" in queue_section
    assert "/api/training/queue/clear-completed" in queue_section
    assert "/api/training/queue/clear-canceled" in queue_section

    assert "training-queue-manager" in html
    assert "training-queue-manager-overview" in html
    assert "training-queue-feedback" in html
    assert 'role="status" aria-live="polite"' in html
    assert "btn-training-queue-view" in html
    assert "training-queue-failure-policy" in html
    assert 'data-queue-filter="actionable">待处理' in html
    assert "training-queue-more-menu" in html
    assert "btn-queue-from-config" in html
    assert "btn-open-history-manager" in html
    assert "未归档 · 最新 6 个训练任务" in html
    assert "queueCurrentTrainingFromConfig" in listener_section
    assert "btn-open-history-manager').addEventListener('click', () => showTrainingView('history'))" in listener_section
    assert "const mainWide = isQueue || isHistory;" in view_section
    assert "workspace.classList.toggle('main-wide', mainWide)" in view_section
    assert "trainingRoot.classList.toggle('history-mode', isHistory)" in view_section
    assert "workspace.classList.toggle('history-mode', isHistory)" in view_section
    assert "history-wide" not in view_section
    assert ".slice(0, 6)" in source

    summary_panel = _section(html, '<section class="panel training-queue-panel"', '<section class="panel task-history-panel">')
    manager_panel = _section(html, '<section id="training-queue-manager"', '<section id="training-history-placeholder"')
    assert "btn-cancel-all-queue" not in summary_panel
    assert "btn-abort-queue-after-current" not in summary_panel
    assert "btn-force-abort-queue" not in summary_panel
    assert "btn-cancel-waiting-queue" not in summary_panel
    assert "btn-clear-finished-queue" not in summary_panel
    assert "btn-clear-completed-queue" not in summary_panel
    assert "btn-clear-canceled-queue" not in summary_panel
    assert "btn-cancel-all-queue" in manager_panel
    assert "btn-abort-queue-after-current" in manager_panel
    assert "btn-force-abort-queue" in manager_panel
    assert "中止后续队列" in manager_panel
    assert "强制中止队列" in manager_panel
    assert "btn-cancel-waiting-queue" in manager_panel
    assert "btn-clear-finished-queue" not in manager_panel
    assert "btn-clear-completed-queue" in manager_panel
    assert "btn-clear-canceled-queue" in manager_panel

    ws_section = _section(source, "function handleWsMessage", "function appendLog")
    assert "case 'queue':" in ws_section
    assert "updateTrainingQueueFromPayload(msg);" in ws_section

    start_section = _section(source, "async function startTrainingUnchecked", "function enterLiveTrainingForNewRun")
    assert "enqueueTrainingFromConfig" in start_section
    assert "chooseTrainingLaunchMode" in source
    assert "await enqueueTrainingFromConfig(variant, preset, methodsSubdir" in current_queue
    assert "options.configFile || deps.currentTrainingConfigFile()" in enqueue_section
    assert "const startPaused = options.startPaused !== false" in enqueue_section
    assert "队列会保存独立运行配置并保持暂停" in enqueue_section
    assert "includeContinueSource === false" in enqueue_section
    assert "const wasDisabled = Boolean(stopBtn?.disabled)" in stop_section
    assert "await pollStatus();" in stop_section
    assert "await loadTrainingQueue();" in stop_section
    assert "setTrainingHealthNotice(message, 'error')" in stop_section
    assert "globalThis.trainingStatusPollFailures = 0" in legacy_source
    assert "globalThis.trainingStatusPollTimer = null" in legacy_source
    assert "globalThis.trainingStatusPollPromise = null" in legacy_source
    assert "globalThis.trainingStatusPollForceReplayMetrics = false" in legacy_source
    assert "if (status.ok === false) throw new Error(status.error || '读取训练状态失败')" in poll_section
    assert "if (trainingStatusPollPromise) return trainingStatusPollPromise;" in poll_section
    assert "trainingStatusPollFailures < 3" in poll_section
    assert "训练状态轮询连续失败" in poll_section
    assert "setTrainingHealthNotice(message, 'error')" in poll_section
    assert "async function enqueueTrainingQueueRequest" in queue_enqueue
    assert "async function enqueueTrainingQueueBatchRequest" in queue_enqueue
    assert "queueBatchApiUnsupported(res)" in queue_enqueue
    assert "enqueueTrainingQueueBatchCompat(requestOptions, res)" in queue_enqueue
    assert "enqueueTrainingQueueBatchRootCompat(options, aliasRes || unsupported)" in queue_enqueue
    assert "enqueueTrainingQueueBatchFallback(options, rootRes || unsupported)" in queue_enqueue
    assert "method not allowed" in queue_enqueue
    assert "status_code" in queue_enqueue
    assert "if (!queueBatchApiUnsupported(unsupported)) throw e;" in queue_enqueue
    assert "async function abortQueueAfterCurrent" in queue_section
    assert "async function forceAbortQueue" in queue_section
    assert "showAppConfirmDialog" in queue_section
    assert "当前正在运行的任务会继续执行到完成" in queue_section
    assert "会立即停止当前正在运行的训练/预处理进程" in queue_section
    assert "on('btn-abort-queue-after-current', 'click', abortQueueAfterCurrent)" in queue_section
    assert "on('btn-force-abort-queue', 'click', forceAbortQueue)" in queue_section
    assert "abortTrainingQueueAfterCurrent(ctx)" in queue_section
    assert "forceAbortTrainingQueue(ctx)" in queue_section
    assert "const abortAfterCurrentBtn = document.getElementById('btn-abort-queue-after-current')" in queue_section
    assert "const forceAbortBtn = document.getElementById('btn-force-abort-queue')" in queue_section
    assert "function queueBackendRunning()" in queue_section
    assert "state.queue.status === 'running'" in queue_section
    assert "deps.getTrainingRuntime()?.state === 'running'" in queue_section
    assert "counts.queued <= 0" in queue_section
    assert "counts.queued + counts.running" in queue_section
    assert "createTomlGroupActionButton('加入队列', () => enqueueTomlGroupToQueue(group)" in group_actions
    assert "queueableTomlGroupFiles(group)" in group_actions
    assert "async function enqueueTomlGroupToQueue" in source
    assert "tomlItemQueueEntry(item, preset)" in source
    assert "tomlGroupQueueFailureLabel(item, failure, failedIndex)" in source
    assert "label: label === '未命名配置文件' ? '' : label" in source
    assert "failure.label || failure.filename" in source
    assert "第 ${fallbackIndex} 个配置" in source
    assert "showTomlGroupQueueConfirmDialog(group, files)" in source
    assert "队列会保持暂停，等待你手动继续" in group_actions
    assert "startPaused: true" in group_actions
    assert "/api/training/queue/batch/start" in queue_api
    assert "/api/training/queue/batch-start" in queue_api
    assert "/api/training/queue', options" in queue_api
    assert "start_paused: Boolean(options.startPaused)" in queue_api
    assert "if (!Object.prototype.hasOwnProperty.call(data, 'status_code')) data.status_code = res.status;" in source
    assert "item?.config_file" in source
    assert ".toml-group-action-btn-queue" in css
    assert "导出单个" in html
    assert "createTomlGroupActionButton('导出分组', () => exportTomlGroup(group)" in group_actions
    assert "exportableTomlGroupFiles(group)" in group_actions
    assert "async function exportTomlGroup" in source
    assert "createTomlZipBlob(entries)" in source
    assert "downloadBlob(blob, filename)" in source
    assert "/api/config/raw?file=${encodeURIComponent(path)}" in source
    assert "uniqueZipEntryName" in source
    assert "ZIP_CRC_TABLE" in source
    assert "内含 ${files.length} 个独立 TOML 文件" in source
    assert ".toml-group-action-btn-export" in css


def test_launch_readiness_panel_is_removed() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    listener_section = _section(source, "function setupEventListeners", "function installBeginnerTooltips")
    action_state = _section(source, "function updateTomlActionState", "function isTomlLocked")

    assert "launch-readiness" not in html
    assert "launch-readiness" not in css
    assert "launchReadiness" not in source
    assert "启动准备" not in html
    assert "启动准备" not in source

    assert "btn-start-from-config" in html
    assert "btn-queue-from-config" in html
    assert "btn-start-from-config').addEventListener('click', startTraining)" in listener_section
    assert "btn-queue-from-config').addEventListener('click', queueCurrentTrainingFromConfig)" in listener_section
    assert "handleLaunchReadinessPrimaryAction" not in source
    assert "btn-start-from-config" in action_state
    assert "startBtn.textContent = sourceMode === 'full_resume'" in action_state
    assert "开始完整续训" in action_state
    assert "开始热启动训练" in action_state
    assert "开始训练" in action_state


def test_config_toolbar_is_first_visible_config_row() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    editor_html = _section(html, '<section class="config-preset-editor">', '<section id="continue-training-source"')
    toolbar_html = _section(html, '<div class="config-toolbar">', '<div id="gpu-picker" class="gpu-picker">')
    form_workspace_css = _section(css, "#tab-config .config-form-workspace,", "#tab-config .config-toolbar {")
    toolbar_css = _section(css, "#tab-config .config-toolbar {", "#tab-config .config-toolbar label")

    assert "config-preset-header" not in editor_html
    assert "CONFIG FORGE" not in editor_html
    assert "CONFIG PRESET" not in editor_html
    assert "配置工作台" not in editor_html
    assert "训练配置" not in editor_html
    assert "config-selection-state\" hidden" in toolbar_html
    hidden_selection_html = _section(toolbar_html, '<div class="config-selection-state" hidden>', '</div>')
    assert "method-select" in toolbar_html
    assert "variant-select" in toolbar_html
    assert "preset-select" in toolbar_html
    assert "preset-select" not in hidden_selection_html
    assert "config-run-preset" in toolbar_html
    assert "运行覆盖" in toolbar_html
    assert "来自 configs/presets.toml" in toolbar_html
    assert ".config-run-preset" in css
    assert 'id="choice-guide"' not in html
    assert "padding: 1rem 1.65rem 1.7rem;" in form_workspace_css
    assert "align-items: center;" in toolbar_css


def test_config_form_uses_navigation_search_and_progressive_disclosure() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    labels_options = (STATIC_DIR / "js" / "config" / "catalog" / "labels-options.js").read_text(encoding="utf-8")

    category_defs = _section(source, "const FORM_CATEGORY_DEFS = [", "const FORM_CATEGORY_SECTION_MAP")
    render_section = _section(source, "function renderConfigForm", "function shouldRenderConfigSection")
    order_section = _section(source, "function appendConfigGroupsByCategory", "function createGroup")
    collect_section = _section(source, "function collectChangedFormValues", "function networkArgInputChanged")
    load_steps = _section(source, "async function loadStepEstimate", "async function loadDatasetEditor")
    defaults = _section(source, "const FORM_UI_DEFAULTS = {", "const OPTIONAL_EMPTY_FIELDS")
    catalog_defaults = _frontend_module_text("js/config/catalog/defaults.js")
    catalog_form_layout = _frontend_module_text("js/config/catalog/form-layout.js")
    catalog_help_training = _frontend_module_text("js/config/catalog/field-help-training.js")
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
    assert "if (selectedConfigDatasetFile !== (currentConfig.dataset_config || ''))" in source
    assert "const params = new URLSearchParams({" in load_steps
    assert "const configFile = currentTrainingConfigFile();" in load_steps
    assert "params.set('config_file', configFile);" in load_steps
    assert "const datasetConfigOverride = selectedDatasetConfigOverride();" in load_steps
    assert "if (datasetConfigOverride !== null) params.set('dataset_config', datasetConfigOverride);" in load_steps
    assert "const data = await api(`/api/config/steps?${params.toString()}`);" in load_steps
    assert "setTomlStatus(\n            applied ? 'ok' : 'error'," in source
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
    for label in ["全 GPU", "Balanced 16G", "FP8 测试", "更省显存", "LoKr 16G", "OOM 兜底"]:
        assert label in source
    assert "merge: {" in source
    assert "blocks_to_swap: 'max'" in source
    assert "selective_checkpoint: 'checkpoint_strength_max'" in source
    quick_presets = _section(source, "globalThis.RESOURCE_QUICK_PRESETS = [", "globalThis.SELECTIVE_CHECKPOINT_STRENGTH")
    assert quick_presets.count("gradient_checkpointing: false") == 6
    assert "globalThis.SELECTIVE_CHECKPOINT_STRENGTH = new Map([" in source
    assert "['mlp_only', 4]" in source
    assert "['every_other', 5]" in source
    assert "function resourceQuickPresetValue(preset, key, value)" in source
    assert "function strongerSelectiveCheckpointValue(current, fallback)" in source
    assert "return Math.max(current, next);" in source
    assert "return currentStrength >= fallbackStrength ? currentKey : fallbackKey;" in source
    assert "NO_DATASET_REGULARIZATION_QUICK_PRESETS" in source
    for label in ["先验基线", "DOP 角色", "遮罩保护", "关闭"]:
        assert label in source
    no_dataset_quick_presets = _section(source, "globalThis.NO_DATASET_REGULARIZATION_QUICK_PRESETS = [", "globalThis.SELECTIVE_CHECKPOINT_STRENGTH")
    assert "prior_preservation_weight: 0.1" in no_dataset_quick_presets
    assert "blank_prompt_preservation: true" in no_dataset_quick_presets
    assert "diff_output_preservation_trigger: 'sks'" in no_dataset_quick_presets
    assert "diff_output_preservation_class: ''" in no_dataset_quick_presets
    assert "inverted_mask_prior_weight: 0.1" in no_dataset_quick_presets
    assert no_dataset_quick_presets.count("use_text_cache: true") == 3
    assert no_dataset_quick_presets.count("cache_llm_adapter_outputs: true") == 3
    assert "function applyNoDatasetRegularizationQuickPreset" in source
    assert "还需要填写 DOP 类提示并重新生成文本缓存" in source
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
    assert "advanced.open = true;" in no_dataset_update
    assert "updateNoDatasetRegularizationModePanel();" in source
    for value in [
        "blocks_to_swap: 12",
        "blocks_to_swap: 16",
        "blocks_to_swap: 23",
        "block_swap_transfer_dtype: 'bf16'",
        "block_swap_transfer_dtype: 'fp8_e4m3'",
        "selective_checkpoint: 'mlp_only'",
        "block_swap_profile_jsonl: 'auto'",
        "memory_probe_jsonl: 'auto'",
        "memory_probe_max_steps: 2",
        "memory_probe_max_steps: 3",
        "lokr_factor_group_size: 8",
        "lokr_project_chunk_bytes: 4194304",
    ]:
        assert value in source
    set_field_section = _section(source, "function setFieldInputValue", "function escapeHtml")
    assert "configDraftValueChanged(key, value, original)" in set_field_section
    assert "configFormState.draftValues.delete(key);" in set_field_section
    assert "input.value = value ?? '';" in set_field_section
    compact_field_css = _section(css, ".config-field-grid-3col .field-main", ".field-label-stack")
    assert "grid-template-rows: auto auto;" in compact_field_css
    assert "row-gap: 0.24rem;" in compact_field_css
    assert "grid-row: 1;" in compact_field_css
    assert "grid-row: 2;" in compact_field_css
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
    assert "LOSS_WEIGHTING_DEPENDENT_FIELDS = new Map([" in source
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
    assert "只在遮罩外区域做先验保留" in catalog_help_training
    assert "block_swap_transfer_dtype: '块交换传输精度'" in source
    assert "block_swap_transfer_dtype: ['bf16', 'fp8_e4m3']" in source
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
    assert "keys: ['blocks_to_swap', 'block_swap_transfer_dtype', 'selective_checkpoint', 'selective_checkpoint_blocks']" in resource_compact
    assert "keys: ['block_swap_profile_jsonl', 'memory_probe_jsonl', 'memory_probe_max_steps']" in resource_compact
    assert "keys: ['peak_probe_jsonl', 'peak_probe_max_steps', 'peak_probe_level']" in resource_compact
    assert "keys: ['gradient_checkpointing', 'unsloth_offload_checkpointing', 'disable_block_swap_for_eval']" in resource_compact
    assert "keys: ['max_data_loader_n_workers', 'vae_chunk_size', 'vae_disable_cache']" in data_resource_compact
    assert "keys: ['dataloader_pin_memory', 'persistent_data_loader_workers']" in data_resource_compact
    assert "config-field-grid-3col config-field-grid-inline-flags" in resource_compact
    assert "config-field-grid-2col config-field-grid-inline-flags" in data_resource_compact
    assert "sample_sampler: 'euler'" in defaults
    assert "sample_sampler: ['euler', 'er_sde', 'lcm']" in options


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
    collect_section = _section(source, "function collectChangedFormValues", "function networkArgInputChanged")
    live_section = _section(source, "function liveConfigFromForm", "function formatFieldName")
    network_args_section = _section(source, "function collectNetworkArgsFromForm", "function formatNetworkArg")

    assert "const rawNetworkArgsChanged = 'network_args' in values;" in collect_section
    assert "{ skipUnchangedInputs: rawNetworkArgsChanged }" in collect_section
    assert "const rawNetworkArgsChanged = configFormState.draftValues.has('network_args');" in live_section
    assert "collectNetworkArgsFromForm(liveConfig, { skipUnchangedInputs: rawNetworkArgsChanged })" in live_section
    assert "function collectNetworkArgsFromForm(baseConfig = currentConfig, options = {})" in network_args_section
    assert "if (options.skipUnchangedInputs && !networkArgInputChanged(input)) continue;" in network_args_section


def test_config_form_keeps_dora_as_lora_addon_and_merges_exclusive_adapters() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    defaults = _section(source, "const FORM_UI_DEFAULTS = {", "const OPTIONAL_EMPTY_FIELDS")
    network_arg_specs = _section(source, "const NETWORK_ARG_FIELD_SPECS = [", "const NETWORK_ARG_FIELD_MAP")
    layout = _section(source, "const FORM_SECTION_DEFS = [", "const STICKY_CONFIG_CATEGORY_IDS")
    merged_fields = _section(source, "const CONFIG_FORM_MERGED_FIELDS = new Set([", "const DEPRECATED_CONFIG_FORM_FIELDS")
    render_section = _section(source, "function renderConfigForm", "function shouldRenderConfigSection")
    collect_section = _section(source, "function collectChangedFormValues", "function networkArgInputChanged")
    live_section = _section(source, "function liveConfigFromForm", "function formatFieldName")
    state_section = _section(source, "function readLoKrEnabled", "function currentLossWeightingScheme")

    assert "lora_adapter_kind: 'lora'" in defaults
    assert "dora_wd: false" in defaults
    assert "use_glora: false" in defaults
    assert "use_vera: false" in defaults
    assert "vera_projection_prng_key: 0" in defaults
    assert "vera_d_initial: 0.1" in defaults
    assert "vera_save_projection: false" in defaults
    assert "'lora_adapter_kind'" in layout
    assert "'dora_wd'" in layout
    assert "'use_loha'" not in layout
    assert "'use_lokr'" not in layout
    assert "'use_glora'" not in layout
    assert "'use_vera'" not in layout
    assert "keys: ['network_dim', 'network_alpha', 'lora_adapter_kind', 'dora_wd', 'lokr_factor', 'vera_projection_prng_key', 'vera_d_initial', 'vera_save_projection']" in source
    assert "{ family: 'lokr', key: 'lokr_factor_group_size', arg: 'lokr_factor_group_size', default: 8, valueType: 'integer' }" in network_arg_specs
    assert "{ family: 'lokr', key: 'lokr_project_chunk_bytes', arg: 'lokr_project_chunk_bytes', default: 4194304, valueType: 'integer' }" in network_arg_specs
    assert "'dora_wd'" not in merged_fields
    assert "'use_glora'" in merged_fields
    assert "'use_loha'" in merged_fields
    assert "'use_lokr'" in merged_fields
    assert "'use_vera'" in merged_fields
    assert "CONFIG_FORM_MERGED_FIELDS?.has?.(key)" in render_section
    assert "function loraAdapterFlagsForKind" in source
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


def test_config_actions_are_de_noised_and_sticky_controls_are_wired() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    listener_section = _section(source, "function setupEventListeners", "function installBeginnerTooltips")

    assert "toml-more-actions" in html
    assert "toml-more-actions-popover" in html
    assert "toml-secondary-actions" not in html
    assert "btn-apply-toml" in html
    assert "btn-toggle-toml-editor" in html
    assert ".toml-more-actions-popover" in css
    toml_current_row_css = _section(css, "#tab-config .toml-file-row {", "#tab-config .toml-current-file {")
    toml_current_file_css = _section(css, "#tab-config .toml-current-file {", "#tab-config .toml-current-file span")
    assert "overflow: visible;" in toml_current_row_css
    assert "z-index: 65;" in toml_current_row_css
    assert "padding: 0 1rem 0.56rem calc(1rem + 4px);" in toml_current_row_css
    assert "position: relative;" in toml_current_file_css
    assert "overflow: visible;" in toml_current_file_css
    assert "box-sizing: border-box;" in toml_current_file_css
    assert "min-width: 0;" in toml_current_file_css
    assert "padding: 0.42rem 0.54rem 0.42rem 0.68rem;" in toml_current_file_css
    assert "box-shadow: inset 3px 0 0 var(--config-accent);" in toml_current_file_css
    assert "grid-template-columns: 1.42rem minmax(0, 1fr);" in css
    assert "width: 1.32rem;" in _section(css, "#tab-config .file-group-drag-handle", "#tab-config .toml-file-group summary")
    assert "min-height: 42px;" in _section(css, "#tab-config .toml-file-item", "#tab-config .toml-file-item:hover")
    toml_actions_css = _section(css, "#tab-config .toml-file-actions", "#tab-config .toml-primary-actions,")
    toml_more_css = _section(css, "#tab-config .toml-more-actions {", "#tab-config .toml-more-actions > summary")
    toml_more_open_css = _section(css, "#tab-config .toml-more-actions[open]", "#tab-config .toml-more-actions > summary")
    toml_popover_css = _section(css, "#tab-config .toml-more-actions-popover", "#tab-config #toml-status")
    assert "overflow: visible;" in toml_actions_css
    assert "z-index: 70;" in toml_actions_css
    assert "position: relative;" in toml_more_css
    assert "z-index: 220;" in toml_more_open_css
    assert "z-index: 220;" in toml_popover_css
    assert "opacity: 0;" in _section(css, ".toml-group-actions", ".toml-group-action-btn,")

    assert "config-sticky-actions" in html
    assert "配置目录" in html
    assert "data-sticky-config-category=\"required\"" in html
    assert "data-sticky-config-category=\"common\"" in html
    assert "data-sticky-config-category=\"preview\"" in html
    assert "data-sticky-config-category=\"optimization\"" in html
    assert "data-sticky-config-category=\"advanced\"" in html
    assert "btn-sticky-config-optimization" in html
    assert "优化</strong>" in html
    assert "btn-sticky-config-advanced" in html
    assert "高级</strong>" in html
    assert "btn-sticky-save-config" not in html
    assert "btn-sticky-start-from-config" not in html
    assert "btn-sticky-queue-from-config" not in html
    assert "data-sticky-config-category" in listener_section
    assert "selectConfigCategory(btn.dataset.stickyConfigCategory, { scrollToForm: true })" in listener_section
    assert "function updateConfigStickyDirectory" in source
    assert "STICKY_CONFIG_CATEGORY_IDS" in source
    assert "new Set(['required', 'common', 'preview', 'optimization', 'advanced'])" in source
    assert "ADVANCED_CATEGORY_DEFAULT_OPEN_GROUPS" in source
    assert "new Set(['缓存与预处理'])" in source
    assert "if (category?.advanced) {" in source
    assert "configFormState.showAdvanced = true;" in source
    assert "(visibleCategories.has(categoryId) || (category.advanced && hasFields))" in source
    assert "configFormState.activeCategory === 'advanced' && ADVANCED_CATEGORY_DEFAULT_OPEN_GROUPS.has(name)" in source
    assert "function scrollConfigFormContentToTop" in source
    assert "scroller.scrollTo({ top: 0, behavior });" in source
    assert "scrollIntoView" not in _section(source, "function selectConfigCategory", "function scrollConfigFormContentToTop")
    assert "function updateConfigStickyPlacement" in source
    assert "--config-sticky-left" in source
    assert "--config-sticky-width" in source
    assert "--config-sticky-safe-space" in source
    assert "--config-left-max-height" in source
    assert "stickyActions.hidden = nextMode !== 'project'" in source
    assert ".config-sticky-actions" in css
    assert ".config-sticky-tab" in css
    sticky_css = _section(css, "#tab-config .config-sticky-actions", "#tab-config .config-sticky-title")
    config_left_css = _section(css, "#tab-config .config-left", "#tab-config .config-direct-editor")
    assert "width: var(--config-sticky-width, min(1040px, calc(100vw - 2rem)));" in sticky_css
    assert "grid-template-columns: auto repeat(5, minmax(118px, 1fr));" in sticky_css
    assert "left: var(--config-sticky-left, 1rem);" in sticky_css
    assert "position: fixed;" in sticky_css
    assert "position: sticky;" not in sticky_css
    assert "max-height: var(--config-left-max-height);" in config_left_css
    assert "overflow-y: auto;" in config_left_css
    assert "overscroll-behavior: contain;" in config_left_css
    assert "padding-bottom: var(--config-sticky-safe-space);" in config_left_css
    assert "min-height: 52px;" in _section(css, "#tab-config .config-sticky-tab", "#tab-config .config-sticky-tab:hover")


def test_resume_queue_button_is_wired() -> None:
    legacy_source = _anima_app_container_text()
    resume_source = _frontend_module_text("js/features/history-detail/resume/panel.js")

    resume_section = _section(resume_source, "function renderResumePanelState", "return { renderResumePanelState")
    history_detail_deps = _section(legacy_source, "function ensureHistoryDetailFeature", "// ── 初始化 ──")
    assert "btn-queue-resume-training" in resume_section
    assert "queueBtn.disabled" in resume_section
    assert "selected.resume_available !== false" in resume_section
    assert "resumeCheckpointRemainingText(selected)" in resume_section
    assert "deps.shouldRenderInlineResumePanel?.() !== true" in resume_section
    assert "resetInlineResumePanel(panel, select, btn, queueBtn, summary, status);" in resume_section
    assert "syncHistoryDetailResumeContent();" in resume_section
    assert "shouldRenderInlineResumePanel" in history_detail_deps

    listener_section = _section(legacy_source, "function setupEventListeners", "function installBeginnerTooltips")
    assert "queueResumeTrainingFromCheckpoint" in legacy_source
    assert "btn-queue-resume-training" in listener_section


def test_sample_prompts_save_uses_current_training_config_context() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    body = _section(source, "async function saveSamplePrompts", "async function importTomlFile")
    prepare_body = _section(source, "async function prepareFormPatchValues", "function shouldSkipUiDefaultField")

    assert "train_config_file: currentTrainingSource.file || currentTomlFile || ''" in body
    assert "await saveSamplePrompts('');" not in prepare_body


def test_config_form_save_reload_and_launch_share_training_config_file() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    save_patch = _section(source, "async function saveFormPatchToToml", "function updateTomlActionState")
    load_config = _section(source, "async function loadConfig", "async function reloadCurrentConfig")
    load_steps = _section(source, "async function loadStepEstimate", "async function loadDatasetEditor")
    run_preflight = _section(source, "async function runPreflight", "function isCliOnlySpdSource")
    start_unchecked = _section(source, "async function startTrainingUnchecked", "async function enqueueTrainingFromConfig")
    current_file = _section(source, "function currentTrainingConfigFile", "function preflightPlainText")

    assert "body: JSON.stringify({ file, values: preparedValues, content })" in save_patch
    assert "await loadConfig();" in save_patch
    assert "currentConfig = data;" in load_config
    assert "renderConfigForm(currentConfig);" in load_config
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
    assert "return currentTrainingSource.file || currentTomlFile || val('toml-file-select') || '';" in current_file


def test_config_training_source_modes_are_audited_before_launch() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    legacy_source = _anima_app_container_text()
    config_source = _frontend_module_text("js/features/anima-app/chunks/37-config-training-source.js")
    action_state = _frontend_module_text("js/features/anima-app/chunks/22-update-toml-action-state.js")
    launch_source = _frontend_module_text("js/features/anima-app/chunks/23-move-current-toml-to-group.js")
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

    for snippet in (
        "mode: 'fresh'",
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


def test_optional_number_fields_can_be_cleared() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    optional_numbers = _section(source, "const OPTIONAL_EMPTY_NUMBER_FIELDS = new Set([", "const FORM_UI_PERSIST_DEFAULT_FIELDS")
    reader = _section(source, "function readFieldInputValue", "function readLoKrEnabled")

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


def test_history_list_marks_queue_tasks() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    queue_label = _section(source, "function historyQueueLabel", "function historyContinueLabel")
    task_item = _section(source, "function createHistoryTaskItem", "function createHistoryActionButton")

    assert "来自队列" in queue_label
    assert "queue_attempt" in queue_label
    assert "historyQueueLabel(task)" in task_item


def test_history_manager_frontend_hooks_are_present() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    legacy_source = _anima_app_container_text()
    chart_source = CHART_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")
    preview_index = _frontend_module_text("js/features/preview/index.js")
    preview_workspace = _frontend_module_text("js/features/preview/workspace.js")
    preview_state = _frontend_module_text("js/features/preview/state.js")
    history_detail_index = _frontend_module_text("js/features/history-detail/index.js")
    history_detail_api = _frontend_module_text("js/features/history-detail/api.js")
    history_detail_dialog = _frontend_module_text("js/features/history-detail/dialog.js")
    history_detail_state = _frontend_module_text("js/features/history-detail/state.js")
    history_detail_workspace = _frontend_module_text("js/features/history-detail/workspace.js")
    history_detail_source = _frontend_feature_text(
        "js/features/history-detail/index.js",
        "js/features/history-detail/api.js",
        "js/features/history-detail/dialog.js",
        "js/features/history-detail/overview.js",
        "js/features/history-detail/resume/index.js",
        "js/features/history-detail/resume/state.js",
        "js/features/history-detail/resume/panel.js",
        "js/features/history-detail/resume/detail.js",
        "js/features/history-detail/resume/actions.js",
        "js/features/history-detail/analysis.js",
        "js/features/history-detail/curve/index.js",
        "js/features/history-detail/curve/data.js",
        "js/features/history-detail/curve/toolbar.js",
        "js/features/history-detail/curve/chart.js",
        "js/features/history-detail/curve/hover.js",
        "js/features/history-detail/system.js",
        "js/features/history-detail/logs.js",
        "js/features/history-detail/config-files.js",
        "js/features/history-detail/workspace.js",
        "js/features/history-detail/ui.js",
    )
    tabs_source = _frontend_module_text("js/features/app-shell/tabs.js")
    history_curve_chart = _frontend_module_text("js/features/history-detail/curve/chart.js")

    history_section = _section(legacy_source, "async function loadTrainingHistoryList()", "function groupHistoryTasks")
    detail_section = history_detail_source
    listener_section = _section(legacy_source, "function setupEventListeners", "function installBeginnerTooltips")
    preview_open_section = _section(preview_index, "async function openTrainingPreview", "function openCurrentTrainingPreview")
    tab_setup_section = _section(tabs_source, "function setupTabs()", "return {")
    sidebar_history_section = _section(legacy_source, "function renderTrainingHistoryList()", "function recentTrainingSidebarTasks")
    recent_sidebar_section = _section(legacy_source, "function recentTrainingSidebarTasks()", "function renderHistoryManager")
    log_append_section = _section(legacy_source, "function appendLogRecord", "async function replayTrainingLogs")
    history_review_mode_section = _section(legacy_source, "function isHistoryReviewMode()", "function openTutorialDialog")
    sidebar_task_item_section = _section(legacy_source, "function createHistoryTaskItem", "function createHistoryActionButton")
    manager_row_section = _section(legacy_source, "function createHistoryManagerRow", "function selectedHistoryConfigGroups")
    collection_card_section = _section(legacy_source, "function createHistoryCollectionWorkbenchCard", "function createHistoryConfigGroupWorkbenchCard")
    config_card_section = _section(legacy_source, "function createHistoryConfigGroupWorkbenchCard", "function historyCollectionNamesForTasks")
    load_task_section = _section(history_detail_index, "async function loadHistoryTask", "function clearHistoryDetailState")
    chart_controls_section = _section(html, '<div class="live-chart-controls"', '<label class="live-chart-field">')
    monitor_view_section = _section(html, '<section id="training-monitor-view"', '<!-- 预览结果工作区')

    assert "training-history-manager" in html
    assert "history-manager-search" in html
    assert "history-collection-search" in html
    assert "history-config-group-search" in html
    assert "集合搜索" in html
    assert "配置组搜索" in html
    assert "history-group-mode" not in html
    assert "集合分组" not in html
    assert "集合管理" in html
    assert '<option value="config">配置分组</option>' not in html
    assert '<option value="flat">平铺列表</option>' not in html
    assert "btn-history-collections-workbench" in html
    assert "btn-preview-training-results" in html
    assert "当前预览" in html
    assert "btn-live-sampling-preview" in html
    assert "途中采样" in html
    assert chart_controls_section.index("btn-live-sampling-preview") < chart_controls_section.index("live-chart-toggle-lr")
    assert monitor_view_section.index('id="history-config-panel"') < monitor_view_section.index('id="history-resume-panel"')
    assert monitor_view_section.index('id="history-resume-panel"') < monitor_view_section.index('class="panel log-panel"')
    assert "预览结果" in html
    assert 'data-tab="preview"' not in html
    assert 'class="preview-workspace-host" hidden aria-hidden="true"' in html
    assert "preview-page-mount" in html
    assert "preview-workspace" in html
    assert "preview-panel-dialog" in html
    assert "preview-dialog-mount" in html
    assert "btn-close-preview-panel" in html
    assert "training-dashboard" in html
    assert "training-run-state" in html
    assert "training-run-summary" in html
    assert "metric-vram-peak" in html
    assert "metric-gpu-peak" in html
    assert "metric-temp" in html
    assert "metric-temp-peak" in html
    assert "metric-eta" in html
    assert "预计完成" in html
    assert "最近训练" in html
    assert "未归档 · 最新 6 个训练任务" in html
    assert "btn-open-history-manager" in html
    assert 'type="module" src="/static/app.js?v=' in html
    assert "import { MetricsChart } from './chart.js?v=module-bootstrap-20260625-" in source
    assert "style.css?v=" in html
    assert "app.js?v=" in html
    assert "history-bulk-bar" in html
    assert "history-bulk-primary-actions" in html
    assert "归档已选" in html
    assert "btn-history-bulk-delete" in html
    assert "设置集合" in html
    assert 'id="history-detail-panel"' not in html
    assert 'id="history-detail-dialog"' in html
    assert "history-detail-dialog-shell" in html
    assert "btn-close-history-detail" in html
    assert "history-detail-tabs" in html
    assert "HISTORY FORGE" in html
    assert "function renderHistoryDetailTabs" in history_detail_dialog
    assert "btn.dataset.historyDetailTab = item.key" in history_detail_dialog
    assert "function historyDetailTabsForPayload" in history_detail_dialog
    assert "task?.job === 'preprocess'" in history_detail_dialog
    assert "['overview', 'logs', 'config_files'].includes(item.key)" in history_detail_dialog
    assert "normalizeVisibleHistoryDetailTab(payload, state.detailTab)" in history_detail_dialog
    assert "mainTaskReturn: null" in history_detail_state
    assert "linked_preprocess_task" in history_detail_dialog
    assert "查阅预处理" in history_detail_dialog
    assert "返回主项目" in history_detail_dialog
    assert "openLinkedPreprocessTask(task, preprocessTask)" in history_detail_dialog
    assert "loadHistoryTaskInDetail(preprocessTaskId, { detailTab: 'overview' })" in history_detail_dialog
    assert "loadHistoryTaskInDetail(target.taskId, { detailTab: target.detailTab || 'overview' })" in history_detail_dialog
    assert history_detail_dialog.index("查阅预处理") < history_detail_dialog.index("deps.createHistoryTaskPreviewButton(task)")
    assert "state.mainTaskReturn = null;" in load_task_section
    assert "{ key: 'overview', label: '概览' }" in history_detail_state
    assert "{ key: 'analysis', label: '训练分析' }" in history_detail_state
    assert "{ key: 'preview', label: '样张与权重' }" in history_detail_state
    assert "{ key: 'config_files', label: '配置与文件' }" in history_detail_state

    assert "renderHistoryManager()" in history_section
    assert "params.set('include_archived', '1');" in history_section
    assert "params.set('limit', '500')" not in history_section
    assert "recentTrainingSidebarTasks" in history_section
    assert "groupHistoryTasks(" not in sidebar_history_section
    assert "task.job === 'training' && !historyTaskIsArchived(task)" in recent_sidebar_section
    assert ".slice(0, 6)" in recent_sidebar_section
    assert "historyManagerFilteredTasks" in history_section
    assert "function historyManagerBaseFilteredTasks" in history_section
    assert "function historyManagerVisibleTasks" in history_section
    assert "normalizeHistoryGroupMode" not in source
    assert "const baseVisible = historyManagerBaseFilteredTasks();" in history_section
    assert "const visible = historyManagerVisibleTasks(baseVisible);" in history_section
    assert "mode === 'config'" not in history_section
    assert "mode === 'flat'" not in history_section
    assert "historyConfigGroupVisibleForSearch" not in source
    assert "uniqueHistoryTasks" in history_section
    assert "createHistoryManagerRow" in history_section
    assert "renderHistoryManagerGrouped" not in source
    assert "resetTrainingExpandedStateOnLeave" in history_section
    assert "collapseVisibleHistoryManagerGroups" not in source
    assert "collapsedHistoryCollections" not in source
    assert "collapsedHistoryConfigGroups" not in source
    assert "historyConfigGroupCollapseKey" not in source
    assert "expandHistoryCollectionConfigGroups" not in source
    assert "historyStatFilterIsActive" in history_section
    assert "archived: 'all'" in history_section
    assert "next.kind = state" in history_section
    assert "createHistoryManagerCollectionSection" not in source
    assert "createHistoryManagerConfigGroupSection" not in source
    assert "list.dataset.groupMode = 'collections';" in history_section
    assert "collection: selectedCollection" in history_section
    assert "history-current-group-content" in history_section
    assert "history-collection-nav" in history_section
    assert "HISTORY_UNGROUPED_COLLECTION_KEY" in source
    assert "selectedHistoryCollectionKey = HISTORY_UNGROUPED_COLLECTION_KEY" in source
    assert "未分类任务" in source
    assert "分组导航" in source
    assert "新建分组" in source
    assert "renderHistoryCollectionsWorkbench" in history_section
    assert "createHistoryCollectionWorkbenchCard" in history_section
    assert "createHistoryConfigGroupWorkbenchCard" in history_section
    assert "historyCollectionWorkbenchTarget" in source
    assert "historyCollectionSettings" in source
    assert "historyCollectionSearch" in source
    assert "historyConfigGroupSearch" in source
    assert "historyTaskMatchesCollectionSearch" in source
    assert "historyTaskMatchesConfigGroupSearch" not in source
    assert "historyCollectionMatchesSearch(collection, terms)" in source
    assert "selectedHistoryCollectionForWorkbench(collections, collectionSearchTerms)" in source
    assert "visibleHistoryCollectionsForSearch(allCollections, collectionSearchTerms)" in source
    assert "collectionSearchTerms.length ? candidates[0] : null" in source
    assert "createHistoryCollectionSearchEmptyCollection" in source
    assert "historySearchTextMatches(historyConfigGroupSearchText(group), configSearchTerms)" in source
    assert "document.getElementById('history-collection-search').addEventListener('input'" in source
    assert "document.getElementById('history-config-group-search').addEventListener('input'" in source
    assert "selectedHistoryCollectionKey" in source
    assert "/api/training/history/collections/settings" in source
    assert "loadHistoryCollectionSettings" in source
    assert "saveHistoryCollectionSettings" in source
    assert "collection_order" in source
    assert "config_group_order" in source
    assert "if (aIndex < 0) return -1;" in source
    assert "if (bIndex < 0) return 1;" in source
    assert "moveHistoryCollection" in source
    assert "moveHistoryConfigGroup" in source
    assert "reorderHistoryCollectionValue" in source
    assert "moveItemNearList" in source
    assert "moveHistoryCollection(collection, 'top', allCollections)" in collection_card_section
    assert "moveHistoryCollection(collection, 'bottom', allCollections)" in collection_card_section
    for direction in ("top", "up", "down", "bottom"):
        assert f"moveHistoryConfigGroup(group, '{direction}', options.groups, options.collection)" in config_card_section
    assert "applySelectedHistoryTasksToCollection" in source
    assert "groupHistoryTasks(scopedTasks)" in history_section
    assert "task?.group" in source
    assert "设置集合" in source
    assert "清除集合" in source
    assert "搜索或新建集合" in source
    assert "未分类" in source
    assert "selectedHistoryCollectionKey === collection.key ? '' : collection.key" not in source
    assert "selectedHistoryCollectionKey = collection.key" in source
    assert "collection.is_ungrouped ? '未分类' : '移入'" in collection_card_section
    assert "if (selectedTaskCount > 0) actions.append(joinSelectedBtn);" in collection_card_section
    assert "加入目标" in source
    assert "查看集合" not in source
    assert "取消查看" not in source
    assert "选择分组" in source
    assert "拖拽分组调整顺序" in collection_card_section
    assert "history-collection-drag-handle" in collection_card_section
    assert "beginHistoryCollectionDrag(event, collection)" in collection_card_section
    assert "dropHistoryCollectionToSort(event, collection, allCollections)" in collection_card_section
    for label in ("置顶", "上移", "下移", "置底"):
        assert f"createHistoryManagerGroupButton('{label}'" in collection_card_section
    for label in ("置顶", "上移", "下移", "置底"):
        assert f"createHistoryManagerGroupButton('{label}'" in config_card_section
    assert "合并查看" in source
    assert "查阅分组详情" in source
    assert "createHistoryActionButton('配置'" in source
    assert "createHistoryTaskConfigButton(task)" in manager_row_section
    assert "createHistoryMoreActions([" in manager_row_section
    assert "compactHistoryPathLabel" in source
    assert "historyCompactGroupMetaParts" in source
    assert "history-compact-meta" in source
    assert "history-more-actions" in source
    assert "loadHistoryTask(task.id, { detailTab: 'config_files' })" in source
    assert "if (options.detailTab)" in source
    assert "normalizeHistoryDetailTab(options.detailTab)" in source
    assert "任务预览" in source
    assert "分组预览" in source
    assert "只查看这一次训练任务的样张和权重" in source
    assert "汇总查看这个配置分组下所有训练任务的样张和权重" in source
    assert "createHistoryConfigGroupMergeButton" in source
    assert "createHistoryConfigGroupPreviewButton" in source
    assert "loadConfigGroupTimeline(group, { skipSelectionDialog: true })" in source
    assert "查阅这个自动配置分组内的训练日志、Loss 曲线和任务明细" in source
    assert "openHistoryConfigGroupPreview(group)" in source
    assert "loadConfigGroupTimeline(group, { skipSelectionDialog: true, detailTab: 'preview' })" in source
    assert "loadHistoryTask(task.id, { detailTab: 'preview' })" in source
    assert "mountPreviewWorkspaceInHistoryDetail" in source
    assert "restorePreviewWorkspaceFromHistoryDetail" in source
    assert "history-detail-preview-mount" in history_detail_workspace
    assert "canPreviewHistoryConfigGroup" in source
    assert "normalizePreviewGroup" in preview_state
    assert "state.selectedGroup = normalizePreviewGroup(options.group)" in preview_index
    assert "openPreviewPanel" in source
    assert "closePreviewPanel" in source
    assert "mountPreviewWorkspaceInDialog" in preview_workspace
    assert "mountPreviewWorkspaceInPage" in preview_workspace
    assert "workspace.openPreviewPanel();" in preview_open_section
    assert "document.querySelector('[data-tab=\"preview\"]')?.click()" not in preview_open_section
    assert "if (nextTab === 'preview')" not in tab_setup_section
    assert "mountPreviewWorkspaceInPage();" not in tab_setup_section
    assert "btn-preview-training-results" in listener_section
    assert "btn-live-sampling-preview" in listener_section
    assert "openCurrentTrainingPreview" in source
    assert "openLiveSamplingPreview" in source
    assert "const historyTaskId = deps.getTrainingViewMode() === 'live'" in preview_index
    assert "state.selectedTaskId = historyTaskId;" in preview_index
    assert "getViewingHistoryTaskId: () => viewingHistoryTaskId" in source
    assert "event?.preventDefault?.()" in source
    assert "event?.stopPropagation?.()" in source
    assert "addEventListener('click', openCurrentTrainingPreview)" in listener_section
    assert "addEventListener('click', openLiveSamplingPreview)" in listener_section
    assert "chooseTimelineTasksForMerge" not in source
    assert "showTimelineTaskSelectionDialog" not in source
    assert "选择要合并查看的训练分组" not in source
    assert "选择合并查看" not in source
    assert "分布在 ${split.size} 个分组" in source
    assert "selectedHistoryTaskIds" in source
    assert "applyHistoryBatchAction" in source
    assert "deleteHistoryTasksThorough" in source
    assert "confirmed: true" in source
    assert "confirm_text: confirmText" not in source
    assert "const confirmText = '彻底删除';" not in source
    assert "title: '确认要删吗'" in source
    assert "confirmText: '确认要删吗'" in source
    assert "输入“彻底删除”确认" not in source
    assert "彻底删除" in source
    assert "runtime_cleanup_errors" in source
    assert "历史记录已删除，部分文件未清理" in source
    assert "detailLines: cleanupErrors" in source
    assert "/api/training/history/batch" in source
    assert "openHistoryDetailDialog" in source
    assert "closeHistoryDetailDialog" in source
    assert "createHistoryDetailFeature(ctx, {" in legacy_source
    for name in (
        "loadHistoryTask",
        "renderHistoryManagerDetail",
        "renderHistoryDetailDialog",
        "closeHistoryDetailDialog",
        "isHistoryDetailDialogOpen",
        "handleHistoryDetailWindowKeydown",
        "loadResumeOptionsForTask",
        "clearResumeOptions",
        "renderResumePanelState",
        "selectedResumeCheckpoint",
        "resumeTrainingFromCheckpoint",
        "selectedHistoryManagerResumeCheckpoint",
        "resumeTrainingFromHistoryDetail",
        "setResumeStatus",
        "getCurrentPayload",
        "getActiveTab",
        "setActiveTab",
    ):
        assert name in history_detail_index
    assert "fetchHistoryTask(ctx, taskId)" in history_detail_index
    assert "/api/training/history/${encodeURIComponent(taskId)}" in history_detail_api
    assert "/api/training/history/${encodeURIComponent(taskId)}/resume-options" in history_detail_api
    assert "/api/preview/weights?task_id=" in history_detail_api
    assert "/api/training/continue-lora/inspect" in history_detail_api
    assert "/api/training/queue/resume" in history_detail_api
    assert "/api/training/resume" in history_detail_api
    assert "inspectContinueLoraWeight: (path) => (" in legacy_source
    assert "正在审查可热启动权重..." in history_detail_source
    assert "reviewHistoryResumeWeights(rawWeights)" in history_detail_source
    assert "inspectHistoryResumeWeight(weightPath)" in history_detail_source
    assert "return historyViewMode !== 'live';" in history_review_mode_section
    assert "Boolean(viewingHistoryTaskId)" not in history_review_mode_section
    assert "main.addEventListener('click', () => openSidebarHistoryTask(task.id))" in sidebar_task_item_section
    assert "createHistoryTaskPreviewButton(task)" in sidebar_task_item_section
    assert "createHistoryActionButton('查看', () => openSidebarHistoryTask(task.id))" in sidebar_task_item_section
    assert "renameHistoryTask(task)" not in sidebar_task_item_section
    assert "archiveHistoryTask(task)" not in sidebar_task_item_section
    assert "deleteHistoryTask(task)" not in sidebar_task_item_section
    assert "main.addEventListener('click', () => loadHistoryTask(task.id))" in manager_row_section
    assert "createHistoryActionButton('查看', () => loadHistoryTask(task.id))" in manager_row_section
    assert "function openSidebarHistoryTask" in source
    assert "renderHistoryTask(payload);" in source
    assert "historyViewMode = 'task';" in source
    assert "await openSidebarHistoryTask(viewingHistoryTaskId);" in source
    assert "showTrainingView('history')" not in load_task_section
    assert "renderHistoryTask(payload)" not in load_task_section
    assert "deps.setViewingHistoryTaskContext({" in load_task_section
    assert "task: payload.task || null" in load_task_section
    assert "dialog.setResumeLoadingForTask(taskId);" in load_task_section
    assert "dialog.renderHistoryManagerDetail(payload, { open: true })" in load_task_section
    assert "await dialog.loadResumeOptionsForTask(taskId);" in load_task_section
    assert "deps.clearViewingHistoryTaskContext?.(state.currentPayload);" in detail_section
    assert "function clearViewingHistoryTaskContext" in source
    assert "currentHistoryTaskForResume = null;" in _section(source, "function clearViewingHistoryTaskContext", "function handleHistoryDetailWindowKeydown")

    assert "renderHistoryDetailDialog" in detail_section
    assert "renderHistoryDetailOverview" in detail_section
    assert "renderHistoryDetailAnalysis" in detail_section
    assert "renderHistoryDetailResume" in detail_section
    assert "renderHistoryDetailChart" in detail_section
    assert "renderHistoryDetailLogs" in detail_section
    assert "renderHistoryDetailSystem" in detail_section
    assert "renderHistoryDetailConfig" in detail_section
    assert "renderHistoryDetailPaths" in detail_section
    assert "renderHistoryDetailConfigFiles" in detail_section
    assert "renderHistoryDetailPathSummary" in detail_section
    assert "historyCurveState" in detail_section
    assert "renderHistoryCurveStats" in detail_section
    assert "renderHistoryCurveToolbar" in detail_section
    assert "renderHistoryCurveMainChart" in detail_section
    assert "createHistoryCurveSvg" in detail_section
    assert "renderHistoryCurveInspector" in detail_section
    assert "renderHistoryCurveSegments" in detail_section
    assert "function historyCurveMetric(" not in detail_section
    assert "曲线指标" not in detail_section
    assert "HISTORY_CURVE_METRICS" in detail_section
    assert "historyCurvePointHasAnyMetric" in detail_section
    assert "historyCurveRawPointHasAnyMetric" in detail_section
    assert ".filter(historyCurveRawPointHasAnyMetric)" in detail_section
    assert ".map(historyCurveNormalizeRawMetricPoint)" in detail_section
    assert "historyCurveMetricStats" in detail_section
    assert "historyCurveMetricRange" in detail_section
    assert "appendHistoryCurveLineSegments" in detail_section
    assert "renderHistoryCurveLegend" not in detail_section
    assert "drawHistoryCurveMetricPoints" in detail_section
    assert "historyCurveStatsWithHover" in detail_section
    assert "updateHistoryCurveHoverLayer" in detail_section
    assert "renderHistoryCurveInspectorRows" in detail_section
    assert "requestAnimationFrame" in detail_section
    assert "scheduleHoverStep" in detail_section
    assert "renderHistoryDetailContent();" not in _section(history_curve_chart, "function createHistoryCurveSvg", "function renderHistoryCurveSegments")
    assert "dual-metric" in detail_section
    assert "history-curve-hover-layer" in detail_section
    assert "loss-axis" in detail_section
    assert "lr-axis" in detail_section
    assert "学习率点" in detail_section
    assert "最后有效学习率" in detail_section
    assert "峰值学习率" in detail_section
    assert "没有可绘制的 Loss 或学习率数据。" in detail_section
    assert "当前范围没有可绘制的 Loss 或学习率点。请调整范围筛选。" in detail_section
    assert "smoothLoss" in detail_section
    assert "smoothLr" in detail_section
    assert "formatSignedLr" in detail_section
    assert "historyCurveSmoothPoints" in detail_section
    assert "historyCurveFilteredPoints" in detail_section
    assert "historyCurveDisplayPoints" in detail_section
    assert "HISTORY_CURVE_RENDER_POINT_LIMIT" in detail_section
    assert "绘图已降采样" in detail_section
    assert "stageBreakBefore" in detail_section
    assert "display_step" in detail_section
    assert "平滑窗口" in detail_section
    assert "最近100点" in detail_section
    assert "最近25%" in detail_section
    assert "自定义 Step" in detail_section
    assert "box.appendChild(createHistorySparkline(lossPoints));" not in source
    assert "historySystemSummary" in detail_section
    assert "historySystemRecords" in detail_section
    assert "HISTORY_SYSTEM_TABLE_RENDER_LIMIT" in detail_section
    assert "historyDetailLimitNotice" in detail_section
    assert "仅显示最近" in detail_section
    assert "syncHistoryLogConsoleState" in detail_section
    assert "renderHistoryLogCommandCard" in detail_section
    assert "复制完整命令" in detail_section
    assert "搜索 Error、Epoch..." in detail_section
    assert "historyLogMatchesLevel" in detail_section
    assert "appendAnsiLogText" in detail_section
    assert "stripAnsiCodes" in detail_section
    assert "/logs/download" in detail_section
    assert "下载完整日志" in detail_section
    assert "最后 VRAM" in detail_section
    assert "峰值 GPU" in detail_section
    assert "无系统采样记录" in detail_section
    assert "system.jsonl" in detail_section
    assert "history-detail-metrics-body" in detail_section
    assert "task.job === 'preprocess'" in detail_section
    assert "renderPreprocessHistoryOverview(payload, box)" in detail_section
    assert "预处理摘要" in detail_section
    assert "预处理文件" in detail_section
    assert "history-preprocess-summary-body" in detail_section
    assert "history-preprocess-stat-grid" in detail_section
    assert "compactHistoryPathName(task.dataset_cache_dir || task.run_dir)" in detail_section
    assert "运行时数据集配置" in detail_section
    assert "日志目录" in detail_section
    assert "task.job === 'training'" in detail_section
    assert "historyDetailSection('任务信息'" not in detail_section
    assert "history-detail-section info" not in detail_section
    assert "loadHistoryResumeWeights" in detail_section
    assert "/api/preview/weights?task_id=" in history_detail_api
    assert "diagnostic" in detail_section
    assert "权重热启动" in detail_section
    assert "optimizer、scheduler 和已完成步数" in detail_section

    assert "btn-history-manager-refresh" in listener_section
    assert "btn-history-collections-workbench" in listener_section
    assert "btn-history-bulk-archive" in listener_section
    assert "btn-live-training" in listener_section
    assert "returnToLiveTraining" in listener_section
    assert "history-filter-kind" in listener_section
    assert "history-group-mode" not in listener_section
    assert "groupMode" not in listener_section
    assert "ensureHistoryDetailFeature().bindHistoryDetailEvents();" in listener_section
    assert "history-detail-tab" in history_detail_dialog
    assert "setHistoryDetailTab(state, btn.dataset.historyDetailTab)" in history_detail_dialog
    assert "btn-close-history-detail" in history_detail_dialog
    assert "logBuffer" in source
    assert "scheduleLogFlush" in log_append_section
    assert "requestAnimationFrame" in log_append_section
    assert "MAX_LOG_LINES" in log_append_section
    assert "lastLrText" not in source
    assert "recordLearningRateChange" not in source
    assert "announceLr" not in source
    assert "updatePointMetadata" in source
    assert "['Loss', formatLossValue(lossPoint.loss)]" in source
    assert "['平滑 Loss', formatLossValue(lossPoint.smoothLoss)]" in source
    assert "['学习率', formatLr(lrPoint.lr)]" in source
    assert "['平滑学习率', formatLr(lrPoint.smoothLr)]" in source
    assert "lr: item.lr" in source
    assert "peakVramUsedGb" in source
    assert "peakGpuUtil" in source
    assert "peakGpuTemp" in source
    assert "renderLiveTrainingDashboard" in source
    assert "function trainingEtaMetricInfo" in source
    assert "parseProgressRateSeconds(msg.rate)" in source
    assert "setEtaMetricText(trainingEtaMetricInfo());" in source
    assert "progressSecondsPerStep" in source
    assert "resetLiveSystemPeaks" in source

    assert ".training-workspace.main-wide" in css
    assert ".training-workspace.main-wide .training-sidebar" in css
    assert ".training-workspace.history-mode .training-main" in css
    assert "#tab-training .training-history-manager" in css
    training_history_css = _section(
        css,
        "#tab-training .training-history-manager {",
        "#tab-training .history-forge-eyebrow {",
    )
    assert '"head content"\n        "stats content"\n        "tools content"\n        "bulk content"\n        ". content"' in training_history_css
    assert "grid-template-rows: auto auto auto auto minmax(0, 1fr);" in training_history_css
    assert "#tab-training .history-manager-head {\n    grid-area: head;\n    align-self: start;" in css
    assert "#tab-training .history-manager-stats {\n    grid-area: stats;\n    align-self: start;" in css
    assert "#tab-training .history-manager-tools {\n    grid-area: tools;\n    align-self: start;" in css
    assert "#tab-training .history-bulk-bar {\n    grid-area: bulk;" in css
    assert "#tab-training .history-forge-eyebrow" in css
    assert ".training-workspace.history-wide" not in css
    assert ".training-dashboard-head" in css
    assert ".training-run-state" in css
    assert ".training-run-summary" in css
    assert ".metric-item-eta .metric-value" in css
    assert ".metric-icon-clock::before" in css
    assert ".history-curve-legend" not in css
    assert ".history-curve-legend-swatch" not in css
    assert ".history-curve-axis-label.loss-axis" in css
    assert ".history-curve-axis-label.lr-axis" in css
    assert ".history-curve-line.loss.smooth" in css
    assert ".history-curve-line.lr.smooth" in css
    assert ".history-curve-hover-layer" in css
    assert "pointer-events: none;" in css
    assert ".history-curve-svg.metric-lr .history-curve-line.smooth" not in css
    assert ".training-panels.training-dashboard" in css
    assert ".metrics-panel,\n.chart-panel" in css
    assert "grid-column: 1 / -1;" in css
    assert ".history-manager-grid" in css
    assert ".history-manager-row" in css
    assert ".history-manager-collection" not in css
    assert ".history-manager-config-group" not in css
    assert "--history-manager-bg" in css
    assert "--history-collection-bg" in css
    assert "--history-config-bg" in css
    assert "--history-row-bg" in css
    assert "--history-row-hover-bg" in css
    assert "--history-level-border" in css
    assert "--history-collection-accent" in css
    assert "--history-config-accent" in css
    assert "--history-ungrouped-accent" in css
    assert ".history-manager-collection.ungrouped" not in css
    assert "border-left: 5px solid var(--history-collection-accent)" not in css
    assert "border-left: 3px solid var(--history-config-accent)" not in css
    assert "background: var(--history-row-bg)" in css
    assert "background: var(--history-row-hover-bg)" in css
    assert ".history-collections-workbench.compact" in css
    assert ".history-compact-meta" in css
    assert ".history-more-actions" in css
    assert ".history-more-actions-menu" in css
    assert ".history-row-state.done" in css
    assert ".history-row-state.running" in css
    assert ".history-row-state.queued" in css
    assert ".history-row-state.interrupted" in css
    assert ".history-collections-workbench" in css
    assert "workbench.className = 'history-collections-workbench compact'" in history_section
    assert "`当前: ${selectedCollection.label}`" in history_section
    assert ".history-collections-body" in css
    assert ".history-collection-nav .history-collection-card" in css
    assert "max-height: min(520px, calc(100vh - 18rem));" in css
    assert ".history-manager-list[data-group-mode=\"collections\"]" in css
    assert "height: min(680px, calc(100vh - 14rem));" in css
    assert "height: calc(100vh - 176px);" in css
    assert "max-height: calc(100vh - 176px);" in css
    assert "--history-training-panel-head-height: 42px;" in css
    assert "#tab-training .history-collection-nav-head {" in css
    nav_head_css = _section(css, "#tab-training .history-collection-nav-head {", "#tab-training .history-collection-nav-head .history-collections-panel-title")
    assert "height: var(--history-training-panel-head-height);" in nav_head_css
    assert "min-height: var(--history-training-panel-head-height);" in nav_head_css
    assert "max-height: var(--history-training-panel-head-height);" in nav_head_css
    assert "padding: 0 0.72rem;" in nav_head_css
    nav_title_css = _section(css, "#tab-training .history-collection-nav-head .history-collections-panel-title", "#tab-training .history-collection-create-btn")
    assert "height: auto;" in nav_title_css
    assert "padding: 0;" in nav_title_css
    assert "background: transparent;" in nav_title_css
    assert "grid-template-rows: auto auto minmax(0, 1fr);" in css
    assert "align-items: stretch;" in css
    assert "overflow: hidden;" in css
    assert "overflow-y: scroll;" in css
    assert "scrollbar-gutter: stable;" in css
    assert "overscroll-behavior: contain;" in css
    assert ".history-collection-card" in css
    assert ".history-collection-card.active" in css
    assert ".history-collection-select-dialog" in css
    assert ".history-collection-select-list" in css
    assert ".history-config-group-card" in css
    assert ".history-config-group-task-list" in css
    assert ".history-manager-group-head" not in css
    assert ".history-manager-group-actions" not in css
    assert ".history-manager-stat.active" in css
    assert ".history-detail-dialog" in css
    assert ".history-detail-overview-dashboard" in css
    assert ".history-detail-overview-dashboard.preprocess-task" in css
    assert ".history-preprocess-stat-grid" in css
    assert ".history-preprocess-summary-body" in css
    assert "align-content: start;" in css
    assert ".history-detail-preview" in css
    assert ".history-detail-preview-mount" in css
    assert ".history-detail-progress" in css
    assert ".history-detail-metrics-body" in css
    assert ".history-detail-analysis" in css
    assert ".history-detail-config-files" in css
    assert ".history-curve-workbench" in css
    assert ".history-curve-toolbar" in css
    assert ".history-curve-svg" in css
    assert ".history-curve-inspector" in css
    assert ".history-curve-segment-line" in css
    assert ".history-detail-limit-note" in css
    assert ".history-command-card" in css
    assert ".history-log-console" in css
    assert ".history-log-toolbar" in css
    assert ".history-log-output.history-detail-pre" in css
    assert ".history-log-line.error" in css
    assert ".history-log-line.warning" in css
    assert ".history-log-line.progress" in css
    assert ".history-log-match.current" in css
    assert ".history-log-ansi-red" in css
    assert ".history-system-trends" in css
    assert ".history-system-table" in css
    assert ".history-detail-section.info" not in css
    assert ".preview-panel-dialog" in css
    assert ".preview-panel-body" in css
    assert ".preview-panel-dialog-sampling .preview-layout" in css
    assert ".preview-panel-dialog-sampling .preview-sidebar" in css
    assert "#tab-training .live-chart-sample-btn" in css

    assert "this.lrColor" in chart_source
    assert "_drawLrLine" in chart_source
    assert "updatePointMetadata" in chart_source
    assert "LR:" in chart_source
    assert "_formatLr" in chart_source
    assert "if (value === undefined || value === null || value === '') return '-';" in chart_source
    assert "if (value === undefined || value === null || value === '') return null;" in source


def test_history_collection_drag_drop_frontend_hooks_are_present() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    workbench = _section(source, "function renderHistoryCollectionsWorkbench", "function renderHistoryManagerStats")
    drag_helpers = _section(source, "function historyDragTaskIdsForGroup", "function createHistoryCollectionWorkbenchCard")
    collection_card = _section(source, "function createHistoryCollectionWorkbenchCard", "function createHistoryConfigGroupWorkbenchCard")
    config_card = _section(source, "function createHistoryConfigGroupWorkbenchCard", "function historyCollectionNamesForTasks")

    assert "globalThis.HISTORY_TASK_DRAG_MIME = 'application/x-anima-history-task-ids';" in source
    assert "globalThis.HISTORY_COLLECTION_DRAG_MIME = 'application/x-anima-history-collection';" in source
    assert "globalThis.HISTORY_CONFIG_GROUP_DRAG_MIME = 'application/x-anima-history-config-group';" in source
    assert "globalThis.historyDragState = {" in source
    for key in ("active: false", "taskIds: []", "sourceGroupKey: ''", "activeDropTarget: ''", "pending: false", "popover: {"):
        assert key in source
    assert "globalThis.historyCollectionDragState = {" in source
    for key in ("sourceValue: ''", "dropPosition: 'after'", "pending: false"):
        assert key in source
    assert "globalThis.historyConfigGroupSortState = {" in source
    for key in ("sourceKey: ''", "collectionKey: ''", "activeDropTarget: ''", "dropPosition: 'after'"):
        assert key in source
    assert "globalThis.historyConfigGroupPointerDrag = null;" in source
    assert "globalThis.historyConfigGroupDropPreviewElement = null;" in source
    assert "globalThis.historyCollectionPointerDrag = null;" in source
    assert "application/x-anima-history-task-ids" in source
    assert "application/x-anima-history-collection" in source
    assert "application/x-anima-history-config-group" in source
    assert "event.dataTransfer.setData(HISTORY_TASK_DRAG_MIME" in source
    assert "event.dataTransfer.setData(HISTORY_COLLECTION_DRAG_MIME" in source
    assert "event.dataTransfer.setData(HISTORY_CONFIG_GROUP_DRAG_MIME" in source
    assert "event.dataTransfer.setData('text/plain'" in source

    assert "collectionList.appendChild(createHistoryCollectionDropzone());" not in workbench
    assert "renderHistoryDropPopover(workbench);" in workbench
    assert "createHistoryCollectionDropzone" not in source
    assert "history-collection-dropzone" not in source
    assert "createHistoryCollectionConfigChip" not in source
    assert "history-collection-config-chip" not in source
    assert "history-collection-create-btn" in workbench
    assert "openHistoryNewCollectionPopover(event, [])" in workbench
    assert "新建分组" in workbench
    assert "historyCompactGroupMetaParts" in source
    assert "history-compact-meta" in source
    assert "createHistoryMoreActions([" in source
    assert "renameHistoryCollection(collection)" in collection_card
    assert "clearHistoryCollection(collection)" in collection_card
    assert "setHistoryCollectionForTasks(collection.tasks, collection.value, collection.label)" not in collection_card
    assert "clearHistoryCollectionForTasks(collection.tasks, collection.label)" not in collection_card
    assert "renameHistoryCollectionOrderValue" in source
    assert "renameHistoryConfigGroupOrderKey" in source
    assert "removeHistoryCollectionSettingValue" in source
    assert "ids.length ? '清空集合' : '删除空集合'" in source

    assert "history-drag-handle" in config_card
    assert "history-config-group-card-head" in config_card
    assert "head.append(select, handle, main, actions)" in config_card
    assert "拖拽配置分组调整顺序或移到右侧分组" in config_card
    assert "history-config-group-drag-handle" in config_card
    assert "handle.draggable = true;" in config_card
    assert "handle.addEventListener('pointerdown'" in config_card
    assert "startHistoryConfigGroupPointerDrag(event, group, options, handle)" in config_card
    assert "startHistoryConfigGroupMouseDrag(event, group, options, handle)" in config_card
    assert "startHistoryConfigGroupTouchDrag(event, group, options, handle)" in config_card
    assert "beginHistoryConfigGroupDrag(event, group, options)" in config_card
    assert "finishHistoryDrag()" in config_card
    assert "historyConfigGroupOrderDragEnter(event, group, card, options)" in config_card
    assert "historyConfigGroupOrderDragLeave(event, group, card)" in config_card
    assert "dropHistoryConfigGroupToSort(event, group, options)" in config_card
    assert "reorderHistoryConfigGroupValue" in source
    assert "function ensureHistoryConfigGroupDropPreview" in drag_helpers
    assert "function placeHistoryConfigGroupDropPreview" in drag_helpers
    assert "function removeHistoryConfigGroupDropPreview" in drag_helpers
    assert "释放后插入到这里" in drag_helpers
    assert "placeHistoryConfigGroupDropPreview(element, historyConfigGroupSortState.dropPosition)" in drag_helpers
    assert "preview.style.top" in drag_helpers
    assert "parent.appendChild(preview)" in drag_helpers
    assert "event.relatedTarget.closest('.history-config-group-card-list')" in drag_helpers
    for label in ("置顶", "上移", "下移", "置底"):
        assert f"createHistoryManagerGroupButton('{label}'" in config_card
    for hook in (
        "function startHistoryConfigGroupPointerDrag",
        "function startHistoryConfigGroupMouseDrag",
        "function startHistoryConfigGroupTouchDrag",
        "function finishHistoryConfigGroupPointerDrag",
        "function historyConfigGroupPointerTargetFromPoint",
        "function historyCollectionDropTargetFromPoint",
        "document.addEventListener('pointermove', drag.onMove, { passive: false })",
        "document.addEventListener('pointerup', drag.onUp, { passive: false })",
        "document.addEventListener('pointercancel', drag.onCancel, { passive: false })",
        "document.addEventListener('mousemove', drag.onMouseMove, { passive: false })",
        "document.addEventListener('mouseup', drag.onMouseUp, { passive: false })",
        "document.addEventListener('touchmove', drag.onTouchMove, { passive: false })",
        "document.addEventListener('touchend', drag.onTouchEnd, { passive: false })",
        "document.addEventListener('touchcancel', drag.onTouchCancel, { passive: false })",
    ):
        assert hook in drag_helpers

    assert "history-collection-drag-handle" in collection_card
    assert "拖拽分组调整顺序" in collection_card
    assert "dragHandle.draggable = true;" in collection_card
    assert "dragHandle.addEventListener('pointerdown'" in collection_card
    assert "beginHistoryCollectionDrag(event, collection)" in collection_card
    assert "startHistoryCollectionPointerDrag(event, collection, allCollections, dragHandle)" in collection_card
    assert "finishHistoryCollectionDrag()" in collection_card
    assert "historyCollectionOrderDragEnter(event, collection, card)" in collection_card
    assert "dropHistoryCollectionToSort(event, collection, allCollections)" in collection_card
    assert "reorderHistoryCollectionValue(source, target, position, allCollections)" in source
    for label in ("置顶", "上移", "下移", "置底"):
        assert f"createHistoryManagerGroupButton('{label}'" in collection_card

    assert "dragenter" in collection_card
    assert "dragover" in collection_card
    assert "dragleave" in collection_card
    assert "dropHistoryTasksToCollection(event, collection.value || '', collection.label)" in collection_card
    assert "collection.value || '__ungrouped__'" in collection_card
    assert "applyHistoryTaskIdsToCollection(taskIds, clean, { clearSelection: true })" in drag_helpers
    assert "historyDraggedTasksAlreadyInCollection(taskIds, clean)" in drag_helpers
    assert "selectedHistoryCollectionKey = clean ? `collection:${clean}` : HISTORY_UNGROUPED_COLLECTION_KEY" in drag_helpers
    for hook in (
        "function startHistoryCollectionPointerDrag",
        "function startHistoryCollectionMouseDrag",
        "function startHistoryCollectionTouchDrag",
        "function finishHistoryCollectionPointerDrag",
        "function historyCollectionEventPoint",
        "function historyCollectionPointerTargetFromPoint",
        "function autoScrollHistoryCollectionPointerDrag",
        "document.addEventListener('pointermove', drag.onMove, { passive: false })",
        "document.addEventListener('pointerup', drag.onUp, { passive: false })",
        "document.addEventListener('pointercancel', drag.onCancel, { passive: false })",
        "document.addEventListener('mousemove', drag.onMouseMove, { passive: false })",
        "document.addEventListener('mouseup', drag.onMouseUp, { passive: false })",
        "document.addEventListener('touchmove', drag.onTouchMove, { passive: false })",
        "document.addEventListener('touchend', drag.onTouchEnd, { passive: false })",
        "document.addEventListener('touchcancel', drag.onTouchCancel, { passive: false })",
        "document.addEventListener('keydown', drag.onKeydown)",
    ):
        assert hook in drag_helpers
    assert "dragHandle.addEventListener('mousedown'" in collection_card
    assert "dragHandle.addEventListener('touchstart'" in collection_card
    assert "startHistoryCollectionMouseDrag(event, collection, allCollections, dragHandle)" in collection_card
    assert "startHistoryCollectionTouchDrag(event, collection, allCollections, dragHandle)" in collection_card

    assert "history-drop-popover" in source
    assert "event.key === 'Escape'" in source
    assert "event.key === 'Enter'" in source
    assert "input.maxLength = 48;" in source
    assert "state.taskIds.length ? `${state.taskIds.length} 条任务归入新分组` : '新建分组'" in source
    assert "defaultHistoryCollectionName" in source
    assert "uniqueHistoryCollectionName" in source

    assert "/api/training/history/batch" in source
    assert "/api/collections/create-and-assign" not in source
    assert "/api/tasks/assign-collection" not in source

    for selector in (
        ".history-config-group-card.draggable",
        ".history-config-group-card.config-sort-active",
        ".history-config-group-card.config-sort-source",
        ".history-config-group-card-head",
        ".history-config-group-drop-preview",
        ".history-config-group-pointer-drag-active",
        ".history-drag-handle",
        ".history-current-group-content",
        ".history-collection-nav",
        ".history-collection-drag-handle",
        ".history-collection-card.nav-card",
        ".history-collection-card.drop-active",
        ".history-collection-card.sort-active",
        ".history-collections-workbench.collection-reordering",
        ".history-collection-pointer-drag-active",
        ".history-collection-drag-image-pointer",
        ".history-collection-create-btn",
        ".history-collections-workbench.dragging",
        ".history-collections-workbench.compact",
        ".history-compact-meta",
        ".history-more-actions",
        ".history-drop-popover",
        "prefers-reduced-motion",
    ):
        assert selector in css
    assert ".history-config-group-card.selected > .history-config-group-card-actions" not in css
    assert "--history-config-group-select-width: 18px;" in css
    assert "--history-config-group-handle-width: 28px;" in css
    assert "grid-template-columns: var(--history-config-group-select-width) var(--history-config-group-handle-width) minmax(0, 1fr);" in css
    config_card_head_css = _section(
        css,
        ".history-config-group-card-head {",
        ".history-config-group-card.draggable {",
    )
    assert "min-height: 32px;" in config_card_head_css
    assert "width: 100%;" in config_card_head_css
    config_card_actions_css = _section(
        css,
        ".history-config-group-card-head > .history-config-group-card-actions {",
        ".history-config-group-card.single-task .history-config-group-card-head > .history-config-group-card-actions {",
    )
    assert "position: absolute;" in config_card_actions_css
    single_task_actions_css = _section(
        css,
        ".history-config-group-card.single-task .history-config-group-card-head > .history-config-group-card-actions {",
        ".history-config-group-card-main strong",
    )
    assert "position: absolute;" not in single_task_actions_css
    assert "transform: translateY(-50%);" in single_task_actions_css
    assert ".history-config-group-card:focus-within > .history-config-group-card-actions" not in css
    assert ".history-manager-row:focus-within .history-row-actions" not in css
    assert ".history-config-group-card-head > .history-config-group-card-actions:focus-within" in css
    assert ".history-row-actions:focus-within" in css
    config_group_select_css = _section(
        css,
        ".history-config-group-select {",
        ".history-config-group-task-list {",
    )
    assert "font-size: 0;" in config_group_select_css
    assert "gap: 0;" in config_group_select_css
    assert "justify-content: center;" in config_group_select_css
    assert ".history-config-group-select input" in css
    config_group_handle_css = _section(
        css,
        ".history-config-group-card-head > .history-drag-handle {",
        ".history-config-group-card.single-task .history-config-group-card-head > .history-drag-handle {",
    )
    assert "min-width: 0;" in config_group_handle_css
    assert "width: 100%;" in config_group_handle_css
    single_task_handle_css = _section(
        css,
        ".history-config-group-card.single-task .history-config-group-card-head > .history-drag-handle {",
        "\n.history-drag-handle {",
    )
    assert "min-width: 0;" in single_task_handle_css


def test_history_detail_overview_uses_full_copyable_paths_and_resume_weights() -> None:
    overview_source = _frontend_module_text("js/features/history-detail/overview.js")
    resume_source = _frontend_module_text("js/features/history-detail/resume/detail.js")
    ui_source = _frontend_module_text("js/features/history-detail/ui.js")
    css = STYLE_CSS.read_text(encoding="utf-8")

    overview = _section(overview_source, "function renderHistoryDetailOverview", "function renderHistoryDetailProgress")
    progress = _section(overview_source, "function renderHistoryDetailProgress", "function renderHistoryDetailPathSummary")
    path_summary = _section(overview_source, "function renderHistoryDetailPathSummary", "return { renderHistoryDetailOverview")
    curve_index_source = _frontend_module_text("js/features/history-detail/curve/index.js")
    resume = _section(resume_source, "function renderHistoryDetailResume", "function renderResumeDiagnosticBlock")
    weights = _section(resume_source, "function renderHistoryResumeWeightOptions", "function formatDiagnosticBool")
    row_helpers = _section(ui_source, "export function historyDetailRow", "export function historyDetailEmptyText")

    assert "icon.className = `metric-icon metric-icon-${iconName}`" in overview
    assert "['平均速度', formatHistoryAverageSpeed(task), 'gauge']" in overview
    assert "['训练总时间', formatHistoryTaskDuration(task), 'time']" in overview
    assert "function formatHistoryAverageSpeed(record)" in overview_source
    assert "function formatHistoryTaskDuration(record)" in overview_source
    assert "ctx.format.formatDuration" in overview_source
    assert "muted: taskFinished && ['队列', '续训'].includes(label) && value === '-'" in overview
    assert "section.classList.toggle('is-complete', finished);" in progress
    assert "historyCurveStatGroup('速度组'" in curve_index_source
    assert "['平均速度', formatHistoryAverageSpeed(task)]" in curve_index_source
    assert "['采样范围', formatAverageSpeedStepRange(task)]" in curve_index_source

    assert "historyDetailRunRoot(task)" in path_summary
    assert "return normalizedHistoryDetailPath(task.run_dir_abs || task.run_dir || '');" in ui_source
    assert "'运行根目录'" in path_summary
    assert "relativeHistoryDetailPath(value, rootPath)" not in path_summary
    assert "copyValue: value" in path_summary
    assert "export function relativeHistoryDetailPath" in ui_source
    assert "export function selectAllTextOnDoubleClick" in ui_source
    assert "range.selectNodeContents(el)" in ui_source
    assert "selectAllTextOnDoubleClick(val)" in row_helpers
    assert "row.appendChild(helpers.copyButton(options.copyValue" in row_helpers

    assert "controls.className = 'history-resume-control-row';" in resume
    assert "fullResume.append(controls, summary);" in resume
    assert "resumeCheckpointRemainingText(selected)" in resume
    assert "selected.resume_available !== false" in resume
    assert "resumeSummaryLine('不可用原因', selected.unavailable_reason)" in resume
    assert "resumeSummaryLine('步数估算', `无法确认剩余步数: ${selected.estimate_error}`)" in resume
    assert "checkpointWeightPaths.has(String(weightPath || '').trim())" in weights
    assert "item.inspect_status === 'ok'" in weights
    assert "item.inspect_compatible !== false" in weights
    assert "审查未通过" in weights
    assert "useBtn.disabled = !canUseWeightDirectly;" in weights
    assert "缺少对应的 checkpoint-state/train_state.json" in weights
    assert "name.textContent = fileNameFromPath(item.name || weightPath)" in weights
    assert "info.append(name, meta);" in weights
    assert "info.append(name, path, meta);" not in weights
    assert "history-resume-weight-actions" in weights
    assert "historyDetailCopyButton(weightPath" in weights

    assert ".history-detail-progress.is-complete .history-detail-progress-bar span" in css
    assert ".history-detail-stat .metric-icon-time::before" in css
    assert ".history-curve-stat-group.speed" in css
    assert ".history-detail-path-summary .history-detail-path-root" in css
    assert ".history-detail-select-all" in css
    assert ".history-detail-copy-btn" in css
    assert ".history-resume-control-row" in css
    assert ".history-resume-hint.warning" in css
    assert ".history-resume-weight-actions" in css


def test_history_detail_config_files_are_tool_ready() -> None:
    legacy_source = _anima_app_container_text()
    config_files_source = _frontend_module_text("js/features/history-detail/config-files.js")
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    config_files = _section(config_files_source, "function renderHistoryDetailConfig", "function renderHistoryDetailConfigFiles")
    path_items = _section(legacy_source, "function runtimePathItems", "function historyStateLabel")

    assert "history-config-viewer" in config_files
    assert "history-config-toolbar" in config_files
    assert "history-config-search" in config_files
    assert "renderHistoryConfigCode(pre, content, searchText, currentMatch)" in config_files
    assert "history-config-token-key" in config_files
    assert "history-config-token-path" in config_files
    assert "history-config-search-hit current" not in config_files
    assert "historyConfigMatchCount(content, searchText)" in config_files
    assert "downloadBlob(new Blob([content], { type: 'text/plain;charset=utf-8' }), filename)" in config_files
    assert "history-detail-file-browser" in config_files
    assert "history-file-root" in config_files
    assert "historyDetailFileRow(task, label, value, artifactKey)" in config_files
    assert "relativeHistoryDetailPath(rawValue, rootPath)" not in config_files
    assert "val.textContent = rawValue" in config_files
    assert "selectAllTextOnDoubleClick(val)" in config_files
    assert "deps.historyArtifactUrl(task, artifactKey)" in config_files
    assert "deps.historyArtifactUrl(task, artifactKey, { download: true })" in config_files
    assert "function makeHistoryArtifactUrl" in legacy_source
    assert "historyArtifactUrl: makeHistoryArtifactUrl" in legacy_source
    assert "choiceHelp, help" in _frontend_module_text("js/config/catalog.js")
    assert "Object.assign(globalThis, ctx.catalog);" in legacy_source

    for artifact in (
        "'runtime-config'",
        "'original-config'",
        "'dataset-config'",
        "'logs'",
        "'metrics'",
        "'system'",
        "'config-snapshot'",
    ):
        assert artifact in path_items
    assert "const runDir = absolutePath(task.run_dir_abs || task.run_dir)" in path_items
    assert "function historyAbsolutePath(value, task = {}, basePath = '')" in path_items
    assert "function historyProjectRoot(task = {})" in path_items
    assert "project_root_abs" in path_items

    assert "module-bootstrap-20260625-" in html
    for selector in (
        ".history-config-viewer",
        ".history-config-toolbar",
        ".history-config-code.history-detail-pre",
        ".history-config-token-key",
        ".history-config-token-path",
        ".history-detail-file-browser",
        ".history-file-root",
        ".history-file-row",
        ".history-file-actions",
        ".history-detail-icon-btn",
        ".history-detail-select-all",
    ):
        assert selector in css
    assert "text-overflow: ellipsis;" not in _section(
        css,
        ".history-detail-path-row code,",
        ".history-detail-select-all {",
    )
    assert ".history-detail-config-files > .history-detail-section" in css
    assert ".history-detail-kv > div" in css
    assert ".history-detail-kv div" not in css


def test_output_scope_group_does_not_expose_unwired_stage_resolution_dialog() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")

    section = _section(source, "title: '输出格式与训练范围'", "title: '方法内部与实验架构'")
    create_group = _section(source, "function createGroup", "function createOpenStageResolutionDialogButton")

    assert "className: 'config-group-output-scope'" in section
    assert "header.appendChild(createOpenStageResolutionDialogButton());" not in create_group
    assert 'id="stage-resolution-dialog"' not in html
    assert 'class="preview-dialog stage-resolution-dialog"' not in html
    assert "stage-resolution-dialog-body" not in html
    assert "btn-open-stage-resolution-dialog" not in _section(source, "function installBeginnerTooltips", "// ── 工具函数 ──")


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
    compatibility_section = _section(source, "function normalizeOptimizerType", "function loraAdapterFlagsMatchConfig")
    load_config_section = _section(source, "async function loadConfig()", "function syncConfigDraftFromForm")
    prepare_section = _section(source, "async function prepareFormPatchValues", "function shouldSkipUiDefaultField")

    assert "function normalizeCameOptimizerArgs(args)" in compatibility_section
    assert "function applyOptimizerCompatibilityPatch(values)" in compatibility_section
    assert "normalizeOptimizerType(optimizerType) !== 'came'" in compatibility_section
    assert "cameBetasNeedPatch(rawBetas)" in compatibility_section
    assert "result[betasIndex] = 'betas=0.9,0.999,0.9999';" in compatibility_section
    assert "const compatibilityPatch = applyConfigCompatibilityDrafts();" in load_config_section
    assert "function applyConfigCompatibilityDrafts()" in load_config_section
    assert "configFormState.draftValues.set(key, value);" in load_config_section
    assert "已自动修正 CAME optimizer_args 的 betas 格式" in load_config_section
    assert "const nextValues = applyOptimizerCompatibilityPatch(values);" in prepare_section


def test_config_catalog_exposes_automagic_and_constant_with_warmup_options() -> None:
    labels_options = _frontend_module_text("js/config/catalog/labels-options.js")
    field_help = _frontend_module_text("js/config/catalog/field-help-training.js")

    assert "lr_scheduler: ['constant', 'constant_with_warmup', 'cosine'" in labels_options
    assert "lr_warmup_steps: '预热步数'" in labels_options
    assert "optimizer_type: ['AdamW', 'CAME', 'Automagic'" in labels_options
    assert "Automagic 属于实验优化器" in field_help
    assert "constant_with_warmup 表示先线性热身再固定" in field_help
    assert "0.05 表示前 5% 的训练步数逐步升到目标学习率" in field_help


def test_balanced_16g_block_swap_fields_are_visible() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    form_layout = _frontend_module_text("js/config/catalog/form-layout.js")
    labels_options = _frontend_module_text("js/config/catalog/labels-options.js")
    guides = _frontend_module_text("js/config/catalog/guides.js")

    optimization_section = _section(form_layout, "title: '显存与速度优化'", "title: '缓存与预处理'")

    for key in (
        "blocks_to_swap",
        "selective_checkpoint",
        "selective_checkpoint_blocks",
        "block_swap_profile_jsonl",
        "memory_probe_jsonl",
        "memory_probe_max_steps",
        "peak_probe_jsonl",
        "peak_probe_max_steps",
        "peak_probe_level",
        "lr_warmup_steps",
        "lokr_factor_group_size",
        "lokr_project_chunk_bytes",
        "disable_block_swap_for_eval",
    ):
        assert f"'{key}'," in optimization_section

    assert "block_swap_profile_jsonl: '块交换 Profile'" in labels_options
    assert "memory_probe_jsonl: '显存探针'" in labels_options
    assert "memory_probe_max_steps: '探针步数'" in labels_options
    assert "peak_probe_jsonl: '峰值探针'" in labels_options
    assert "peak_probe_max_steps: '峰值探针步数'" in labels_options
    assert "peak_probe_level: '峰值探针粒度'" in labels_options
    assert "lr_warmup_steps: '预热步数'" in labels_options
    assert "lokr_factor_group_size: 'LoKr 分组'" in labels_options
    assert "lokr_project_chunk_bytes: 'LoKr 张量切块阈值'" in labels_options
    assert "use_glora: '启用 GLoRA'" in labels_options
    assert "use_vera: '启用 VeRA'" in labels_options
    assert "vera_projection_prng_key: 'VeRA 投影随机种子'" in labels_options
    assert "vera_d_initial: 'VeRA d 初始值'" in labels_options
    assert "vera_save_projection: '保存 VeRA 投影矩阵'" in labels_options
    assert "selective_checkpoint: '选择性重算'" in labels_options
    assert "selective_checkpoint_blocks: '定点重算块'" in labels_options
    assert "disable_block_swap_for_eval: '评估时暂停交换块'" in labels_options
    assert "selective_checkpoint: ['off', 'adapter_aware', 'peak_blocks_adapter_aware', 'mlp_layer1_only', 'peak_blocks_mlp_layer1', 'peak_blocks_mlp', 'mlp_only', 'every_other']" in labels_options
    assert "memory_probe_jsonl: ['off', 'auto']" in labels_options
    assert "peak_probe_jsonl: ['off', 'auto']" in labels_options
    assert "peak_probe_level: ['block', 'ops', 'lokr', 'full']" in labels_options
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
    numeric_field_section = _section(source, "function isNumericField", "function isIntegerNumericField")
    integer_field_section = _section(source, "function isIntegerNumericField", "function allowsNegativeNumberField")
    negative_field_section = _section(source, "function allowsNegativeNumberField", "function createSelectInput")
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


def test_config_workbench_manager_is_right_column() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    layout = _section(css, "#tab-config .config-forge-layout", "#tab-config .config-preset-manager,")
    manager = _section(css, "#tab-config .config-preset-manager {", "#tab-config .config-sidebar-project")
    editor = _section(css, "#tab-config .config-preset-editor {", "#tab-config .config-preset-header")
    compact = _section(css, "@media (max-width: 900px)", "@media (max-width: 520px)")
    phone = _section(css, "@media (max-width: 520px)", "@media (max-width: 640px)")

    assert "左侧训练配置工作台 + 右侧配置预设管理" in html
    assert "grid-template-columns: minmax(0, 1fr) clamp(260px, 24vw, 360px);" in layout
    assert "grid-column: 2;" in manager
    assert "isolation: isolate;" in manager
    assert "z-index: 60;" in manager
    assert "overflow: visible;" in manager
    assert "border-left: 1px solid var(--config-border-strong);" in manager
    assert "grid-column: 1;" in editor
    assert "grid-template-columns: minmax(0, 1fr) minmax(260px, 300px);" in compact
    assert "grid-template-columns: 1fr;" in phone


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
    source = APP_JS.read_text(encoding="utf-8")
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
    assert "释放后插入到这里" in drag_helpers
    assert "clearFileGroupDropIndicators({ keepPreview: true })" in drag_helpers
    assert "placeFileGroupDropPreview(node, position)" in drag_helpers
    assert "handle.addEventListener('pointerdown'" in handle_body
    assert "handle.addEventListener('mousedown'" in handle_body
    assert "if (fileGroupPointerDrag) {" in handle_body
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
    assert "fileGroupActiveDropTargetNode === node && fileGroupActiveDropPosition === normalizedPosition" in source

    assert drop_targets.count("registerFileGroupDropTarget") == 4
    assert "position: 'inside'" in drop_targets
    assert "configFileDropIndex(group, targetFile, placeAfter, payload.file)" in drop_targets
    assert "configGroupDropIndex(options.getSortableGroups(), group.id, placeAfter, payload.groupId)" in drop_targets

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
    api_helpers = _section(source, "async function api(url, opts = {})", "function val(id)")
    listener_section = _section(source, "function setupEventListeners", "function installBeginnerTooltips")

    assert "classList.contains('active')" in tab_active
    assert "closest('#tab-datasets')" not in tab_active
    assert "DATASET_PRESET_REQUEST_TIMEOUT_MS" in source
    assert "async function datasetPresetApi" in api_helpers
    assert "数据集预设请求超时" in api_helpers
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
    assert "btn-config-dataset-dialog-refresh').addEventListener('click', () => loadDatasetPresets({ selectCurrent: false, manage: false }))" in listener_section
    assert "const params = new URLSearchParams({" in load_editor
    assert "const configFile = currentTrainingConfigFile();" in load_editor
    assert "params.set('config_file', configFile);" in load_editor
    assert "function selectedDatasetConfigOverride" in source
    assert "const datasetConfigOverride = selectedDatasetConfigOverride();" in load_editor
    assert "if (datasetConfigOverride !== null) params.set('dataset_config', datasetConfigOverride);" in load_editor
    assert "params.set('dataset_config', selectedConfigDatasetFile);" not in load_editor
    assert "datasetConfig = selectedConfigDatasetFile || currentConfig.dataset_config" not in load_editor
    assert "api(`/api/config/datasets?${params.toString()}`)" in load_editor


def test_config_toml_manager_excludes_dataset_groups() -> None:
    source = APP_JS.read_text(encoding="utf-8")
    css = STYLE_CSS.read_text(encoding="utf-8")

    load_toml = _section(source, "async function loadTomlFileList", "async function loadOutputRuns")
    toml_render = _section(source, "function renderTomlFileGroups", "function createTomlGroupActions")
    file_button = _section(source, "function createTomlFileButton", "function updateTomlSelectionUI")
    save_as_groups = _section(source, "function saveAsTargetGroups", "async function moveTomlFileToGroup")
    helper_section = _section(source, "function isDatasetConfigGroup", "function populateTomlFileSelect")
    create_group = _section(source, "async function createTomlGroup", "async function renameTomlGroup")
    movable_groups = _section(source, "function getMovableTomlGroups", "function deleteTomlGroupButtonTitle")

    assert "/api/config/file-groups?kind=training" in load_toml
    assert "tomlFileGroups = filterTrainingTomlGroups(groups);" in load_toml
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
    assert "createFileGroupDragHandle" in file_button
    assert "placeTomlFile" in source
    assert "placeTomlGroup" in source
    assert "createTomlGroupOrderActions" not in source
    assert "createTomlFileOrderButton" not in source
    assert "toml-file-order-btn" not in css
    assert "const trainingGroups = filterTrainingTomlGroups(tomlFileGroups);" in save_as_groups
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
    experimental_factory = _section(source, "function createDatasetExperimentalFeaturesEditor", "function createDatasetRowSettingsEditor")
    notice_factory = _section(source, "function createDatasetExperimentalNotice", "function createDatasetExperimentalAdvancedBody")
    advanced_body_factory = _section(source, "function createDatasetExperimentalAdvancedBody", "function datasetExperimentalOpenKey")
    inline_help_factory = _section(source, "function datasetExperimentalOpenKey", "function createDatasetIsRegEditor")
    is_reg_factory = _section(source, "function createDatasetIsRegEditor", "Object.assign(globalThis")
    caption_extension_factory = _section(source, "function createDatasetCaptionExtensionEditor", "function createDatasetNlTagMixEditor")
    mix_factory = _section(source, "function createDatasetNlTagMixEditor", "function normalizeCaptionSourceMode")
    help_specs = _section(source, "function datasetLocalHelpSpec", "function createDatasetHelpNode")
    caption_source_factory = _section(source, "function createDatasetRowCaptionSourceModeEditor", "function createDatasetRowSettingInput")
    normalize_factory = _section(source, "function normalizeNlTagMix", "function updateDatasetDefault")
    payload_factory = _section(source, "function datasetRowsForPayload", "function normalizeDatasetRowSettings")
    row_update_factory = _section(source, "function updateDatasetEditorRowSetting", "function updateDatasetEditorRowNlTagMix")

    assert "通用标注设置" in defaults_editor
    assert "这里只保留 keep_tokens" in defaults_editor
    assert "文本标注扩展名等兼容项在每组数据集的高级区配置" in defaults_editor
    assert "['caption_extension', 'text']" not in defaults_editor
    assert "['keep_tokens', 'number']" in defaults_editor
    assert "['prefer_json_caption', 'switch', 'switch']" not in defaults_editor

    assert "createDatasetEditorItem(row, index)" in source
    assert "dataset-editor-item" in item_factory
    assert "createDatasetEditorRow(row, index, item)" in item_factory
    assert "createDatasetExperimentalFeaturesEditor(row, index)" in item_factory
    assert "createDatasetExperimentalFeaturesEditor(row, index)" not in row_factory
    assert "createDatasetRowCaptionSourceModeEditor(settings, index)" in row_factory
    assert "createDatasetNlTagMixEditor(row, index)" in row_factory
    assert "实验性/高级/旧功能" in experimental_factory
    assert "dataset-experimental-features" in experimental_factory
    assert "createDatasetExperimentalAdvancedBody(row, index, overviewHelp)" in experimental_factory
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

    trigger_clone_factory = _section(source, "function createDatasetTriggerCloneEditor", "function normalizeCaptionSourceMode")
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
    normalize_factory = _section(source, "function normalizeDatasetEditorRows", "function normalizeDatasetRowSettings")
    defaults_factory = _section(source, "function normalizeDatasetDefaults", "function updateDatasetDefault")
    row_update_factory = _section(source, "function updateDatasetEditorRow(index", "function updateDatasetEditorRowSetting")

    assert "createDatasetExperimentalAdvancedBody(row, index, overviewHelp)" in experimental_factory
    assert "createDatasetPathFilterEditor(row, index)" in advanced_body_factory
    assert "递归扫描子目录 / recursive" in filter_factory
    assert "路径筛选 / path_pattern" in filter_factory
    assert "recursive: row.recursive !== false && row.recursive !== 'false'" in normalize_factory
    assert "path_pattern: String(row.path_pattern || '*').trim() || '*'" in normalize_factory
    assert "recursive: row.recursive" in normalize_factory
    assert "path_pattern: row.path_pattern" in normalize_factory
    assert "rows[index].image_dir = '';" in row_update_factory
    assert "rows[index].cache_dir = '';" in row_update_factory
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
