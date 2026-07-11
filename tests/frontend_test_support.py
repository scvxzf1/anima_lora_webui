"""Shared helpers for split training frontend state tests."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import pytest


STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"
REPO_ROOT = Path(__file__).resolve().parents[1]
APP_JS_PATH = STATIC_DIR / "app.js"
CHART_JS = STATIC_DIR / "chart.js"
INDEX_HTML = STATIC_DIR / "index.html"

# Critical workflow anchors used by save/queue/start/history/preview/settings paths.
# Keep this intentionally small; do not expand into a full DOM inventory.
CRITICAL_WORKFLOW_DOM_IDS = frozenset(
    {
        # config workflow
        "btn-load-config",
        "btn-save-toml",
        "btn-queue-from-config",
        "btn-start-from-config",
        "config-form",
        # training control
        "btn-stop-training",
        "status-indicator",
        "status-text",
        # global settings
        "btn-save-global-settings",
        "tab-settings",
        # history / preview
        "btn-refresh-history",
        "btn-preview-training-results",
        "task-history-list",
    }
)



# Workflow DOM contracts (required/optional). Keep small; do not inventory all ids.
WORKFLOW_DOM_CONTRACTS: dict[str, dict[str, frozenset[str]]] = {
    "queue": {
        "required": frozenset(
            {
                "btn-training-queue-view",
                "btn-refresh-queue",
                "btn-open-queue-manager",
                "training-queue-list",
                "training-queue-summary",
            }
        ),
        "optional": frozenset(
            {
                "btn-toggle-queue-pause",
                "training-queue-manager",
                "btn-cancel-all-queue",
                "btn-clear-completed-queue",
            }
        ),
    },
    "history": {
        "required": frozenset(
            {
                "btn-training-history-view",
                "btn-refresh-history",
                "btn-open-history-manager",
                "task-history-list",
                "btn-history-manager-refresh",
            }
        ),
        "optional": frozenset(
            {
                "btn-history-collections-workbench",
                "history-manager-search",
                "history-show-archived",
            }
        ),
    },
    "preview": {
        "required": frozenset(
            {
                "tab-preview",
                "btn-preview-training-results",
                "btn-refresh-preview",
                "preview-workspace",
                "preview-grid",
            }
        ),
        "optional": frozenset(
            {
                "preview-page-mount",
                "btn-save-preview-settings",
                "preview-settings-status",
            }
        ),
    },
    "settings": {
        "required": frozenset(
            {
                "tab-settings",
                "btn-save-global-settings",
                "global-output-root",
                "global-pretrained-model-path",
                "global-configs-root",
            }
        ),
        "optional": frozenset(
            {
                "global-ui-scale",
                "global-qwen3-path",
                "global-vae-path",
            }
        ),
    },
}


def workflow_dom_contract(name: str) -> dict[str, frozenset[str]]:
    """Return required/optional DOM id sets for a named workflow bucket."""
    if name not in WORKFLOW_DOM_CONTRACTS:
        raise KeyError(f"unknown workflow DOM contract: {name}")
    return WORKFLOW_DOM_CONTRACTS[name]

def missing_dom_ids_in_html(html: str, dom_ids: set[str] | frozenset[str]) -> set[str]:
    """Return DOM ids that do not appear as id="..." attributes in HTML."""
    return {dom_id for dom_id in dom_ids if f'id="{dom_id}"' not in html}


STYLE_CSS_PATH = STATIC_DIR / "style.css"
MODULE_IMPORT_RE = re.compile(
    r"""(?:(?:import|export)\s+(?:[^'"]*?\s+from\s+)?|import\(\s*)['"]([^'"]+\.js(?:\?[^'"]*)?)['"]"""
)
CSS_IMPORT_RE = re.compile(r"""@import\s+(?:url\()?['"]?([^'")]+\.css(?:\?[^'")]+)?)['"]?\)?\s*;""")
GLOBAL_THIS_ASSIGN_RE = re.compile(
    r"(?<![\w$])globalThis\.([A-Za-z_$][\w$]*)\s*(?:\|\|=|&&=|\?\?=|=)(?!=)"
)
GLOBAL_THIS_OBJECT_ASSIGN_RE = re.compile(r"Object\.assign\(\s*globalThis\s*,")

ANIMA_APP_GLOBAL_THIS_BASELINE = {
    "js/features/anima-app/index.js": (0, 0),
    "js/features/anima-app/runtime.js": (0, 0),
    "js/features/anima-app/chunks/01-scope-state.js": (0, 0),
    "js/features/anima-app/chunks/01a-image-test-feature.js": (0, 0),
    "js/features/anima-app/chunks/02-ensure-history-detail-feature.js": (0, 0),
    "js/features/anima-app/chunks/03-parse-network-arg-entry.js": (0, 0),
    "js/features/anima-app/chunks/04-create-config-group-entry.js": (0, 0),
    "js/features/anima-app/chunks/05-create-stage-resolution-summary.js": (0, 0),
    "js/features/anima-app/chunks/05a-no-dataset-regularization-mode.js": (0, 0),
    "js/features/anima-app/chunks/06-stronger-selective-checkpoint-value.js": (0, 0),
    "js/features/anima-app/chunks/07-render-config-dataset-picker-dialog.js": (0, 0),
    "js/features/anima-app/chunks/08-origin-closest.js": (0, 0),
    "js/features/anima-app/chunks/09-setup-config-group-drop-target.js": (0, 0),
    "js/features/anima-app/chunks/10-create-dataset-config-input.js": (0, 0),
    "js/features/anima-app/chunks/10a-dataset-inline-help.js": (0, 0),
    "js/features/anima-app/chunks/11-create-dataset-editor-row.js": (0, 0),
    "js/features/anima-app/chunks/12-create-dataset-row-caption-source-mode-editor.js": (0, 0),
    "js/features/anima-app/chunks/13-update-dataset-editor-rows-setting-value.js": (0, 0),
    "js/features/anima-app/chunks/14-lora-adapter-kind-from-config.js": (0, 0),
    "js/features/anima-app/chunks/15-append-sample-prompt-row.js": (0, 0),
    "js/features/anima-app/chunks/16-load-output-run-config.js": (0, 0),
    "js/features/anima-app/chunks/17-apply-selected-dataset-preset-to-current-config.js": (0, 0),
    "js/features/anima-app/chunks/18-delete-dataset-preset-group.js": (0, 0),
    "js/features/anima-app/chunks/19-current-sample-prompt-text.js": (0, 0),
    "js/features/anima-app/chunks/20-can-drop-toml-file-to-group.js": (0, 0),
    "js/features/anima-app/chunks/21-update-toml-selection-ui.js": (0, 0),
    "js/features/anima-app/chunks/22-update-toml-action-state.js": (0, 0),
    "js/features/anima-app/chunks/23-move-current-toml-to-group.js": (0, 0),
    "js/features/anima-app/chunks/24-show-preflight-pending-dialog.js": (0, 0),
    "js/features/live-log/index.js": (0, 0),
    "js/features/preflight-dialog/index.js": (0, 0),
    "js/features/anima-app/chunks/25-update-progress.js": (0, 0),
    "js/features/anima-app/chunks/26-load-global-settings.js": (0, 0),
    "js/features/anima-app/chunks/26a-status-polling.js": (0, 0),
    "js/features/anima-app/chunks/26a-global-settings.js": (0, 0),
    "js/features/anima-app/chunks/26b-preview-view.js": (0, 0),
    "js/features/anima-app/chunks/26c-queue-view.js": (0, 0),
    "js/features/anima-app/chunks/26d-history-list.js": (0, 0),
    "js/features/anima-app/chunks/27-render-history-collections-workbench.js": (0, 0),
    "js/features/anima-app/chunks/28-history-collection-search-text.js": (0, 0),
    "js/features/anima-app/chunks/29-start-history-config-group-pointer-drag.js": (0, 0),
    "js/features/anima-app/chunks/30-start-history-collection-pointer-drag.js": (0, 0),
    "js/features/anima-app/chunks/31-create-history-collection-workbench-card.js": (0, 0),
    "js/features/anima-app/chunks/32-history-task-collection-label.js": (0, 0),
    "js/features/anima-app/chunks/33-create-history-task-item.js": (0, 0),
    "js/features/anima-app/chunks/34-show-history-collection-select-dialog.js": (0, 0),
    "js/features/anima-app/chunks/35-render-config-group-timeline.js": (0, 0),
    "js/features/anima-app/chunks/36-setup-event-listeners.js": (0, 0),
    "js/features/training-source/index.js": (0, 0),
    "js/features/anima-app/chunks/37-config-training-source.js": (0, 0),
}
LEGACY_GLOBALS_RELATIVE = "js/features/anima-app/legacy-globals.js"
LEGACY_GLOBALS_PATH = STATIC_DIR / LEGACY_GLOBALS_RELATIVE
LEGACY_GLOBALS_REPO_SCAN_ROOTS = (
    REPO_ROOT / "anima_lora",
    REPO_ROOT / "gui",
    REPO_ROOT / "library",
    REPO_ROOT / "scripts",
    REPO_ROOT / "web",
)
LEGACY_GLOBALS_REPO_SCAN_FILES = (
    REPO_ROOT / "inference.py",
    REPO_ROOT / "tasks.py",
    REPO_ROOT / "train.py",
)
LEGACY_GLOBALS_REPO_SCAN_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".mjs",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
}
GLOBAL_THIS_ALLOWED_OUTSIDE_ANIMA_APP = {
    "app.js": (1, 0),
}
GLOBAL_THIS_ZERO_WRITE_PREFIXES = (
    "js/features/anima-app/features/",
    "js/features/anima-app/helpers/",
    "js/features/anima-app/runtime/",
    "js/features/app-shell/",
    "js/features/config-form/",
    "js/features/dataset-editor/",
    "js/features/global-settings/",
    "js/features/history-detail/",
    "js/features/history-list/",
    "js/features/image-test/",
    "js/features/live-log/",
    "js/features/live-training/",
    "js/features/preflight-dialog/",
    "js/features/preview/",
    "js/features/queue/",
    "js/features/sample-prompts/",
    "js/features/toml-manager/",
    "js/features/training-launch/",
    "js/features/training-source/",
    "js/features/weight-analysis/",
    "js/shared/",
)


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


def _assert_imports_from(source: str, module_path: str, names: tuple[str, ...]) -> None:
    pattern = re.compile(
        r"import\s+\{(?P<body>[^}]*)\}\s+from\s+['\"]"
        + re.escape(module_path)
        + r"(?:\?[^'\"]*)?['\"]",
        re.S,
    )
    imported: set[str] = set()
    for match in pattern.finditer(source):
        for raw_name in match.group("body").split(","):
            name = raw_name.strip()
            if not name:
                continue
            imported.add(name.split(" as ", 1)[0].strip())
    missing = [name for name in names if name not in imported]
    assert not missing, f"missing imports from {module_path}: {missing}"


def _frontend_feature_text(*relative_paths: str) -> str:
    return "\n".join(_frontend_module_text(relative_path) for relative_path in relative_paths)


def _anima_app_container_text() -> str:
    graph = _frontend_module_graph()
    paths = [
        STATIC_DIR / "js/features/anima-app/index.js",
        *sorted((STATIC_DIR / "js/features/anima-app/chunks").glob("*.js")),
        # Feature modules that former chunk truth now re-exports.
        # Keep queue view-mode before status-polling so legacy section scans still work.
        STATIC_DIR / "js/features/global-settings/settings.js",
        STATIC_DIR / "js/features/history-list/list.js",
        STATIC_DIR / "js/features/preview/compat-api.js",
        STATIC_DIR / "js/features/queue/view-mode.js",
        STATIC_DIR / "js/features/preflight-dialog/index.js",
        STATIC_DIR / "js/features/live-log/index.js",
        STATIC_DIR / "js/features/live-training/status-polling.js",
        STATIC_DIR / "js/features/app-shell/startup.js",
        STATIC_DIR / "js/features/config-form/index.js",
        STATIC_DIR / "js/features/config-form/stage-resolution.js",
        STATIC_DIR / "js/features/config-form/stage-resolution-model.js",
        STATIC_DIR / "js/features/config-form/stage-resolution-ui.js",
        STATIC_DIR / "js/features/config-form/stage-resolution-ui-widgets.js",
        STATIC_DIR / "js/features/config-form/stage-resolution-ui-dialog.js",
        STATIC_DIR / "js/features/config-form/form-fields.js",
        STATIC_DIR / "js/features/config-form/form-fields-adapters.js",
        STATIC_DIR / "js/features/config-form/form-fields-sample.js",
        STATIC_DIR / "js/features/config-form/form-fields-ui.js",

        STATIC_DIR / "js/features/sample-prompts/row-ui.js",
        STATIC_DIR / "js/features/toml-manager/mode.js",
        STATIC_DIR / "js/features/history-list/workbench-cards.js",
        STATIC_DIR / "js/features/dataset-editor/row.js",
        STATIC_DIR / "js/features/dataset-editor/row-settings.js",
        STATIC_DIR / "js/features/dataset-editor/row-settings-basic.js",
        STATIC_DIR / "js/features/dataset-editor/row-settings-experimental.js",
        STATIC_DIR / "js/features/dataset-editor/dataset-render.js",

        STATIC_DIR / "js/features/dataset-editor/row-fields.js",
        STATIC_DIR / "js/features/dataset-editor/preview.js",
        STATIC_DIR / "js/features/dataset-editor/config-input.js",
        STATIC_DIR / "js/features/dataset-editor/item-drag.js",
        STATIC_DIR / "js/features/dataset-editor/preset-page.js",
        STATIC_DIR / "js/features/toml-manager/file-group-drag-core.js",
        STATIC_DIR / "js/features/toml-manager/file-group-drag-targets.js",
        STATIC_DIR / "js/features/dataset-editor/load.js",
        STATIC_DIR / "js/features/config-form/no-dataset-regularization.js",
        STATIC_DIR / "js/features/config-form/dataset-picker-dialog.js",
        STATIC_DIR / "js/features/config-form/step-estimate.js",
        STATIC_DIR / "js/features/live-training/dashboard-ui.js",
        STATIC_DIR / "js/features/toml-manager/actions.js",
        STATIC_DIR / "js/features/toml-manager/drag.js",
        STATIC_DIR / "js/features/toml-manager/drag-core.js",
        STATIC_DIR / "js/features/toml-manager/drag-actions.js",
        STATIC_DIR / "js/features/toml-manager/drag-render.js",

        STATIC_DIR / "js/features/training-launch/index.js",
        STATIC_DIR / "js/features/app-shell/event-listeners.js",
        STATIC_DIR / "js/features/app-shell/event-listeners-contract.js",
        STATIC_DIR / "js/features/app-shell/event-listeners-setup.js",
        STATIC_DIR / "js/features/app-shell/beginner-tooltips.js",
    ]
    for path in paths:
        assert path.resolve() in graph, f"{path.relative_to(STATIC_DIR).as_posix()} is not reachable"
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _module_cache_token(specifier: str) -> str | None:
    return parse_qs(urlparse(specifier).query).get("v", [None])[0]


def _style_cache_token(specifier: str) -> str | None:
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


def _setup_event_dom_contract() -> dict[str, set[str]]:
    source = _frontend_feature_text("js/features/app-shell/event-listeners.js", "js/features/app-shell/event-listeners-contract.js", "js/features/app-shell/event-listeners-setup.js", "js/features/app-shell/beginner-tooltips.js")
    contract: dict[str, set[str]] = {}
    for key in ("required", "optional"):
        match = re.search(rf"{key}:\s*Object\.freeze\(\[(.*?)\]\)", source, re.S)
        assert match, f"missing setup event DOM contract bucket: {key}"
        contract[key] = set(re.findall(r"'([^']+)'", match.group(1)))
    return contract


def _config_training_source_dom_contract() -> dict[str, set[str]]:
    source = _frontend_module_text("js/features/training-source/index.js")
    contract: dict[str, set[str]] = {}
    for key in ("required", "optional"):
        match = re.search(rf"{key}:\s*Object\.freeze\(\[(.*?)\]\)", source, re.S)
        assert match, f"missing config training source DOM contract bucket: {key}"
        contract[key] = set(re.findall(r"'([^']+)'", match.group(1)))
    return contract


def node_syntax_check(relative_module: str) -> subprocess.CompletedProcess[str]:
    """Run `node --check` on a static JS module when node is available."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node is not available on PATH")
    target = (STATIC_DIR / relative_module).resolve()
    assert target.is_file(), f"missing static module for node --check: {relative_module}"
    assert STATIC_DIR.resolve() in target.parents or target == STATIC_DIR.resolve()
    return subprocess.run(
        [node, "--check", str(target)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _literal_get_element_by_id_targets(source: str) -> set[str]:
    return set(re.findall(r"document\.getElementById\('([^']+)'\)", source))


def _global_this_write_counts(path: Path) -> tuple[int, int]:
    source = path.read_text(encoding="utf-8")
    return (
        len(GLOBAL_THIS_ASSIGN_RE.findall(source)),
        len(GLOBAL_THIS_OBJECT_ASSIGN_RE.findall(source)),
    )


def _global_this_write_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if GLOBAL_THIS_ASSIGN_RE.search(line) or GLOBAL_THIS_OBJECT_ASSIGN_RE.search(line):
            relative = path.relative_to(STATIC_DIR).as_posix()
            lines.append(f"{relative}:{lineno}: {line.strip()}")
    return lines


def _legacy_globals_repo_scan_paths() -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen or resolved == LEGACY_GLOBALS_PATH.resolve():
            return
        seen.add(resolved)
        ordered.append(resolved)

    for root in LEGACY_GLOBALS_REPO_SCAN_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in LEGACY_GLOBALS_REPO_SCAN_SUFFIXES:
                continue
            add(path)

    for path in LEGACY_GLOBALS_REPO_SCAN_FILES:
        if path.is_file():
            add(path)

    return ordered


