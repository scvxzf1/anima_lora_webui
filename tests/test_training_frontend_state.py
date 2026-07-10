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
    "js/features/sample-prompts/",
    "js/features/toml-manager/",
    "js/features/training-launch/",
    "js/features/history-detail/",
    "js/features/image-test/",
    "js/features/preview/",
    "js/features/queue/",
    "js/features/weight-analysis/",
    "js/shared/",
)



"""Facade module: frontend state tests split into test_training_frontend_*.py."""

from tests.frontend_test_support import *  # noqa: F403

def test_frontend_state_tests_are_split_into_domain_modules() -> None:
    assert Path(__file__).with_name("test_training_frontend_modules.py").is_file()
    assert Path(__file__).with_name("test_training_frontend_config_ui.py").is_file()
