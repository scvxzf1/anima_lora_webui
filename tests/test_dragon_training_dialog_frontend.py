from __future__ import annotations

from tests.frontend_test_support import STATIC_DIR, node_syntax_check


def _read(relative: str) -> str:
    return (STATIC_DIR / relative).read_text(encoding="utf-8")


def test_training_launch_dialog_exposes_scannable_decision_structure() -> None:
    controls = _read("js/dragon-ui/pages/training-controls.js")

    assert 'aria-labelledby="dragon-training-dialog-title"' in controls
    assert 'aria-describedby="dragon-training-dialog-body"' in controls
    assert 'class="dragon-training-overview"' in controls
    assert 'class="dragon-training-metrics"' in controls
    assert 'class="dragon-training-launch-context"' in controls
    assert 'class="dragon-training-check-badge"' in controls
    assert 'class="dragon-training-plan"' in controls
    assert 'aria-label="关闭弹窗"' in controls
    assert "if (event.target === dialog) dialog.close('cancel');" in controls
    assert "cancelText ? '' : 'autofocus'" in controls
    confirmation = controls[controls.index("function showLaunchConfirmation"):controls.index("function showResultDialog")]
    assert confirmation.index("renderLaunchPlan(action, context)") < confirmation.index("renderPreflightChecks(preflight)")


def test_training_launch_dialog_handles_status_and_untrusted_payloads() -> None:
    controls = _read("js/dragon-ui/pages/training-controls.js")

    assert "function escapeHtml(value)" in controls
    assert ".replaceAll('<', '&lt;')" in controls
    assert "function preflightTone(payload)" in controls
    assert "const levelOrder = { error: 0, warning: 1, info: 2, ok: 3 };" in controls
    assert "escapeHtml(item.path)" in controls
    assert "escapeHtml(payload?.message || payload?.error" in controls
    assert "root.dataset.trainingBusy === 'true'" in controls
    assert "actionButtons.forEach((item) => { item.disabled = true; });" in controls
    assert "catch (error)" in controls


def test_training_launch_dialog_has_responsive_and_reduced_motion_styles() -> None:
    css = _read("css/dragon/04-dragon-config.css")

    assert "body[data-dragon-ui] #dragon-root .dragon-training-dialog-header" in css
    assert ".dragon-training-dialog-heading" in css
    assert ".dragon-training-overview" in css
    assert ".dragon-training-launch-context" in css
    assert ".dragon-training-check-copy" in css
    assert ".dragon-training-dialog-close:focus-visible" in css
    assert "scrollbar-gutter: stable;" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert ".dragon-training-launch-context { grid-template-columns: 1fr; }" in css


def test_training_controls_module_is_valid_javascript() -> None:
    result = node_syntax_check("js/dragon-ui/pages/training-controls.js")
    assert result.returncode == 0, result.stderr
