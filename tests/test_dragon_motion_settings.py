from __future__ import annotations

from pathlib import Path

import toml

from web.services import settings_service


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"


def _patch_settings(tmp_path: Path, monkeypatch) -> Path:
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    return settings_file


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def test_dragon_motion_defaults_enabled_and_roundtrips(tmp_path, monkeypatch):
    settings_file = _patch_settings(tmp_path, monkeypatch)

    assert settings_service.get_global_settings()["dragon_motion_enabled"] is True

    saved = settings_service.save_global_settings({"dragon_motion_enabled": False})

    assert saved["dragon_motion_enabled"] is False
    assert settings_service.get_global_settings()["dragon_motion_enabled"] is False
    assert toml.loads(settings_file.read_text(encoding="utf-8"))["global"]["dragon_motion_enabled"] is False


def test_partial_global_save_preserves_disabled_dragon_motion(tmp_path, monkeypatch):
    _patch_settings(tmp_path, monkeypatch)
    settings_service.save_global_settings({"dragon_motion_enabled": False})

    saved = settings_service.save_global_settings({"output_root": "output/next"})

    assert saved["dragon_motion_enabled"] is False


def test_dragon_motion_runtime_disables_effect_costs():
    motion = _read("js/dragon-ui/motion.js")
    animations = _read("js/dragon-ui/animations.js")
    router = _read("js/dragon-ui/router.js")
    entry = _read("js/dragon-ui/index.js")
    base_css = _read("css/dragon/01-dragon-base.css")
    animation_css = _read("css/dragon/05-dragon-animations.css")

    assert "prefers-reduced-motion: reduce" in motion
    assert "data.dragonMotion" not in motion
    assert "dataset.dragonMotion" in motion
    assert "dragon-motion-change" in motion
    assert "delete document.documentElement.dataset.dragonMotion" in motion
    assert "if (!isDragonMotionEnabled())" in animations
    assert "revealAll();" in animations
    assert "typeof window.IntersectionObserver !== 'function'" in animations
    assert "new window.IntersectionObserver" in animations
    assert "typeof window.requestAnimationFrame === 'function'" in animations
    assert "window.setTimeout(callback, 16)" in animations
    assert "if (currentPage && currentWrapper && isDragonMotionEnabled())" in router
    assert "behavior: dragonScrollBehavior()" in router
    assert "initDragonMotion(globalSettings || {})" in entry
    assert "window.addEventListener('dragon-motion-change', handleMotionChange)" in entry
    assert 'html[data-dragon-motion="disabled"]' in base_css
    assert 'html[data-dragon-motion="disabled"]' in animation_css


def test_global_settings_exposes_and_applies_motion_toggle():
    settings_page = _read("js/dragon-ui/pages/global-settings.js")

    assert "['dragon_motion_enabled', '启用 Dragon 动态效果', 'boolean'" in settings_page
    assert "applyDragonMotionSetting(payload);" in settings_page
    assert "return key === 'dragon_motion_enabled';" in settings_page
