"""测试 UI 缩放设置功能。"""

import sys
from pathlib import Path

import toml

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


UI_OVERRIDE_KEYS = (
    "ui_scale_config",
    "ui_scale_datasets",
    "ui_scale_training",
    "ui_scale_weight_analysis",
    "ui_scale_image_test",
    "ui_scale_settings",
    "ui_scale_environment",
    "ui_scale_history_overview",
    "ui_scale_history_analysis",
    "ui_scale_history_preview",
    "ui_scale_history_logs",
    "ui_scale_history_config_files",
)


def test_settings_service_imports():
    """测试settings_service可以正常导入"""
    from web.services import settings_service
    assert hasattr(settings_service, 'DEFAULT_UI_SCALE')
    assert settings_service.DEFAULT_UI_SCALE == 100
    assert hasattr(settings_service, 'GLOBAL_UI_KEYS')
    assert 'ui_scale' in settings_service.GLOBAL_UI_KEYS
    for key in UI_OVERRIDE_KEYS:
        assert key in settings_service.GLOBAL_UI_KEYS


def test_normalize_ui_setting():
    """测试UI设置归一化函数"""
    from web.services.settings_service import _normalize_ui_setting

    # 正常值
    assert _normalize_ui_setting('ui_scale', 100) == 100
    assert _normalize_ui_setting('ui_scale', 150) == 150

    # 边界值
    assert _normalize_ui_setting('ui_scale', 25) == 25
    assert _normalize_ui_setting('ui_scale', 400) == 400

    # 超出范围
    assert _normalize_ui_setting('ui_scale', 10) == 25  # 小于最小值，限制到最小值
    assert _normalize_ui_setting('ui_scale', 500) == 400  # 大于最大值，限制到最大值

    # 无效值
    assert _normalize_ui_setting('ui_scale', 'invalid') == 100
    assert _normalize_ui_setting('ui_scale', None) == 100

    # 可选覆盖值
    assert _normalize_ui_setting('ui_scale_config', 95) == 95
    assert _normalize_ui_setting('ui_scale_config', 10) == 25
    assert _normalize_ui_setting('ui_scale_config', 500) == 400
    assert _normalize_ui_setting('ui_scale_config', '') is None
    assert _normalize_ui_setting('ui_scale_config', None) is None
    assert _normalize_ui_setting('ui_scale_config', 'invalid') == 100

    # 未知的key
    assert _normalize_ui_setting('unknown_key', 100) is None


def test_default_global_settings():
    """测试默认全局设置包含ui_scale"""
    from web.services.settings_service import _default_global_settings

    defaults = _default_global_settings()
    assert 'ui_scale' in defaults
    assert defaults['ui_scale'] == 100
    for key in UI_OVERRIDE_KEYS:
        assert key in defaults
        assert defaults[key] == ""


def test_ui_scale_override_settings_roundtrip(tmp_path, monkeypatch):
    """测试独立界面比例覆盖值会保存、读取并支持清除。"""
    from web.services import settings_service

    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)

    saved = settings_service.save_global_settings(
        {
            "output_root": "output/runs",
            "ui_scale": 120,
            "ui_scale_config": 92,
            "ui_scale_training": 95,
            "ui_scale_history_logs": 88,
            "ui_scale_history_preview": "",
        }
    )

    assert saved["ui_scale"] == 120
    assert saved["ui_scale_config"] == 92
    assert saved["ui_scale_training"] == 95
    assert saved["ui_scale_history_logs"] == 88
    assert saved["ui_scale_history_preview"] == ""
    assert saved["ui_scale_history_analysis"] == ""

    raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    assert raw["global"]["ui_scale"] == 120
    assert raw["global"]["ui_scale_config"] == 92
    assert raw["global"]["ui_scale_training"] == 95
    assert raw["global"]["ui_scale_history_logs"] == 88
    assert "ui_scale_history_preview" not in raw["global"]

    cleared = settings_service.save_global_settings(
        {
            "ui_scale_config": "",
            "ui_scale_history_logs": None,
        }
    )

    assert cleared["ui_scale_config"] == ""
    assert cleared["ui_scale_history_logs"] == ""
    raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    assert "ui_scale_config" not in raw["global"]
    assert "ui_scale_history_logs" not in raw["global"]


if __name__ == '__main__':
    test_settings_service_imports()
    test_normalize_ui_setting()
    test_default_global_settings()
    print("✓ 所有测试通过")
