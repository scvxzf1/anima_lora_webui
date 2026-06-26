"""测试UI缩放设置功能"""
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_settings_service_imports():
    """测试settings_service可以正常导入"""
    from web.services import settings_service
    assert hasattr(settings_service, 'DEFAULT_UI_SCALE')
    assert settings_service.DEFAULT_UI_SCALE == 100
    assert hasattr(settings_service, 'GLOBAL_UI_KEYS')
    assert 'ui_scale' in settings_service.GLOBAL_UI_KEYS


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

    # 未知的key
    assert _normalize_ui_setting('unknown_key', 100) is None


def test_default_global_settings():
    """测试默认全局设置包含ui_scale"""
    from web.services.settings_service import _default_global_settings

    defaults = _default_global_settings()
    assert 'ui_scale' in defaults
    assert defaults['ui_scale'] == 100


if __name__ == '__main__':
    test_settings_service_imports()
    test_normalize_ui_setting()
    test_default_global_settings()
    print("✓ 所有测试通过")
