#!/usr/bin/env python3
"""测试配置目录外置功能"""

import sys
from pathlib import Path

# 确保可以导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library.env import (
    get_configs_root,
    get_training_history_root,
    get_training_queue_root,
    project_root,
)

def test_config_paths():
    """测试配置路径解析"""
    print("=== 配置目录外置功能测试 ===\n")

    # 测试项目根目录
    root = project_root()
    print(f"项目根目录: {root}")

    # 测试配置根目录
    configs = get_configs_root()
    print(f"配置根目录: {configs}")
    print(f"  - 是否为绝对路径: {configs.is_absolute()}")
    print(f"  - 目录是否存在: {configs.exists()}")

    # 测试训练历史目录
    history = get_training_history_root()
    print(f"\n训练历史目录: {history}")
    print(f"  - 是否为绝对路径: {history.is_absolute()}")
    print(f"  - 目录是否存在: {history.exists()}")

    # 测试训练队列目录
    queue = get_training_queue_root()
    print(f"\n训练队列目录: {queue}")
    print(f"  - 是否为绝对路径: {queue.is_absolute()}")
    print(f"  - 目录是否存在: {queue.exists()}")

    # 验证关键文件
    print("\n=== 验证关键文件 ===")
    base_toml = configs / "base.toml"
    print(f"base.toml: {base_toml}")
    print(f"  - 存在: {base_toml.exists()}")

    presets_toml = configs / "presets.toml"
    print(f"presets.toml: {presets_toml}")
    print(f"  - 存在: {presets_toml.exists()}")

    web_settings = configs / "web-ui-settings.toml"
    print(f"web-ui-settings.toml: {web_settings}")
    print(f"  - 存在: {web_settings.exists()}")

    history_collections = history / "collections.json"
    print(f"collections.json: {history_collections}")
    print(f"  - 存在: {history_collections.exists()}")

    queue_file = queue / "queue.json"
    print(f"queue.json: {queue_file}")
    print(f"  - 存在: {queue_file.exists()}")

    # 测试 settings_service
    print("\n=== 测试 settings_service ===")
    try:
        from web.services.settings_service import CONFIGS_DIR, SETTINGS_FILE, get_global_settings
        print(f"settings_service.CONFIGS_DIR: {CONFIGS_DIR}")
        print(f"settings_service.SETTINGS_FILE: {SETTINGS_FILE}")
        print(f"  - SETTINGS_FILE 存在: {SETTINGS_FILE.exists()}")

        # 尝试加载全局设置
        settings = get_global_settings()
        print(f"\n全局设置加载成功:")
        print(f"  - output_root: {settings.get('output_root', 'N/A')}")
        print(f"  - training_history_root: {settings.get('training_history_root', 'N/A')}")
        print(f"  - training_queue_root: {settings.get('training_queue_root', 'N/A')}")
    except Exception as e:
        print(f"settings_service 测试失败: {e}")

    # 测试 training_service
    print("\n=== 测试 training_service ===")
    try:
        from web.services.training_service import HISTORY_DIR, QUEUE_DIR
        print(f"training_service.HISTORY_DIR: {HISTORY_DIR}")
        print(f"training_service.QUEUE_DIR: {QUEUE_DIR}")
        print(f"  - HISTORY_DIR 存在: {HISTORY_DIR.exists()}")
        print(f"  - QUEUE_DIR 存在: {QUEUE_DIR.exists()}")
    except Exception as e:
        print(f"training_service 测试失败: {e}")

    print("\n=== 测试完成 ===")
    return True

if __name__ == "__main__":
    try:
        test_config_paths()
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
