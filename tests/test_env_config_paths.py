"""测试配置目录外置功能。"""

import os
from pathlib import Path

import pytest

from library.env import get_configs_root


def _reload_env_with_project_root(monkeypatch, root: Path):
    """Reload library.env with a temp project root so local settings do not leak in."""
    from importlib import reload
    import library.env

    monkeypatch.delenv("ANIMA_HOME", raising=False)
    module = reload(library.env)
    monkeypatch.setattr(module, "project_root", lambda: root)
    return module


def test_get_configs_root_default():
    """测试默认配置根目录（未设置环境变量时）。"""
    # 清除环境变量
    old_value = os.environ.pop("ANIMA_CONFIGS_ROOT", None)
    try:
        # 在已经加载 .env 的情况下，需要重新导入来获取默认值
        # 但由于 .env 已经加载，这里只验证函数存在且可调用
        root = get_configs_root()
        assert root.is_absolute()
        assert root.exists() or root.name == "configs"  # 可能还不存在
    finally:
        if old_value is not None:
            os.environ["ANIMA_CONFIGS_ROOT"] = old_value


def test_get_configs_root_env_relative(monkeypatch, tmp_path):
    """测试相对路径环境变量（相对于项目根）。"""
    project = tmp_path / "project"
    project.mkdir()
    (project / "configs").mkdir()
    monkeypatch.setenv("ANIMA_CONFIGS_ROOT", "configs")

    library_env = _reload_env_with_project_root(monkeypatch, project)

    root = library_env.get_configs_root()
    assert root == (project / "configs").resolve()


def test_get_configs_root_env_absolute(monkeypatch, tmp_path):
    """测试绝对路径环境变量。"""
    test_configs = tmp_path / "absolute_configs"
    test_configs.mkdir()

    monkeypatch.setenv("ANIMA_CONFIGS_ROOT", str(test_configs))

    library_env = _reload_env_with_project_root(monkeypatch, tmp_path)

    root = library_env.get_configs_root()
    assert root == test_configs.resolve()


def test_get_training_history_root_env(monkeypatch, tmp_path):
    """测试训练历史目录环境变量。"""
    history = tmp_path / "history"
    history.mkdir()

    monkeypatch.setenv("ANIMA_TRAINING_HISTORY_ROOT", str(history))

    library_env = _reload_env_with_project_root(monkeypatch, tmp_path)

    root = library_env.get_training_history_root()
    assert root == history.resolve()


def test_get_training_queue_root_fallback_to_configs(monkeypatch, tmp_path):
    """测试未设置队列目录时回退到 configs_root。"""
    test_configs = tmp_path / "configs"
    test_configs.mkdir()
    (test_configs / "web-training-queue").mkdir()

    monkeypatch.setenv("ANIMA_CONFIGS_ROOT", str(test_configs))
    monkeypatch.delenv("ANIMA_TRAINING_QUEUE_ROOT", raising=False)

    library_env = _reload_env_with_project_root(monkeypatch, tmp_path)

    queue = library_env.get_training_queue_root()
    assert queue == test_configs / "web-training-queue"


def test_get_training_queue_root_env(monkeypatch, tmp_path):
    """测试训练队列目录环境变量。"""
    queue = tmp_path / "queue"
    queue.mkdir()

    monkeypatch.setenv("ANIMA_TRAINING_QUEUE_ROOT", str(queue))

    library_env = _reload_env_with_project_root(monkeypatch, tmp_path)

    root = library_env.get_training_queue_root()
    assert root == queue.resolve()


def test_configs_root_with_env_expansion(monkeypatch, tmp_path):
    """测试环境变量扩展（$HOME, ~）。"""
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    test_configs = home_dir / "configs"
    test_configs.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("ANIMA_CONFIGS_ROOT", "$HOME/configs")

    library_env = _reload_env_with_project_root(monkeypatch, tmp_path)

    root = library_env.get_configs_root()
    assert root == test_configs.resolve()
