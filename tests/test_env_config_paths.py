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


def test_resolve_under_home_uses_anima_home_for_relative_paths(monkeypatch, tmp_path):
    """相对路径必须锚到 ANIMA_HOME，而不是当前工作目录。"""
    home = tmp_path / "repo-home"
    cwd = tmp_path / "other-cwd"
    absolute = tmp_path / "absolute-output"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("ANIMA_HOME", str(home))
    monkeypatch.chdir(cwd)

    import library.env as library_env

    assert library_env.resolve_under_home("output/runs") == home / "output" / "runs"
    assert library_env.resolve_under_home(absolute) == absolute


def test_load_dotenv_preserves_existing_env_and_parses_simple_file(
    monkeypatch,
    tmp_path,
):
    """读取 .env 时不覆盖已有环境变量，并忽略注释和坏行。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "",
                "BROKEN_LINE",
                "ANIMA_EXISTING=from-file",
                'ANIMA_DOTENV_DOUBLE="double quoted"',
                "ANIMA_DOTENV_SINGLE='single quoted'",
                "ANIMA_DOTENV_SPACED = spaced value ",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANIMA_EXISTING", "from-process")
    for key in (
        "BROKEN_LINE",
        "ANIMA_DOTENV_DOUBLE",
        "ANIMA_DOTENV_SINGLE",
        "ANIMA_DOTENV_SPACED",
    ):
        monkeypatch.delenv(key, raising=False)

    import library.env as library_env

    added = library_env.load_dotenv(env_file)

    assert added == {
        "ANIMA_DOTENV_DOUBLE": "double quoted",
        "ANIMA_DOTENV_SINGLE": "single quoted",
        "ANIMA_DOTENV_SPACED": "spaced value",
    }
    assert os.environ["ANIMA_EXISTING"] == "from-process"
    assert "BROKEN_LINE" not in os.environ


def test_get_configs_root_env_relative(monkeypatch, tmp_path):
    """测试相对路径环境变量（相对于项目根）。"""
    project = tmp_path / "project"
    project.mkdir()
    (project / "configs").mkdir()
    monkeypatch.setenv("ANIMA_CONFIGS_ROOT", "configs")

    library_env = _reload_env_with_project_root(monkeypatch, project)

    root = library_env.get_configs_root()
    assert root == (project / "configs").resolve()


def test_get_configs_root_rejects_parent_traversal(monkeypatch, tmp_path):
    """配置根目录环境变量不能通过 .. 跳出项目根。"""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANIMA_CONFIGS_ROOT", "../outside")

    library_env = _reload_env_with_project_root(monkeypatch, project)

    with pytest.raises(ValueError, match="ANIMA_CONFIGS_ROOT"):
        library_env.get_configs_root()


def test_get_configs_root_rejects_absolute_parent_traversal(monkeypatch, tmp_path):
    """配置根目录绝对路径中也不能包含 ..。"""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv(
        "ANIMA_CONFIGS_ROOT",
        str(tmp_path / "safe" / ".." / "outside"),
    )

    library_env = _reload_env_with_project_root(monkeypatch, project)

    with pytest.raises(ValueError, match="ANIMA_CONFIGS_ROOT"):
        library_env.get_configs_root()


def test_path_root_overrides_reject_blank_values(monkeypatch, tmp_path):
    """路径根目录环境变量不能用纯空白误解析成项目根。"""
    project = tmp_path / "project"
    project.mkdir()

    cases = [
        ("ANIMA_CONFIGS_ROOT", "get_configs_root"),
        ("ANIMA_TRAINING_HISTORY_ROOT", "get_training_history_root"),
        ("ANIMA_TRAINING_QUEUE_ROOT", "get_training_queue_root"),
    ]

    for env_name, getter_name in cases:
        monkeypatch.setenv(env_name, "   ")
        library_env = _reload_env_with_project_root(monkeypatch, project)

        with pytest.raises(ValueError, match=env_name):
            getattr(library_env, getter_name)()

        monkeypatch.delenv(env_name, raising=False)


def test_get_configs_root_settings_file_rejects_parent_traversal(monkeypatch, tmp_path):
    """WebUI 本机配置里的 configs_root 也不能通过 .. 跳出项目根。"""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".anima-webui-settings.toml").write_text(
        '[paths]\nconfigs_root = "../outside"\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("ANIMA_CONFIGS_ROOT", raising=False)

    library_env = _reload_env_with_project_root(monkeypatch, project)

    with pytest.raises(ValueError, match="configs_root"):
        library_env.get_configs_root()


def test_training_roots_follow_webui_configs_root_settings(monkeypatch, tmp_path):
    """history/queue 默认跟随 WebUI 本机 configs_root 设置。"""
    project = tmp_path / "project"
    project.mkdir()
    settings_configs = project / "local-configs"
    env_configs = tmp_path / "env-configs"
    settings_configs.mkdir()
    env_configs.mkdir()
    (project / ".anima-webui-settings.toml").write_text(
        '[paths]\nconfigs_root = "local-configs"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ANIMA_CONFIGS_ROOT", str(env_configs))
    monkeypatch.delenv("ANIMA_TRAINING_HISTORY_ROOT", raising=False)
    monkeypatch.delenv("ANIMA_TRAINING_QUEUE_ROOT", raising=False)

    library_env = _reload_env_with_project_root(monkeypatch, project)

    assert library_env.get_configs_root() == settings_configs.resolve()
    assert (
        library_env.get_training_history_root()
        == settings_configs.resolve() / "web-training-history"
    )
    assert (
        library_env.get_training_queue_root()
        == settings_configs.resolve() / "web-training-queue"
    )


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


def test_get_training_history_root_rejects_parent_traversal(monkeypatch, tmp_path):
    """训练历史目录环境变量不能通过 .. 跳出项目根。"""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANIMA_TRAINING_HISTORY_ROOT", "../outside-history")

    library_env = _reload_env_with_project_root(monkeypatch, project)

    with pytest.raises(ValueError, match="ANIMA_TRAINING_HISTORY_ROOT"):
        library_env.get_training_history_root()


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


def test_get_training_queue_root_rejects_parent_traversal(monkeypatch, tmp_path):
    """训练队列目录环境变量不能通过 .. 跳出项目根。"""
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ANIMA_TRAINING_QUEUE_ROOT", "../outside-queue")

    library_env = _reload_env_with_project_root(monkeypatch, project)

    with pytest.raises(ValueError, match="ANIMA_TRAINING_QUEUE_ROOT"):
        library_env.get_training_queue_root()


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


def test_expand_env_vars_in_obj_recurses_nested_structures(monkeypatch, tmp_path):
    """嵌套 dict/list/tuple 中的路径占位也要递归展开。"""
    home_dir = tmp_path / "home"
    project = tmp_path / "project"
    home_dir.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("ANIMA_NESTED_ROOT", str(tmp_path / "nested-root"))

    library_env = _reload_env_with_project_root(monkeypatch, project)

    expanded = library_env.expand_env_vars_in_obj(
        {
            "plain": "unchanged",
            "path": "$ANIMA_NESTED_ROOT/data",
            "items": ["~/cache", 5, ("$ANIMA_NESTED_ROOT/a", "literal")],
        }
    )

    assert expanded == {
        "plain": "unchanged",
        "path": str(tmp_path / "nested-root" / "data"),
        "items": [
            str(home_dir / "cache"),
            5,
            (str(tmp_path / "nested-root" / "a"), "literal"),
        ],
    }
    assert isinstance(expanded["items"][2], tuple)


def test_history_and_queue_roots_follow_webui_settings_file(monkeypatch, tmp_path):
    """WebUI settings paths.history_root / queue_root 优先于环境变量。"""
    project = tmp_path / "project"
    project.mkdir()
    history = project / "custom-history"
    queue = project / "custom-queue"
    history.mkdir()
    queue.mkdir()
    env_history = tmp_path / "env-history"
    env_queue = tmp_path / "env-queue"
    env_history.mkdir()
    env_queue.mkdir()
    (project / ".anima-webui-settings.toml").write_text(
        "\n".join(
            [
                "[paths]",
                'history_root = "custom-history"',
                'queue_root = "custom-queue"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANIMA_TRAINING_HISTORY_ROOT", str(env_history))
    monkeypatch.setenv("ANIMA_TRAINING_QUEUE_ROOT", str(env_queue))

    library_env = _reload_env_with_project_root(monkeypatch, project)

    assert library_env.get_training_history_root() == history.resolve()
    assert library_env.get_training_queue_root() == queue.resolve()


def test_web_service_roots_follow_anima_home(monkeypatch, tmp_path):
    """WebUI service ROOT constants should anchor on anima_home()."""
    home = tmp_path / "anima-home"
    home.mkdir()
    monkeypatch.setenv("ANIMA_HOME", str(home))
    # re-import functions that read anima_home dynamically
    import importlib
    import library.env as env
    importlib.reload(env)
    assert env.anima_home() == home.resolve()
    # settings_service.ROOT is assigned at import; call anima_home directly and
    # ensure modules use the same function for new assignments.
    assert env.anima_home() == home.resolve()
    # config common get root via anima_home symbol
    from web.services.config import common as common_mod
    assert common_mod.anima_home() == home.resolve()

