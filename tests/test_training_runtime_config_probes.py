"""Web runtime auto probe path tests"""

from __future__ import annotations

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

def test_block_swap_profile_auto_config_targets_current_history_task(
    tmp_path, monkeypatch
):
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    history_dir = tmp_path / "configs" / "web-training-history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    config_path = tmp_path / "output" / "runs" / "demo" / "config.runtime.toml"
    config_path.parent.mkdir(parents=True)
    profile_path = history_dir / "task-new" / "block_swap_profile.jsonl"

    config_path.write_text('block_swap_profile_jsonl = "auto"\n', encoding="utf-8")
    training_service._resolve_block_swap_profile_auto_config(
        str(config_path), profile_path
    )
    cfg = toml.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["block_swap_profile_jsonl"] == str(profile_path)

    old_profile_path = history_dir / "task-old" / "block_swap_profile.jsonl"
    config_path.write_text(
        toml.dumps({"block_swap_profile_jsonl": str(old_profile_path)}),
        encoding="utf-8",
    )
    training_service._resolve_block_swap_profile_auto_config(
        str(config_path), profile_path
    )
    cfg = toml.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["block_swap_profile_jsonl"] == str(profile_path)

    explicit_profile_path = tmp_path / "logs" / "explicit.jsonl"
    config_path.write_text(
        toml.dumps({"block_swap_profile_jsonl": str(explicit_profile_path)}),
        encoding="utf-8",
    )
    training_service._resolve_block_swap_profile_auto_config(
        str(config_path), profile_path
    )
    cfg = toml.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["block_swap_profile_jsonl"] == str(explicit_profile_path)

def test_memory_probe_auto_config_targets_current_history_task(tmp_path, monkeypatch):
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    history_dir = tmp_path / "configs" / "web-training-history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    config_path = tmp_path / "output" / "runs" / "demo" / "config.runtime.toml"
    config_path.parent.mkdir(parents=True)
    probe_path = history_dir / "task-new" / "memory_probe.jsonl"

    config_path.write_text('memory_probe_jsonl = "auto"\n', encoding="utf-8")
    training_service._resolve_memory_probe_auto_config(str(config_path), probe_path)
    cfg = toml.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["memory_probe_jsonl"] == str(probe_path)

    old_probe_path = history_dir / "task-old" / "memory_probe.jsonl"
    config_path.write_text(
        toml.dumps({"memory_probe_jsonl": str(old_probe_path)}),
        encoding="utf-8",
    )
    training_service._resolve_memory_probe_auto_config(str(config_path), probe_path)
    cfg = toml.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["memory_probe_jsonl"] == str(probe_path)

    explicit_probe_path = tmp_path / "logs" / "explicit-memory.jsonl"
    config_path.write_text(
        toml.dumps({"memory_probe_jsonl": str(explicit_probe_path)}),
        encoding="utf-8",
    )
    training_service._resolve_memory_probe_auto_config(str(config_path), probe_path)
    cfg = toml.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["memory_probe_jsonl"] == str(explicit_probe_path)

    args = [
        "python",
        "train.py",
        "--memory_probe_jsonl",
        "auto",
        "--block_swap_profile_jsonl=auto",
    ]
    resolved = training_service._resolve_memory_probe_auto_arg(args, probe_path)
    assert resolved[3] == str(probe_path)
    assert resolved[-1] == "--block_swap_profile_jsonl=auto"

def test_peak_probe_auto_config_targets_current_history_task(tmp_path, monkeypatch):
    _patch_runtime_service_paths(monkeypatch, tmp_path)
    history_dir = tmp_path / "configs" / "web-training-history"
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    config_path = tmp_path / "output" / "runs" / "demo" / "config.runtime.toml"
    config_path.parent.mkdir(parents=True)
    probe_path = history_dir / "task-new" / "peak_probe.jsonl"

    config_path.write_text('peak_probe_jsonl = "auto"\n', encoding="utf-8")
    training_service._resolve_peak_probe_auto_config(str(config_path), probe_path)
    cfg = toml.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["peak_probe_jsonl"] == str(probe_path)

    old_probe_path = history_dir / "task-old" / "peak_probe.jsonl"
    config_path.write_text(
        toml.dumps({"peak_probe_jsonl": str(old_probe_path)}),
        encoding="utf-8",
    )
    training_service._resolve_peak_probe_auto_config(str(config_path), probe_path)
    cfg = toml.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["peak_probe_jsonl"] == str(probe_path)

    explicit_probe_path = tmp_path / "logs" / "explicit-peak.jsonl"
    config_path.write_text(
        toml.dumps({"peak_probe_jsonl": str(explicit_probe_path)}),
        encoding="utf-8",
    )
    training_service._resolve_peak_probe_auto_config(str(config_path), probe_path)
    cfg = toml.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["peak_probe_jsonl"] == str(explicit_probe_path)

    args = [
        "python",
        "train.py",
        "--peak_probe_jsonl",
        "auto",
        "--memory_probe_jsonl=auto",
    ]
    resolved = training_service._resolve_peak_probe_auto_arg(args, probe_path)
    assert resolved[3] == str(probe_path)
    assert resolved[-1] == "--memory_probe_jsonl=auto"

