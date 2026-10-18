"""Bounded, isolated tests for the checkpoint / weight retention engine."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

from web.services import data_management_service as module
from web.services.data_management_retention import (
    DataManagementRetentionService,
    RetentionPolicy,
    classify_file,
    progress_of,
)
from web.services.data_management_service import DataManagementError, DataManagementService


def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DataManagementRetentionService, Path]:
    project = tmp_path / "project"
    runs = project / "output" / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setattr(DataManagementService, "_project_root", staticmethod(lambda: project.resolve()))
    monkeypatch.setattr(module.settings_service, "resolve_output_root", lambda: runs)
    monkeypatch.setattr(module, "get_configs_root", lambda: project / "configs")
    monkeypatch.setattr(module, "get_training_history_root", lambda: project / "history")
    monkeypatch.setattr(module, "get_training_queue_root", lambda: project / "queue")
    app = web.Application()
    app["training_service"] = SimpleNamespace(
        status="idle", current_task_dir=None, current_output_dir="",
        current_runtime_info={}, _queue={"items": []},
    )
    app["image_test_service"] = None
    app["tagging_service"] = None
    return DataManagementRetentionService(app), runs


def _unit(runs: Path, rel: str) -> Path:
    path = runs / rel
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint(unit: Path, base: str, step: int, size: int = 100) -> None:
    (unit / f"{base}-step{step:08d}.safetensors").write_bytes(b"w" * size)
    state = unit / f"{base}-step{step:08d}-state"
    state.mkdir()
    (state / "optimizer.bin").write_bytes(b"o" * (size * 3))
    (state / "train_state.json").write_bytes(b"{}")


def _final(unit: Path, base: str, size: int = 100) -> None:
    (unit / f"{base}.safetensors").write_bytes(b"f" * size)


def _decisions(response: dict, unit_rel: str) -> dict[str, str]:
    group = next(item for item in response["groups"] if item["relative_path"] == unit_rel)
    return {item["name"]: item["action"] for item in group["items"]}


def test_progress_parsing_is_natural_and_date_safe():
    assert progress_of("anima-no-ckpt-step1500.safetensors") == 1500
    assert progress_of("anima-no-ckpt-step1500-step00001000.safetensors") == 1000
    assert progress_of("model_step_00500.safetensors") == 500
    assert progress_of("checkpoint-500") == 500
    assert progress_of("default_20260904-1315", allow_fallback=False) is None
    assert progress_of("adapter-000500.safetensors") == 500
    assert classify_file("status.tsv")[0] == "protected"
    assert classify_file("optimizer.bin")[0] == "optimizer_state"
    assert classify_file("random_states_0.pkl")[0] == "optimizer_state"
    assert classify_file("run_final.safetensors")[0] == "final_weight"


def test_keep_top_n_is_applied_per_experiment_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    first = _unit(runs, "exp-a/ckpt/anima")
    second = _unit(runs, "exp-b/ckpt/lokr")
    for step in (200, 400, 600, 1000):
        _checkpoint(first, "anima", step)
    for step in (100, 200, 300):
        _checkpoint(second, "lokr", step)

    response = service.preview("project-output", ["runs/exp-a", "runs/exp-b"],
                               {"mode": "keep_top_n", "keep_count": 2, "clean_optimizer_states": False})

    first_actions = _decisions(response, "runs/exp-a/ckpt/anima")
    second_actions = _decisions(response, "runs/exp-b/ckpt/lokr")
    kept_weights = [name for name, action in first_actions.items()
                    if action == "keep" and name.endswith(".safetensors")]
    assert kept_weights == ["anima-step00001000.safetensors", "anima-step00000600.safetensors"]
    assert sum(1 for name, action in first_actions.items()
                if action == "delete" and name.endswith(".safetensors")) == 2
    assert sum(1 for name, action in second_actions.items()
                if action == "keep" and name.endswith(".safetensors")) == 2
    assert sum(1 for action in second_actions.values() if action == "delete") == 1
    assert response["summary"]["group_count"] == 2
    assert response["summary"]["matched_group_count"] == 2


def test_delete_top_m_removes_the_earliest_progress_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    for step in (100, 200, 300, 400):
        _checkpoint(unit, "lora", step)

    response = service.preview("project-output", ["runs/exp"], {
        "mode": "delete_top_m", "delete_count": 2, "sort_by": "progress",
        "sort_order": "asc", "clean_optimizer_states": False,
    })
    actions = _decisions(response, "runs/exp/ckpt/lora")
    assert [name for name, action in actions.items() if action == "delete"] == [
        "lora-step00000100.safetensors", "lora-step00000200.safetensors",
    ]


def test_keep_first_last_removes_only_the_middle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    for step in (100, 200, 300, 400, 500):
        _checkpoint(unit, "lora", step)

    response = service.preview("project-output", ["runs/exp"], {
        "mode": "keep_first_last", "keep_first": 1, "keep_last": 1, "clean_optimizer_states": False,
    })
    actions = _decisions(response, "runs/exp/ckpt/lora")
    kept_weights = [name for name, action in actions.items()
                    if action == "keep" and name.endswith(".safetensors")]
    assert sorted(kept_weights) == ["lora-step00000100.safetensors", "lora-step00000500.safetensors"]
    assert sum(1 for action in actions.values() if action == "delete") == 3


def test_delete_all_checkpoints_keeps_the_exported_final_weight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/anima")
    _final(unit, "anima-step1500")
    for step in (500, 1000):
        _checkpoint(unit, "anima-step1500", step)

    response = service.preview("project-output", ["runs/exp"], {
        "mode": "delete_all_checkpoints", "clean_optimizer_states": True,
    })
    actions = _decisions(response, "runs/exp/ckpt/anima")
    assert actions["anima-step1500.safetensors"] == "keep"
    assert sum(1 for action in actions.values() if action == "delete") == 4


def test_safety_lock_keeps_one_weight_without_a_final_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    for step in (100, 200):
        _checkpoint(unit, "checkpoint", step)

    response = service.preview("project-output", ["runs/exp"], {
        "mode": "delete_all_checkpoints", "clean_optimizer_states": False,
    })
    group = next(item for item in response["groups"] if item["relative_path"] == "runs/exp/ckpt/lora")
    kept_weights = [item for item in group["items"]
                    if item["action"] == "keep" and item["role"] in {"checkpoint", "final_weight"}]
    assert len(kept_weights) == 1
    assert kept_weights[0]["reason"].startswith("保底安全锁")


def test_optimizer_states_and_patterns_are_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    for step in (100, 200, 300):
        _checkpoint(unit, "lora", step)
    (unit / "lora_best.safetensors").write_bytes(b"b" * 10)

    keep_states = service.preview("project-output", ["runs/exp"], {
        "mode": "keep_top_n", "keep_count": 1, "clean_optimizer_states": False,
    })
    state_actions = _decisions(keep_states, "runs/exp/ckpt/lora")
    assert any(name.endswith("-state") and action == "keep" for name, action in state_actions.items())

    protected = service.preview("project-output", ["runs/exp"], {
        "mode": "delete_top_m", "delete_count": 3, "keep_weight_patterns": ["*best*"],
    })
    actions = _decisions(protected, "runs/exp/ckpt/lora")
    assert actions["lora_best.safetensors"] == "keep"
    assert actions["lora-step00000100.safetensors"] in {"keep", "delete"}


def test_protected_files_and_foreign_checkpoint_dirs_are_never_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    _checkpoint(unit, "lora", 100)
    (unit / "status.tsv").write_bytes(b"metrics")
    (unit / "console.log").write_bytes(b"log")
    (unit / "preview.png").write_bytes(b"png")
    foreign = unit / "checkpoint-500"
    foreign.mkdir()
    (foreign / "weights.safetensors").write_bytes(b"w")
    (foreign / "console.log").write_bytes(b"log")

    response = service.preview("project-output", ["runs/exp"], {"mode": "delete_all_checkpoints"})
    actions = _decisions(response, "runs/exp/ckpt/lora")
    assert actions["status.tsv"] == "keep"
    assert actions["console.log"] == "keep"
    assert actions["preview.png"] == "keep"
    assert actions["checkpoint-500"] == "keep"
    assert response["blocked"]


def test_cleanup_requires_the_same_plan_revision_and_item_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    for step in (100, 200):
        _checkpoint(unit, "lora", step)

    policy = {"mode": "keep_top_n", "keep_count": 1, "clean_optimizer_states": False}
    preview = service.preview("project-output", ["runs/exp"], policy)
    expected = [item["relative_path"] for group in preview["groups"]
                for item in group["items"] if item["action"] == "delete"]

    with pytest.raises(DataManagementError, match="重新预览"):
        service.cleanup("project-output", ["runs/exp"], policy, expected, "stale-revision", True)
    with pytest.raises(DataManagementError, match="重新预览"):
        service.cleanup("project-output", ["runs/exp"], policy, expected[:0], preview["plan_revision"], True)
    with pytest.raises(DataManagementError, match="confirmed"):
        service.cleanup("project-output", ["runs/exp"], policy, expected, preview["plan_revision"], False)


def test_cleanup_deletes_only_the_planned_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    for step in (100, 200):
        _checkpoint(unit, "lora", step)
    (unit / "status.tsv").write_bytes(b"metrics")

    policy = {"mode": "keep_top_n", "keep_count": 1, "clean_optimizer_states": True}
    preview = service.preview("project-output", ["runs/exp"], policy)
    expected = [item["relative_path"] for group in preview["groups"]
                for item in group["items"] if item["action"] == "delete"]
    result = service.cleanup("project-output", ["runs/exp"], policy, expected, preview["plan_revision"], True)

    assert result["ok"] is True
    assert not (unit / "lora-step00000100.safetensors").exists()
    assert not (unit / "lora-step00000100-state").exists()
    assert (unit / "lora-step00000200.safetensors").exists()
    # 保留的检查点仍按开关统一清理优化器 / 恢复状态
    assert not (unit / "lora-step00000200-state").exists()
    assert (unit / "status.tsv").exists()
    assert set(result["deleted"]) == set(expected)


def test_scope_rejects_nested_duplicate_and_missing_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    _unit(runs, "exp/ckpt/lora")
    policy = {"mode": "keep_top_n", "keep_count": 1}

    nested = service.preview("project-output", ["runs/exp", "runs/exp/ckpt"], policy)
    assert nested["scope"]["dirs"] == ["runs/exp"]

    with pytest.raises(DataManagementError, match="重复"):
        service.preview("project-output", ["runs/exp", "runs/exp"], policy)
    with pytest.raises(DataManagementError, match="不存在"):
        service.preview("project-output", ["runs/missing"], policy)
    with pytest.raises(DataManagementError):
        service.preview("models", ["whatever"], policy)


def test_policy_validation_rejects_out_of_range_values():
    with pytest.raises(DataManagementError, match="keep_count"):
        RetentionPolicy.from_payload({"keep_count": -1})
    with pytest.raises(DataManagementError, match="sort_by"):
        RetentionPolicy.from_payload({"sort_by": "random"})
    with pytest.raises(DataManagementError, match="mode"):
        RetentionPolicy.from_payload({"mode": "destroy_all"})
    with pytest.raises(DataManagementError, match="保留规则"):
        RetentionPolicy.from_payload({"keep_weight_patterns": ["*" * 100]})
    policy = RetentionPolicy.from_payload({"keep_weight_patterns": "best, *final*, ema"})
    assert policy.keep_weight_patterns == ("best", "*final*", "ema")


def test_symlinked_artifacts_are_skipped_silently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    _checkpoint(unit, "lora", 100)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (unit / "lora-step00000050.safetensors").symlink_to(outside)

    response = service.preview("project-output", ["runs/exp"], {"mode": "delete_all_checkpoints"})
    actions = _decisions(response, "runs/exp/ckpt/lora")
    assert actions["lora-step00000050.safetensors"] == "keep"
    assert outside.exists()
    # 软链接默认静默跳过：既不删除，也不进入刺眼的“阻止项”列表
    assert response["blocked"] == []


def test_cache_and_analysis_artifacts_are_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    _checkpoint(unit, "lora", 100)
    cache_dir = unit / "cache"
    cache_dir.mkdir()
    (cache_dir / "latent.npz").write_bytes(b"c" * 40)
    (unit / "train_cache.npz").write_bytes(b"n" * 20)
    dump_dir = unit / "dump"
    dump_dir.mkdir()
    (dump_dir / "activation.npy").write_bytes(b"a" * 30)
    (unit / "gradients.npy").write_bytes(b"g" * 10)
    (unit / "summary.json").write_bytes(b"{}")

    default = service.preview("project-output", ["runs/exp"], {"mode": "keep_top_n", "keep_count": 5})
    default_actions = _decisions(default, "runs/exp/ckpt/lora")
    assert default_actions["cache"] == "keep"
    assert default_actions["train_cache.npz"] == "keep"
    assert default_actions["dump"] == "keep"
    assert default_actions["gradients.npy"] == "keep"

    cache_only = service.preview("project-output", ["runs/exp"], {
        "mode": "keep_top_n", "keep_count": 5, "clean_data_cache": True,
    })
    cache_actions = _decisions(cache_only, "runs/exp/ckpt/lora")
    assert cache_actions["cache"] == "delete"
    assert cache_actions["train_cache.npz"] == "delete"
    assert cache_actions["dump"] == "keep"
    assert cache_actions["gradients.npy"] == "keep"

    analysis_only = service.preview("project-output", ["runs/exp"], {
        "mode": "keep_top_n", "keep_count": 5, "clean_analysis_tensors": True,
    })
    analysis_actions = _decisions(analysis_only, "runs/exp/ckpt/lora")
    assert analysis_actions["cache"] == "keep"
    assert analysis_actions["dump"] == "delete"
    assert analysis_actions["gradients.npy"] == "delete"
    assert analysis_actions["summary.json"] == "keep"

    both = service.preview("project-output", ["runs/exp"], {
        "mode": "keep_top_n", "keep_count": 5, "clean_optimizer_states": False,
        "clean_data_cache": True, "clean_analysis_tensors": True,
    })
    assert both["summary"]["delete_count"] == 4
    assert _decisions(both, "runs/exp/ckpt/lora")["summary.json"] == "keep"


def test_exclude_units_keeps_a_whole_experiment_and_changes_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    first = _unit(runs, "exp-a/ckpt/lora")
    second = _unit(runs, "exp-b/ckpt/lora")
    for step in (100, 200, 300):
        _checkpoint(first, "a", step)
        _checkpoint(second, "b", step)
    policy = {"mode": "keep_top_n", "keep_count": 1, "clean_optimizer_states": False}

    full = service.preview("project-output", ["runs/exp-a", "runs/exp-b"], policy)
    excluded = service.preview("project-output", ["runs/exp-a", "runs/exp-b"], policy,
                               ["runs/exp-b/ckpt/lora"])

    assert full["summary"]["delete_count"] == 4
    assert excluded["summary"]["delete_count"] == 2
    assert excluded["summary"]["excluded_group_count"] == 1
    assert excluded["plan_revision"] != full["plan_revision"]
    group = next(item for item in excluded["groups"] if item["relative_path"] == "runs/exp-b/ckpt/lora")
    assert group["excluded"] is True
    assert all(item["action"] == "keep" for item in group["items"])
    assert any(item["reason"] == "已豁免：手动跳过该实验" for item in group["items"])


def test_exclude_units_rejects_invalid_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    _unit(runs, "exp/ckpt/lora")
    policy = {"mode": "keep_top_n", "keep_count": 1}
    with pytest.raises(DataManagementError, match="exclude_units"):
        service.preview("project-output", ["runs/exp"], policy, "not-a-list")
    with pytest.raises(DataManagementError, match="exclude_units"):
        service.preview("project-output", ["runs/exp"], policy, ["../escape"])
    with pytest.raises(DataManagementError, match="exclude_units"):
        service.preview("project-output", ["runs/exp"], policy, [""])

def test_large_batch_returns_a_short_revision_and_accepts_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A few hundred targets must not produce a multi-KB revision token."""
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    for step in range(1, 61):
        (unit / f"lora-step{step:08d}.safetensors").write_bytes(b"w" * 8)
        state = unit / f"lora-step{step:08d}-state"
        state.mkdir()
        (state / "optimizer.bin").write_bytes(b"o" * 8)
        (state / "train_state.json").write_bytes(b"{}")

    policy = {"mode": "keep_top_n", "keep_count": 2, "clean_optimizer_states": True}
    preview = service.preview("project-output", ["runs/exp"], policy)
    assert preview["summary"]["delete_count"] == 118
    assert len(preview["plan_revision"]) <= 96
    expected = [item["relative_path"] for group in preview["groups"]
                for item in group["items"] if item["action"] == "delete"]

    result = service.cleanup("project-output", ["runs/exp"], policy, expected,
                             preview["plan_revision"], True)
    assert result["ok"] is True
    assert len(result["deleted"]) == 118
    assert (unit / "lora-step00000060.safetensors").exists()
    assert not (unit / "lora-step00000001.safetensors").exists()



def test_cache_directory_with_weight_or_unknown_file_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    _checkpoint(unit, "lora", 100)
    cache_dir = unit / "cache"
    cache_dir.mkdir()
    (cache_dir / "exported.safetensors").write_bytes(b"w" * 20)
    (cache_dir / "research.bin").write_bytes(b"b" * 20)

    response = service.preview("project-output", ["runs/exp"], {
        "mode": "keep_top_n", "keep_count": 2, "clean_data_cache": True,
    })
    actions = _decisions(response, "runs/exp/ckpt/lora")
    assert actions["cache"] == "keep"
    assert any(item["relative_path"].endswith("/cache") for item in response["blocked"])


def test_more_than_batch_limit_is_rejected_without_truncating_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    for step in range(1, 403):
        (unit / f"lora-step{step:08d}.safetensors").write_bytes(b"w")

    with pytest.raises(DataManagementError, match="超过单次上限"):
        service.preview("project-output", ["runs/exp"], {
            "mode": "delete_all_checkpoints", "clean_optimizer_states": False,
        })
    assert len(list(unit.glob("*.safetensors"))) == 402



def test_execution_rejects_target_changed_after_plan_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    target = unit / "lora-step00000001.safetensors"
    target.write_bytes(b"old")
    (unit / "lora-step00000002.safetensors").write_bytes(b"new")
    policy = {"mode": "delete_all_checkpoints", "clean_optimizer_states": False}
    preview = service.preview("project-output", ["runs/exp"], policy)
    expected = [item["relative_path"] for group in preview["groups"]
                for item in group["items"] if item["action"] == "delete"]
    original_revalidate = service.base._revalidate_target
    mutated = False

    def mutate_before_revalidate(spec, candidate, expected_state=None):
        nonlocal mutated
        if not mutated:
            mutated = True
            target.write_bytes(b"replacement-with-different-size")
        return original_revalidate(spec, candidate, expected_state)

    monkeypatch.setattr(service.base, "_revalidate_target", mutate_before_revalidate)
    result = service.cleanup("project-output", ["runs/exp"], policy, expected,
                             preview["plan_revision"], True)
    assert result["deleted"] == []
    assert result["blocked"]
    assert target.read_bytes() == b"replacement-with-different-size"



def test_optimizer_bundle_with_user_file_is_not_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    _checkpoint(unit, "lora", 100)
    state = unit / "lora-step00000100-state"
    (state / "console.log").write_text("keep me", encoding="utf-8")

    response = service.preview("project-output", ["runs/exp"], {
        "mode": "keep_top_n", "keep_count": 2, "clean_optimizer_states": True,
    })
    actions = _decisions(response, "runs/exp/ckpt/lora")
    assert actions[state.name] == "keep"
    assert any(item["relative_path"].endswith(state.name) for item in response["blocked"])


def test_directory_content_change_after_plan_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    # A checkpoint is only a delete candidate when every artifact inside it is a
    # recognized one; a foreign file (weight.safetensors) makes the whole unit
    # "keep - protected" and never reaches the revalidation guard under test.
    _checkpoint(unit, "lora", 1)
    target = unit / "lora-step00000001-state"
    weight = target / "optimizer.bin"
    policy = {"mode": "delete_all_checkpoints", "clean_optimizer_states": True}
    preview = service.preview("project-output", ["runs/exp"], policy)
    expected = [item["relative_path"] for group in preview["groups"]
                for item in group["items"] if item["action"] == "delete"]
    original_revalidate = service.base._revalidate_target
    changed = False

    def change_child_before_revalidate(spec, candidate, expected_state=None):
        nonlocal changed
        if not changed:
            changed = True
            weight.write_bytes(b"new-value-after-plan")
        return original_revalidate(spec, candidate, expected_state)

    monkeypatch.setattr(service.base, "_revalidate_target", change_child_before_revalidate)
    result = service.cleanup("project-output", ["runs/exp"], policy, expected,
                             preview["plan_revision"], True)
    assert result["deleted"] == []
    assert any("目录内容" in item["reason"] for item in result["blocked"])
    assert weight.read_bytes() == b"new-value-after-plan"


def test_fingerprint_scan_failure_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    _checkpoint(unit, "lora", 100)
    state = unit / "lora-step00000100-state"
    monkeypatch.setattr(service.base, "_fingerprint_tree", lambda _path: (_ for _ in ()).throw(OSError("denied")))

    response = service.preview("project-output", ["runs/exp"], {"mode": "delete_all_checkpoints"})
    assert response["summary"]["delete_count"] == 0
    assert response["blocked"]
    assert state.exists()


def test_exact_batch_limit_is_not_rejected_or_marked_truncated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs = _service(tmp_path, monkeypatch)
    unit = _unit(runs, "exp/ckpt/lora")
    for step in range(1, 402):
        (unit / f"checkpoint-step{step:08d}.safetensors").write_bytes(b"w")
    response = service.preview("project-output", ["runs/exp"], {
        "mode": "delete_all_checkpoints", "ensure_at_least_one_weight": True,
        "clean_optimizer_states": False,
    })
    assert response["summary"]["delete_count"] == 400
    group = next(item for item in response["groups"] if item["relative_path"] == "runs/exp/ckpt/lora")
    assert group["truncated"] is False
