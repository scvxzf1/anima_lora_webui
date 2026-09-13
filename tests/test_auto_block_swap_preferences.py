from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from library.training.auto_block_swap.config import add_arguments, configuration_errors, probe_arguments
from library.training.auto_block_swap.policy import host_swap_limit, SwapSearch, Measurement
from library.training.auto_block_swap.preferences import gpu_reserve_bytes
from library.training.auto_block_swap.online_policy import OnlineSwapPolicy
from library.training.auto_block_swap.resources import GIB, HostMemory


def test_reserve_percent_is_of_total_capacity_with_hard_safety_floor():
    args = SimpleNamespace(auto_block_swap_vram_reserve_percent=25)
    assert gpu_reserve_bytes(args, 16 * GIB) == 4 * GIB
    args.auto_block_swap_vram_reserve_percent = 0
    assert gpu_reserve_bytes(args, 16 * GIB) == GIB
    assert gpu_reserve_bytes(SimpleNamespace(), 24 * GIB) >= 2.4 * GIB


@pytest.mark.parametrize("value", [-1, 91, float("nan"), float("inf"), True, "bad", None])
def test_invalid_percent_is_rejected(value):
    errors = configuration_errors({
        "auto_block_swap": True, "auto_block_swap_vram_reserve_percent": value,
    })
    assert any("reserve_percent" in e for e in errors)


def test_preference_scope_and_disabled_compatibility():
    assert configuration_errors({"auto_block_swap": False, "auto_block_swap_preference": "old"}) == []
    assert any("preference" in e for e in configuration_errors({
        "auto_block_swap": True, "auto_block_swap_preference": "invalid",
    }))
    assert any("Krea-2" in e for e in configuration_errors({
        "auto_block_swap": True, "auto_block_swap_preference": "ram",
    }))


def test_parser_and_probe_preserve_resource_options(tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([
        "--auto_block_swap", "--auto_block_swap_vram_reserve_percent", "25.5",
        "--auto_block_swap_preference", "ram",
    ])
    probe = probe_arguments(args, tmp_path, blocks=2)
    assert probe.auto_block_swap_vram_reserve_percent == 25.5
    assert probe.auto_block_swap_preference == "ram"
    assert args.auto_block_swap and not probe.auto_block_swap


@pytest.mark.parametrize("family", [{}, {"model_family": None}, {"model_family": "krea2"}])
def test_preference_respects_model_family_resolution(monkeypatch, family):
    monkeypatch.setenv("ANIMA_MODEL_FAMILY", "krea2_raw")
    assert configuration_errors({
        "auto_block_swap": True, "auto_block_swap_preference": "ram",
        "auto_block_swap_mode": "dynamic", **family,
    }) == []


def test_small_ram_inventory_accounts_for_only_exchange_pairs():
    assert host_swap_limit([10] * 28, available=180, reserve=20) == 0
    assert host_swap_limit([10] * 28, available=180, reserve=20, sparse=True) == 4
    assert host_swap_limit([10] * 28, available=69, reserve=20, sparse=True) == 0


@pytest.mark.parametrize("preference,expected", [("balanced", 2), ("vram", 4), ("ram", 0)])
def test_startup_preference_only_selects_measured_safe_candidates(preference, expected):
    search = SwapSearch(4, preference=preference)
    for blocks, seconds in [(4, 10.8), (2, 10), (0, 10.5)]:
        search.observe(Measurement(blocks, True, seconds))
    assert search.select() == expected


def paired(preference, candidate_seconds, *, current=4, direction=-1):
    policy = OnlineSwapPolicy(4, interval=4, warmup=0, preference=preference)
    policy.current = current
    policy.direction = direction
    for step, seconds in enumerate([10] * 4 + [candidate_seconds] * 4 + [10] * 4, 1):
        decision = policy.observe("shape", seconds, step=step, remaining=1000, promotion_floor=0)
        if decision:
            policy.changed(decision.blocks)
    return policy


def test_ram_preference_accepts_resource_saving_without_speed_gain():
    assert paired("ram", 10.5).current == 2
    assert paired("balanced", 10.5).current == 4
    assert not paired("ram", 11.01).last_verdict["accepted"]


def test_vram_preference_does_not_spend_more_vram_for_small_speed_gain():
    assert not paired("vram", 9.5).last_verdict["accepted"]
    assert paired("vram", 10.5, current=2, direction=1).current == 4


def test_sparse_master_growth_guard_keeps_host_reserve(monkeypatch):
    from library.training.auto_block_swap import online_memory
    from library.training.auto_block_swap.online_policy import Decision

    control = SimpleNamespace(
        block_bytes=[GIB] * 6,
        offloader=SimpleNamespace(_cpu_weight_masters=[{"base": 1}, {}, {}, {}, {}, {"base": 1}]),
    )
    monkeypatch.setattr(online_memory, "host_memory", lambda: HostMemory(16 * GIB, 4 * GIB, 0, 0, 0, 0))
    assert not online_memory.check_master_growth(control, Decision(2, "explore"))
    with pytest.raises(RuntimeError, match="both host RAM and GPU"):
        online_memory.check_master_growth(control, Decision(2, "gpu_pressure"))
    assert online_memory.check_master_growth(control, Decision(0, "explore"))


@pytest.mark.parametrize("preference,maximum", [("ram", 3), ("balanced", 0)])
def test_after_load_uses_sparse_host_budget_and_scope(monkeypatch, preference, maximum):
    from library.training.auto_block_swap import probe
    from library.training.auto_block_swap.process import ProbeComplete

    monkeypatch.setattr(probe, "model_block_bytes", lambda *a: [GIB] * 28)
    monkeypatch.setattr(probe, "host_memory", lambda: HostMemory(16 * GIB, 14 * GIB, 0, 0, 0, 0))
    model = SimpleNamespace(offloader=SimpleNamespace(master_scope="all"))
    args = SimpleNamespace(
        model_family="krea2_raw", auto_block_swap_preference=preference,
        _auto_swap_probe={"inventory": True}, blocks_to_swap=3,
    )
    with pytest.raises(ProbeComplete) as result:
        probe.after_model_load(args, model)
    assert result.value.result["host_swap_limit"] == maximum
    assert model.offloader.master_scope == ("participating" if preference == "ram" else "all")


def test_probe_and_fixed_monitor_use_custom_total_vram_reserve(monkeypatch):
    from library.training.auto_block_swap import monitor, probe
    from library.training.auto_block_swap.process import ProbeComplete

    args = SimpleNamespace(
        _auto_swap_probe={"inventory": False}, _auto_swap_resolved=True,
        blocks_to_swap=2, auto_block_swap_vram_reserve_percent=25,
        _auto_swap_report="summary.json",
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=16 * GIB))
    monkeypatch.setattr(probe, "representative_batches", lambda _: [{}])
    samples = [{"headroom": 3 * GIB, "case": 0, "update": i, "seconds": 1.0} for i in range(3)]
    monkeypatch.setattr(probe, "_probe_updates", lambda *a: samples)
    state = SimpleNamespace(
        args=args, accelerator=SimpleNamespace(
            device=torch.device("cuda"), num_processes=1, unwrap_model=lambda x: x,
        ), train_ctx=SimpleNamespace(train_text_encoder=False),
        train_dataloader=SimpleNamespace(dataset=Mock()), current_epoch=SimpleNamespace(value=0),
        network=Mock(), unet=SimpleNamespace(), text_encoder=None, optimizer_train_fn=Mock(),
    )
    with pytest.raises(ProbeComplete) as result:
        probe.run_probe(Mock(), state)
    assert result.value.result["gpu_reserve"] == 4 * GIB
    assert not result.value.result["safe"]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (3 * GIB, 16 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: GIB)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: GIB)
    monkeypatch.setattr(monitor, "host_memory", lambda: HostMemory(16 * GIB, 10 * GIB, 0, 0, 0, 0))
    monitor.observe_training(args, "cuda", 16)
    assert args._auto_swap_warned


def test_resource_preferences_survive_web_patch_and_toml_reload(monkeypatch, tmp_path):
    import tomllib
    from tests.web_config_test_support import _patch_config_service_paths, _write_minimal_config_tree
    from web.services import config_service

    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    values = {
        "auto_block_swap_vram_reserve_percent": 25.5,
        "auto_block_swap_preference": "ram",
    }
    ok, message, content, changed, warnings = config_service.patch_raw_file_values(
        "configs/imported/lora.toml", values,
    )
    assert ok, message
    assert set(values) <= set(changed)
    assert not any("auto_block_swap" in warning for warning in warnings)
    saved = tomllib.loads((configs / "imported/lora.toml").read_text())
    assert saved == tomllib.loads(content)
    assert all(saved[key] == value for key, value in values.items())
