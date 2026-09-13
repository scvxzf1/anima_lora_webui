from argparse import ArgumentParser, Namespace
import copy
import json

import pytest

from library.training.auto_block_swap.config import (
    add_arguments,
    configuration_errors,
    probe_arguments,
)
from library.training.auto_block_swap import coordinator


def args(**updates):
    values = dict(
        auto_block_swap=True,
        blocks_to_swap=7,
        auto_block_swap_max_trials=3,
        auto_block_swap_timeout=100,
        seed=42,
        output_dir="output/user",
        model_family="anima",
        network_module="networks.lora_anima",
        network_args=None,
        use_vae_cache=True,
        use_text_cache=True,
        gradient_accumulation_steps=4,
        max_train_steps=100,
    )
    values.update(updates)
    return Namespace(**values)


def test_cli_explicit_opt_in_and_disable():
    parser = ArgumentParser()
    add_arguments(parser)
    assert not parser.parse_args([]).auto_block_swap
    assert parser.parse_args(["--auto_block_swap"]).auto_block_swap
    assert not parser.parse_args(
        ["--auto_block_swap", "--no-auto_block_swap"]
    ).auto_block_swap


def test_system_swap_io_limit_is_configurable():
    parser = ArgumentParser()
    add_arguments(parser)
    parsed = parser.parse_args(["--auto_block_swap", "--auto_block_swap_swap_io_limit_mb", "512"])
    assert parsed.auto_block_swap_swap_io_limit_mb == 512
    assert not configuration_errors(args(auto_block_swap_swap_io_limit_mb=512))
    assert any(
        "swap_io_limit" in error
        for error in configuration_errors(args(auto_block_swap_swap_io_limit_mb=-1))
    )


@pytest.mark.parametrize(
    "update",
    [
        {"use_vae_cache": False},
        {"use_text_cache": False},
        {"network_args": ["foo=bar"]},
        {"network_module": "custom"},
        {"use_moe_style": "shared_A"},
        {"use_ortho": True},
        {"lora_adapter_kind": "lokr"},
        {"base_compute": "convrot"},
        {"cpu_offload_checkpointing": True},
        {"prior_preservation_weight": 0.1},
        {"inverted_mask_prior_weight": 0.1},
        {"artist_filter": "example"},
        {"debug_dataset": True},
        {"stage_schedule": [1]},
        {"resume": "state"},
        {"sample_prompts": "prompt.txt", "sample_at_first": True},
        {"auto_block_swap_max_trials": 0},
        {"auto_block_swap_timeout": 0},
    ],
)
def test_unsupported_contract_fails_closed(update):
    assert configuration_errors(args(**update))
    assert not configuration_errors(args(auto_block_swap=False, **update))


def test_single_process_only():
    assert not configuration_errors(args())
    assert configuration_errors(args(), world_size=2)


def test_probe_arguments_isolate_outputs_without_changing_training_math(tmp_path):
    original = args(
        log_with="wandb",
        huggingface_token="secret",
        sample_prompts="x",
        save_state=True,
    )
    before = copy.deepcopy(vars(original))
    probe = probe_arguments(original, tmp_path, blocks=12)
    assert vars(original) == before
    assert probe.gradient_accumulation_steps == 4
    assert probe.max_train_steps == 100
    assert probe.blocks_to_swap == 12
    assert not probe.auto_block_swap
    assert probe.log_with is None and probe.huggingface_token is None
    # Disabled preview prompts still contribute compile sequence bounds.
    assert probe.sample_prompts == "x" and not probe.save_state
    assert not probe.sample_at_first and not probe.sample_every_n_steps
    assert probe.output_dir == str(tmp_path / "output")
    assert probe.skip_cache_check is False
    json.dumps(vars(probe))


def test_disabled_preview_preserves_compile_resolution_inputs(tmp_path):
    from library.training.train_bootstrap import (
        collect_compile_resolutions,
        normalize_sample_args,
    )

    original = args(
        output_dir=str(tmp_path / "formal"),
        sample_prompts=["subject --w 768 --h 1024"],
        sample_at_first=False,
        sample_every_n_steps=0,
    )
    probe = probe_arguments(original, tmp_path / "trial", blocks=12)
    normalize_sample_args(original)
    normalize_sample_args(probe)
    expected = collect_compile_resolutions(sample_prompts=original.sample_prompts)
    assert expected
    assert collect_compile_resolutions(sample_prompts=probe.sample_prompts) == expected


def test_full_coordinator_runs_inventory_search_confirmation(tmp_path):
    calls = []

    def runner(probe, directory, *, timeout):
        calls.append((probe, directory, timeout))
        if probe._auto_swap_probe["inventory"]:
            return {
                "status": "inventory",
                "model_swap_limit": 26,
                "host_swap_limit": 24,
                "block_count": 28,
            }
        blocks = probe.blocks_to_swap
        return {
            "status": "ok",
            "safe": True,
            "seconds": blocks,
            "headroom": 100,
            "host_min_available": 10,
            "host_reserve": 1,
            "swap_io_bytes": 0,
        }

    original = args()
    assert coordinator.run_calibration(original, tmp_path, runner=runner) == 18
    assert [p.blocks_to_swap for p, _, _ in calls] == [1, 24, 22, 18, 18]
    assert original.blocks_to_swap == 7
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "selected"
    assert summary["selected_blocks"] == 18


@pytest.mark.parametrize("status", ["error", "timeout", "host_limit", "killed"])
def test_non_oom_failure_is_not_a_gpu_search_boundary(status):
    with pytest.raises(RuntimeError, match=status):
        coordinator.measurement({"status": status}, 10)


def test_host_paging_is_not_counted_as_free_memory():
    with pytest.raises(RuntimeError, match="paging"):
        coordinator.measurement(
            {
                "status": "ok",
                "safe": True,
                "seconds": 1,
                "headroom": 100,
                "host_min_available": 10,
                "host_reserve": 1,
                "swap_io_bytes": 128 * 1024**2,
            },
            10,
        )


@pytest.mark.parametrize("status", ["ok", "cuda_oom"])
@pytest.mark.parametrize("gpu_safe", [False, True])
@pytest.mark.parametrize("paging", [False, True])
def test_host_failure_precedes_gpu_search_boundary(status, gpu_safe, paging):
    with pytest.raises(RuntimeError, match="RAM/paging"):
        coordinator.measurement(
            {
                "status": status,
                "safe": gpu_safe,
                "host_min_available": 10 if paging else 1,
                "host_reserve": 1,
                "swap_io_bytes": 128 * 1024**2 if paging else 0,
            },
            10,
        )


def test_cuda_oom_with_healthy_host_is_recoverable_boundary():
    assert not coordinator.measurement(
        {
            "status": "cuda_oom",
            "host_min_available": 10,
            "host_reserve": 1,
            "swap_io_bytes": 0,
        },
        10,
    ).safe


def test_manual_mode_does_not_probe(monkeypatch):
    monkeypatch.setattr(
        coordinator, "_validate_launch", lambda _: pytest.fail("manual invoked AUTO")
    )
    original = args(auto_block_swap=False)
    coordinator.calibrate_if_requested(original)
    assert original.blocks_to_swap == 7


@pytest.mark.parametrize("update", [
    {"model_family": "anima"}, {"model_family": "z_image"},
    {"compile_block_scope": "all"}, {"auto_block_swap_interval": 3},
    {"auto_block_swap_interval": 257}, {"auto_block_swap_interval": "bad"},
])
def test_dynamic_contract_rejects_unsupported_modes(update):
    values = {"model_family": "krea2_raw", "auto_block_swap_mode": "dynamic", **update}
    assert configuration_errors(args(**values))


def test_dynamic_cli_preserves_startup_default():
    parser = ArgumentParser()
    add_arguments(parser)
    assert parser.parse_args([]).auto_block_swap_mode == "startup"
    parsed = parser.parse_args(["--auto_block_swap", "--auto_block_swap_mode", "dynamic"])
    assert parsed.auto_block_swap_interval == 8
    assert not configuration_errors(args(model_family="krea2_raw", auto_block_swap_mode="dynamic"))
