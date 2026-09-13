import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import auto_block_swap_probe as experiment


def result(blocks, *, safe=True, status="ok"):
    return {
        "blocks": blocks,
        "status": status,
        "safe": safe,
        "seconds": 1 + blocks / 10,
        "headroom": 10,
        "host_min_available": 10,
        "host_reserve": 1,
        "swap_io_bytes": 0,
        "elapsed_seconds": 5,
    }


def test_no_reserve_only_removes_gpu_margin_threshold():
    sample = result(12, safe=False)
    assert experiment.ablation_measurement(sample, 12, "no-reserve").safe
    assert not experiment.ablation_measurement(sample, 12, "no-bisection").safe
    sample["swap_io_bytes"] = 128 * 1024**2
    with pytest.raises(RuntimeError, match="RAM/paging"):
        experiment.ablation_measurement(sample, 12, "no-reserve")
    assert not experiment.ablation_measurement(
        result(0, safe=False, status="cuda_oom"), 0, "no-reserve"
    ).safe


def test_shared_prefix_is_reused_but_confirmation_is_independent(tmp_path, monkeypatch):
    prior = tmp_path / "full"
    prior.mkdir()
    (prior / "summary.json").write_text(json.dumps({"trials": [result(4), result(2)]}))
    calls = []

    def runner(args, root, label, blocks, *a, **k):
        calls.append((label, blocks))
        return result(blocks)

    monkeypatch.setattr(experiment, "run_candidate", runner)
    experiment.ablation(SimpleNamespace(), tmp_path, "fixed-stride", 4, 3, "full")
    report = json.loads((tmp_path / "fixed-stride/summary.json").read_text())
    assert [row["blocks"] for row in report["trials"]] == [4, 3, 2]
    assert calls == [
        ("fixed-stride/trial-01-swap-3", 3),
        ("fixed-stride/confirm-swap-2", 2),
    ]
    assert report["reused_trials"] == 2
    assert report["measured_candidate_seconds"] == 15
    assert report["confirmation_passed"]


def test_formal_acceptance_requires_confirmation_and_isolates_output(
    tmp_path, monkeypatch
):
    import train

    prior = tmp_path / "full"
    prior.mkdir()
    source = prior / "summary.json"
    source.write_text(json.dumps({"status": "calibrating"}))
    args = SimpleNamespace(auto_block_swap=True, config_snapshot=False)
    with pytest.raises(ValueError, match="confirmed"):
        experiment.run_formal(args, tmp_path, "formal", "full", 6)
    source.write_text(
        json.dumps(
            {
                "status": "selected",
                "selected_blocks": 4,
                "confirmation": {"safe": True},
            }
        )
    )
    calls = []
    saved_steps = 6

    def train_stub(args):
        import torch
        from safetensors.torch import save_file

        calls.append(args)
        save_file(
            {"weight": torch.ones(1)},
            str(Path(args.output_dir) / "final.safetensors"),
            metadata={"ss_steps": str(saved_steps)},
        )

    monkeypatch.setattr(
        train, "AnimaTrainer", lambda: SimpleNamespace(train=train_stub)
    )
    experiment.run_formal(args, tmp_path, "formal", "full", 6)
    assert calls == [args]
    assert args._auto_swap_resolved and args.auto_block_swap
    assert args.blocks_to_swap == 4
    assert args.max_train_steps == 6
    assert args.max_train_epochs is None
    assert args.save_every_n_steps is None
    assert args.save_every_n_epochs is None
    assert args.output_dir == str(tmp_path / "formal")
    assert not args.save_state and not args.save_state_on_train_end
    assert args.resume is None and not args.config_snapshot
    assert (tmp_path / "formal/completion.json").exists()
    with pytest.raises(FileExistsError):
        experiment.run_formal(args, tmp_path, "formal", "full", 6)
    saved_steps = 5
    with pytest.raises(RuntimeError, match="completed 5, expected 6"):
        experiment.run_formal(args, tmp_path, "stopped", "full", 6)
    assert not (tmp_path / "stopped/completion.json").exists()
