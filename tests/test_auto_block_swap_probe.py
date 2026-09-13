from types import SimpleNamespace as NS

import pytest
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState, PartialState

from library.training.auto_block_swap import probe


class Buckets(torch.utils.data.Dataset):
    def __init__(self):
        self.bucket_manager = NS(
            resos=[(512, 512), (1024, 1024)], buckets=[list(range(3)), list(range(4))]
        )
        self.buckets_indices = [
            NS(bucket_index=0, bucket_batch_size=2, batch_index=1),
            NS(bucket_index=0, bucket_batch_size=2, batch_index=0),
            NS(bucket_index=1, bucket_batch_size=2, batch_index=0),
        ]

    def __len__(self):
        return len(self.buckets_indices)

    def __getitem__(self, index):
        return torch.ones(2, 4) * (index + 1)


def test_representatives_cover_resolutions_and_full_batches():
    group = torch.utils.data.ConcatDataset([Buckets(), Buckets()])
    cases = probe.representative_batches(group)
    assert [case["index"] for case in cases] == [2, 5, 1, 4]
    assert all(case["batch_size"] == 2 for case in cases)


@pytest.mark.parametrize(
    "family,attribute",
    [("anima", "blocks"), ("krea2_raw", "blocks"), ("z_image", "layers")],
)
def test_inventory_uses_real_block_count(family, attribute):
    model = NS(
        **{attribute: torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(5)])}
    )
    assert probe.model_block_bytes(model, family) == [64] * 5


def test_inventory_unknown_family_fails_closed():
    with pytest.raises(ValueError):
        probe.model_block_bytes(NS(blocks=[]), "unknown")


@pytest.mark.parametrize("resolved", [False, True])
def test_dropout_requires_existing_sidecar_before_model_load(
    monkeypatch, tmp_path, resolved
):
    from library.inference import uncond

    path = tmp_path / "uncond.safetensors"
    monkeypatch.setattr(uncond, "default_uncond_path", lambda: path)
    dataset = Buckets()
    dataset.subsets = [NS(caption_dropout_rate=0.1)]
    group = torch.utils.data.ConcatDataset([dataset])
    args = NS(
        model_family="anima",
        _auto_swap_probe=None if resolved else {"inventory": True},
        _auto_swap_resolved=resolved,
    )
    with pytest.raises(ValueError, match="existing Anima uncond text cache"):
        probe.check_dataset(args, group, None)
    assert not path.exists()
    path.touch()
    probe.check_dataset(args, group, None)


@pytest.mark.parametrize("family,rate", [("anima", 0), ("krea2_raw", 0.1)])
def test_no_uncond_requirement_for_other_conditioning(monkeypatch, family, rate):
    from library.inference import uncond

    monkeypatch.setattr(
        uncond,
        "default_uncond_path",
        lambda: pytest.fail("unexpected Anima cache lookup"),
    )
    dataset = Buckets()
    dataset.subsets = [NS(caption_dropout_rate=rate)]
    probe.check_dataset(
        NS(model_family=family, _auto_swap_probe={"inventory": True}),
        torch.utils.data.ConcatDataset([dataset]),
        None,
    )


@pytest.mark.parametrize("resolved", [False, True])
def test_uncond_loader_never_stages_for_auto(monkeypatch, tmp_path, resolved):
    from library.inference import uncond
    from library.training.uncond_sidecar import ensure_uncond_crossattn

    path = tmp_path / "uncond.safetensors"
    monkeypatch.setattr(uncond, "default_uncond_path", lambda: path)
    monkeypatch.setattr(
        uncond, "stage_uncond_sidecar", lambda *a, **k: pytest.fail("AUTO staged cache")
    )
    cached = torch.ones(1, 512, 4)
    monkeypatch.setattr(uncond, "load_uncond_crossattn", lambda *a, **k: cached)
    trainer = NS(_state=NS(uncond_crossattn_1=None))
    args = NS(
        _auto_swap_probe=None if resolved else {"inventory": False},
        _auto_swap_resolved=resolved,
    )
    with pytest.raises(ValueError, match="cannot stage"):
        ensure_uncond_crossattn(trainer, args, NS(device="cpu"), torch.bfloat16)
    assert not path.exists()
    path.touch()
    ensure_uncond_crossattn(trainer, args, NS(device="cpu"), torch.bfloat16)
    assert trainer._state.uncond_crossattn_1 is cached


@pytest.mark.parametrize("accumulation", [1, 3])
@pytest.mark.parametrize("updates", [3, 5])
def test_probe_executes_real_production_backward_and_adam_updates(
    monkeypatch, accumulation, updates, tmp_path
):
    # Only CUDA telemetry is mocked. The production _run_step, Accelerate
    # accumulation, autograd, AdamW state allocation and scheduler are real.
    AcceleratorState._reset_state(reset_partial_state=True)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation)
    network = torch.nn.Linear(4, 1)
    original = network.weight.detach().clone()
    raw_optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(raw_optimizer, lambda _: 1)
    network, optimizer, scheduler = accelerator.prepare(
        network, raw_optimizer, scheduler
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (10000, 12000))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 2000)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 1000)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 2000)
    dataset = torch.utils.data.ConcatDataset([Buckets()])
    state = NS(
        args=NS(
            gradient_accumulation_steps=accumulation,
            dataloader_pin_memory=False,
            max_grad_norm=0,
            debug_finite_checks=False,
            max_train_steps=100,
            _auto_swap_probe={"updates_per_case": updates, "directory": str(tmp_path)},
        ),
        accelerator=accelerator,
        network=network,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        training_model=network,
        train_ctx=NS(),
        text_encoder=None,
        unet=None,
        train_dataloader=NS(collate_fn=lambda batch: batch[0]),
        current_step=NS(value=0),
        global_step=0,
        profile_started=False,
        is_tracking=False,
        on_step_start_for_network=lambda *a: None,
    )
    trainer = NS(
        _cudagraph_mark_step=False,
        _state=NS(personalization_observer={}),
        on_step_start=lambda *a, **k: None,
        process_batch=lambda _, batch, **k: network(batch).square().mean(),
        run_after_backward=lambda _: None,
    )
    try:
        samples = probe._probe_updates(
            trainer, state, probe.representative_batches(dataset), dataset
        )
        assert len(samples) == 2 * updates
        assert state.global_step == 2 * updates
        assert all(
            float(s["step"]) == 2 * updates for s in raw_optimizer.state.values()
        )
        assert [s["update"] for s in samples] == list(range(updates)) * 2
        assert all(s["loss"] >= 0 for s in samples)
        import json

        assert json.loads((tmp_path / "updates.json").read_text())["updates"] == samples
        assert not torch.equal(original, network.weight)
        assert all(sample["seconds"] > 0 for sample in samples)
        assert all(sample["headroom"] == 11000 for sample in samples)
    finally:
        accelerator.end_training()
        AcceleratorState._reset_state(reset_partial_state=True)
        PartialState._reset_state()
