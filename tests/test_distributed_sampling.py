from __future__ import annotations

import datetime
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from library.anima import training as anima_training


class _FakeAccelerator:
    def __init__(
        self,
        process_index: int,
        num_processes: int,
        *,
        remote_failure: bool = False,
    ) -> None:
        self.process_index = process_index
        self.num_processes = num_processes
        self.is_main_process = process_index == 0
        self.device = "cpu"
        self.wait_count = 0
        self.remote_failure = remote_failure

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return nullcontext()

    def wait_for_everyone(self) -> None:
        self.wait_count += 1

    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        assert reduction == "sum"
        return tensor + int(self.remote_failure)


class _FakeDiT:
    def __init__(self) -> None:
        self.inference_switches = 0
        self.training_switches = 0
        self.prepare_calls = 0

    def switch_block_swap_for_inference(self) -> None:
        self.inference_switches += 1

    def switch_block_swap_for_training(self) -> None:
        self.training_switches += 1

    def prepare_block_swap_before_forward(self) -> None:
        self.prepare_calls += 1


def _run_sample(
    monkeypatch,
    tmp_path,
    *,
    process_index,
    num_processes,
    prompt_count,
    remote_failure=False,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("placeholder\n", encoding="utf-8")
    args = SimpleNamespace(
        sample_at_first=False,
        sample_every_n_epochs=1,
        sample_every_n_steps=None,
        sample_prompts=str(prompt_file),
        output_dir=str(tmp_path / "output"),
        disable_block_swap_for_eval=False,
    )
    prompts = [
        {"enum": index, "prompt": f"prompt {index}"} for index in range(prompt_count)
    ]
    sampled: list[int] = []
    decoded: list[int] = []

    monkeypatch.setattr(
        anima_training.train_util, "load_prompts", lambda _path: prompts
    )
    monkeypatch.setattr(
        anima_training, "_ensure_sample_compile_range", lambda *_args: False
    )
    monkeypatch.setattr(anima_training.torch.cuda, "is_available", lambda: False)

    def fake_sample(*call_args, **_kwargs):
        prompt = call_args[8]
        sampled.append(prompt["enum"])
        return str(tmp_path / f"{prompt['enum']}.pt")

    monkeypatch.setattr(anima_training, "_sample_image_inference", fake_sample)
    monkeypatch.setattr(
        anima_training,
        "decode_samples_for_live_preview",
        lambda accelerator, *_args, **_kwargs: decoded.append(
            accelerator.process_index
        ),
    )

    accelerator = _FakeAccelerator(
        process_index, num_processes, remote_failure=remote_failure
    )
    dit = _FakeDiT()
    anima_training.sample_images(
        accelerator,
        args,
        1,
        150,
        dit,
        object(),
        None,
        object(),
        object(),
    )
    return accelerator, dit, sampled, decoded


def test_sample_prompts_are_sharded_by_process(monkeypatch, tmp_path) -> None:
    rank_0 = _run_sample(
        monkeypatch,
        tmp_path / "rank0",
        process_index=0,
        num_processes=2,
        prompt_count=4,
    )
    rank_1 = _run_sample(
        monkeypatch,
        tmp_path / "rank1",
        process_index=1,
        num_processes=2,
        prompt_count=4,
    )

    assert rank_0[2] == [0, 2]
    assert rank_1[2] == [1, 3]
    assert rank_0[3] == [0]
    assert rank_1[3] == []
    assert rank_0[0].wait_count == rank_1[0].wait_count == 2
    assert rank_0[1].prepare_calls == rank_1[1].prepare_calls == 2


def test_process_without_a_prompt_waits_for_main_process(monkeypatch, tmp_path) -> None:
    rank_0 = _run_sample(
        monkeypatch,
        tmp_path / "rank0",
        process_index=0,
        num_processes=2,
        prompt_count=1,
    )
    rank_1 = _run_sample(
        monkeypatch,
        tmp_path / "rank1",
        process_index=1,
        num_processes=2,
        prompt_count=1,
    )

    assert rank_0[2] == [0]
    assert rank_1[2] == []
    assert rank_0[3] == [0]
    assert rank_1[3] == []
    assert rank_0[0].wait_count == rank_1[0].wait_count == 2
    assert rank_1[1].prepare_calls == 0
    assert rank_1[1].inference_switches == rank_1[1].training_switches == 1


def test_single_process_samples_every_prompt(monkeypatch, tmp_path) -> None:
    accelerator, dit, sampled, decoded = _run_sample(
        monkeypatch,
        tmp_path,
        process_index=0,
        num_processes=1,
        prompt_count=3,
    )

    assert sampled == [0, 1, 2]
    assert decoded == [0]
    assert accelerator.wait_count == 2
    assert dit.prepare_calls == 3


def test_remote_missing_prompt_stops_all_processes_before_barriers(
    monkeypatch, tmp_path
) -> None:
    accelerator, dit, sampled, decoded = _run_sample(
        monkeypatch,
        tmp_path,
        process_index=0,
        num_processes=2,
        prompt_count=2,
        remote_failure=True,
    )

    assert sampled == []
    assert decoded == []
    assert accelerator.wait_count == 0
    assert dit.inference_switches == 0


class _DistributedAccelerator(_FakeAccelerator):
    def wait_for_everyone(self) -> None:
        super().wait_for_everyone()
        dist.barrier()

    def reduce(self, tensor: torch.Tensor, reduction: str) -> torch.Tensor:
        assert reduction == "sum"
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced


def _gloo_sampling_worker(
    rank: int,
    world_size: int,
    init_method: str,
    output_dir: str,
    fail_rank_one: bool,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=20),
    )
    root = Path(output_dir)
    prompts = [{"enum": index, "prompt": f"prompt {index}"} for index in range(4)]
    try:
        anima_training.train_util.load_prompts = lambda _path: prompts
        anima_training._ensure_sample_compile_range = lambda *_args: False
        anima_training.torch.cuda.is_available = lambda: False

        def fake_sample(*call_args, **_kwargs):
            prompt_index = call_args[8]["enum"]
            if fail_rank_one and rank == 1 and prompt_index == 1:
                raise RuntimeError("injected rank 1 sampling failure")
            Path(root, f"sampled-{rank}-{prompt_index}").touch()
            return str(root / f"{prompt_index}.pt")

        def fake_decode(*_args, **_kwargs):
            sampled = {path.name for path in root.glob("sampled-*")}
            assert sampled == {
                "sampled-0-0",
                "sampled-0-2",
                "sampled-1-1",
                "sampled-1-3",
            }
            Path(root, "decoded").touch()

        anima_training._sample_image_inference = fake_sample
        anima_training.decode_samples_for_live_preview = fake_decode
        args = SimpleNamespace(
            sample_at_first=False,
            sample_every_n_epochs=1,
            sample_every_n_steps=None,
            sample_prompts=str(root / "prompts.txt"),
            output_dir=str(root / "output"),
            disable_block_swap_for_eval=False,
        )
        accelerator = _DistributedAccelerator(rank, world_size)
        try:
            anima_training.sample_images(
                accelerator,
                args,
                1,
                150,
                _FakeDiT(),
                object(),
                None,
                object(),
                object(),
            )
        except RuntimeError as exc:
            if not fail_rank_one:
                raise
            if rank == 1:
                assert str(exc) == "injected rank 1 sampling failure"
            else:
                assert str(exc) == "Sample generation failed on another process"
            Path(root, f"failed-{rank}").touch()
            return
        if fail_rank_one:
            raise AssertionError("distributed sampling failure was not propagated")
        assert Path(root, "decoded").exists()
        Path(root, f"finished-{rank}").touch()
    finally:
        dist.destroy_process_group()


@pytest.mark.integration
@pytest.mark.focused
def test_two_process_gloo_shards_before_main_process_decode(tmp_path) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    (tmp_path / "prompts.txt").write_text("placeholder\n", encoding="utf-8")
    world_size = 2
    mp.spawn(
        _gloo_sampling_worker,
        args=(
            world_size,
            f"file://{tmp_path / 'gloo-init'}",
            str(tmp_path),
            False,
        ),
        nprocs=world_size,
        join=True,
    )

    assert (tmp_path / "finished-0").exists()
    assert (tmp_path / "finished-1").exists()


@pytest.mark.integration
@pytest.mark.focused
def test_two_process_gloo_propagates_sampling_failure_without_barrier_hang(
    tmp_path,
) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is unavailable")

    (tmp_path / "prompts.txt").write_text("placeholder\n", encoding="utf-8")
    world_size = 2
    mp.spawn(
        _gloo_sampling_worker,
        args=(
            world_size,
            f"file://{tmp_path / 'gloo-init'}",
            str(tmp_path),
            True,
        ),
        nprocs=world_size,
        join=True,
    )

    assert (tmp_path / "failed-0").exists()
    assert (tmp_path / "failed-1").exists()
