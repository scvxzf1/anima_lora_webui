"""Online batch policy and CUDA retry contracts, without requiring a GPU."""

import argparse
import random
from contextlib import nullcontext
import weakref

import pytest
import torch

from library.preprocess.adaptive_batch import AutoBatcher, batch_size_arg
from library.preprocess.batch_config import resolve_cache_batch_sizes
from library.preprocess.batch_policy import AutoBatchPolicy, BatchMeasurement
from library.runtime.cli import add_io_args


def observe(policy, *, budget=10000, rate=None):
    batch = policy.prepare(budget)
    policy.success(
        BatchMeasurement(batch, batch / (rate or batch), 100, 100 + batch * 100, budget)
    )
    return batch


def test_warmup_then_predictive_jump_and_bound():
    policy = AutoBatchPolicy()
    sizes = [observe(policy) for _ in range(9)]
    assert sizes == [1, 1, 1, 2, 2, 4, 4, 16, 16]
    assert policy.current == 32


def test_cold_start_and_tail_do_not_promote():
    policy = AutoBatchPolicy()
    observe(policy)
    assert not policy.samples
    policy.current = 4
    policy.success(BatchMeasurement(2, 0.1, 100, 300, 10000))
    assert policy.current == 4 and not policy.samples


def test_plateau_settles_on_smaller_near_best_batch():
    policy = AutoBatchPolicy()
    for _ in range(10):
        observe(policy, rate=10)
    assert policy.settled and policy.current == 1
    for _ in range(50):
        observe(policy, rate=10)
    assert policy.current == 1


def test_memory_pressure_decreases_batch_and_reopens_after_relief():
    policy = AutoBatchPolicy(throughput=False)
    for _ in range(15):
        observe(policy)
    assert policy.current == 32
    assert policy.prepare(600) == 4
    for _ in range(3):
        observe(policy, budget=600)
    assert policy.settled
    for _ in range(17):
        observe(policy, budget=10000)
    assert policy.current > 4


def test_oom_brackets_without_repeating_failed_size():
    policy = AutoBatchPolicy(predict=False, throughput=False)
    seen = []
    for _ in range(50):
        batch = policy.prepare(10000)
        seen.append(batch)
        if batch > 6:
            policy.failure(batch)
        else:
            observe(policy)
    assert seen.count(8) == 1 and seen.count(7) == 1
    assert policy.current == 6
    with pytest.raises(ValueError, match="cannot be recovered"):
        policy.failure(1)


@pytest.mark.parametrize(
    "value, expected", [("auto", "auto"), (" AUTO ", "auto"), (3, 3), ("2", 2)]
)
def test_batch_parser(value, expected):
    assert batch_size_arg(value) == expected


@pytest.mark.parametrize("value", [0, -1, "bad", True, 1.5, None])
def test_batch_parser_rejects_invalid(value):
    with pytest.raises(ValueError, match="positive integer"):
        batch_size_arg(value)


def test_cli_auto_is_opt_in():
    parser = argparse.ArgumentParser()
    add_io_args(parser, include_batch_size=True, allow_auto_batch_size=True)
    assert (
        parser.parse_args(["--dir", ".", "--batch_size", "auto"]).batch_size == "auto"
    )
    parser = argparse.ArgumentParser()
    add_io_args(parser, include_batch_size=True)
    with pytest.raises(SystemExit):
        parser.parse_args(["--dir", ".", "--batch_size", "auto"])


@pytest.mark.parametrize(
    "profile, expected",
    [
        ("auto", ("auto", "auto")),
        ("low_vram", (1, 4)),
        ("balanced", (2, 8)),
        ("speed", (4, 16)),
        ("unknown", ("auto", "auto")),
    ],
)
def test_config_profiles_and_single_override(profile, expected):
    values = {
        "preprocess_memory_profile": profile,
        "preprocess_vae_cache_batch_size": "auto",
    }
    assert resolve_cache_batch_sizes(values) == expected
    values["preprocess_text_cache_batch_size"] = "3"
    assert resolve_cache_batch_sizes(values) == (expected[0], 3)


def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 200)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_retry_releases_traceback_and_never_loses_or_duplicates_items(monkeypatch):
    fake_cuda(monkeypatch)
    events, refs, cleanups = [], [], []
    runner = AutoBatcher(
        "cuda",
        label="test",
        predict=False,
        throughput=False,
        on_event=events.append,
        cleanup=lambda: cleanups.append(all(ref() is None for ref in refs)),
    )
    monkeypatch.setattr(runner, "_budget", lambda: 10000)

    def encode(items):
        if len(items) > 2:
            failed_tensor = torch.ones(5)
            refs.append(weakref.ref(failed_tensor))
            raise torch.cuda.OutOfMemoryError("injected")
        return torch.tensor(items)

    result = runner.encode_all(list(range(80)), encode)
    assert result.tolist() == list(range(80))
    assert cleanups and all(cleanups)
    assert any(event["status"] == "oom" for event in events)
    assert runner.batch_size == 2


def test_batch_one_and_non_oom_errors_propagate(monkeypatch):
    fake_cuda(monkeypatch)
    runner = AutoBatcher("cuda", label="test")
    monkeypatch.setattr(runner, "_budget", lambda: 10000)

    def oom(_items):
        raise torch.cuda.OutOfMemoryError("one cannot fit")

    with pytest.raises(torch.cuda.OutOfMemoryError, match="one cannot fit") as exc:
        runner.encode_all([1], oom)
    assert "batch=1 cannot fit" in exc.value.__notes__[0]

    def other(_items):
        raise RuntimeError("not an OOM")

    with pytest.raises(RuntimeError, match="not an OOM"):
        runner.encode_all([1, 2], other)


def test_cpu_auto_is_bounded_and_optional_outputs_preserved():
    runner = AutoBatcher("cpu", label="test")
    sizes = []

    def encode(items):
        sizes.append(len(items))
        return torch.tensor(items), None

    tensor, optional = runner.encode_all(list(range(9)), encode)
    assert sizes == [1] * 9
    assert tensor.tolist() == list(range(9)) and optional is None


def test_budget_counts_reusable_reservations_and_allocator_limit(monkeypatch):
    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (2 * gib, 10 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 5 * gib)
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda _: 0.6)
    runner = AutoBatcher("cuda:0", label="test")
    assert runner._budget() == 5 * gib


def test_empty_encoding_request_has_explicit_contract():
    with pytest.raises(ValueError, match="empty encoding request"):
        AutoBatcher("cpu", label="empty").encode_all([], lambda _: None)


def test_telemetry_cannot_change_caption_shuffle_rng(monkeypatch):
    from library.preprocess import adaptive_batch

    monkeypatch.setattr(adaptive_batch.logger, "info", lambda *_: random.random())
    runner = AutoBatcher("cpu", label="rng", on_event=lambda _: random.random())
    state = random.getstate()
    runner._emit("oom", 4)
    assert random.getstate() == state


def test_prepare_without_success_cannot_expire_cooldown():
    policy = AutoBatchPolicy()
    for _ in range(10):
        observe(policy, rate=10)
    for _ in range(30):
        policy.prepare(20000)
    assert policy.current == 1 and policy.stable_steps < 16


def test_cleanup_error_still_clears_allocator_but_does_not_retry(monkeypatch):
    fake_cuda(monkeypatch)
    cleared = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))

    def cleanup():
        raise RuntimeError("model cleanup failed")

    runner = AutoBatcher("cuda", label="broken", cleanup=cleanup)
    runner.policy.current = 2
    monkeypatch.setattr(runner, "_budget", lambda: 10000)

    def encode(_):
        raise torch.cuda.OutOfMemoryError("injected")

    with pytest.raises(RuntimeError, match="model cleanup failed"):
        runner.encode_all([1, 2], encode)
    assert cleared == [True]


@pytest.mark.parametrize(
    "module_name, required",
    [
        ("scripts.preprocess.cache_latents", ["--dir", ".", "--vae", "unused"]),
        (
            "scripts.preprocess.cache_text_embeddings",
            ["--dir", ".", "--qwen3", "unused"],
        ),
        (
            "scripts.krea2.preprocess_te_cache",
            ["--dir", ".", "--cache_dir", ".", "--qwen3", "unused"],
        ),
        (
            "scripts.z_image.preprocess_te_cache",
            ["--dir", ".", "--cache_dir", ".", "--qwen3", "unused"],
        ),
    ],
)
@pytest.mark.parametrize("extra, expected", [([], "auto"), (["--batch_size", "3"], 3)])
def test_cache_cli_defaults_to_auto_and_preserves_explicit_batch(
    monkeypatch, module_name, required, extra, expected
):
    import importlib

    module = importlib.import_module(module_name)
    parse_args = argparse.ArgumentParser.parse_args
    parsed = []

    class ParsedBeforeModelLoad(Exception):
        pass

    def capture(parser):
        parsed.append(parse_args(parser, required + extra))
        raise ParsedBeforeModelLoad

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(ParsedBeforeModelLoad):
        module.main()
    assert parsed[0].batch_size == expected


def test_cache_api_defaults_to_auto():
    from inspect import signature
    from library.preprocess import cache_latents, cache_text_embeddings

    for cache in (cache_latents, cache_text_embeddings):
        assert signature(cache).parameters["batch_size"].default == "auto"
