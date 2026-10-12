"""CPU-only standalone contracts through the real low-memory encoder runtime."""

from contextlib import contextmanager
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from library.models.qwen_image_2_1 import inference_conditions as conditions
from library.models.qwen_image_2_1 import inference_runner as runner
from library.models.qwen_image_2_1 import inference_text_encoder as encoding


class _Core(torch.nn.Linear):
    def __init__(self, events):
        super().__init__(2, 2)
        self.events = events

    def to(self, *, device, dtype):
        # Record GPU placement without making any real CUDA calls.
        self.events.append(("core", device, dtype))
        return self


class _Encoder(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.model = _Core(events)
        self.lm_head = torch.nn.Linear(2, 2)
        self.events = events

    def to(self, device):
        assert device == "cpu", "the whole encoder must never move to CUDA"
        self.events.append(("encoder", device))
        return self


class _EditValue:
    def __init__(self, index):
        self.index = index

    def unsqueeze(self, dim):
        assert dim == 0
        return self

    def to(self, *, device, dtype):
        # Output placement is recorded by the caller; no real CUDA allocation.
        return torch.zeros((1, 2, 4) if self.index == 0 else (1, 2), dtype=dtype)


@pytest.fixture
def setup_encoding(monkeypatch):
    import accelerate
    from accelerate import hooks

    events, residents, calls = [], [], []
    # Do not query the real GPU's availability, memory or allocator.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty"))
    monkeypatch.setattr(encoding, "clean_memory_on_device", lambda device: events.append(("clean", device)))
    monkeypatch.setattr(accelerate, "cpu_offload", lambda core, **kw: events.append(("install", kw)))
    monkeypatch.setattr(hooks, "remove_hook_from_module", lambda core, **kw: events.append(("remove", kw)))

    def load(path, *, dtype, device):
        assert (path, dtype, device) == ("encoder", torch.bfloat16, "cpu")
        value = _Encoder(events)
        residents.append(weakref.ref(value))
        events.append("load")
        return value

    monkeypatch.setattr(runner, "load_qwen_image_2_1_text_encoder", load)
    monkeypatch.setattr(conditions, "load_qwen_image_2_1_text_encoder", load)
    monkeypatch.setattr(conditions, "QwenImage21EditTokenizeStrategy", lambda path: object())
    return events, residents, calls


def _encode(monkeypatch, setup, task, device, *, cpu=False, free_bytes=0, fail_at=None):
    events, residents, calls = setup
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda requested: (free_bytes, free_bytes))
    args = SimpleNamespace(text_encoder="encoder", runtime_dtype="bf16", text_encoder_cpu=cpu,
                           prompt="positive", negative_prompt="negative")

    def encode(model, text):
        expected = torch.device("cpu") if cpu else device
        assert model._qwen_execution_device == expected
        assert model._qwen_cache_offloaded == (expected.type == "cuda" and free_bytes == 0)
        assert model.lm_head.weight.device.type == "cpu"
        calls.append(text)
        if text == fail_at:
            raise RuntimeError(f"{text} encoding failed")

    if task == "t2i":
        def t2i(model, path, text, **kwargs):
            encode(model, text)
            return torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)
        monkeypatch.setattr(runner, "_encode_prompt", t2i)
        return runner.prepare_qwen_image_2_1_text(args, device, torch.bfloat16)

    images = (object(), object(), object(), object())

    def edit(model, tokenizer, text, references):
        assert references is images  # Keep all ordered refs for both encodes.
        encode(model, text)
        return tuple(_EditValue(index) for index in range(3))
    monkeypatch.setattr(conditions, "encode_edit_prompt", edit)
    return conditions._prepare_edit_text(args, images, device, torch.bfloat16)


@pytest.mark.parametrize("task", ["t2i", "edit"])
@pytest.mark.parametrize("device,cpu,free_bytes,offloaded", [
    ("cuda:2", False, 0, True),
    ("cuda:2", False, 1024**3, False),
    ("cuda:2", True, 0, False),
    ("cpu", False, 0, False),
])
def test_standalone_automatic_placement_and_explicit_cpu(
    monkeypatch, setup_encoding, task, device, cpu, free_bytes, offloaded
):
    events, residents, calls = setup_encoding
    positive, negative = _encode(monkeypatch, setup_encoding, task, torch.device(device),
                                 cpu=cpu, free_bytes=free_bytes)
    assert calls == ["positive", "negative"]
    assert positive is not negative
    assert residents[0]() is None
    names = [event[0] if isinstance(event, tuple) else event for event in events]
    assert ("install" in names) == offloaded
    assert ("remove" in names) == offloaded
    if offloaded:
        assert names == ["load", "install", "remove", "encoder", "empty", "clean"]
        assert events[1][1] == {"execution_device": torch.device(device), "offload_buffers": False}
        assert events[2][1] == {"recurse": True}
    else:
        assert events[1] == ("core", torch.device("cpu" if cpu else device), torch.bfloat16)
    assert events[-1] == ("clean", torch.device(device))


@pytest.mark.parametrize("task", ["t2i", "edit"])
@pytest.mark.parametrize("fail_at", ["positive", "negative"])
def test_standalone_encoding_error_removes_hooks_and_releases_encoder(
    monkeypatch, setup_encoding, task, fail_at
):
    events, residents, calls = setup_encoding
    with pytest.raises(RuntimeError, match=f"{fail_at} encoding failed"):
        _encode(monkeypatch, setup_encoding, task, torch.device("cuda:2"), fail_at=fail_at)
    assert calls == (["positive"] if fail_at == "positive" else ["positive", "negative"])
    assert residents[0]() is None
    names = [event[0] if isinstance(event, tuple) else event for event in events]
    assert names == ["load", "install", "remove", "encoder", "empty", "clean"]


@pytest.mark.parametrize("encode_error", [False, True])
@pytest.mark.parametrize("failed_cleanup", ["collect", "clean"])
def test_outer_cleanup_retains_encoding_error_and_attempts_both_steps(
    monkeypatch, encode_error, failed_cleanup
):
    events = []
    primary = ValueError("encoding failed")
    secondary = RuntimeError(f"{failed_cleanup} failed")
    args = SimpleNamespace(text_encoder="encoder", runtime_dtype="bf16")

    @contextmanager
    def runtime(model, **_kwargs):
        yield model

    def cleanup(name):
        events.append(name)
        if name == failed_cleanup:
            raise secondary

    def encode(_model):
        if encode_error:
            raise primary
        return torch.zeros(1, 2, 4)

    monkeypatch.setattr(encoding, "loaded_text_encoder_for_cache", runtime)
    monkeypatch.setattr(encoding, "gc", SimpleNamespace(collect=lambda: cleanup("collect")))
    monkeypatch.setattr(encoding, "clean_memory_on_device", lambda _device: cleanup("clean"))
    with pytest.raises((ValueError, RuntimeError)) as raised:
        encoding.encode_with_text_encoder(
            args, torch.device("cpu"), encode,
            loader=lambda *_args, **_kwargs: torch.nn.Linear(2, 2),
        )
    assert raised.value is (primary if encode_error else secondary)
    assert events == ["collect", "clean"]
    if encode_error:
        assert any(str(secondary) in note for note in primary.__notes__)


@pytest.mark.parametrize("allocator_error", [False, True])
def test_allocator_error_does_not_retain_encoder_until_another_collection(
    monkeypatch, allocator_error
):
    references = []
    args = SimpleNamespace(text_encoder="encoder", runtime_dtype="bf16")

    def load(*_args, **_kwargs):
        encoder = _Encoder([])
        references.append(weakref.ref(encoder))
        return encoder

    def encode(model):
        assert model._qwen_execution_device == torch.device("cpu")
        raise ValueError("encoding failed")

    def clean(_device):
        if allocator_error:
            raise RuntimeError("allocator cleanup failed")

    monkeypatch.setattr(encoding, "clean_memory_on_device", clean)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        try:
            encoding.encode_with_text_encoder(args, torch.device("cpu"), encode, loader=load)
        except ValueError as error:
            assert str(error) == "encoding failed"
            if allocator_error:
                assert any("allocator cleanup failed" in note for note in error.__notes__)
        else:
            pytest.fail("encoding error must propagate")
        # No retained exception reference or extra collection may hide a cycle.
        assert references[0]() is None
    finally:
        if was_enabled:
            gc.enable()
