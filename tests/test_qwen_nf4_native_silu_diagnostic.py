"""Native SiLU diagnostic contracts; CPU checks are not GPU acceptance."""
import json
import os
from types import SimpleNamespace

import pytest
import torch

from scripts.quantization.qwen_nf4_native_silu import (
    native_silu_backward,
    native_silu_forward,
    patch_native_silu,
)


def _model(activations):
    model = torch.nn.Module()
    model.transformer_blocks = torch.nn.ModuleList()
    for activation in activations:
        block = torch.nn.Module()
        block.img_mlp = torch.nn.Sequential()
        block.img_mlp.activation_fn = activation
        model.transformer_blocks.append(block)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "sliced"])
def test_native_metadata_and_first_order_autograd(dtype, layout):
    x = torch.linspace(-3, 3, 24, dtype=dtype).reshape(4, 6)
    if layout == "transposed":
        x = x.t()
    elif layout == "sliced":
        x = x[:, ::2]
    x = x.detach().requires_grad_(True)
    gradient = torch.linspace(-1, 1, x.numel(), dtype=dtype).reshape(x.shape)
    expected = torch.nn.functional.silu(x)
    actual = native_silu_forward(x)
    assert actual.dtype == expected.dtype and actual.stride() == expected.stride()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_grad = torch.autograd.grad(expected, x, gradient)[0]
    actual_grad = torch.autograd.grad(actual, x, gradient)[0]
    assert actual_grad.dtype == expected_grad.dtype
    assert actual_grad.stride() == expected_grad.stride()
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    torch.library.opcheck(native_silu_forward, (x,))
    torch.library.opcheck(native_silu_backward, (gradient, x.detach()))


def test_patch_preserves_objects_and_state_and_validates_before_mutation():
    activations = [torch.nn.SiLU(), torch.nn.SiLU()]
    model = _model(activations)
    parameter = torch.nn.Parameter(torch.tensor([2.]))
    model.register_parameter("sentinel", parameter)
    original_state = model.state_dict()
    metadata = patch_native_silu(model)
    assert metadata["patch_count"] == 2 and metadata["first_order_only"]
    assert list(model.state_dict()) == list(original_state)
    assert model.sentinel is parameter
    for index, original in enumerate(activations):
        assert model.transformer_blocks[index].img_mlp.activation_fn is original
    torch.testing.assert_close(model.state_dict(), original_state, rtol=0, atol=0)
    broken = _model([torch.nn.SiLU(), torch.nn.SiLU(inplace=True)])
    first = broken.transformer_blocks[0].img_mlp.activation_fn
    before = first.forward
    with pytest.raises(ValueError, match="inplace=False"):
        patch_native_silu(broken)
    assert first.forward == before
    wrong = _model([torch.nn.SiLU(), torch.nn.GELU()])
    before = wrong.transformer_blocks[0].img_mlp.activation_fn.forward
    with pytest.raises(TypeError, match="nn.SiLU"):
        patch_native_silu(wrong)
    assert wrong.transformer_blocks[0].img_mlp.activation_fn.forward == before


def test_double_backward_is_rejected():
    x = torch.tensor([-.5, 1.5], requires_grad=True)
    first = torch.autograd.grad(native_silu_forward(x).sum(), x, create_graph=True)[0]
    with pytest.raises(RuntimeError, match="autograd formula"):
        torch.autograd.grad(first.sum(), x)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("workspace", [None, ":16:8"])
def test_probe_failure_keeps_evidence_and_restores_deterministic_flags(tmp_path, monkeypatch, deterministic, workspace):
    from scripts.quantization import probe_qwen_nf4_compile as probe
    from scripts.quantization import qwen_nf4_compile_acceptance as acceptance

    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    args = SimpleNamespace(native_layer_norm_targets=("img_norm1", "img_norm2"),
        native_layer_norm=False, native_silu=True, deterministic=deterministic,
        scope="all", count=24, restore_mode="foreach", dynamic=False,
        persistent_reductions=False)

    def fail(loader, **kwargs):
        model = loader()
        assert model.transformer_blocks[0].img_mlp.activation_fn is activation
        if deterministic:
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        else:
            assert torch.are_deterministic_algorithms_enabled() == old_enabled
            assert torch.is_deterministic_algorithms_warn_only_enabled() == old_warn
        raise RuntimeError("intentional diagnostic failure")

    activation = torch.nn.SiLU()
    monkeypatch.setattr(acceptance, "run_case", fail)
    from library.models.qwen_image_2_1 import weights
    monkeypatch.setattr(weights, "load_qwen_image_2_1_transformer",
        lambda *a, **k: _model([activation]))
    with pytest.raises(RuntimeError, match="intentional diagnostic failure"):
        probe._run_probe(args, "unused-local-model", tmp_path)
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == "failed"
    assert report["native_silu_patches"]["eager"]["patch_count"] == 1
    assert report["production_gate"] == "not approved"
    assert torch.are_deterministic_algorithms_enabled() == old_enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled() == old_warn
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == workspace


def test_cli_defaults_leave_diagnostic_flags_disabled(tmp_path, monkeypatch):
    from scripts.quantization import probe_qwen_nf4_compile as probe

    captured = []
    monkeypatch.setattr("sys.argv", ["probe", "--model", "unused", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(probe, "_run_probe", lambda args, *a: captured.append(args))
    probe.main()
    assert not captured[0].native_silu
    assert not captured[0].deterministic


def test_success_patches_both_runs_and_records_metadata(tmp_path, monkeypatch):
    from scripts.quantization import probe_qwen_nf4_compile as probe
    from scripts.quantization import qwen_nf4_compile_acceptance as acceptance
    from library.models.qwen_image_2_1 import weights

    args = SimpleNamespace(native_layer_norm_targets=("img_norm1", "img_norm2"),
        native_layer_norm=False, native_silu=True, deterministic=False,
        scope="all", count=24, restore_mode="foreach", dynamic=False,
        persistent_reductions=False)
    activations = []

    def load(*a, **k):
        activation = torch.nn.SiLU()
        activations.append(activation)
        return _model([activation])

    def run(loader, **kwargs):
        model = loader()
        x = torch.tensor([-.5, .5], requires_grad=True)
        output = model.transformer_blocks[0].img_mlp.activation_fn(x)
        assert "qwen_nf4_diagnostic" in str(output.grad_fn)
        return {"records": [], "initial_adapter": {}}

    monkeypatch.setattr(weights, "load_qwen_image_2_1_transformer", load)
    monkeypatch.setattr(acceptance, "run_case", run)
    monkeypatch.setattr(acceptance, "compare_runs", lambda *a, **k: {})
    probe._run_probe(args, "unused-local-model", tmp_path)
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == "passed" and len(activations) == 2
    for label in ("eager", "compiled"):
        assert report["native_silu_patches"][label]["patch_count"] == 1
        assert report["runs"][label]["diagnostic_config"]["native_silu"]["first_order_only"]
        assert (tmp_path / f"{label}.pt").is_file()
