"""CPU-only checks for the disposable Anima base-dx counterfactual."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from bench.adaptive_runtime import anima_backward_counterfactual as cf
from bench.adaptive_runtime import diagnose_anima_backward as driver
from bench.adaptive_runtime import probe_anima_train as probe
from library.training.adaptive_runtime.islands import install_precision_islands
from networks.lora_modules.lora import LoRAModule


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(4, 8, bias=False)
        self.layer2 = torch.nn.Linear(8, 4, bias=False)

    def forward(self, x):
        return self.layer2(torch.nn.functional.gelu(self.layer1(x)))


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()


class Toy(torch.nn.Module):
    num_blocks = 1

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block()])

    def forward(self, x):
        return self.blocks[0].mlp(x)


def prepared(*, mode):
    torch.manual_seed(19)
    model = Toy()
    assignments = {name: "fp16" for name in cf.target_names(1)}
    model.requires_grad_(False)
    install_precision_islands(model, assignments)
    if mode == "fp32":
        assert len(cf.install(model, {"assignments": assignments}, block_count=1)) == 2
    loras = []
    for name in sorted(assignments):
        module = model.get_submodule(name)
        lora = LoRAModule(name.replace(".", "_"), module, lora_dim=2, alpha=2)
        lora.apply_to()
        loras.append(lora)
    return model, loras


@pytest.mark.parametrize("scale", [1, 1024])
def test_actual_lora_mlp_checkpoint_forward_and_gradients(scale):
    baseline, native_loras = prepared(mode="native")
    candidate, cf_loras = prepared(mode="fp32")
    x0 = torch.tensor([[0.3, -0.7, 1.1, 0.2]], requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    y0 = checkpoint(baseline, x0, use_reentrant=False)
    y1 = checkpoint(candidate, x1, use_reentrant=False)
    assert torch.equal(y0, y1)
    (y0.square().mean() * scale).backward()
    (y1.square().mean() * scale).backward()
    # Layer2 sees the same incoming gradient; layer1 legitimately sees the
    # changed base dx through layer2 even though its own LoRA rule is unchanged.
    assert torch.equal(native_loras[1].lora_up.weight.grad,
                       cf_loras[1].lora_up.weight.grad)
    assert not torch.equal(native_loras[0].lora_up.weight.grad,
                           cf_loras[0].lora_up.weight.grad)
    assert x1.grad is not None and torch.isfinite(x1.grad).all()
    assert x0.grad is not None and torch.isfinite(x0.grad).all()


@pytest.mark.parametrize("scale", [1, 1024])
def test_base_dx_uses_current_half_weight_in_fp32(scale):
    module = torch.nn.Linear(3, 2, bias=False).half().requires_grad_(False)
    x = torch.tensor([[0.11, -0.22, 0.33]], requires_grad=True)
    baseline = torch.nn.functional.linear(x.half(), module.weight).float()
    out = cf.CurrentWeightBaseDx.apply(x, module)
    assert torch.equal(out, baseline)
    incoming = torch.tensor([[0.17, -0.23]]) * scale
    out.backward(incoming)
    assert torch.equal(x.grad, incoming.float() @ module.weight.float())
    assert module.weight.grad is None


def test_rebound_data_is_used_and_invalid_current_weight_fails_closed():
    module = torch.nn.Linear(3, 2, bias=False).half().requires_grad_(False)
    x = torch.ones(1, 3, requires_grad=True)
    out = cf.CurrentWeightBaseDx.apply(x, module)
    module.weight.data = torch.full_like(module.weight, 2)
    out.backward(torch.ones_like(out))
    assert torch.equal(x.grad, torch.full_like(x, 4))
    for invalid in ("shape", "dtype", "identity"):
        module = torch.nn.Linear(3, 2, bias=False).half().requires_grad_(False)
        x = torch.ones(1, 3, requires_grad=True)
        out = cf.CurrentWeightBaseDx.apply(x, module)
        if invalid == "shape":
            module.weight.data = torch.ones(2, 4, dtype=torch.float16)
        elif invalid == "dtype":
            module.weight.data = module.weight.data.float()
        else:
            module.weight = torch.nn.Parameter(module.weight.detach().clone(), requires_grad=False)
        with pytest.raises(RuntimeError, match="identity/shape/dtype/device drift"):
            out.backward(torch.ones_like(out))


def test_invalid_suite_and_assignment_rejected(tmp_path):
    weights, inputs = tmp_path / "weights", tmp_path / "inputs"
    weights.touch()
    inputs.touch()
    base = dict(output=tmp_path / "worker" / "result.json", weights=weights, inputs=inputs,
                precision="fp16-islands", steps=1, swap=0, loss_scale=1024,
                fp32_pattern=[], fp16_only_pattern=["blocks.*.mlp.*"],
                single_sigma=0.5, record_replay=False, replay_reference=tmp_path / "ref",
                backward_counterfactual_suite=True, base_backward_mode="fp32")
    probe.validate_args(SimpleNamespace(**base))
    for change in ({"precision": "fp32-reference"}, {"single_sigma": 0.4},
                   {"capture_local_vjp": True}, {"backward_counterfactual_suite": False}):
        with pytest.raises(ValueError):
            probe.validate_args(SimpleNamespace(**{**base, **change}))
    model, _ = prepared(mode="native")
    with pytest.raises(ValueError, match="exact MLP-only"):
        cf.install(model, {"assignments": {"blocks.0.mlp.layer1": "fp16"}}, block_count=1)


def test_driver_rejects_forged_mode_before_comparison(tmp_path, monkeypatch):
    directory = tmp_path / "baseline"
    directory.mkdir()
    result = {"status": "ok"}
    monkeypatch.setattr(driver, "_complete", lambda *a, **kw: True)
    monkeypatch.setattr(driver, "read_capture", lambda *a: (
        {"signature": {"precision": "fp16-islands"}}, result))
    with pytest.raises(ValueError, match="signature/mode/source"):
        driver._validate_capture(directory, result, precision="fp16-islands", mode="native",
                                 environment={"gpu_uuid": "GPU-test", "helper_sha256": "x"})


@pytest.mark.parametrize("forgery", ["signature", "inventory", "replay", "prediction"])
def test_direct_comparison_rejects_forged_evidence(tmp_path, monkeypatch, forgery):
    baseline, candidate = tmp_path / "baseline", tmp_path / "counterfactual"
    baseline.mkdir()
    candidate.mkdir()
    manifest = {"signature": {"precision": "fp16-islands", "loss_scale": 1024},
                "initial_adapter_sha256": "initial", "expected_gradients": ["gradient.a"],
                "cases": [{"sigma": 0.5, "replay": {"state_sha256": "state"}}]}
    other = {**manifest, "signature": dict(manifest["signature"]),
             "cases": [dict(manifest["cases"][0])]}
    if forgery == "signature":
        other["signature"]["backward_counterfactual_scope"] = "forged"
    elif forgery == "inventory":
        other["expected_gradients"] = ["gradient.other"]
    elif forgery == "replay":
        other["cases"][0]["replay"] = {"state_sha256": "other"}
    monkeypatch.setattr(driver, "read_capture", lambda path: (
        manifest if path.parent == baseline else other, {}))
    monkeypatch.setattr(driver, "load_case", lambda path, case, gradients: {
        "prediction": torch.tensor([1.0 if path.parent == baseline else
                                    2.0 if forgery == "prediction" else 1.0]),
        "input_gradient": torch.ones(1), "gradient.a": torch.ones(1)})
    with pytest.raises(ValueError, match="mismatch|differ"):
        driver._direct_equivalence(baseline, candidate)
