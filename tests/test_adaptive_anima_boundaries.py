"""CPU-only contracts for sampled block-boundary evidence."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from bench.adaptive_runtime.anima_block_boundaries import (
    BlockBoundaryCapture, MAX_BYTES, SCOPE, analyze, read_capture,
)
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime import diagnose_anima_boundaries as driver
from bench.adaptive_runtime.anima_backward_counterfactual import target_names


class Block(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def _forward(self, x):
        return x * self.value

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(2), Block(3), Block(4)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def signature(precision="fp32-reference", scale=1):
    return {"steps": 1, "sigmas": [0.5], "block_boundary_scope": SCOPE,
            "precision": precision, "fp32_modules": [], "loss_scale": scale,
            "block_boundary_helper_sha256": "a" * 64}


def capture(tmp_path, name, *, scale=1, precision="fp32-reference", max_bytes=MAX_BYTES):
    model = Model()
    evidence = BlockBoundaryCapture(tmp_path / name, model, signature=signature(precision, scale),
                                    uuid="GPU-test", scale=scale, replay={"same": "replay"},
                                    block_count=3, sample_tokens=4, max_bytes=max_bytes)
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 6, 1).requires_grad_()
    try:
        y = model(x)
        (y.sum() * scale).backward()
        evidence.close()
    finally:
        evidence.restore()
    return tmp_path / name


def test_checkpoint_capture_is_nonperturbing_and_unscales(tmp_path):
    directory = capture(tmp_path, "reference")
    manifest, tensors = read_capture(directory)
    assert len(manifest["boundaries"]) == 4
    assert [row["indices"] for row in manifest["boundaries"]] == [[0, 1, 3, 5]] * 4
    assert torch.equal(tensors["boundary_00.gradient"], torch.full((1, 4, 1), 24.0))
    assert torch.equal(tensors["boundary_03.gradient"], torch.ones(1, 4, 1))
    assert not Model().blocks[0]._forward.__name__.startswith("capture")


def test_control_zero_difference(tmp_path):
    ref = capture(tmp_path, "reference")
    trial = capture(tmp_path, "candidate", scale=1024, precision="fp16-islands")
    ctrl = capture(tmp_path, "control", scale=1024, precision="fp16-islands")
    report = analyze(ref, trial, ctrl)
    assert report["first_exceedance_reverse_order"] is None
    assert all(row["gradient"]["relative_l2"] == 0
               and row["control_gradient"]["relative_l2"] == 0
               for row in report["boundaries_reverse"])


def test_budget_and_incomplete_capture(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        capture(tmp_path, "too-small", max_bytes=1)
    model = Model()
    evidence = BlockBoundaryCapture(tmp_path / "missing", model, signature=signature(),
                                    uuid="GPU-test", scale=1, replay=None,
                                    block_count=3, sample_tokens=4)
    with pytest.raises(ValueError, match="Missing"):
        evidence.close()


def test_duplicate_forward_and_broken_chain(tmp_path):
    model = Model()
    evidence = BlockBoundaryCapture(tmp_path / "duplicate", model, signature=signature(),
                                    uuid="GPU-test", scale=1, replay=None,
                                    block_count=3, sample_tokens=4)
    x = torch.ones(1, 1, 1, 6, 1, requires_grad=True)
    try:
        model(x)
        with pytest.raises(ValueError, match="Duplicate"):
            model(x)
    finally:
        evidence.restore()
    model = Model()
    evidence = BlockBoundaryCapture(tmp_path / "broken", model, signature=signature(),
                                    uuid="GPU-test", scale=1, replay=None,
                                    block_count=3, sample_tokens=4)
    try:
        model.blocks[0](x)
        with pytest.raises(ValueError, match="Nonadjacent"):
            model.blocks[1](x)
    finally:
        evidence.restore()


def test_copied_boundary_is_rejected_even_when_values_match(tmp_path):
    model = Model()
    evidence = BlockBoundaryCapture(tmp_path / "copied", model, signature=signature(),
                                    uuid="GPU-test", scale=1, replay=None,
                                    block_count=3, sample_tokens=4)
    x = torch.ones(1, 1, 1, 6, 1, requires_grad=True)
    try:
        output = model.blocks[0](x)
        with pytest.raises(ValueError, match="Nonadjacent"):
            model.blocks[1](output.clone())
    finally:
        evidence.restore()


def test_tampered_manifest_and_payload_rejected(tmp_path):
    directory = capture(tmp_path, "reference")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["boundaries"][1]["indices"][0] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="samples changed"):
        read_capture(directory)
    manifest["boundaries"][1]["indices"][0] = 0
    manifest_path.write_text(json.dumps(manifest))
    payload = directory / "boundaries.safetensors"
    with payload.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="hash/budget"):
        read_capture(directory)


def test_captured_hash_is_sha256(tmp_path):
    directory = capture(tmp_path, "reference")
    manifest, _ = read_capture(directory)
    assert manifest["sha256"] == sha256_file(directory / manifest["file"])


def test_hooks_leave_original_gradients_and_forwards_unchanged(tmp_path):
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 6, 1)
    baseline = x.detach().clone().requires_grad_()
    expected = Model()(baseline)
    expected.square().sum().backward()
    model = Model()
    actual = x.detach().clone().requires_grad_()
    evidence = BlockBoundaryCapture(tmp_path / "unchanged", model,
                                    signature=signature(), uuid="GPU-test", scale=1,
                                    replay={}, block_count=3, sample_tokens=4)
    try:
        output = model(actual)
        output.square().sum().backward()
        evidence.close()
    finally:
        evidence.restore()
    assert torch.equal(output, expected)
    assert torch.equal(actual.grad, baseline.grad)
    assert all(not block._forward.__name__.startswith("capture") for block in model.blocks)


def test_full_backward_hooks_preserve_adjacent_cotangents(tmp_path):
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 6, 1)
    baseline = x.detach().clone().requires_grad_()
    expected_model = Model()
    for block in expected_model.blocks:
        block.register_full_backward_hook(lambda *_: None)
    expected = expected_model(baseline)
    expected.square().sum().backward()

    model = Model()
    backward_handles = [block.register_full_backward_hook(lambda *_: None)
                        for block in model.blocks]
    evidence = BlockBoundaryCapture(tmp_path / "real-hooks", model,
                                    signature=signature(), uuid="GPU-test", scale=1,
                                    replay={}, block_count=3, sample_tokens=4)
    observed = {}
    previous = [None]
    forward_handles = []
    for index, block in enumerate(model.blocks):
        def observe(_module, args, output, index=index):
            if index:
                assert args[0] is not previous[0]
                args[0].register_hook(lambda grad, index=index:
                                      observed.setdefault(("input", index), grad.detach().clone()))
            output.register_hook(lambda grad, index=index:
                                 observed.setdefault(("output", index), grad.detach().clone()))
            previous[0] = output
        forward_handles.append(block.register_forward_hook(observe))

    actual = x.detach().clone().requires_grad_()
    try:
        output = model(actual)
        output.square().sum().backward()
        evidence.close()
    finally:
        evidence.restore()
        for handle in backward_handles + forward_handles:
            handle.remove()
    assert torch.equal(output, expected)
    assert torch.equal(actual.grad, baseline.grad)
    assert len(observed) == 5
    for index in (1, 2):
        assert torch.equal(observed[("output", index - 1)], observed[("input", index)])
    _, tensors = read_capture(tmp_path / "real-hooks")
    indices = tensors["boundary_01.indices"]
    sampled = observed[("output", 0)].flatten(1, -2).index_select(1, indices)
    assert torch.equal(tensors["boundary_01.gradient"], sampled)


@pytest.mark.parametrize("field,value", [
    ("scale", 1024), ("sample_tokens", 513), ("sample_tokens", 0),
    ("max_bytes", MAX_BYTES + 1), ("max_bytes", 0),
])
def test_invalid_capture_contract_fails_before_output(tmp_path, field, value):
    kwargs = {"scale": 1, field: value}
    with pytest.raises(ValueError, match="contract"):
        BlockBoundaryCapture(tmp_path / "invalid", Model(), signature=signature(),
                             uuid="GPU-test", replay={}, block_count=3, **kwargs)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("field", ["index", "shape", "token_layout"])
def test_manifest_layout_tamper_rejected(tmp_path, field):
    directory = capture(tmp_path, "reference")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["boundaries"][1][field] = {"index": 0, "shape": [1, 2, 3],
                                         "token_layout": "forged"}[field]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract|samples changed|Insufficient block tokens"):
        read_capture(directory)


def _driver_fixture(tmp_path, monkeypatch, *, mutate=None, monitor_failure=False):
    weights, inputs = tmp_path / "weights", tmp_path / "inputs"
    weights.write_bytes(b"weights")
    inputs.write_bytes(b"inputs")
    args = SimpleNamespace(weights=weights, inputs=inputs, output_dir=tmp_path / "output",
                           gpu_uuid="GPU-test", swap=20, worker_timeout=10)
    monkeypatch.setattr(driver, "sha256_file", lambda _: "h" * 64)
    monkeypatch.setattr(driver, "_complete", lambda *_a, **_kw: True)
    targets = target_names()
    assignments = {name: "fp16" for name in targets}
    assignments.update({f"other.{index}": "fp32" for index in range(321)})
    launched = []

    def capture_data(name):
        reference = name == "reference"
        control = name == "control"
        modes = None if reference else {key: "fp32" if control else value
                                            for key, value in assignments.items()}
        signature_data = {
            "steps": 1, "sigmas": [0.5], "block_boundary_scope": SCOPE,
            "block_boundary_helper_sha256": "h" * 64, "probe_sha256": "h" * 64,
            "weights_sha256": "h" * 64, "inputs_sha256": "h" * 64,
            "backend": "torch", "checkpoint": "full", "rank": 4, "alpha": 4,
            "loss_scale": 1 if reference else 1024,
            "precision": "fp32-reference" if reference else "fp16-islands",
            "fp32_modules": [] if reference else sorted(k for k, v in modes.items() if v == "fp32"),
        }
        replay = {"snapshot_sha256": "r" * 64}
        capture_manifest = {"signature": signature_data, "cases": [{"sigma": 0.5,
                            "replay": replay}], "initial_adapter_sha256": "i" * 64,
                            "expected_gradients": ["gradient.lora"]}
        boundary = {"signature": signature_data, "uuid": "GPU-test", "block_count": 28,
                    "sample_tokens": 512, "replay": replay}
        worker = {"swap": 20, "precision_manifest": None if reference else
                  {"assignments": modes}}
        if mutate and name == mutate[0]:
            mutate[1](capture_manifest, boundary, worker)
        return capture_manifest, boundary, worker

    data = {name: capture_data(name) for name in ("reference", "candidate", "control")}
    monkeypatch.setattr(driver, "read_capture", lambda path: (data[path.parent.name][0],
                                                               data[path.parent.name][2]))
    monkeypatch.setattr(driver.boundaries, "read_capture", lambda path:
                        (data[path.parent.name][1], {}))
    monkeypatch.setattr(driver.boundaries, "analyze", lambda *_: {"status": "analyzed"})

    def launch(argv, directory, *_a):
        launched.append((directory.name, argv))
        return {"status": "monitor_failed" if monitor_failure else "ok"}

    monkeypatch.setattr(driver, "launch_if_idle", launch)
    if monitor_failure:
        monkeypatch.setattr(driver, "_complete", lambda result, *_a, **_kw:
                            result["status"] == "ok")
    return args, launched


def test_driver_three_worker_cpu_contract_and_no_plan(tmp_path, monkeypatch):
    args, launched = _driver_fixture(tmp_path, monkeypatch)
    report = driver.run(args, gpu_check=lambda _: {"uuid": "GPU-test"},
                        comparator=lambda *_: {"status": "compared"})
    assert report["status"] == "completed"
    assert [name for name, _ in launched] == ["reference", "candidate", "control"]
    commands = [argv for _, argv in launched]
    assert [argv[argv.index("--loss-scale") + 1] for argv in commands] == ["1", "1024", "1024"]
    assert [argv.count("--fp16-only-pattern") for argv in commands] == [0, 1, 0]
    assert [argv.count("--fp32-pattern") for argv in commands] == [0, 0, 1]
    assert all("--capture-block-boundaries" in argv for argv in commands)
    assert not list(args.output_dir.glob("*plan*"))


@pytest.mark.parametrize("mutate", [
    ("candidate", lambda cap, _b, _w: cap.update(initial_adapter_sha256="different")),
    ("control", lambda cap, _b, _w: cap["cases"][0].update(replay={"forged": True})),
    ("candidate", lambda _c, _b, w: w["precision_manifest"]["assignments"].update(
        {next(iter(target_names())): "fp32"})),
    ("control", lambda _c, _b, w: w["precision_manifest"]["assignments"].update(
        {next(iter(target_names())): "fp16"})),
])
def test_driver_rejects_cross_worker_identity_or_plan(tmp_path, monkeypatch, mutate):
    args, launched = _driver_fixture(tmp_path, monkeypatch, mutate=mutate)
    report = driver.run(args, gpu_check=lambda _: {"uuid": "GPU-test"})
    assert report["status"] == "failed" and "stop_reason" in report
    assert not (args.output_dir / "boundary-analysis.json").exists()
    assert len(launched) >= 2


def test_driver_stops_after_monitor_failure(tmp_path, monkeypatch):
    args, launched = _driver_fixture(tmp_path, monkeypatch, monitor_failure=True)
    report = driver.run(args, gpu_check=lambda _: {"uuid": "GPU-test"})
    assert report["status"] == "failed"
    assert [name for name, _ in launched] == ["reference"]
    assert not (args.output_dir / "boundary-analysis.json").exists()
