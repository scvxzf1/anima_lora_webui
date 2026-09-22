import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch

from bench.adaptive_runtime.replay import MODE, TrainingReplay, tensor_digest, validate_replay
from bench.adaptive_runtime.compare_training import compare
from bench.adaptive_runtime.training_capture import TrainingCapture


def record(tmp_path, *, steps=1, max_bytes=1024**3, mask=False):
    torch.manual_seed(42)
    network = torch.nn.Linear(4, 3)
    network.register_buffer("mask", torch.ones(1), persistent=False)
    signature = {"precision": "fp32-reference", "comparison_mode": MODE, "inputs_sha256": "same"}
    root = tmp_path / "reference"
    archive = TrainingReplay(root / "training-replay", signature=signature, uuid="test-device",
                             steps=steps, max_bytes=max_bytes)
    capture = TrainingCapture(root / "training-capture", network=network, signature=signature, steps=steps)
    initial = copy.deepcopy(network)
    for step in range(1, steps + 1):
        noisy = torch.randn(2, 4)
        noisy, target, prompts, sigma, evidence = archive.prepare(
            network, noisy=noisy, target=torch.zeros(2, 3),
            prompts=[torch.ones(2, 4)] + ([torch.tensor([[True, False]])] if mask else []),
            sigma=torch.tensor([0.2]), step=step)
        prediction = network(noisy) * network.mask
        (prediction - target).square().mean().backward()
        capture.capture(network=network, prediction=prediction, noisy=noisy, step=step, sigma=0.2,
                        replay=evidence)
        network.zero_grad()
        with torch.no_grad():
            network.weight.add_(0.25)
            network.mask.add_(0.5)
    archive.close()
    capture.close()
    (root / "result.json").write_text(json.dumps({
        "status": "ok", "precision": "fp32-reference", "uuid": "test-device", "replay_mode": "record",
        "inputs_sha256": "same", "updates": [{"step": s} for s in range(1, steps + 1)],
    }))
    return archive, initial, signature


def test_replay_restores_buffers_rng_and_each_pre_step_state(tmp_path):
    rng = random.getstate()
    try:
        archive, network, signature = record(tmp_path, steps=2)
        source_bytes = (archive.directory / "manifest.json").read_bytes()
        player = TrainingReplay(tmp_path / "unused", source=archive.directory, signature=signature,
                                uuid="test-device", steps=2)
        noisy = torch.randn(2, 4, dtype=torch.bfloat16)
        for step in (1, 2):
            with torch.no_grad():
                network.weight.fill_(99)
                network.mask.fill_(0)
            random.seed(99)
            torch.manual_seed(99)
            values = player.prepare(network, noisy=noisy, target=torch.zeros(2, 3),
                                    prompts=[torch.zeros(2, 4)], sigma=torch.zeros(1), step=step)
            restored, target, prompts, sigma, actual = values
            assert actual == archive.manifest["cases"][step - 1]["evidence"]
            assert restored.dtype == torch.float32 and restored.is_leaf and restored.requires_grad
            assert network.mask.item() == 1 + (step - 1) * 0.5
            assert sigma.item() == pytest.approx(0.2)
            assert torch.equal(prompts[0], torch.ones(2, 4))
            assert torch.equal(target, torch.zeros(2, 3))
        player.close()
        assert source_bytes == (archive.directory / "manifest.json").read_bytes()
    finally:
        random.setstate(rng)


@pytest.mark.parametrize("corruption", ["uuid", "status", "precision", "steps", "contract", "worker"])
def test_replay_rejects_wrong_reference(tmp_path, corruption):
    archive, _, signature = record(tmp_path)
    path = archive.directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if corruption == "worker":
        (archive.directory.parent / "result.json").write_text('{"status":"error"}')
    elif corruption == "contract":
        manifest["signature"]["inputs_sha256"] = "wrong"
    elif corruption == "precision":
        manifest["signature"]["precision"] = "bf16"
    else:
        manifest[corruption] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reference contract"):
        TrainingReplay(tmp_path / "unused", source=archive.directory, signature=signature,
                       uuid="test-device", steps=1)


def test_replay_rejects_digest_corruption_and_budget(tmp_path):
    archive, network, signature = record(tmp_path)
    path = archive.directory / "step-000001.safetensors"
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    player = TrainingReplay(tmp_path / "unused", source=archive.directory, signature=signature,
                            uuid="test-device", steps=1)
    with pytest.raises(ValueError, match="digest"):
        player.prepare(network, noisy=torch.zeros(2, 4), target=torch.zeros(2, 3),
                       prompts=[torch.ones(2, 4)], sigma=torch.zeros(1), step=1)
    with pytest.raises(ValueError, match="budget"):
        record(tmp_path / "small", max_bytes=1)


def test_fixed_state_comparison_and_mismatched_evidence(tmp_path):
    archive, network, signature = record(tmp_path)
    signature = {**signature, "precision": "fp16-islands"}
    player = TrainingReplay(tmp_path / "unused", source=archive.directory, signature=signature,
                            uuid="test-device", steps=1)
    root = tmp_path / "candidate"
    recorder = TrainingCapture(root / "training-capture", network=network, signature=signature, steps=1)
    noisy, target, _, _, evidence = player.prepare(
        network, noisy=torch.zeros(2, 4), target=torch.zeros(2, 3), prompts=[torch.zeros(2, 4)],
        sigma=torch.zeros(1), step=1)
    prediction = network(noisy) * network.mask
    (prediction - target).square().mean().backward()
    recorder.capture(network=network, prediction=prediction, noisy=noisy, step=1, sigma=0.2, replay=evidence)
    recorder.close()
    (root / "result.json").write_text(json.dumps({
        "status": "ok", "precision": "fp16-islands", "uuid": "test-device", "replay_mode": "replay",
        "inputs_sha256": "same", "updates": [{"step": 1}],
    }))
    reference = archive.directory.parent / "training-capture"
    report = compare(reference, recorder.directory)
    assert report["within_experimental_tolerances"] is True
    assert report["scope"] == "fixed_pre_step_state_and_inputs_not_quality_certificate"
    assert report["production_ready"] is False
    manifest = json.loads((recorder.directory / "manifest.json").read_text())
    manifest["cases"][0]["replay"]["state_sha256"] = "0" * 64
    (recorder.directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="per-step replay evidence"):
        compare(reference, recorder.directory)


@pytest.mark.parametrize("change", [{"disposable_probe": False}, {"capture_training": False},
                                    {"inputs": None}, {"resume": "state"},
                                    {"checkpoint_every_step": True}, {"capture_linear": ["a"]},
                                    {"precision": "bf16"}])
def test_replay_cli_rejects_unsafe_combinations(change):
    args = {"record_replay": True, "replay_reference": None, "disposable_probe": True,
            "capture_training": True, "inputs": "input", "resume": None,
            "checkpoint_every_step": False, "capture_linear": [], "precision": "fp32-reference"}
    with pytest.raises(ValueError):
        validate_replay(SimpleNamespace(**{**args, **change}))


def test_tensor_digest_distinguishes_dtype_and_scalar_buffers():
    assert tensor_digest({"a": torch.tensor(1.0)}) != tensor_digest({"a": torch.tensor(1)})


@pytest.mark.parametrize("wrong_dtype", [False, True])
def test_replay_preserves_boolean_text_mask(tmp_path, wrong_dtype):
    archive, network, signature = record(tmp_path, mask=True)
    player = TrainingReplay(tmp_path / "unused", source=archive.directory, signature=signature,
                            uuid="test-device", steps=1)
    prompts = [torch.zeros(2, 4), torch.zeros(1, 2, dtype=torch.float32 if wrong_dtype else torch.bool)]
    kwargs = dict(noisy=torch.zeros(2, 4), target=torch.zeros(2, 3), prompts=prompts,
                  sigma=torch.zeros(1), step=1)
    if wrong_dtype:
        with pytest.raises(ValueError, match="tensor/state/input contract"):
            player.prepare(network, **kwargs)
    else:
        _, _, restored, _, evidence = player.prepare(network, **kwargs)
        assert restored[1].dtype == torch.bool
        assert torch.equal(restored[1], torch.tensor([[True, False]]))
        assert evidence == archive.manifest["cases"][0]["evidence"]
