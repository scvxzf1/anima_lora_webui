import json
from pathlib import Path

import pytest
import torch

from bench.adaptive_runtime.anima_update_comparison import (
    UpdateCapture, compare_update, sha256_file,
)
from bench.adaptive_runtime.replay import MODE
from bench.adaptive_runtime.training_capture import TrainingCapture
from library.training.auto_block_swap.process import write_result


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_unet_blocks_0_mlp_layer1 = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.lora_unet_blocks_0_mlp_layer1.weight.copy_(torch.tensor([[1.0, -2.0], [0.5, 3.0]]))


def make_worker(root, precision, gradient, replay_mode):
    root.mkdir()
    network = Adapter()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    signature = {"family": "anima", "precision": precision, "comparison_mode": MODE,
                 "capture_update": True, "steps": 1, "sigmas": [0.5],
                 "weights_sha256": "w" * 64, "inputs_sha256": "i" * 64,
                 "update_source_sha256": sha256_file(
                     Path(__import__(UpdateCapture.__module__, fromlist=["x"]).__file__))}
    replay = {key: key[0] * 64 for key in
              ("snapshot_sha256", "state_sha256", "inputs_sha256", "rng_sha256")}
    training = TrainingCapture(root / "training-capture", network=network,
                               signature=signature, steps=1)
    update = UpdateCapture(root / "update-capture", network=network,
                           signature=signature, uuid="GPU-123")
    network.lora_unet_blocks_0_mlp_layer1.weight.grad = torch.tensor(gradient, dtype=torch.float32)
    noisy = torch.tensor([1.0], requires_grad=True)
    noisy.grad = torch.tensor([0.5])
    training.capture(network=network, prediction=torch.tensor([2.0]), noisy=noisy,
                     step=1, sigma=0.5, replay=replay)
    update.before(network=network, optimizer=optimizer, step=1, sigma=0.5,
                  replay=replay, training_case=training.manifest["cases"][0], clip_max_norm=1.0)
    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
    optimizer.step()
    update.after(network=network, optimizer=optimizer, gradient_norm=norm)
    training.close()
    update.close()
    write_result(root / "result.json", {"status": "ok", "precision": precision,
                 "uuid": "GPU-123", "weights_sha256": "w" * 64,
                 "inputs_sha256": "i" * 64, "signature": signature,
                 "replay_mode": replay_mode, "updates": [{"step": 1}]})
    update.bind_result(root)
    return root


def test_real_adamw_updates_with_different_gradients(tmp_path):
    reference = make_worker(tmp_path / "ref", "fp32-reference", [[1, 2], [3, 4]], "record")
    candidate = make_worker(tmp_path / "trial", "fp16-islands", [[-2, 1], [4, 3]], "replay")
    result = compare_update(reference, candidate)
    assert result["global"]["finite"]
    assert result["global"]["absolute_l2"] > 0
    assert result["groups"]["mlp"]["absolute_l2"] > 0
    assert result["groups"]["other"]["absolute_l2"] == 0


def test_rejects_nonempty_moments_before_step(tmp_path):
    network = Adapter()
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    param = next(network.parameters())
    optimizer.state[param]["step"] = torch.tensor(1.0)
    update = UpdateCapture(tmp_path / "update", network=network,
                           signature={"comparison_mode": MODE, "capture_update": True}, uuid="GPU-123")
    replay = {key: "a" * 64 for key in
              ("snapshot_sha256", "state_sha256", "inputs_sha256", "rng_sha256")}
    with pytest.raises(ValueError, match="nonempty"):
        update.before(network=network, optimizer=optimizer, step=1, sigma=0.5,
                      replay=replay, training_case={"step": 1, "replay": replay, "sha256": "b" * 64},
                      clip_max_norm=1.0)


def test_rejects_tampered_capture_contract(tmp_path):
    reference = make_worker(tmp_path / "ref", "fp32-reference", [[1, 2], [3, 4]], "record")
    candidate = make_worker(tmp_path / "trial", "fp16-islands", [[2, 1], [4, 3]], "replay")
    path = candidate / "update-capture" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["cases"][0]["optimizer_contract"]["state"] = "nonzero"
    write_result(path, manifest)
    with pytest.raises(ValueError, match="contract"):
        compare_update(reference, candidate)
