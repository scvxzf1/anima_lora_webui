"""Bounded full-forward/loss-gradient evidence from fixed-input short probes."""

import hashlib
from pathlib import Path

from safetensors.torch import save_file
import torch

from library.training.auto_block_swap.process import write_result


def parameter_digest(network):
    digest = hashlib.sha256()
    for name, parameter in network.named_parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


class TrainingCapture:
    def __init__(self, directory, *, network, signature, steps, max_bytes=1024**3):
        if type(steps) is not int or steps < 1 or max_bytes < 1:
            raise ValueError("Positive capture steps and byte budget required")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.max_bytes, self.used_bytes = max_bytes, 0
        gradients = [f"gradient.{name}" for name, p in network.named_parameters() if p.requires_grad]
        if not gradients:
            raise ValueError("Training capture requires trainable parameters")
        self.manifest = {"schema": "adaptive_training_capture_v1", "status": "capturing",
                         "scope": "full_dit_short_training_trajectory_not_quality_certificate",
                         "signature": signature, "steps": steps,
                         "initial_adapter_sha256": parameter_digest(network),
                         "expected_gradients": gradients, "cases": []}
        self._write()

    def _write(self):
        write_result(self.directory / "manifest.json", self.manifest)

    def capture(self, *, network, prediction, noisy, step, sigma, replay=None):
        if step != len(self.manifest["cases"]) + 1 or step > self.manifest["steps"]:
            raise ValueError("Training capture requires contiguous steps")
        if noisy.grad is None:
            raise ValueError("Missing full-model input gradient")
        tensors = {"prediction": prediction, "input_gradient": noisy.grad}
        tensors.update({f"gradient.{name}": p.grad for name, p in network.named_parameters()
                        if p.requires_grad and p.grad is not None})
        if set(tensors) - {"prediction", "input_gradient"} != set(self.manifest["expected_gradients"]):
            raise ValueError("Missing or changed trainable parameter gradients")
        size = sum(t.numel() * 4 for t in tensors.values())
        if self.used_bytes + size > self.max_bytes:
            raise ValueError("Training capture byte budget exceeded")
        if not all(torch.isfinite(t.detach()).all() for t in tensors.values()):
            raise ValueError("Training capture requires finite tensors")
        tensors = {name: value.detach().float().cpu().contiguous() for name, value in tensors.items()}
        name = f"step-{step:06d}.safetensors"
        path = self.directory / name
        save_file(tensors, str(path))
        self.used_bytes += size
        case = {"step": step, "sigma": sigma, "file": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if replay is not None:
            case["replay"] = replay
        self.manifest["cases"].append(case)
        self.manifest["tensor_bytes"] = self.used_bytes
        self._write()

    def close(self):
        self.manifest["status"] = (
            "captured" if len(self.manifest["cases"]) == self.manifest["steps"] else "partial"
        )
        self._write()
