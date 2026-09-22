"""Bounded pre-forward state/input replay for disposable DiT diagnostics."""

import hashlib
import json
from pathlib import Path
import random

from safetensors.torch import load_file, save_file
import torch

from bench.adaptive_runtime.capture import capture_path
from library.training.auto_block_swap.process import write_result


MODE = "fixed_pre_step_state_and_inputs"


def validate_replay(args):
    if not (args.record_replay or args.replay_reference):
        return
    if (not args.disposable_probe or not args.capture_training or not args.inputs
            or args.resume or args.checkpoint_every_step or args.capture_linear):
        raise ValueError("Replay requires disposable real-input training capture without checkpoints")
    if args.record_replay and args.precision != "fp32-reference":
        raise ValueError("Replay recording requires FP32 reference")


def contract(signature):
    return {k: v for k, v in signature.items() if k not in {"precision", "fp32_modules", "loss_scale"}}


def network_tensors(network):
    # Include nonpersistent buffers such as the timestep mask, absent from state_dict.
    return {**{f"parameter.{k}": v for k, v in network.named_parameters()},
            **{f"buffer.{k}": v for k, v in network.named_buffers()}}


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def evidence(tensors, python_rng, snapshot_sha256):
    return {"snapshot_sha256": snapshot_sha256,
            "state_sha256": tensor_digest({k: v for k, v in tensors.items()
                                            if k.startswith(("parameter.", "buffer."))}),
            "inputs_sha256": tensor_digest({k: v for k, v in tensors.items() if k.startswith("input.")}),
            "rng_sha256": hashlib.sha256((tensor_digest({k: v for k, v in tensors.items()
                                                          if k.startswith("rng.")})
                                           + json.dumps(python_rng)).encode()).hexdigest()}


def tuple_tree(value):
    return tuple(tuple_tree(v) for v in value) if isinstance(value, list) else value


class TrainingReplay:
    def __init__(self, directory, *, signature, uuid, steps, source=None, max_bytes=1024**3):
        self.source = Path(source) if source else None
        self.directory = self.source or Path(directory)
        self.used_bytes, self.max_bytes, self.count = 0, max_bytes, 0
        if not uuid or type(steps) is not int or steps < 1 or max_bytes < 1:
            raise ValueError("Replay requires GPU identity, positive steps and byte budget")
        if self.source:
            self.manifest = json.loads((self.directory / "manifest.json").read_text())
            worker = json.loads((self.directory.parent / "result.json").read_text())
            m = self.manifest
            if (m.get("schema") != "adaptive_training_replay_v1" or m.get("status") != "captured"
                    or m.get("uuid") != uuid or worker.get("uuid") != uuid
                    or worker.get("status") != "ok" or worker.get("precision") != "fp32-reference"
                    or worker.get("replay_mode") != "record"
                    or m.get("signature", {}).get("precision") != "fp32-reference"
                    or contract(m.get("signature", {})) != contract(signature)
                    or m.get("steps") != steps
                    or [c.get("step") for c in m.get("cases", [])] != list(range(1, steps + 1))):
                raise ValueError("Replay requires a complete same-device FP32 reference contract")
        else:
            if signature.get("precision") != "fp32-reference":
                raise ValueError("Replay recording requires FP32 reference")
            self.directory.mkdir(parents=True, exist_ok=False)
            self.manifest = {"schema": "adaptive_training_replay_v1", "status": "capturing",
                             "signature": signature, "uuid": uuid, "steps": steps, "cases": []}
            self._write()

    def _write(self):
        write_result(self.directory / "manifest.json", self.manifest)

    def prepare(self, network, *, noisy, target, prompts, sigma, step):
        if step != self.count + 1 or step > self.manifest["steps"]:
            raise ValueError("Replay requires contiguous steps")
        state = network_tensors(network)
        inputs = {"input.noisy": noisy, "input.target": target, "input.sigma": sigma,
                  **{f"input.prompt.{i}": p for i, p in enumerate(prompts)}}
        live = {**state, **inputs, "rng.cpu": torch.get_rng_state()}
        if noisy.is_cuda:
            live["rng.cuda"] = torch.cuda.get_rng_state(noisy.device)
        if self.source:
            tensors, case = self._read(step, live)
            with torch.no_grad():
                for name, value in state.items():
                    value.copy_(tensors[name])
            inputs = {k: tensors[k].to(device=noisy.device) for k in inputs}
            torch.set_rng_state(tensors["rng.cpu"])
            if noisy.is_cuda:
                torch.cuda.set_rng_state(tensors["rng.cuda"], noisy.device)
            random.setstate(tuple_tree(case["python_rng"]))
            restored = {**network_tensors(network), **inputs, "rng.cpu": torch.get_rng_state()}
            if noisy.is_cuda:
                restored["rng.cuda"] = torch.cuda.get_rng_state(noisy.device)
            actual = evidence(restored, random.getstate(), case["evidence"]["snapshot_sha256"])
            if actual != case["evidence"]:
                raise ValueError("Replay restored state/input/RNG digest mismatch")
        else:
            actual = self._record(step, live)
        self.count += 1
        return (inputs["input.noisy"].detach().requires_grad_(True), inputs["input.target"],
                [inputs[f"input.prompt.{i}"] for i in range(len(prompts))],
                inputs["input.sigma"], actual)

    def _record(self, step, live):
        size = sum(v.numel() * v.element_size() for v in live.values())
        if self.used_bytes + size > self.max_bytes or size > 512 * 1024**2:
            raise ValueError("Replay byte budget exceeded")
        if not all(torch.isfinite(v.detach()).all() for v in live.values()):
            raise ValueError("Replay requires finite tensors")
        if any(v.dtype != torch.float32 and not (k.startswith("input.prompt.") and v.dtype == torch.bool)
               for k, v in live.items() if k.startswith("input.")):
            raise ValueError("Replay reference inputs must be FP32 or boolean masks")
        tensors = {k: v.detach().cpu().contiguous().clone() for k, v in live.items()}
        path = self.directory / f"step-{step:06d}.safetensors"
        python_rng = random.getstate()
        save_file(tensors, str(path))
        actual = evidence(tensors, python_rng, hashlib.sha256(path.read_bytes()).hexdigest())
        self.used_bytes += size
        self.manifest["cases"].append({"step": step, "file": path.name,
                                       "python_rng": python_rng, "evidence": actual})
        self.manifest["tensor_bytes"] = self.used_bytes
        self._write()
        return actual

    def _read(self, step, live):
        case = self.manifest["cases"][step - 1]
        path = capture_path(self.directory, case["file"])
        size = path.stat().st_size
        if size > 512 * 1024**2 or self.used_bytes + size > self.max_bytes:
            raise ValueError("Replay byte budget exceeded")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != case["evidence"]["snapshot_sha256"]:
            raise ValueError("Replay snapshot digest mismatch")
        tensors = load_file(str(path))
        if (set(tensors) != set(live)
                or any(t.shape != live[k].shape for k, t in tensors.items())
                or any(t.dtype != (torch.float32 if k.startswith("input.")
                                    and not (k.startswith("input.prompt.") and live[k].dtype == torch.bool)
                                    else live[k].dtype)
                       for k, t in tensors.items())
                or not all(torch.isfinite(t).all() for t in tensors.values())
                or evidence(tensors, case["python_rng"], digest) != case["evidence"]):
            raise ValueError("Replay tensor/state/input contract mismatch")
        self.used_bytes += size
        return tensors, case

    def close(self):
        if not self.source:
            self.manifest["status"] = "captured" if self.count == self.manifest["steps"] else "partial"
            self._write()
