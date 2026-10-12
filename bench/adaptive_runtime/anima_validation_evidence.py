"""Check the source state/replay artifacts behind a validation worker."""

import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from bench.adaptive_runtime import anima_validation_updates as updates
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.capture import capture_path
from bench.adaptive_runtime.replay import TrainingReplay, evidence, tensor_digest


def _network_state(tensors):
    return {k: v for k, v in tensors.items() if k.startswith(("parameter.", "buffer."))}


def validate_warmup_bundle(worker_dir, bundle):
    worker_dir = Path(worker_dir)
    result = json.loads((worker_dir / "result.json").read_text())
    training = json.loads((worker_dir / "training-capture/manifest.json").read_text())
    update = json.loads((worker_dir / "update-capture/manifest.json").read_text())
    provenance = bundle.manifest["provenance"]
    if (provenance.get("warmup_signature") != result["signature"]
            or provenance.get("warmup_inputs_sha256") != result["inputs_sha256"]
            or provenance.get("training_case_sha256") != training["cases"][0]["sha256"]
            or provenance.get("update_case_sha256") != update["cases"][0]["sha256"]):
        raise ValueError("State bundle warmup provenance mismatch")
    replay_dir = worker_dir / "training-replay"
    replay = json.loads((replay_dir / "manifest.json").read_text())
    case = replay["cases"][0]
    path = capture_path(replay_dir, case["file"])
    tensors = load_file(str(path))
    if (evidence(tensors, case["python_rng"], sha256_file(path)) != case["evidence"]
            or tensor_digest(_network_state(tensors))
            != tensor_digest(_network_state(bundle.tensors["initial"]))):
        raise ValueError("Warmup replay differs from initial bundle")
    delta = load_file(str(capture_path(worker_dir / "update-capture", update["cases"][0]["file"])))
    params = {k: v for k, v in bundle.tensors["initial"].items() if k.startswith("parameter.")}
    if (set(delta) != {"delta." + name.removeprefix("parameter.") for name in params}
            or any(not torch.equal(delta["delta." + name.removeprefix("parameter.")],
                                   bundle.tensors["nonzero"][name] - value)
                   for name, value in params.items())):
        raise ValueError("Nonzero bundle differs from actual warmup update")


def validate_validation_sources(worker_dir, reference_dir, bundle):
    manifest, worker, cases = updates._read(worker_dir)
    signature = worker["signature"]
    if (signature.get("validation_state_identity") != bundle.manifest["identity"]
            or worker["uuid"] != bundle.manifest["uuid"]):
        raise ValueError("Worker state identity differs from source bundle")
    reference_dir = Path(reference_dir)
    replay_dir = reference_dir / "training-replay"
    if replay_dir.is_symlink() or (replay_dir / "manifest.json").is_symlink():
        raise ValueError("Unsafe replay source")
    reader = TrainingReplay(Path(worker_dir) / "training-replay", signature=signature,
                            uuid=worker["uuid"], steps=6, source=replay_dir)
    ref_result = json.loads((reference_dir / "result.json").read_text())
    if reader.manifest["signature"] != ref_result.get("signature"):
        raise ValueError("Replay source worker signature mismatch")
    used = 0
    for (label, sigma), (case, _), replay_case in zip(
            updates.CASES, cases, reader.manifest["cases"], strict=True):
        path = capture_path(replay_dir, replay_case["file"])
        used += path.stat().st_size
        if path.is_symlink() or path.stat().st_size > 512 * 1024**2 or used > 1024**3:
            raise ValueError("Replay source file budget/path mismatch")
        digest = sha256_file(path)
        if digest != replay_case["evidence"]["snapshot_sha256"]:
            raise ValueError("Replay source snapshot digest mismatch")
        tensors = load_file(str(path))
        actual = evidence(tensors, replay_case["python_rng"], digest)
        state = bundle.manifest["states"][label]
        expected_state = {"bundle_manifest_sha256": bundle.manifest_sha256,
                          "state_file_sha256": state["sha256"], "state_label": label,
                          "parameter_sha256": state["parameter_sha256"],
                          "optimizer_sha256": state["optimizer_sha256"]}
        if (actual != replay_case["evidence"] or actual != case["replay"]
                or case["state_evidence"] != expected_state or case["optimizer_options"] != state["groups"]
                or tensor_digest(_network_state(tensors))
                != tensor_digest(_network_state(bundle.tensors[label]))
                or "input.sigma" not in tensors
                or not torch.equal(tensors["input.sigma"], torch.tensor([sigma], dtype=torch.float32))
                or not all(bool(torch.isfinite(value).all()) for value in tensors.values())):
            raise ValueError("Replay or optimizer evidence differs from source artifacts")
    return {"state_bundle_sha256": bundle.manifest_sha256,
            "replay_manifest_sha256": sha256_file(replay_dir / "manifest.json"),
            "validated_cases": len(manifest["cases"])}
