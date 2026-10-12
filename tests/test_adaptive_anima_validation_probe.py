"""CPU integration for independent Anima validation state/update cases."""

from types import SimpleNamespace

import pytest
import torch

from bench.adaptive_runtime import probe_anima_train as probe
from bench.adaptive_runtime.anima_validation_probe import train_validation
from bench.adaptive_runtime.anima_validation_updates import CASES, compare_validation_updates
from library.training.auto_block_swap.process import write_result


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor(0.5))


def model_for(network):
    def model(noisy, sigma, prompt, padding_mask):
        assert noisy.ndim == 5 and padding_mask.shape == (1, 1, 2, 2)
        return noisy * network.adapter + sigma.reshape(1, 1, 1, 1, 1) * network.adapter
    return model


def args(tmp_path, **overrides):
    weights = tmp_path / "weights.safetensors"
    inputs = tmp_path / "inputs.safetensors"
    weights.touch(exist_ok=True)
    inputs.touch(exist_ok=True)
    values = dict(weights=weights, inputs=inputs, output=tmp_path / "worker" / "result.json",
                  precision="fp32-reference", swap=0, steps=1, loss_scale=1.0,
                  fp32_pattern=[], fp16_only_pattern=[], record_replay=True,
                  replay_reference=None, single_sigma=0.5, capture_update=True,
                  capture_local_vjp=False, capture_block_boundaries=False,
                  capture_block_vjp=False, block_vjp_indices=None,
                  local_vjp_profile="linear", backward_counterfactual_suite=False,
                  base_backward_mode="native", validation_states=None,
                  export_validation_states=None)
    return SimpleNamespace(**(values | overrides))


def report(inputs_sha="c" * 64, *, precision="fp32-reference", replay_mode="record"):
    return {"uuid": "cpu-fixture", "weights_sha256": "a" * 64,
            "inputs_sha256": inputs_sha, "fp32_modules": [], "updates": [],
            "precision": precision, "replay_mode": replay_mode}


def inputs(value):
    return {"latents": torch.full((1, 1, 1, 2, 2), value),
            "noise": torch.zeros(1, 1, 1, 2, 2), "prompt": torch.ones(1, 2, 3)}


def finish(arg, result, resources):
    probe.close_resources(resources)
    result["status"] = "ok"
    write_result(arg.output, result)
    for resource in resources:
        if hasattr(resource, "bind_result"):
            resource.bind_result(arg.output.parent)


def _validation_pair(tmp_path):
    state_dir = tmp_path / "states"
    warmup = args(tmp_path, output=tmp_path / "warmup" / "result.json",
                  export_validation_states=state_dir)
    warmup.output.parent.mkdir()
    network = Network()
    resources = []
    warmup_result = report(inputs_sha="b" * 64)
    probe.train_steps(warmup, model_for(network), network, inputs(1), warmup_result, resources)
    finish(warmup, warmup_result, resources)
    assert warmup_result["adapter_max_delta"] > 0

    reference = args(tmp_path, output=tmp_path / "reference" / "result.json",
                     steps=6, single_sigma=None, validation_states=state_dir)
    reference.output.parent.mkdir()
    ref_network = Network()
    ref_resources = []
    ref_result = report()
    train_validation(reference, model_for(ref_network), ref_network, inputs(2), ref_result, ref_resources)
    finish(reference, ref_result, ref_resources)
    assert [(x["state_label"], x["sigma"]) for x in ref_resources[-1].manifest["cases"]] == list(CASES)
    assert len({x["prestate_evidence"]["parameter_sha256"] for x in ref_result["updates"][:3]}) == 1
    assert len({x["prestate_evidence"]["optimizer_sha256"] for x in ref_result["updates"][3:]}) == 1
    assert all(x["effective_update"] and x["adapter_max_delta"] > 0 for x in ref_result["updates"])

    candidate = args(tmp_path, output=tmp_path / "candidate" / "result.json", steps=6,
                     single_sigma=None, validation_states=state_dir,
                     precision="fp16-islands", replay_reference=reference.output.parent / "training-replay",
                     record_replay=False)
    candidate.output.parent.mkdir()
    candidate_network = Network()
    candidate_resources = []
    candidate_result = report(precision="fp16-islands", replay_mode="replay")
    train_validation(candidate, model_for(candidate_network), candidate_network,
                     inputs(2), candidate_result, candidate_resources)
    finish(candidate, candidate_result, candidate_resources)
    comparison = compare_validation_updates(reference.output.parent, candidate.output.parent)
    assert comparison["status"] == "compared" and len(comparison["cases"]) == 6
    assert all(case["global"]["absolute_l2"] == 0 for case in comparison["cases"])
    return warmup, reference, candidate, state_dir, ref_result


def test_export_then_six_independent_validation_cases_and_replay(tmp_path):
    from bench.adaptive_runtime.anima_validation_evidence import validate_validation_sources, validate_warmup_bundle
    from bench.adaptive_runtime.anima_validation_state import StateBundle

    warmup, reference, candidate, state_dir, result = _validation_pair(tmp_path)
    bundle = StateBundle(state_dir, identity=result["signature"]["validation_state_identity"], uuid=result["uuid"])
    validate_warmup_bundle(warmup.output.parent, bundle)
    assert validate_validation_sources(candidate.output.parent, reference.output.parent, bundle)["validated_cases"] == 6


def test_validation_rejects_missing_or_modified_replay_source(tmp_path):
    from bench.adaptive_runtime.anima_validation_evidence import validate_validation_sources
    from bench.adaptive_runtime.anima_validation_state import StateBundle

    _, reference, candidate, state_dir, result = _validation_pair(tmp_path)
    bundle = StateBundle(state_dir, identity=result["signature"]["validation_state_identity"], uuid=result["uuid"])
    path = reference.output.parent / "training-replay/step-000001.safetensors"
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises((ValueError, RuntimeError)):
        validate_validation_sources(candidate.output.parent, reference.output.parent, bundle)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        validate_validation_sources(candidate.output.parent, reference.output.parent, bundle)


@pytest.mark.parametrize("changes", [
    {"steps": 5}, {"capture_update": False}, {"single_sigma": 0.5},
    {"capture_local_vjp": True}, {"record_replay": False}, {"resume": True},
])
def test_validation_rejects_invalid_modes(tmp_path, changes):
    arg = args(tmp_path, **({"steps": 6, "single_sigma": None,
                             "validation_states": tmp_path / "states"} | changes))
    with pytest.raises(ValueError):
        probe.validate_args(arg)


def test_export_rejects_non_reference_or_wrong_sigma(tmp_path):
    for changes in ({"precision": "bf16"}, {"single_sigma": 0.2}, {"steps": 2}):
        arg = args(tmp_path, export_validation_states=tmp_path / "states", **changes)
        with pytest.raises(ValueError):
            probe.validate_args(arg)


def test_cli_state_modes_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as error:
        probe.main(["--weights", str(tmp_path / "weights.safetensors"),
                    "--inputs", str(tmp_path / "inputs.safetensors"),
                    "--output", str(tmp_path / "worker" / "result.json"),
                    "--precision", "fp32-reference",
                    "--export-validation-states", str(tmp_path / "export"),
                    "--validation-states", str(tmp_path / "export")])
    assert error.value.code == 2
