"""FP64 gradient-error attribution for validated fixed-state Anima captures."""

from __future__ import annotations

import math
import re

import torch

from bench.adaptive_runtime.compare_training import load_case, read_capture


BLOCK = re.compile(r"^gradient\.lora_unet_blocks_(\d+)_(.+)\.(lora_down|lora_up)\.weight$")


def parameter_group(name):
    match = BLOCK.fullmatch(name)
    if not match:
        return "other", "other", "other"
    block, component, direction = match.groups()
    family = next((part for part in ("mlp", "self_attn", "cross_attn")
                   if component.startswith(part + "_") or component == part), "other")
    return f"block-{int(block):03d}", family, direction.removeprefix("lora_")


class Accumulator:
    def __init__(self):
        self.ref_square = self.candidate_square = self.diff_square = self.dot = 0.0
        self.elements = self.ref_zeros = self.candidate_zeros = self.both_zeros = 0
        self.parameters = 0

    def add(self, ref, candidate):
        if ref.shape != candidate.shape or ref.dtype != torch.float32 or candidate.dtype != torch.float32:
            raise ValueError("Gradient shape/dtype contract changed")
        if not torch.isfinite(ref).all() or not torch.isfinite(candidate).all():
            raise ValueError("Nonfinite gradient diagnostic input")
        a, b = ref.double(), candidate.double()
        self.ref_square += float(a.square().sum())
        self.candidate_square += float(b.square().sum())
        self.diff_square += float((b - a).square().sum())
        self.dot += float((a * b).sum())
        self.elements += a.numel()
        self.ref_zeros += int((a == 0).sum())
        self.candidate_zeros += int((b == 0).sum())
        self.both_zeros += int(((a == 0) & (b == 0)).sum())
        self.parameters += 1

    def result(self, total_diff_square):
        ref_l2 = math.sqrt(self.ref_square)
        candidate_l2 = math.sqrt(self.candidate_square)
        diff_l2 = math.sqrt(self.diff_square)
        relative = diff_l2 / ref_l2 if ref_l2 else (0.0 if not candidate_l2 else None)
        cosine = (self.dot / (ref_l2 * candidate_l2) if ref_l2 and candidate_l2 else
                  1.0 if not ref_l2 and not candidate_l2 else 0.0)
        return {"absolute_l2": diff_l2, "reference_l2": ref_l2,
                "candidate_l2": candidate_l2, "relative_l2": relative,
                "cosine": max(-1.0, min(1.0, cosine)), "diff_square": self.diff_square,
                "diff_square_fraction": self.diff_square / total_diff_square if total_diff_square else 0.0,
                "elements": self.elements, "parameters": self.parameters,
                "reference_zero_elements": self.ref_zeros,
                "candidate_zero_elements": self.candidate_zeros,
                "both_zero_elements": self.both_zeros}


def diagnose(reference, candidate, comparison):
    if comparison.get("status") != "compared" or not comparison.get("scope", "").startswith("fixed_pre_step"):
        raise ValueError("Verified fixed-state comparison required")
    ref_manifest, _ = read_capture(reference)
    trial_manifest, _ = read_capture(candidate)
    if len(comparison.get("cases", [])) != ref_manifest["steps"]:
        raise ValueError("Comparison step count changed")
    groups = {axis: {} for axis in ("step", "block", "family", "direction")}
    per_step = {}
    overall = Accumulator()
    unknown = []
    for ref_case, trial_case in zip(ref_manifest["cases"], trial_manifest["cases"], strict=True):
        step_groups = {axis: {} for axis in ("block", "family", "direction")}
        step_total = Accumulator()
        a = load_case(reference, ref_case, ref_manifest["expected_gradients"])
        b = load_case(candidate, trial_case, trial_manifest["expected_gradients"])
        if set(a) != set(b):
            raise ValueError("Gradient inventory changed")
        for name in sorted(n for n in a if n.startswith("gradient.")):
            block, family, direction = parameter_group(name)
            if block == "other":
                unknown.append(name)
            overall.add(a[name], b[name])
            step_total.add(a[name], b[name])
            labels = {"step": str(ref_case["step"]), "block": block,
                      "family": family, "direction": direction}
            for axis, label in labels.items():
                groups[axis].setdefault(label, Accumulator()).add(a[name], b[name])
                if axis != "step":
                    step_groups[axis].setdefault(label, Accumulator()).add(a[name], b[name])
        per_step[str(ref_case["step"])] = {
            "sigma": ref_case["sigma"], "overall": step_total.result(step_total.diff_square),
            "groups": {axis: {label: item.result(step_total.diff_square)
                              for label, item in sorted(values.items())}
                       for axis, values in step_groups.items()},
        }
    return {"status": "diagnosed", "production_ready": False,
            "scope": "fixed_pre_step_state_and_inputs_not_trajectory_certificate",
            "source_signature": ref_manifest["signature"],
            "sigmas": [case["sigma"] for case in ref_manifest["cases"]],
            "overall": overall.result(overall.diff_square),
            "per_step": per_step,
            "groups": {axis: {label: item.result(overall.diff_square)
                              for label, item in sorted(values.items())}
                       for axis, values in groups.items()},
            "unknown_parameters": sorted(set(unknown)),
            "zero_counts_are_observations_not_underflow_classification": True}
