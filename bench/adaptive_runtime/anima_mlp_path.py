"""Sampled, observational GELU-path diagnostics for the Anima local VJP probe."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from library.training.adaptive_runtime.precision import vector_metrics


PROFILE = "mlp-path"
BLOCKS = (0, 5, 27)
NAMES = tuple(f"blocks.{block}.mlp.layer{layer}" for block in BLOCKS for layer in (1, 2))


def sample_row(entry, data):
    keys = ("y", "g") if entry["name"].endswith(".layer1") else ("x", "gx")
    return {"name": entry["name"], "approximate": entry["approximate"],
            "data": {key: data[key] for key in keys}}


def activation_metadata(modules, name):
    block = name.split(".")[1]
    activation = modules.get(f"blocks.{block}.mlp.activation")
    if type(activation) is not torch.nn.GELU or activation.approximate not in {"none", "tanh"}:
        raise ValueError(f"MLP path requires supported GELU: {name}")
    return activation.approximate


def gelu_vjp(z, gy, approximate):
    with torch.enable_grad(), torch.autocast(z.device.type, enabled=False):
        source = z.detach().float().requires_grad_(True)
        output = F.gelu(source, approximate=approximate)
        return torch.autograd.grad(output, source, gy.detach().float())[0].detach()


def analyze_path(reference, candidate, *, device):
    results = []
    for first, second in zip(range(0, len(NAMES), 2), range(1, len(NAMES), 2), strict=True):
        r1, c1 = reference[first], candidate[first]
        r2, c2 = reference[second], candidate[second]
        if (r1["name"] != NAMES[first] or r2["name"] != NAMES[second]
                or c1["name"] != r1["name"] or c2["name"] != r2["name"]
                or r1["approximate"] != r2["approximate"]
                or c1["approximate"] != c2["approximate"]
                or r1["approximate"] != c1["approximate"]):
            raise ValueError("MLP path inventory or activation mismatch")
        approximate = r1["approximate"]
        zref, zcand = r1["data"]["y"].to(device), c1["data"]["y"].to(device)
        aref, acand = r2["data"]["x"].to(device), c2["data"]["x"].to(device)
        gyref, gycand = r2["data"]["gx"].to(device), c2["data"]["gx"].to(device)
        gzref, gzcand = r1["data"]["g"].to(device), c1["data"]["g"].to(device)
        vjps = {"reference": gelu_vjp(zref, gyref, approximate),
                "z_only": gelu_vjp(zcand, gyref, approximate),
                "gy_only": gelu_vjp(zref, gycand, approximate),
                "both": gelu_vjp(zcand, gycand, approximate)}
        def measure(a, b):
            return vector_metrics([(a, b)])
        results.append({"block": int(r1["name"].split(".")[1]), "approximate": approximate,
                        "forward_fidelity_reference": measure(F.gelu(zref, approximate=approximate), aref),
                        "forward_fidelity_candidate": measure(F.gelu(zcand, approximate=approximate), acand),
                        "reference_vjp_fidelity": measure(vjps["reference"], gzref),
                        "candidate_vjp_fidelity": measure(vjps["both"], gzcand),
                        "drift": {"z": measure(zref, zcand), "a": measure(aref, acand),
                                  "gy": measure(gyref, gycand), "gz": measure(gzref, gzcand)},
                        "fixed_gy_z_only": measure(vjps["reference"], vjps["z_only"]),
                        "fixed_z_gy_only": measure(vjps["reference"], vjps["gy_only"]),
                        "both": measure(vjps["reference"], vjps["both"])})
    return results
