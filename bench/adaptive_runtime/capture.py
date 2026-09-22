"""Bounded real-activation fixtures for local frozen-Linear calibration."""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path

from safetensors.torch import save_file
import torch

from library.training.auto_block_swap.process import write_result


def select_rows(value, limit):
    if limit < 1 or value.ndim < 1 or value.numel() == 0:
        raise ValueError("Row sampling requires nonempty input and a positive limit")
    flat = value.detach().reshape(-1, value.shape[-1])
    if len(flat) <= limit:
        indices = torch.arange(len(flat), device=flat.device)
    else:
        # Include the largest-magnitude row as well as distributed positions.
        extreme = flat.abs().amax(dim=1).argmax().reshape(1)
        regular = torch.linspace(0, len(flat) - 1, max(0, limit - 1), device=flat.device).long()
        indices = torch.unique(torch.cat((regular, extreme)), sorted=True)
    return flat.index_select(0, indices).float().cpu().contiguous(), indices.cpu().tolist()


class LinearCapture:
    def __init__(self, model, names, directory, report, *, max_cases=3, max_rows=64, max_bytes=1024**3):
        if min(max_cases, max_rows, max_bytes) < 1 or not names or len(set(names)) != len(names):
            raise ValueError("Capture requires unique names and positive budgets")
        modules = [model.get_submodule(name) for name in names]
        if len({id(m) for m in modules}) != len(modules):
            raise ValueError("Aliased modules cannot be captured twice")
        estimate = 0
        for module in modules:
            if type(module) is not torch.nn.Linear or any(p.requires_grad for p in module.parameters()):
                raise ValueError("Capture supports plain frozen Linear base weights only")
            estimate += sum(p.numel() * p.element_size() for p in module.parameters())
            estimate += max_rows * module.in_features * 4 * max_cases
        if estimate > max_bytes:
            raise ValueError("Requested capture exceeds its byte budget")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.report, self.max_cases, self.max_rows = report, max_cases, max_rows
        self.handles = []
        self.manifest = {"schema": "adaptive_linear_capture_v1", "status": "capturing",
                         "precision_calibrated": False, "max_cases": max_cases,
                         "row_limit": max_rows, "estimated_tensor_bytes": estimate,
                         "units": [{"name": n, "cases": []} for n in names]}
        self._write()
        try:
            for index, module in enumerate(modules):
                self.handles.append(module.register_forward_pre_hook(partial(self._capture, index)))
        except Exception:
            self.close()
            raise

    def _write(self):
        self.manifest["source"] = {k: self.report.get(k) for k in (
            "scope", "precision", "inputs_sha256", "uuid", "fp32_modules",
        )}
        write_result(self.directory / "manifest.json", self.manifest)

    def _capture(self, index, module, args):
        if self.report.get("stage") != "forward":
            return
        unit = self.manifest["units"][index]
        step = len(self.report.get("updates", [])) + 1
        if len(unit["cases"]) >= self.max_cases or any(c["step"] == step for c in unit["cases"]):
            return
        if (len(args) != 1 or not isinstance(args[0], torch.Tensor) or args[0].ndim < 1
                or args[0].shape[-1] != module.in_features or args[0].numel() == 0):
            raise ValueError("Unexpected Linear capture input contract")
        with torch.no_grad():
            value = args[0]
            if not torch.isfinite(value).all():
                raise ValueError("Refusing a nonfinite upstream calibration input")
            rows, indices = select_rows(value, self.max_rows)
            if not unit["cases"]:
                state = {"weight": module.weight.detach().cpu().contiguous()}
                if module.bias is not None:
                    state["bias"] = module.bias.detach().cpu().contiguous()
                if not all(torch.isfinite(t).all() for t in state.values()):
                    raise ValueError("Nonfinite captured Linear weights")
                unit["weights"] = f"unit-{index:03d}.weights.safetensors"
                save_file(state, str(self.directory / unit["weights"]))
            filename = f"unit-{index:03d}.case-{len(unit['cases']):03d}.safetensors"
            save_file({"input": rows}, str(self.directory / filename))
            unit["cases"].append({"file": filename, "step": step, "rows": indices,
                                  "original_shape": list(value.shape), "dtype": str(value.dtype),
                                  "max_abs": float(value.abs().max()), "sigma": self.report.get("sigma")})
        self._write()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.manifest["status"] = (
            "captured" if all(len(u["cases"]) == self.max_cases for u in self.manifest["units"])
            else "partial"
        )
        self._write()


def read_manifest(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if (not isinstance(manifest, dict) or manifest.get("schema") != "adaptive_linear_capture_v1"
            or manifest.get("status") != "captured"):
        raise ValueError("A completed Linear capture is required")
    units = manifest.get("units")
    count = manifest.get("max_cases")
    if (not isinstance(units, list) or not units or type(count) is not int or count < 1
            or not isinstance(manifest.get("source"), dict)):
        raise ValueError("Invalid Linear capture manifest")
    names = set()
    for unit in units:
        if (not isinstance(unit, dict) or not isinstance(unit.get("name"), str)
                or not unit["name"] or unit["name"] in names
                or not isinstance(unit.get("cases"), list) or len(unit["cases"]) != count):
            raise ValueError("Invalid or duplicate captured unit")
        names.add(unit["name"])
        capture_path(root, unit.get("weights"))
        steps = set()
        for case in unit["cases"]:
            if (not isinstance(case, dict) or type(case.get("step")) is not int
                    or case["step"] < 1 or case["step"] in steps):
                raise ValueError("Invalid or duplicate captured case")
            steps.add(case["step"])
            capture_path(root, case.get("file"))
    return manifest


def capture_path(directory, relative):
    root = Path(directory).resolve()
    if not isinstance(relative, str) or not relative or relative == ".":
        raise ValueError("Capture artifact must have a relative filename")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not (root / path).resolve().is_relative_to(root):
        raise ValueError("Capture artifact escapes its directory")
    return root / path
