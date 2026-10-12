"""Bounded sampled base-Linear cotangent capture and fixed-input VJP analysis."""

from __future__ import annotations

from fnmatch import fnmatchcase
import json
import math
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime import anima_mlp_path
from bench.adaptive_runtime.compare_training import compare, read_capture
from library.training.adaptive_runtime.precision import vector_metrics
from library.training.auto_block_swap.process import write_result


BLOCKS = (0, 5, 14, 27)
COMPONENTS = ("mlp.layer1", "self_attn.qkv_proj")
NAMES = tuple(f"blocks.{block}.{part}" for block in BLOCKS for part in COMPONENTS)
MAX_BYTES = 512 * 1024**2
TOKENS = 64


def sampled_indices(token_count):
    if token_count < TOKENS:
        raise ValueError("Too few tokens for fixed 64-token local sample")
    return torch.linspace(0, token_count - 1, TOKENS, dtype=torch.float64).long()


def _entry_path(directory, filename):
    if (not isinstance(filename, str) or not filename or filename != Path(filename).name
            or filename in {".", ".."} or not filename.endswith(".safetensors")):
        raise ValueError("Unsafe local VJP filename")
    root = Path(directory).resolve()
    path = root / filename
    if path.resolve().parent != root or not path.is_file():
        raise ValueError("Local VJP path escapes capture directory")
    return path


def _tensor_bytes(in_features, out_features, dtype, bias, *, input_cotangent=False):
    element = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}[dtype]
    return (out_features * in_features + (out_features if bias else 0)) * element + \
           TOKENS * (in_features * (2 if input_cotangent else 1) + 2 * out_features) * 4 + TOKENS * 8


def _weight_storage(weight):
    value = weight.detach().cpu().contiguous()
    bf16 = value.to(torch.bfloat16)
    return bf16 if torch.equal(bf16.to(value.dtype), value) else value


class LocalVJPCapture:
    def __init__(self, directory, model, network, *, signature, uuid, scale, replay):
        if (signature.get("steps") != 1 or signature.get("sigmas") != [0.5]
                or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 1):
            raise ValueError("Local VJP requires one fixed sigma, finite loss scale")
        self.directory = Path(directory)
        self.scale = scale
        self.samples = {}
        self.layouts = {}
        self.wrappers = []
        self.entries = []
        self.used = 0
        self.profile = signature.get("local_vjp_profile", "linear")
        if self.profile not in {"linear", anima_mlp_path.PROFILE}:
            raise ValueError("Unsupported local VJP profile")
        self.names = NAMES if self.profile == "linear" else anima_mlp_path.NAMES
        self.activation = {}
        modules = dict(model.named_modules())
        adapters = set(network.unet_loras)
        for name in self.names:
            module = modules.get(name)
            adapter = getattr(getattr(module, "forward", None), "__self__", None)
            if (type(module) is not torch.nn.Linear or adapter not in adapters
                    or getattr(getattr(adapter, "org_forward", None), "__self__", None) is not module):
                raise ValueError(f"Missing LoRA-patched base Linear: {name}")
            if (module.weight.dtype not in {torch.bfloat16, torch.float16, torch.float32}
                    or (module.bias is not None and module.bias.dtype not in
                        {torch.bfloat16, torch.float16, torch.float32})):
                raise ValueError("Unsupported local VJP base dtype")
            if self.profile == anima_mlp_path.PROFILE:
                up = getattr(getattr(adapter, "lora_up", None), "weight", None)
                if up is None or not bool(torch.count_nonzero(up.detach()) == 0):
                    raise ValueError("MLP path requires zero-initialized LoRA up weights")
                self.activation[name] = anima_mlp_path.activation_metadata(modules, name)
            size = _tensor_bytes(module.in_features, module.out_features, module.weight.dtype,
                                 module.bias is not None, input_cotangent=self.profile == anima_mlp_path.PROFILE)
            if self.used + size > MAX_BYTES:
                raise ValueError("Local VJP 512MiB preflight budget exceeded")
            self.used += size
            self.entries.append((name, module, adapter))
        self.entries = [(name, module, adapter, _weight_storage(module.weight),
                         None if module.bias is None else _weight_storage(module.bias),
                         str(module.weight.dtype)) for name, module, adapter in self.entries]
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {"schema": "anima_local_vjp_v1", "status": "capturing",
                         "signature": signature, "uuid": uuid, "sigma": signature["sigmas"][0],
                         "loss_scale": scale, "budget_bytes": MAX_BYTES,
                         "replay": replay,
                         "planned_tensor_bytes": self.used, "layers": []}
        write_result(self.directory / "manifest.json", self.manifest)
        for name, module, adapter, _, _, _ in self.entries:
            original = adapter.org_forward

            def wrapped(x, _original=original, _name=name, _module=module):
                if (x.ndim not in (3, 5) or x.shape[0] != 1 or x.shape[-1] != _module.in_features
                        or math.prod(x.shape[1:-1]) < TOKENS):
                    raise ValueError(f"Unexpected local Linear input shape: {_name}, {tuple(x.shape)}")
                y = _original(x)
                if y.shape != (*x.shape[:-1], _module.out_features):
                    raise ValueError(f"Unexpected local Linear shape: {_name}")
                indices = sampled_indices(math.prod(x.shape[1:-1])).to(x.device)
                xs = x.detach().flatten(1, -2).index_select(1, indices).float().cpu().contiguous()
                ys = y.detach().flatten(1, -2).index_select(1, indices).float().cpu().contiguous()
                previous = self.samples.get(_name)
                if previous is None:
                    self.samples[_name] = {"x": xs, "y": ys, "indices": indices.cpu(), "g": None}
                    if self.profile == anima_mlp_path.PROFILE:
                        self.samples[_name]["gx"] = None
                    self.layouts[_name] = list(x.shape)
                elif (self.layouts[_name] != list(x.shape) or not torch.equal(previous["x"], xs)
                      or not torch.equal(previous["y"], ys)):
                    raise ValueError(f"Checkpoint recomputation changed local values: {_name}")
                if y.requires_grad:
                    def capture_grad(gradient):
                        item = self.samples[_name]
                        gs = gradient.detach().flatten(1, -2).index_select(1, indices)
                        gs = gs.float().div(self.scale).cpu().contiguous()
                        if item["g"] is None:
                            item["g"] = gs
                        elif not torch.equal(item["g"], gs):
                            raise ValueError(f"Repeated local cotangent changed: {_name}")
                        return gradient
                    y.register_hook(capture_grad)
                if self.profile == anima_mlp_path.PROFILE and x.requires_grad:
                    def capture_input_grad(gradient):
                        item = self.samples[_name]
                        value = gradient.detach().flatten(1, -2).index_select(1, indices)
                        value = value.float().div(self.scale).cpu().contiguous()
                        if item["gx"] is None:
                            item["gx"] = value
                        elif not torch.equal(item["gx"], value):
                            raise ValueError(f"Repeated local input cotangent changed: {_name}")
                        return gradient
                    x.register_hook(capture_input_grad)
                return y

            adapter.org_forward = wrapped
            self.wrappers.append((adapter, original))

    def restore(self):
        for adapter, original in self.wrappers:
            adapter.org_forward = original
        self.wrappers.clear()

    def close(self):
        self.restore()
        if set(self.samples) != set(self.names) or any(
                v["g"] is None or (self.profile == anima_mlp_path.PROFILE and v["gx"] is None)
                for v in self.samples.values()):
            raise ValueError("Incomplete local base Linear forward/cotangent capture")
        for name, module, _, weight, bias, original_dtype in self.entries:
            item = self.samples[name]
            tensors = {**item, "weight": weight}
            if bias is not None:
                tensors["bias"] = bias
            if any(not bool(torch.isfinite(t).all()) for t in tensors.values()):
                raise ValueError(f"Nonfinite local VJP tensor: {name}")
            if (item["x"].shape != (1, TOKENS, module.in_features)
                    or item["y"].shape != (1, TOKENS, module.out_features)
                    or item["g"].shape != (1, TOKENS, module.out_features)
                    or (self.profile == anima_mlp_path.PROFILE and
                        item["gx"].shape != (1, TOKENS, module.in_features))):
                raise ValueError(f"Local VJP shape changed: {name}")
            file = name.replace(".", "_") + ".safetensors"
            path = self.directory / file
            save_file(tensors, str(path))
            if path.stat().st_size > MAX_BYTES:
                raise ValueError("Local VJP file exceeds budget")
            self.manifest["layers"].append({"name": name, "file": file,
                  "sha256": sha256_file(path), "weight_shape": list(weight.shape),
                  "weight_dtype": str(weight.dtype), "original_weight_dtype": original_dtype,
                  "bias": bias is not None, "sample_shape": list(item["x"].shape),
                  "original_input_shape": self.layouts[name],
                  "token_layout": "row_major_flatten_nonbatch_nonfeature_axes",
                  "indices": item["indices"].tolist(),
                  **({"input_cotangent": True, "zero_up": True,
                      "approximate": self.activation[name]} if self.profile == anima_mlp_path.PROFILE else {})})
        if sum((self.directory / e["file"]).stat().st_size for e in self.manifest["layers"]) > MAX_BYTES:
            raise ValueError("Local VJP disk budget exceeded")
        self.manifest["status"] = "captured"
        write_result(self.directory / "manifest.json", self.manifest)


def read_manifest(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    profile = manifest.get("signature", {}).get("local_vjp_profile", "linear")
    if profile not in {"linear", anima_mlp_path.PROFILE}:
        raise ValueError("Unknown local VJP profile")
    names = NAMES if profile == "linear" else anima_mlp_path.NAMES
    if (manifest.get("schema") != "anima_local_vjp_v1" or manifest.get("status") != "captured"
            or [entry.get("name") for entry in manifest.get("layers", [])] != list(names)
            or type(manifest.get("planned_tensor_bytes")) is not int
            or not 0 < manifest["planned_tensor_bytes"] <= MAX_BYTES
            or manifest.get("budget_bytes") != MAX_BYTES
            or manifest.get("sigma") != 0.5
            or not isinstance(manifest.get("uuid"), str)):
        raise ValueError("Incomplete or over-budget local VJP manifest")
    total = 0
    for entry in manifest["layers"]:
        path = _entry_path(directory, entry.get("file"))
        if (entry["file"] != entry["name"].replace(".", "_") + ".safetensors"
                or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
                or type(entry.get("bias")) is not bool
                or entry.get("weight_dtype") not in {"torch.bfloat16", "torch.float16", "torch.float32"}
                or entry.get("original_weight_dtype") not in
                    {"torch.bfloat16", "torch.float16", "torch.float32"}
                or not isinstance(entry.get("weight_shape"), list)
                or len(entry["weight_shape"]) != 2
                or any(type(v) is not int or v <= 0 for v in entry["weight_shape"])
                or entry.get("sample_shape") != [1, TOKENS, entry["weight_shape"][1]]
                or not isinstance(entry.get("original_input_shape"), list)
                or len(entry["original_input_shape"]) not in (3, 5)
                or any(type(v) is not int or v <= 0 for v in entry["original_input_shape"])
                or entry["original_input_shape"][0] != 1
                or entry["original_input_shape"][-1] != entry["weight_shape"][1]
                or entry.get("token_layout") != "row_major_flatten_nonbatch_nonfeature_axes"
                or entry.get("indices") is None or len(entry["indices"]) != TOKENS
                or any(type(v) is not int or v < 0 for v in entry["indices"])
                or entry["indices"] != sorted(set(entry["indices"]))):
            raise ValueError("Invalid local VJP entry metadata")
        if profile == anima_mlp_path.PROFILE:
            if (entry.get("input_cotangent") is not True or entry.get("zero_up") is not True
                    or entry.get("approximate") not in {"none", "tanh"}):
                raise ValueError("Invalid MLP path metadata")
        elif any(key in entry for key in ("input_cotangent", "zero_up", "approximate")):
            raise ValueError("Unexpected MLP path metadata")
        if entry["indices"] != sampled_indices(math.prod(entry["original_input_shape"][1:-1])).tolist():
            raise ValueError("Local VJP indices do not match original token layout")
        expected = _tensor_bytes(entry["weight_shape"][1], entry["weight_shape"][0],
                                 {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16,
                                  "torch.float32": torch.float32}[entry["weight_dtype"]], entry["bias"],
                                 input_cotangent=profile == anima_mlp_path.PROFILE)
        size = path.stat().st_size
        if size < 8 or size > expected + 2048:
            raise ValueError("Local VJP file/header exceeds expected budget")
        with path.open("rb") as stream:
            header_bytes = int.from_bytes(stream.read(8), "little")
        if header_bytes > 2048 or header_bytes + 8 > size:
            raise ValueError("Local VJP oversized or truncated header")
        total += size
        if total > MAX_BYTES:
            raise ValueError("Local VJP disk budget exceeded")
    return manifest


def load_layer(directory, entry):
    path = _entry_path(directory, entry["file"])
    if sha256_file(path) != entry["sha256"]:
        raise ValueError("Local VJP path or hash mismatch")
    with safe_open(str(path), framework="pt", device="cpu") as source:
        data = {key: source.get_tensor(key) for key in source.keys()}
    expected_keys = {"x", "y", "g", "weight", "indices"} | ({"bias"} if entry["bias"] else set())
    if entry.get("input_cotangent") is True:
        expected_keys.add("gx")
    if (set(data) != expected_keys
            or data["weight"].shape != torch.Size(entry["weight_shape"])
            or str(data["weight"].dtype) != entry["weight_dtype"]
            or data["x"].shape != torch.Size(entry["sample_shape"])
            or data["y"].shape != (1, TOKENS, entry["weight_shape"][0])
            or data["g"].shape != data["y"].shape
            or data["indices"].shape != (TOKENS,)
            or data["indices"].tolist() != entry["indices"]
            or data["indices"].dtype != torch.int64
            or any(data[key].dtype != torch.float32 for key in ("x", "y", "g"))
            or ("gx" in expected_keys and (data["gx"].shape != data["x"].shape
                                           or data["gx"].dtype != torch.float32))
            or (entry["bias"] and (data["bias"].shape != (entry["weight_shape"][0],)
                                    or data["bias"].dtype != data["weight"].dtype))):
        raise ValueError("Local VJP tensor metadata mismatch")
    if any(not bool(torch.isfinite(t).all()) for t in data.values()):
        raise ValueError("Nonfinite local VJP input")
    return data


def metrics(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError("Mismatched local VJP result shape")
    return vector_metrics([(reference, actual)])


def local_analysis(item, *, device):
    x = item["x"].to(device).float()
    g = item["g"].to(device).float()
    w = item["weight"].to(device).float()
    bias = item.get("bias")
    b = None if bias is None else bias.to(device).float()
    if not all(bool(torch.isfinite(t).all()) for t in (x, g, w, *([b] if b is not None else []))):
        raise ValueError("Nonfinite local VJP input")
    with torch.autocast(device.type, enabled=False):
        y32 = F.linear(x, w, b)
        dx32 = g @ w
        w16 = w.half()
        b16 = None if b is None else b.half()
        y16 = F.linear(x.half(), w16, b16).float()
        weight_dx = g @ w16.float()
        results = {"reference_y_vs_recomputed_fp32": metrics(item["y"].to(device), y32),
                   "forward_actual_fp16": metrics(y32, y16),
                   "weight_only_dx": metrics(dx32, weight_dx),
                   "weight_rounding": metrics(w, w16.float())}
        for scale in (1, 64, 1024):
            scaled = g * scale
            cast = scaled.half()
            case = {"g_scaled_zero": int((scaled != 0).logical_and(cast == 0).sum()),
                    "g_scaled_subnormal": int((cast != 0).logical_and(cast.abs() < 2**-14).sum()),
                    "g_scaled_overflow": int(torch.isinf(cast).sum())}
            if case["g_scaled_overflow"]:
                case.update(finite=False, reason="scaled_cotangent_fp16_overflow")
            else:
                gcast_dx = (cast.float() / scale) @ w16.float()
                actual_x = x.detach().float().requires_grad_(True)
                actual_y = F.linear(actual_x.half(), w16, b16).float()
                actual_y.backward(scaled)
                actual_dx = actual_x.grad / scale
                case.update(g_cast_only_dx=metrics(dx32, gcast_dx),
                            actual_fp16_dx=metrics(dx32, actual_dx))
                case["finite"] = all(value["finite"] for value in
                                     (case["g_cast_only_dx"], case["actual_fp16_dx"]))
                if not case["finite"]:
                    case["reason"] = "fp16_local_vjp_nonfinite"
            results[str(scale)] = case
    return results


def validate_training_pair(reference, candidate, comparison):
    ref_root, trial_root = Path(reference).parent, Path(candidate).parent
    verified = compare(ref_root / "training-capture", trial_root / "training-capture")
    if verified != comparison or not verified["scope"].startswith("fixed_pre_step"):
        raise ValueError("Local VJP comparison is not bound to actual training captures")
    pairs = []
    for root, expected_precision, scale in ((ref_root, "fp32-reference", 1),
                                           (trial_root, "fp16-islands", 1024)):
        capture, worker = read_capture(root / "training-capture")
        local = read_manifest(root / "local-vjp")
        if (capture["steps"] != 1 or capture["signature"].get("sigmas") != [0.5]
                or capture["signature"].get("precision") != expected_precision
                or capture["signature"].get("loss_scale") != scale
                or worker.get("loss_scale") != scale
                or local.get("loss_scale") != scale or local["signature"] != capture["signature"]
                or local.get("replay") != capture["cases"][0].get("replay")
                or local["uuid"] != worker["uuid"]):
            raise ValueError("Local VJP capture/replay/configuration identity mismatch")
        pairs.append((local, worker))
    if (pairs[0][0]["signature"].get("local_vjp_profile", "linear") !=
            pairs[1][0]["signature"].get("local_vjp_profile", "linear")):
        raise ValueError("Local VJP profile mismatch")
    assignments = (pairs[1][1].get("precision_manifest") or {}).get("assignments", {})
    expected = {name: "fp16" if fnmatchcase(name, "blocks.*.mlp.*") else "fp32"
                for name in assignments}
    if (not assignments or assignments != expected or "fp16" not in expected.values()
            or sorted(pairs[1][0]["signature"].get("fp32_modules", [])) !=
                sorted(name for name, dtype in assignments.items() if dtype == "fp32")):
        raise ValueError("Local VJP candidate must use MLP-only FP16")
    for entry in pairs[1][0]["layers"]:
        if entry["original_weight_dtype"] != "torch." + (
                "float16" if assignments.get(entry["name"]) == "fp16" else "float32"):
            raise ValueError("Local VJP layer dtype disagrees with candidate manifest")
    if any(entry["original_weight_dtype"] != "torch.float32" for entry in pairs[0][0]["layers"]):
        raise ValueError("Local VJP reference must use FP32 base Linear")
    return pairs[0][0], pairs[1][0]


def analyze(reference, candidate, comparison, *, device):
    ref, trial = validate_training_pair(reference, candidate, comparison)
    excluded = {"precision", "fp32_modules", "loss_scale"}
    a = {k: v for k, v in ref["signature"].items() if k not in excluded}
    b = {k: v for k, v in trial["signature"].items() if k not in excluded}
    if (a != b or ref["uuid"] != trial["uuid"] or ref["sigma"] != trial["sigma"]
            or ref["signature"].get("precision") != "fp32-reference"
            or trial["signature"].get("precision") != "fp16-islands"
            or ref["signature"].get("local_vjp_source_sha256") != sha256_file(Path(__file__))
            or (ref["signature"].get("local_vjp_profile") == anima_mlp_path.PROFILE and
                ref["signature"].get("mlp_path_source_sha256") !=
                    sha256_file(Path(anima_mlp_path.__file__)))
            or comparison.get("reference", {}).get("uuid") != ref["uuid"]
            or comparison.get("candidate", {}).get("uuid") != trial["uuid"]):
        raise ValueError("Local VJP reference/candidate identity mismatch")
    layers = []
    path_reference, path_candidate = [], []
    for ra, ca in zip(ref["layers"], trial["layers"], strict=True):
        if (ra["name"] != ca["name"] or ra["indices"] != ca["indices"]
                or ra["weight_shape"] != ca["weight_shape"]
                or ra["original_input_shape"] != ca["original_input_shape"]):
            raise ValueError("Local VJP layer inventory mismatch")
        r, c = load_layer(reference, ra), load_layer(candidate, ca)
        target_dtype = {"torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16,
                        "torch.float32": torch.float32}[ca["original_weight_dtype"]]
        expected = r["weight"].to(target_dtype).to(c["weight"].dtype)
        if not torch.equal(expected, c["weight"]):
            raise ValueError("Base weights changed beyond candidate dtype rounding")
        if (r.get("bias") is None) != (c.get("bias") is None):
            raise ValueError("Base bias inventory changed")
        if "bias" in r and not torch.equal(r["bias"].to(target_dtype).to(c["bias"].dtype), c["bias"]):
            raise ValueError("Base bias changed beyond candidate dtype rounding")
        local = local_analysis(r, device=device)
        drift = {key: metrics(r[key], c[key]) for key in
                 (("x", "y", "g", "gx") if "gx" in r else ("x", "y", "g"))}
        if "gx" in r:
            path_reference.append(anima_mlp_path.sample_row(ra, r))
            path_candidate.append(anima_mlp_path.sample_row(ca, c))
        layers.append({"name": ra["name"], "original_input_shape": ra["original_input_shape"],
                       "local_fixed_cotangent": local,
                       "whole_network_sample_drift": drift,
                       "candidate_weight_rounding": metrics(r["weight"].float(), c["weight"].float())})
    result = {"status": "analyzed", "production_ready": False,
            "scope": "sampled_64_token_local_replay_kernel_shape_not_whole_network_bound_or_quality_certificate",
            "uuid": ref["uuid"], "sigma": ref["sigma"], "layers": layers}
    if path_reference:
        result["mlp_path"] = anima_mlp_path.analyze_path(path_reference, path_candidate, device=device)
        result["scope"] += "_gelu_single_direction_not_causal_attribution"
    return result
