"""Convert a local, ordinary BF16 transformer file to canonical mixed NF4."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def convert(*, family, source, output, threads=None, policy="main-mlp"):
    import torch
    from safetensors import safe_open
    from library.quantization.format import inspect_mixed_nf4
    from library.quantization.mixed_nf4 import save_mixed_nf4
    source, output = Path(source).expanduser(), Path(output).expanduser()
    if policy not in ("main-mlp", "all-linear"):
        raise ValueError("policy must be main-mlp or all-linear")
    if policy == "all-linear" and family != "qwen_image_2_1":
        raise ValueError("all-linear policy is supported only for qwen_image_2_1")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if not source.is_file() or source.suffix != ".safetensors":
        raise ValueError("Source must be a local single BF16 .safetensors file; directories are unsupported")
    if threads is not None:
        if threads <= 0:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
    if inspect_mixed_nf4(source) is not None:
        raise ValueError("Source is already mixed NF4")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        if (any("nf4" in key.lower() or "quant" in key.lower() for key in (handle.metadata() or {}))
                or any(handle.get_slice(key).get_dtype() != "BF16" for key in handle.keys())):
            raise ValueError("Source must contain ordinary BF16 tensors, without quantization")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if family == "krea2_raw":
        from library.models.krea2_raw.weights import load_krea2_dit
        model = load_krea2_dit(source, dtype=torch.bfloat16, device="cpu")
    elif family == "z_image":
        from library.models.z_image.weights import load_z_image_transformer
        model = load_z_image_transformer(str(source), dtype=torch.bfloat16, device="cpu")
    elif family == "qwen_image_2_1":
        from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_transformer
        model = load_qwen_image_2_1_transformer(str(source), dtype=torch.bfloat16, device="cpu")
    else:
        raise ValueError(f"Unsupported family: {family}")
    kwargs = {"family": family, "source_sha256": digest.hexdigest()}
    if policy == "all-linear":
        kwargs["policy"] = "all-linear-nf4-v1"
    return save_mixed_nf4(model, output, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("krea2_raw", "z_image", "qwen_image_2_1"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--policy", choices=("main-mlp", "all-linear"), default="main-mlp")
    args = parser.parse_args(argv)
    print(convert(**vars(args)))


if __name__ == "__main__":
    main()
