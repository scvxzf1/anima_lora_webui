"""Bounded real-shape component timing, separate from full-step ranking."""

from pathlib import Path
import argparse
import torch
from .manifest import GPU_UUID, validate_device, write_json, software_manifest
from library.runtime.convrot.gemm import quantize_activation_absmax_int8, int8_mm_scaled
from library.runtime.convrot.triton_kernels import quantize_rowwise, int8_matmul_scaled
from library.runtime.convrot.rht import group_rht, normalized_hadamard


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(20):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 20


def extra_peak(fn):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    del output
    return peak


def run_shape(m, k, n):
    torch.manual_seed(99)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randint(-127, 128, (k, n), device="cuda", dtype=torch.int8)
    sw = torch.rand(n, device="cuda", dtype=torch.float32) * 0.002
    q, sx = quantize_activation_absmax_int8(x)
    qt, st = quantize_rowwise(x)
    yr = int8_mm_scaled(
        q, sx, w, sw, weight_layout="kn", prefer="int_mm", out_dtype=torch.bfloat16
    )
    yt = int8_matmul_scaled(q, sx, w, sw)
    exact = {
        "codes": torch.equal(q, qt),
        "scale": torch.equal(sx, st),
        "output": torch.equal(yr, yt),
    }
    if not all(exact.values()):
        raise RuntimeError(f"local parity failed {(m, k, n)} {exact}")
    h = normalized_hadamard(256, device=x.device, dtype=x.dtype)
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)

    def dx():
        folded = dy.float() * sw
        return group_rht((folded @ w.float().T).to(torch.bfloat16), 256, hadamard=h)

    timing = {
        "gemm_torch_extra_peak_bytes": extra_peak(
            lambda: int8_mm_scaled(
                q,
                sx,
                w,
                sw,
                weight_layout="kn",
                prefer="int_mm",
                out_dtype=torch.bfloat16,
            )
        ),
        "gemm_triton_extra_peak_bytes": extra_peak(
            lambda: int8_matmul_scaled(q, sx, w, sw)
        ),
        "quant_torch_ms": measure(lambda: quantize_activation_absmax_int8(x)),
        "quant_triton_ms": measure(lambda: quantize_rowwise(x)),
        "gemm_scale_torch_ms": measure(
            lambda: int8_mm_scaled(
                q,
                sx,
                w,
                sw,
                weight_layout="kn",
                prefer="int_mm",
                out_dtype=torch.bfloat16,
            )
        ),
        "gemm_scale_triton_ms": measure(lambda: int8_matmul_scaled(q, sx, w, sw)),
        "rht_ms": measure(lambda: group_rht(x, 256, hadamard=h)),
        "fp32_dx_including_cast_scale_rht_ms": measure(dx),
    }
    return {
        "m": m,
        "k": k,
        "n": n,
        "scope": "real_1024_shape" if m in [4096, 512] else "synthetic_bucket_shape",
        "exact": exact,
        **timing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    device = validate_device(GPU_UUID)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    shapes = [
        (4096, 2048, 6144),
        (4096, 2048, 2048),
        (512, 1024, 4096),
        (4096, 2048, 8192),
        (4096, 8192, 2048),
        (4032, 2048, 2048),
        (4200, 2048, 2048),
    ]
    results = []
    for shape in shapes:
        results.append(run_shape(*shape))
        write_json(
            args.out,
            {
                "status": "RUNNING",
                "device": device,
                "timing": "CUDA Events; pre-generated KN; 5 warmup/20 iterations; no setup cost",
                "results": results,
            },
        )
        print(results[-1], flush=True)
    write_json(
        args.out,
        {
            "status": "PASS",
            "manifest": software_manifest(),
            "device": device,
            "timing": "CUDA Events; pre-generated KN; 5 warmup/20 iterations; no setup cost",
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
