# Krea-2 self-contained NF4 v2

Status: Implemented and verified.

Date: 2026-08-09

## Result

Krea-2 NF4 now has a versioned, self-contained safetensors format. It preserves
the existing 264 `Linear4bit` payloads and quantization states, and adds the 159
non-Linear model tensors that legacy NF4 overlays obtained from the BF16 base.
The added model state is 5.44 MiB, so the resulting file remains 6.62 GB.

Generated artifact:

```text
models/diffusion_models/krea2_raw_nf4_self_contained.safetensors
```

SHA-256:

```text
ee2aedcce6a0f145584c90514a42ef202d9a814e22a30fcbd5f11cd8cd900338
```

Build or rebuild without requantizing:

```bash
.venv/bin/python scripts/krea2/build_self_contained_nf4.py
```

The builder writes a temporary file in the destination directory and atomically
renames it. It never modifies the BF16 or legacy NF4 source files and refuses to
overwrite an existing output unless `--overwrite` is supplied.

## Compatibility

`load_krea2_dit` supports all three existing modes:

1. BF16 strict load, with optional online NF4 quantization.
2. Legacy NF4 v1 overlay plus the BF16 base.
3. Self-contained NF4 v2 supplied either as `nf4_path` or directly as
   `pretrained_model_name_or_path` / `dit_path`.

The loader identifies v2 from safetensors metadata, not the filename. It rejects
unknown versions, partial metadata, incomplete model state, mismatched Linear
paths/shapes, and missing weight, bias, or quantization-state tensors.

Recommended training configuration:

```toml
model_family = "krea2_raw"
pretrained_model_name_or_path = "models/diffusion_models/krea2_raw_nf4_self_contained.safetensors"
base_compute = "nf4"
```

`nf4_prequantized_path` remains available for legacy v1 overlays but is not
needed when the base path is a v2 file.

## Verification

| Check | Result |
| --- | --- |
| Legacy NF4 payload | 1,594 tensors, all bit-identical in v2 |
| Added model state | 159 tensors, all bit-identical to BF16 source |
| Total v2 keys | 1,753 |
| Direct full-model CPU load | 264 Linear4bit, 159 model tensors, zero meta parameters |
| Direct-load peak RSS | 1.67 GB |
| Direct-load elapsed time (original CPU shell) | 129.67 s |
| Direct-load elapsed time (meta shell, 2026-08-10) | 0.386 s |
| PG199 real LoRA smoke, 256 px, 2 steps | loss 0.0239 -> 0.0228, gradients nonzero, finite |
| PG199 smoke allocator peak | 7.71 GB |
| RTX 3080, swap20, synthetic 256 px, 2 steps | loss 5.0938 -> 4.9375, gradients nonzero, finite |
| RTX 3080 allocator peak | 4.873 GB |
| PG199 Flash + compile + full checkpoint + swap26 | first forward/backward/optimizer step passed |
| Compile smoke | loss 5.78125, grad norm 0.265625, finite |

The two-step ablation probe's aggregate verdict reports false because its
first-five versus last-five monotonicity rule requires a longer run. The actual
two training steps completed and satisfied the intended smoke criteria.

## Scope

This is a packaging and loading change, not a new quantizer. Training math,
4-bit codes, quantization state, LoRA targets, block swap, and checkpoint
semantics are unchanged. The original loader spent about two minutes creating
and immediately discarding full-size FP32 weights for 264 `Linear4bit` shells.
Constructing those shells on the meta device reduced the measured CPU
direct-load time to 0.386 seconds without leaving meta parameters or buffers.
The loader also avoids reading the 25 GB BF16 payload or running online
quantization.
