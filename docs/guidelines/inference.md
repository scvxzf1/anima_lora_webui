# Inference Reference

## Basic usage

```bash
python inference.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --lora_weight output/anima_lora.safetensors \
    --prompt "your prompt" \
    --image_size 1024 1024 \
    --infer_steps 50 \
    --guidance_scale 5.0 \
    --save_path images
```

Or use the canned `make test` target, which inherits the `TEST_COMMON` prompt and flags and runs against the latest bakeable LoRA in `output/`.

## Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--lora_weight` | — | LoRA weight path(s); multiple allowed, space-separated |
| `--lora_multiplier` | 1.0 | LoRA strength (one per weight or a single scalar for all) |
| `--infer_steps` | 50 | Denoising steps |
| `--guidance_scale` | 5.0 | CFG scale |
| `--flow_shift` | — | Flow-matching schedule shift (see `inference.py --help`) |
| `--sampler` | euler | `euler` (deterministic ODE) or `er_sde` (stochastic) |
| `--attn_mode` | torch | Attention backend: `torch`, `flash`, `flex`, `sageattn`, `xformers` |
| `--from_file` | — | Batch prompts from a text file |
| `--interactive` | off | Interactive prompt mode |
| `--fp8` | off | FP8 quantization for DiT |
| `--compile` | off | `torch.compile` speedup |
| `--spectrum` | off | Spectrum acceleration — see [`../methods/spectrum.md`](../methods/spectrum.md) |
| `--pooled_text_proj` | — | Path to distilled modulation-guidance MLP — see [`../methods/mod-guidance.md`](../methods/mod-guidance.md) |
| `--prefix_weight` | — | Prefix-tuning vectors (also used for reference inversion) |
| `--postfix_weight` | — | Postfix-tuning vectors |

## P-GRAFT inference

Loads LoRA as dynamic hooks instead of a static merge, allowing mid-denoising cutoff:

```bash
python inference.py ... \
    --pgraft \
    --lora_cutoff_step 37    # LoRA active for steps 0–36, disabled 37+
```

## Prompt file format

```
a girl standing in a field --w 1024 --h 1024 --s 50 --g 5.0
another prompt --seed 42 --flow_shift 4.0
```

## LoRA in ComfyUI

Plain anima LoRA `.safetensors` files use kohya-ss `lora_unet_` key naming and load directly into ComfyUI's stock `LoraLoader` — no conversion step. For HydraLoRA / FeRA / ReFT and prefix/postfix checkpoints (extra `router.*`, `reft_*`, stacked `lora_ups.N.*` keys that stock loader drops), use the `custom_nodes/comfyui-hydralora/` Anima Adapter Loader node.
