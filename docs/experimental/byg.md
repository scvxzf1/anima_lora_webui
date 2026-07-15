# BYG Experimental Training

状态：实验
适用版本：当前 main；可运行边界以 `tasks.py --help` 和实时源码为准

BYG is currently wired as an experimental training path; dedicated source-concat inference is still a placeholder.

## Status

| Item | Current state |
|---|---|
| Data sidecars | `python tasks.py exp-byg-data` builds `post_image_dataset/byg/<stem>_byg.safetensors` |
| Training | `python tasks.py exp-byg` runs `configs/methods/byg.toml` |
| Output weight | Plain Anima LoRA, loadable with normal `--lora_weight` |
| Dedicated test target | `python tasks.py exp-test-byg` exits with a placeholder message |

## Workflow

```bash
python tasks.py exp-byg-data
python tasks.py exp-byg
```

`exp-byg-data` reads images from `image_dataset/` and writes BYG edit-tuple
sidecars to `post_image_dataset/byg/`. The regular VAE/text caches are still
the normal `post_image_dataset/lora` style caches.

## Current limits

- BYG owns several DiT forwards per step, so `configs/methods/byg.toml` keeps
  `blocks_to_swap = 0`.
- `unsloth_offload_checkpointing` must stay `false`; the config documents why
  the reentrant checkpoint path drops gradients for BYG's closed-over LoRA
  parameters.
- The dedicated BYG inference patch is not wired yet. Until then, use the
  trained checkpoint as a plain LoRA with `--lora_weight`.
