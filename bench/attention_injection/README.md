# Attention Injection Phase 1 Probe

This bench checks the hypothesis that LoRA injection can amplify attention
signals during Anima forward passes. It does not train or patch the main model.

It runs fixed-sigma forwards on cached training samples and writes:

- `attention_events.jsonl`: Q/K/V, sampled logits, softmax entropy/max-prob,
  attention output, output projection, and RMS ratios per attention module.
- `adapter_events.jsonl`: LoRA delta/base/output RMS ratios for injected Linear
  modules.
- `outputs.jsonl`: final DiT output stats per arm.
- `result.json`: standard bench envelope with grouped summaries.

Example:

```bash
python -m bench.attention_injection.probe \
  --dit models/diffusion_models/anima-base-v1.0.safetensors \
  --adapter output/ckpt/example.safetensors \
  --data-dir post_image_dataset/lora \
  --bucket 128x192 \
  --num-samples 2 \
  --sigmas 0.1 0.4 0.7 \
  --attn_mode torch \
  --label phase1
```

If `--adapter` is omitted, only the `base` arm runs. With an adapter, the probe
runs `base` (`network.set_multiplier(0)`) and `adapted`
(`network.set_multiplier(1)`) on the same noisy latents.

Self-attention logits are sampled evenly by default (`--max-logit-tokens 512`)
to avoid materializing full `B x H x Q x K` matrices for large buckets.
