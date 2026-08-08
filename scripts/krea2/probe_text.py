"""R-verify: Krea-2-Raw 文本链路 (阶段 1) 真火测试.

验证:
1. Qwen3-VL-4B 加载 (单 safetensors + bundled config, PG199 bf16).
2. tokenize 单 prompt -> input_ids/attn_mask shape, ChatML 模板 pad 到 541+suffix.
3. encode -> hiddens (B, L, 12, 2560), mask (B, L) bool, 真实长度反映在 mask.
4. 喂 Krea-2 DiT forward (context=hiddens, txtmask=mask), 输出 shape 对齐 + 有限.
5. R1 对照: padding 契约用 mask 屏蔽 (非 anima zero-sink), 验证 mask 正确性.

PG199 bf16. Qwen3-VL 8.4GB + DiT 26.3GB = ~35GB, 超过 PG199 32GB 单卡.
所以分两段: TE 在 GPU 跑完 -> hiddens/mask 移到 CPU -> TE 释放 -> DiT 上 GPU.
"""
from __future__ import annotations

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # PG199

import sys
import time
from pathlib import Path

import torch
from einops import rearrange, repeat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402
from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402
from library.models.krea2_raw.strategy import (  # noqa: E402
    KREA2_PAD_LENGTH,
    KREA2_PREFIX_IDX,
    Krea2TextEncodingStrategy,
    Krea2TokenizeStrategy,
    load_krea2_text_encoder,
)

KREA2_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"
KREA2_TE = ROOT / "models" / "text_encoders" / "qwen3vl_4b_bf16.safetensors"


def prepare(img: torch.Tensor, txtlen: int, patch: int, txtmask: torch.Tensor):
    """移植自 krea-ai/krea-2 sampling.prepare (同 probe_dit.py)."""
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask, imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"=== 阶段 1: Krea-2 文本链路 真火测试 (PG199, {dtype}) ===")

    # 1. 加载 Qwen3-VL TE
    print("\n--- 1. 加载 Qwen3-VL-4B ---")
    t0 = time.time()
    te_model, tokenizer = load_krea2_text_encoder(
        str(KREA2_TE), dtype=dtype, device="cuda"
    )
    print(f"TE 加载耗时: {time.time()-t0:.2f}s")
    te_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"TE 显存 peak: {te_peak:.2f}GB")

    # 2. tokenize 单 prompt
    print("\n--- 2. tokenize 单 prompt ---")
    tok = Krea2TokenizeStrategy()
    prompts = ["a red circle on blue background"]
    tokens = tok.tokenize(prompts)
    input_ids, attn_mask = tokens[0], tokens[1]
    print(f"input_ids: {tuple(input_ids.shape)} (期望 (1, 541+suffix_len))")
    print(f"attn_mask: {tuple(attn_mask.shape)}")
    # 真实 token 数 (含 suffix): sum True
    real_len = attn_mask[0].sum().item()
    print(f"真实 token 数 (含 suffix): {real_len}")
    # 验证 prefix 34 token 是 system prompt (应全 True)
    print(f"prefix (前 34) 全 True: {attn_mask[0, :34].all().item()}")

    # 3. encode -> hiddens + mask
    print("\n--- 3. encode -> hiddens (B, L, 12, 2560) + mask ---")
    enc = Krea2TextEncodingStrategy()
    t1 = time.time()
    with torch.inference_mode():
        [hiddens, txtmask] = enc.encode_tokens(tok, [te_model], tokens)
    print(f"encode 耗时: {time.time()-t1:.2f}s")
    print(f"hiddens: {tuple(hiddens.shape)} (期望 (1, L-34, 12, 2560))")
    print(f"txtmask: {tuple(txtmask.shape)} (期望 (1, L-34))")
    print(f"hiddens dtype: {hiddens.dtype}")
    print(f"hiddens 有限: {torch.isfinite(hiddens).all().item()}")
    real_len_post = txtmask[0].sum().item()
    print(f"切 prefix 后真实 token 数: {real_len_post} (期望 real_len-34={real_len-34})")

    shape_ok = (
        hiddens.shape[2] == 12
        and hiddens.shape[3] == 2560
        and txtmask.dtype == torch.bool
    )
    finite = torch.isfinite(hiddens).all().item()

    # R1: 验证 padding 位 (mask=False) 的 hiddens 是否非零 (Krea-2 不置零, 应有值)
    # 若 anima 式置零, padding 位 hiddens 应为 0
    pad_mask = ~txtmask[0]  # True = padding 位
    if pad_mask.any():
        pad_vals = hiddens[0][pad_mask]  # (N_pad, 12, 2560)
        print(f"\nR1: padding 位 hiddens (不置零, 应有值):")
        print(f"  padding token 数: {pad_mask.sum().item()}")
        print(f"  padding hiddens abs max: {pad_vals.abs().max().item():.6f}")
        print(f"  padding hiddens abs mean: {pad_vals.abs().mean().item():.6f}")
        krea2_no_zero = pad_vals.abs().max().item() > 0
        print(f"  Krea-2 不二次置零 padding (R1 契约): {krea2_no_zero}")

    # 4. 把 hiddens/mask 移到 CPU, 释放 TE, 加载 DiT
    print("\n--- 4. 释放 TE, 加载 DiT ---")
    hiddens = hiddens.to("cpu")
    txtmask = txtmask.to("cpu")
    del te_model, tokens, enc
    torch.cuda.empty_cache()
    print(f"TE 释放后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    dit = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=True)
    dit = dit.to(device)
    torch.cuda.reset_peak_memory_stats()

    # 5. 喂 DiT forward
    print("\n--- 5. DiT forward (真实 Qwen3-VL context) ---")
    patch = dit.config.patch
    channels = dit.config.channels
    h = w = 256
    latent_h, latent_w = h // 8, w // 8
    torch.manual_seed(0)
    noise = torch.randn(1, channels, latent_h, latent_w, device=device, dtype=dtype)
    context = hiddens.to(device)
    txtmask_d = txtmask.to(device)
    img_tokens, pos, mask = prepare(noise, context.shape[1], patch, txtmask_d)
    t = torch.tensor([0.5], device=device, dtype=dtype)

    t2 = time.time()
    with torch.inference_mode():
        out = dit(img=img_tokens, context=context, t=t, pos=pos, mask=mask)
    fwd_t = time.time() - t2
    dit_peak = torch.cuda.max_memory_allocated() / 1e9
    exp_out = (1, (latent_h // patch) * (latent_w // patch), patch * patch * channels)
    print(f"DiT forward 输出: {tuple(out.shape)} (期望 {exp_out})")
    print(f"forward 耗时: {fwd_t*1000:.0f}ms, DiT peak {dit_peak:.2f}GB")
    dit_shape_ok = tuple(out.shape) == exp_out
    dit_finite = torch.isfinite(out).all().item()

    # 结论
    print("\n=== 结论 ===")
    ok = shape_ok and finite and dit_shape_ok and dit_finite
    print(f"阶段 1 文本链路通过: {ok}")
    print(f"  hiddens (B,L,12,2560): {shape_ok}")
    print(f"  hiddens 有限:          {finite}")
    print(f"  DiT forward shape 对齐: {dit_shape_ok}")
    print(f"  DiT forward 有限:       {dit_finite}")
    print(f"\n基线 (PG199 bf16):")
    print(f"  TE (Qwen3-VL-4B) 显存: {te_peak:.2f}GB")
    print(f"  DiT (256×256) peak:    {dit_peak:.2f}GB")
    print(f"  context shape: (1, {context.shape[1]}, 12, 2560)")
    print(f"  DiT forward: {fwd_t*1000:.0f}ms")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
