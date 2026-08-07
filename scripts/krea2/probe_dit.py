"""R-verify: Krea-2 DiT 单 latent forward shape 对齐验证 (阶段 2 出口).

验证:
1. SingleStreamDiT 用 SingleMMDiTConfig.krea2_raw() 构造,strict 加载
   krea2_raw_bf16.safetensors 全部 430 keys,无 missing/unexpected.
2. 单 latent forward 通过:用合成 context (模拟 Qwen3-VL 12 层 MFA 输出)
   + 3D pos + mask,跑一次 forward,输出 shape = (B, L_img, patch*patch*channels).
3. 数值有限性:输出无 NaN/Inf.
4. dtype 路径:bf16 forward 通过 (生产 dtype).

通过门槛:strict 加载 0 missing/0 unexpected + forward shape 对齐 + 数值有限.
不验证语义正确性 (那需要完整 TE+采样, 阶段 5 再做).
"""
from __future__ import annotations

import os

# 必须在 import torch 之前设置 CUDA 设备选择:
#   - CUDA_DEVICE_ORDER=PCI_BUS_ID: 让 index 与 nvidia-smi 一致 (默认
#     FASTEST_FIRST 会把 RTX 3080 排到 device 0, 使 PG199=1 失效).
#   - CUDA_VISIBLE_DEVICES=1: 选 PG199 (32GB), DiT 26.3GB bf16 必须用它.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import time
from pathlib import Path

import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402
from library.models.krea2_raw.weights import (  # noqa: E402
    inspect_dit_config,
    load_krea2_dit,
)

KREA2_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"


def prepare(img: torch.Tensor, txtlen: int, patch: int, txtmask: torch.Tensor):
    """移植自 krea-ai/krea-2 sampling.prepare: patchify latent + 构造 pos/mask."""
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    from einops import repeat
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

    print(f"=== 阶段 2: Krea-2 DiT 单 latent forward 验证 (PG199, {dtype}) ===")
    print(f"DiT: {KREA2_DIT} ({KREA2_DIT.stat().st_size / 1e9:.1f} GB)")

    # 1. 反推 config 核验
    print("\n--- 1. 权重 shape 反推 config 核验 ---")
    inferred = inspect_dit_config(KREA2_DIT)
    expected = SingleMMDiTConfig.krea2_raw()
    print(f"反推: {inferred}")
    config_mismatches = []
    for k, v in inferred.items():
        if k == "headdim":  # 反推的派生量, 不在 SingleMMDiTConfig 字段里
            continue
        exp = getattr(expected, k, None)
        if exp is None:
            config_mismatches.append((k, v, "<field not in config>"))
        elif v != exp:
            config_mismatches.append((k, v, exp))
    if config_mismatches:
        print(f"!! config 不匹配: {config_mismatches}")
        return 1
    print(f"config 与 SingleMMDiTConfig.krea2_raw() 全对齐 ✓")

    # 2. strict 加载
    print("\n--- 2. strict 加载 (path B, 原生命名) ---")
    t0 = time.time()
    model = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=True)
    load_t = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"加载耗时: {load_t:.2f}s, 参数量: {n_params/1e9:.2f}B")
    # strict=True 已在 load_krea2_dit 内部断言; 走到这里即 0 missing/0 unexpected.
    print("strict 加载通过 (0 missing / 0 unexpected) ✓")

    # 3. 移到 GPU
    model = model.to(device)

    # 4. 合成输入 + forward
    print("\n--- 3. 单 latent forward (合成 context) ---")
    patch = model.config.patch
    channels = model.config.channels
    txtdim = model.config.txtdim
    txtlayers = model.config.txtlayers

    # 用 256x256 起步: latent (1,16,32,32) -> patchify -> (1,256,64).
    # Krea-2 DiT 12.8B bf16 = 25.6GB 权重, 加激活/attention 中间量在 PG199 32GB
    # 上需要控制分辨率; 256x256 是验证 shape 对齐的最小合理尺寸.
    h = w = 256
    latent_h, latent_w = h // 8, w // 8  # VAE 8x 压缩
    noise = torch.randn(1, channels, latent_h, latent_w, device=device, dtype=dtype)

    # 合成 context: (B, L_txt, num_txt_layers, txtdim)
    # Qwen3-VL 输出 12 层 hidden states stack 在 dim=2 (R1 定论).
    txtlen = 77  # 典型 CLIP-style 长度; Krea-2 实际用 Qwen3-VL tokenizer
    txtmask = torch.ones(1, txtlen, device=device, dtype=torch.bool)
    context = torch.randn(1, txtlen, txtlayers, txtdim, device=device, dtype=dtype) * 0.02

    # patchify + 构造 pos/mask
    img_tokens, pos, mask = prepare(noise, txtlen, patch, txtmask)
    print(f"img_tokens: {tuple(img_tokens.shape)} (期望 B={1}, L_img={latent_h*latent_w//patch//patch}, D={channels*patch*patch})")
    print(f"context:    {tuple(context.shape)}")
    print(f"pos:        {tuple(pos.shape)}")
    print(f"mask:       {tuple(mask.shape)}")

    t = torch.tensor([0.5], device=device, dtype=dtype)  # timestep

    t1 = time.time()
    with torch.inference_mode():
        out = model(img=img_tokens, context=context, t=t, pos=pos, mask=mask)
    fwd_t = time.time() - t1

    # 期望输出 shape: (1, L_img, patch*patch*channels) = (1, 1024, 64)
    exp_l_img = (latent_h // patch) * (latent_w // patch)
    exp_out = (1, exp_l_img, patch * patch * channels)
    print(f"\nforward 输出: {tuple(out.shape)} (期望 {exp_out})")
    print(f"forward 耗时: {fwd_t*1000:.0f}ms")

    shape_ok = tuple(out.shape) == exp_out
    finite = torch.isfinite(out).all().item()
    print(f"shape 对齐: {shape_ok}")
    print(f"数值有限 (无 NaN/Inf): {finite}")
    if not finite:
        print(f"!! 输出含 NaN/Inf: min={out.min().item()}, max={out.max().item()}")

    # 5. 不同分辨率 forward (验证 pos/mask 构造对齐)
    #    用 256/512 两种; 768+ 在 PG199 32GB 上 12.8B bf16 + 激活会 OOM,
    #    阶段 4 训练热测再用 block swap 跑大分辨率.
    print("\n--- 4. 多分辨率 forward shape 核验 ---")
    multi_ok = True
    for h, w in [(256, 256), (512, 512)]:
        lh, lw = h // 8, w // 8
        noise2 = torch.randn(1, channels, lh, lw, device=device, dtype=dtype)
        img2, pos2, mask2 = prepare(noise2, txtlen, patch, txtmask)
        with torch.inference_mode():
            out2 = model(img=img2, context=context, t=t, pos=pos2, mask=mask2)
        exp2 = (1, (lh // patch) * (lw // patch), patch * patch * channels)
        finite2 = torch.isfinite(out2).all().item()
        ok2 = tuple(out2.shape) == exp2 and finite2
        print(f"[{h}x{w}] out={tuple(out2.shape)} exp={exp2} finite={finite2} ok={ok2}")
        multi_ok = multi_ok and ok2
        del out2, noise2
        torch.cuda.empty_cache()

    # 结论
    print("\n=== 结论 ===")
    ok = shape_ok and finite and multi_ok
    print(f"阶段 2 出口通过 (单 latent forward shape 对齐 + 数值有限): {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
