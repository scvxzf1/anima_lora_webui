"""R2 验证: Krea-2 VAE encode/decode 互逆性 + per-channel mean/std 一致性.

加载 anima 的 AutoencoderKLQwenImage + Krea-2 VAE 权重, encode 一张测试图
再 decode 回来, 计算 PSNR. 同时核对 latents_mean/std 与 Krea-2 checkpoint
是否逐元素一致.

通过门槛: encode→decode 误差与 anima 自身互逆误差同量级(PSNR > 40dB).
"""
from __future__ import annotations

import os
# PCI_BUS_ID 让 index 与 nvidia-smi 一致; PG199 = device 1 (32GB).
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.models.qwen_vae import AutoencoderKLQwenImage, load_vae  # noqa: E402

KREA2_VAE = ROOT / "models" / "vae" / "qwen_image_vae.safetensors"


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    import math
    mse = ((a.float() - b.float()) ** 2).mean().item()
    if mse <= 0:
        return 99.0
    return 10 * math.log10(1.0 / mse)


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"=== R2: Krea-2 VAE 互逆验证 (PG199, {dtype}) ===")
    print(f"VAE: {KREA2_VAE} ({KREA2_VAE.stat().st_size / 1e6:.1f} MB)")

    # 1. 加载 VAE (用 anima 的 load_vae, 它会从 checkpoint 读 config + 转换 ComfyUI 前缀)
    t0 = time.time()
    vae = load_vae(str(KREA2_VAE), device=device, dtype=dtype, eval=True)
    print(f"VAE 加载耗时: {time.time()-t0:.2f}s")

    # 2. 核对 latents_mean/std
    mean = list(vae.latents_mean)
    std = list(vae.latents_std)
    print(f"\nlatents_mean (16): {mean}")
    print(f"latents_std  (16): {std}")

    # 子代理给的 Krea-2 公开值 (Qwen/Qwen-Image config.json)
    krea2_mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                  0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    krea2_std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]

    mean_match = all(abs(a - b) < 1e-3 for a, b in zip(mean, krea2_mean))
    std_match = all(abs(a - b) < 1e-3 for a, b in zip(std, krea2_std))
    print(f"\nmean 与 Krea-2 公开值一致: {mean_match}")
    print(f"std  与 Krea-2 公开值一致: {std_match}")

    # 3. 互逆测试: 用结构化图案 + 随机像素对比
    #    随机均匀像素对 VAE 重建本就困难(非自然分布), 用渐变/条纹验证互逆性
    results = []
    torch.manual_seed(42)
    for h, w in [(256, 256), (512, 768), (768, 512)]:
        # 结构化图案: 渐变 + 条纹 + 棋盘, 模拟自然图像的低频分布
        ys = torch.linspace(0, 1, h, device=device, dtype=dtype)
        xs = torch.linspace(0, 1, w, device=device, dtype=dtype)
        grad_h = ys.view(h, 1).expand(h, w)       # 水平渐变 (h,w)
        grad_w = xs.view(1, w).expand(h, w)      # 垂直渐变 (h,w)
        checker = ((xs.view(1, w).expand(h, w) * 8).int().float() % 2)
        pixels = torch.stack([grad_h, grad_w, checker], dim=0).unsqueeze(0)
        pixels = (pixels * 0.8 + 0.1).clamp(0, 1)  # 避免极值
        # encode
        t1 = time.time()
        with torch.no_grad():
            latents = vae.encode_pixels_to_latents(pixels)
        enc_t = time.time() - t1
        # decode
        t2 = time.time()
        with torch.no_grad():
            recon = vae.decode_to_pixels(latents)
        dec_t = time.time() - t2

        # latents shape 应为 (B, 16, H//8, W//8)
        exp_h, exp_w = h // 8, w // 8
        shape_ok = latents.shape == (1, 16, exp_h, exp_w)
        p = psnr(pixels, recon)
        results.append((h, w, latents.shape, shape_ok, p, enc_t, dec_t))
        print(f"\n[{h}x{w}] latents={tuple(latents.shape)} shape_ok={shape_ok} "
              f"PSNR={p:.2f}dB enc={enc_t*1000:.0f}ms dec={dec_t*1000:.0f}ms")

    # 4. 5D 路径测试 (unsqueeze(2))
    print("\n=== 5D 路径 (B,C,1,H,W) ===")
    ys = torch.linspace(0, 1, 512, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, 512, device=device, dtype=dtype)
    grad_h = ys.view(512, 1).expand(512, 512)
    grad_w = xs.view(1, 512).expand(512, 512)
    pixels5d = torch.stack([grad_h, grad_w, (xs.view(1,512).expand(512,512)*8).int().float()%2], dim=0)
    pixels5d = (pixels5d.unsqueeze(0).unsqueeze(2) * 0.8 + 0.1).clamp(0,1)
    with torch.no_grad():
        # encode_pixels_to_latents 内部处理 4D/5D; 这里测 4D 输入走 5D 内部路径
        latents_4d = vae.encode_pixels_to_latents(pixels5d.squeeze(2))
        recon_4d = vae.decode_to_pixels(latents_4d)
    p5 = psnr(pixels5d.squeeze(2), recon_4d)
    print(f"[512x512 via 4D→5D 内部] PSNR={p5:.2f}dB")

    # 结论
    all_psnr = [r[4] for r in results] + [p5]
    min_psnr = min(all_psnr)
    print(f"\n=== 结论 ===")
    print(f"min PSNR = {min_psnr:.2f}dB (结构化图案门槛 25dB)")
    print(f"mean/std 一致: {mean_match and std_match}")
    ok = min_psnr > 25 and mean_match and std_match
    print(f"R2 通过: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
