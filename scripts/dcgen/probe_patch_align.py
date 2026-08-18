"""DC-Gen POC 探针 1：Patch Embedding Alignment dry-run。

复现论文阶段 0 的核心对齐（局部，不加载整 DiT）：
  target = spatial_downsample(old_x_embedder(z_old, mask_old), new_grid)
  pred   = new_x_embedder(z_new, mask_new)
  loss   = MSE(pred, target)

只把旧 checkpoint 的 ``net.x_embedder.proj.1.weight`` 读进教师 PatchEmbed，
新 PatchEmbed 随机初始化并在冻结教师下训练。验证：
1. 旧/新 patch 特征的网格、通道契约（旧 2048×68 -> 新 2048×33）。
2. 下采样后新旧网格一致（256² 图：旧 (1,16,16,2048) --pool2--> (1,8,8,2048)）。
3. 新 embedder 可训练，MSE 显著下降。

真实 DC-Gen 还会在下一阶段联合对齐 output head，并最后做 rank-256 LoRA
端到端训练；本探针只证明输入层对齐这条链路在本仓库模型代码上可跑通。
"""
from __future__ import annotations

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.anima.models import PatchEmbed  # noqa: E402
from library.io.cache_names import latent_cache_suffix  # noqa: E402
from library.models.latent_space import ANIMA_F8C16_P2, DCGEN_F32C32_P1  # noqa: E402

DIT_PATH = Path(
    "/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/"
    "anima-preview3-base.safetensors"
)
CACHE_DIR = ROOT / "scripts" / "dcgen" / "_out" / "dual_latent_cache"
N_STEPS = 200
LR = 1e-3


def load_old_x_embedder_weight(device: torch.device, dtype: torch.dtype):
    from safetensors import safe_open

    with safe_open(str(DIT_PATH), framework="pt", device="cpu") as f:
        w = f.get_slice("net.x_embedder.proj.1.weight")[:].to(device=device, dtype=dtype)
    print(f"old x_embedder weight: {tuple(w.shape)}")
    return w


def spatial_downsample_patch_features(x_bt_h_w_d: torch.Tensor, scale: int = 2):
    """5D patch feature (B,T,H,W,D) -> (B,T,H//scale,W//scale,D), avg pool."""
    b, t, h, w, d = x_bt_h_w_d.shape
    x = x_bt_h_w_d.permute(0, 1, 4, 2, 3).reshape(b * t, d, h, w)
    x = torch.nn.functional.avg_pool2d(x, kernel_size=scale, stride=scale)
    _, _, h2, w2 = x.shape
    return x.reshape(b, t, d, h2, w2).permute(0, 1, 3, 4, 2)


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=== DC-Gen POC 1: patch embedding alignment (dry-run) ===")
    size = 256
    old_suffix = latent_cache_suffix(ANIMA_F8C16_P2.name)
    new_suffix = latent_cache_suffix(DCGEN_F32C32_P1.name)
    z_old = torch.from_numpy(
        np.load(CACHE_DIR / f"probe_{size:04d}x{size:04d}{old_suffix}")[
            f"latents_{size//8}x{size//8}"
        ]
    ).to(device=device, dtype=dtype)
    z_new = torch.from_numpy(
        np.load(CACHE_DIR / f"probe_{size:04d}x{size:04d}{new_suffix}")[
            f"latents_{size//32}x{size//32}"
        ]
    ).to(device=device, dtype=dtype)
    print(f"z_old {tuple(z_old.shape)} z_new {tuple(z_new.shape)}")

    # 教师：旧 x_embedder（冻结）
    old_embed = PatchEmbed(
        spatial_patch_size=ANIMA_F8C16_P2.patch_spatial,
        temporal_patch_size=ANIMA_F8C16_P2.patch_temporal,
        in_channels=ANIMA_F8C16_P2.patch_embed_in_channels,
        out_channels=2048,
    ).to(device=device, dtype=dtype)
    old_embed.proj[1].weight.data.copy_(load_old_x_embedder_weight(device, dtype))
    old_embed.eval()
    for p in old_embed.parameters():
        p.requires_grad_(False)

    # 学生：新 x_embedder（随机初始化，可训练）
    new_embed = PatchEmbed(
        spatial_patch_size=DCGEN_F32C32_P1.patch_spatial,
        temporal_patch_size=DCGEN_F32C32_P1.patch_temporal,
        in_channels=DCGEN_F32C32_P1.patch_embed_in_channels,
        out_channels=2048,
    ).to(device=device, dtype=dtype)
    print(
        f"old embed in_features={old_embed.dim} new embed in_features={new_embed.dim}"
    )

    # 5D 输入 + padding mask
    z_old_5d = z_old.unsqueeze(2)
    z_new_5d = z_new.unsqueeze(2)
    mask_old = torch.zeros(1, 1, 1, z_old.shape[-2], z_old.shape[-1], device=device, dtype=dtype)
    mask_new = torch.zeros(1, 1, 1, z_new.shape[-2], z_new.shape[-1], device=device, dtype=dtype)
    z_old_5d = torch.cat([z_old_5d, mask_old], dim=1)
    z_new_5d = torch.cat([z_new_5d, mask_new], dim=1)

    with torch.no_grad():
        feat_old = old_embed(z_old_5d)
        target = spatial_downsample_patch_features(feat_old, scale=2)
        print(f"feat_old {tuple(feat_old.shape)} -> target {tuple(target.shape)}")

    opt = torch.optim.Adam(new_embed.parameters(), lr=LR)
    initial = None
    final = None
    for step in range(1, N_STEPS + 1):
        opt.zero_grad()
        pred = new_embed(z_new_5d)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
        val = float(loss.detach())
        if initial is None:
            initial = val
        final = val
        if step == 1 or step % 40 == 0 or step == N_STEPS:
            print(f"step {step:4d}  mse {val:.6f}")

    assert tuple(pred.shape) == tuple(target.shape) == (1, 1, 8, 8, 2048)
    print(f"\ninitial {initial:.6f} -> final {final:.6f}")
    if final >= initial * 0.7:
        raise SystemExit("FAIL: loss barely decreased, alignment not learning")
    print("OK: patch embedding alignment dry-run passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
