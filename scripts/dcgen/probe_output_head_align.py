"""DC-Gen POC 探针 2：阶段 1 output-head alignment dry-run。

论文 3.3：先训 patch embedder，再"冻结 DiT 主干，联合微调 patch embedder
与 output head"——目标函数不是对齐旧模型 velocity，而是新潜空间上的标准
flow-matching loss（Eq. 2）。新输出 32 通道 vs 旧 16 通道的不匹配因此不存在。

本探针：
1. 加载完整旧 DiT（f8/c16/p2）作为主干权重来源。
2. 构造新几何 DiT（f32/c32/p1），复制所有形状匹配的主干权重；
   x_embedder / final_layer（形状不同）保持随机初始化。
3. 冻结主干，只训练 x_embedder + final_layer。
4. 在新 latent 上做 rectified-flow 一步对齐，验证 loss 显著下降。

这是 dry-run：文本条件用随机 context，latent 用合成图，只证明训练几何闭环。
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

from library.anima.models import Anima  # noqa: E402
from library.anima.weights import load_anima_model  # noqa: E402
from library.io.cache_names import latent_cache_suffix  # noqa: E402
from library.models.latent_space import DCGEN_F32C32_P1  # noqa: E402

DIT_PATH = Path(
    "/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/"
    "anima-preview3-base.safetensors"
)
CACHE_DIR = ROOT / "scripts" / "dcgen" / "_out" / "dual_latent_cache"
N_STEPS = 40
LR = 1e-3


def make_new_dit(device: torch.device, dtype: torch.dtype) -> Anima:
    spec = DCGEN_F32C32_P1
    return Anima(
        max_img_h=512,
        max_img_w=512,
        max_frames=128,
        in_channels=spec.latent_channels,
        out_channels=spec.latent_channels,
        patch_spatial=spec.patch_spatial,
        patch_temporal=spec.patch_temporal,
        vae_spatial_compression=spec.vae_spatial_compression,
    ).to(device=device, dtype=dtype)


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=== DC-Gen POC 2: output-head alignment dry-run (stage 1) ===")
    print("--- 加载旧 DiT (f8/c16/p2) 作为主干权重来源 ---")
    old_dit = load_anima_model(
        device=device,
        dit_path=str(DIT_PATH),
        attn_mode="torch",
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    old_dit.eval()
    old_state = old_dit.state_dict()

    print("--- 构造新 DiT (f32/c32/p1) 并复制主干权重 ---")
    new_dit = make_new_dit(device, dtype)
    new_state = new_dit.state_dict()
    copied, skipped = [], []
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k].copy_(v)
            copied.append(k)
        elif k in new_state:
            skipped.append((k, tuple(new_state[k].shape), tuple(v.shape)))
    print(f"copied {len(copied)}/{len(new_state)} params")
    for k, ns, os_ in skipped:
        print(f"  skip(shape mismatch) {k}: new {ns} old {os_}")
    del old_dit, old_state
    torch.cuda.empty_cache()

    # 冻结主干，只训 x_embedder + final_layer
    for p in new_dit.parameters():
        p.requires_grad_(False)
    for p in new_dit.x_embedder.parameters():
        p.requires_grad_(True)
    for p in new_dit.final_layer.parameters():
        p.requires_grad_(True)
    trainable = [n for n, p in new_dit.named_parameters() if p.requires_grad]
    print(f"trainable: {len(trainable)} modules -> {trainable[:6]}{'...' if len(trainable)>6 else ''}")

    # 数据：新 latent（scaled）+ 随机 context
    size = 256
    suffix = latent_cache_suffix(DCGEN_F32C32_P1.name)
    z1 = torch.from_numpy(
        np.load(CACHE_DIR / f"probe_{size:04d}x{size:04d}{suffix}")[
            f"latents_{size//32}x{size//32}"
        ]
    ).to(device=device, dtype=dtype)
    print(f"z1 (image latent) {tuple(z1.shape)}")

    context = torch.randn(1, 64, 1024, device=device, dtype=dtype)
    padding_mask = torch.zeros(1, 1, z1.shape[-2], z1.shape[-1], device=device, dtype=dtype)

    opt = torch.optim.Adam(trainable_params := [p for n, p in new_dit.named_parameters() if p.requires_grad], lr=LR)
    del trainable_params

    initial = final = None
    new_dit.train()
    for step in range(1, N_STEPS + 1):
        opt.zero_grad()
        x0 = torch.randn_like(z1)
        t = torch.full((1, 1), 0.5, device=device, dtype=dtype)
        xt = (1 - 0.5) * x0 + 0.5 * z1
        target = z1 - x0  # velocity v = x1 - x0
        pred = new_dit(
            xt.unsqueeze(2), t, context, padding_mask=padding_mask
        ).squeeze(2)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
        val = float(loss.detach())
        if initial is None:
            initial = val
        final = val
        if step == 1 or step % 10 == 0 or step == N_STEPS:
            print(f"step {step:3d}  mse {val:.6f}")

    assert tuple(pred.shape) == (1, 32, 8, 8)
    print(f"\ninitial {initial:.6f} -> final {final:.6f}")
    if final >= initial * 0.8:
        raise SystemExit("FAIL: stage-1 loss barely decreased")
    print("OK: output-head alignment dry-run passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
