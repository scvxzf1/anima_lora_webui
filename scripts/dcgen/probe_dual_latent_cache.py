"""DC-Gen POC 探针 0：双 latent 缓存（anima f8c16 vs dcgen f32c32）。

验证两件事：
1. 同一批图片在旧 VAE 和新 DC-AE 下的 latent 形状契约：
   anima   256² -> (B,16,32,32)，1024² -> (B,16,128,128)
   dcgen   256² -> (B,32,8,8)，  1024² -> (B,32,32,32)
2. 两种空间按各自 cache_suffix 落盘、读回无损，且文件不会互相覆盖。

本探针只做缓存几何，不做 DiT / alignment / 训练。
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

from library.io.cache_names import latent_cache_suffix  # noqa: E402
from library.models.dc_ae import encode_images_to_latents, load_dc_ae  # noqa: E402
from library.models.latent_space import ANIMA_F8C16_P2, DCGEN_F32C32_P1  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402

ANIMA_VAE_PATH = Path(
    "/home/scv/nvme0n1p1/ComfyUI/models/vae/anima/anima_vae.safetensors"
)
OUT_DIR = ROOT / "scripts" / "dcgen" / "_out" / "dual_latent_cache"


def make_probe_image(h: int, w: int, device: torch.device, dtype: torch.dtype):
    """[-1,1] 渐变 + 棋盘，带高频边缘，避免全黑掩盖形状问题。"""
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    ch0 = ys.view(-1, 1).expand(h, w)
    ch1 = xs.view(1, -1).expand(h, w)
    ch2 = (((xs * 8).long()) % 2).to(dtype) * 2 - 1
    ch2 = ch2.view(1, -1).expand(h, w)
    return torch.stack([ch0, ch1, ch2], dim=0).unsqueeze(0)


def write_npz(path: Path, latents: torch.Tensor, latent_h: int, latent_w: int) -> None:
    key = f"latents_{latent_h}x{latent_w}"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: latents.detach().float().cpu().numpy()})


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=== DC-Gen POC 0: dual latent cache ===")
    # --- 旧 Anima VAE ---
    anima_vae = load_vae(str(ANIMA_VAE_PATH), device=device, dtype=dtype, eval=True)
    # --- 新 DC-AE ---
    dc_ae = load_dc_ae(device=device, dtype=dtype)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for size in (256, 1024):
        img = make_probe_image(size, size, device, dtype)

        with torch.no_grad():
            z_old = anima_vae.encode_pixels_to_latents(img)
            if z_old.dim() == 5:
                z_old = z_old.squeeze(2)
            z_new = encode_images_to_latents(dc_ae, img)

        expect_old = (1, 16, size // 8, size // 8)
        expect_new = (1, 32, size // 32, size // 32)
        assert tuple(z_old.shape) == expect_old, (tuple(z_old.shape), expect_old)
        assert tuple(z_new.shape) == expect_new, (tuple(z_new.shape), expect_new)

        old_suffix = latent_cache_suffix(ANIMA_F8C16_P2.name)
        new_suffix = latent_cache_suffix(DCGEN_F32C32_P1.name)
        old_path = OUT_DIR / f"probe_{size:04d}x{size:04d}{old_suffix}"
        new_path = OUT_DIR / f"probe_{size:04d}x{size:04d}{new_suffix}"

        write_npz(old_path, z_old, size // 8, size // 8)
        write_npz(new_path, z_new, size // 32, size // 32)

        # 读回校验
        old_back = np.load(old_path)[f"latents_{size//8}x{size//8}"]
        new_back = np.load(new_path)[f"latents_{size//32}x{size//32}"]
        assert np.allclose(old_back, z_old.float().cpu().numpy(), atol=1e-6)
        assert np.allclose(new_back, z_new.float().cpu().numpy(), atol=1e-6)

        print(
            f"{size:4d}px  anima {tuple(z_old.shape)} -> {old_path.name} "
            f"| dcgen {tuple(z_new.shape)} -> {new_path.name}"
        )
        print(
            f"        z_old mean {float(z_old.float().mean()):.4f} std {float(z_old.float().std()):.4f}"
            f" | z_new mean {float(z_new.float().mean()):.4f} std {float(z_new.float().std()):.4f}"
        )

    # token 数对照：同分辨率下 dcgen = anima 的 1/4
    t_old = (1024 // 8 // 2) ** 2
    t_new = (1024 // 32 // 1) ** 2
    print(f"\n1024² block tokens: anima {t_old} vs dcgen {t_new} (ratio {t_new/t_old:.2f})")
    assert t_new * 4 == t_old
    print("OK: dual latent cache probe passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
