"""DC-Gen POC 探针 4：锻造基座 save/load roundtrip。

验证 library/models/dcgen_forge.py 的完整分发链路：
  old DiT (f8/c16/p2) -> forge_new_dit (复制主干 + 新输入/输出层)
  -> save_forged_dit (safetensors + dcgen metadata)
  -> load_forged_dit (从 metadata 重建几何)
  -> 同输入 forward 与保存前一致。

输入/输出层此时仍是随机初始化（未经过对齐训练），本探针只验证
checkpoint 几何与数值无损，不验证生成质量。
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

from library.anima.weights import load_anima_model  # noqa: E402
from library.io.cache_names import latent_cache_suffix  # noqa: E402
from library.models.dcgen_forge import (  # noqa: E402
    forge_new_dit,
    load_forged_dit,
    save_forged_dit,
)
from library.models.latent_space import DCGEN_F32C32_P1  # noqa: E402

DIT_PATH = Path(
    "/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/"
    "anima-preview3-base.safetensors"
)
CACHE_DIR = ROOT / "scripts" / "dcgen" / "_out" / "dual_latent_cache"
OUT_PATH = ROOT / "scripts" / "dcgen" / "_out" / "forged" / "anima_dcgen_f32c32_dryrun.safetensors"


@torch.no_grad()
def _forward(dit, z1, context, padding_mask, t):
    return dit(z1.unsqueeze(2), t, context, padding_mask=padding_mask).squeeze(2)


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=== DC-Gen POC 4: forged base save/load roundtrip ===")
    old_dit = load_anima_model(
        device=device, dit_path=str(DIT_PATH), attn_mode="torch",
        loading_device=device, dit_weight_dtype=dtype,
    )
    old_dit.eval()

    print("--- forge new DiT (copy backbone, fresh in/out heads) ---")
    new_dit, report = forge_new_dit(old_dit, device=device, dtype=dtype)
    print(f"copied {report.copied}/{report.total}, skipped {len(report.skipped)}")
    for k, ns, os_ in report.skipped:
        print(f"  {k}: new {ns} old {os_}")
    del old_dit
    torch.cuda.empty_cache()

    # 输入：新空间 latent + 随机 context
    size = 256
    suffix = latent_cache_suffix(DCGEN_F32C32_P1.name)
    z1 = torch.from_numpy(
        np.load(CACHE_DIR / f"probe_{size:04d}x{size:04d}{suffix}")[
            f"latents_{size//32}x{size//32}"
        ]
    ).to(device=device, dtype=dtype)
    context = torch.randn(1, 64, 1024, device=device, dtype=dtype)
    padding_mask = torch.zeros(1, 1, z1.shape[-2], z1.shape[-1], device=device, dtype=dtype)
    t = torch.full((1, 1), 0.5, device=device, dtype=dtype)

    new_dit.eval()
    before = _forward(new_dit, z1, context, padding_mask, t)

    print(f"--- save forged base -> {OUT_PATH} ---")
    save_forged_dit(new_dit, OUT_PATH, spec=DCGEN_F32C32_P1)
    print(f"file size: {OUT_PATH.stat().st_size/2**30:.2f} GiB")
    del new_dit
    torch.cuda.empty_cache()

    print("--- load forged base from checkpoint metadata ---")
    loaded, meta = load_forged_dit(OUT_PATH, device=device, dtype=dtype)
    loaded.eval()
    print(f"metadata: {meta}")
    after = _forward(loaded, z1, context, padding_mask, t)

    max_diff = float((before - after).abs().max())
    print(f"forward max |delta| = {max_diff:.3e}")
    assert tuple(before.shape) == tuple(after.shape) == (1, 32, 8, 8)
    assert meta["dcgen_space"] == "dcgen_f32c32"
    assert int(meta["latent_channels"]) == 32
    assert int(meta["patch_spatial"]) == 1
    assert int(meta["vae_spatial_compression"]) == 32
    assert max_diff == 0.0, "bf16 roundtrip must be lossless"
    print("OK: forged base save/load roundtrip passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
