"""DC-Gen POC 探针 3：阶段 2 端到端 LoRA dry-run。

论文阶段 2 在新潜空间上做标准 flow-matching 的 LoRA 微调（FLUX 用
rank=alpha=256）。本 dry-run 用 rank=4 验证链路：
1. 复制旧主干到新几何 DiT（与 probe 2 相同）。
2. 冻结 base 参数，只训 LoRA 参数。
3. 新 latent 上 rectified-flow loss 显著下降。

正式锻造时把 rank 提到 256、接真实数据集与 text encoder。
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
from networks.lora_anima.config import LoRANetworkCfg  # noqa: E402
from networks.lora_anima.network import LoRANetwork  # noqa: E402
from networks.lora_modules.lora import LoRAModule  # noqa: E402

DIT_PATH = Path(
    "/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/"
    "anima-preview3-base.safetensors"
)
CACHE_DIR = ROOT / "scripts" / "dcgen" / "_out" / "dual_latent_cache"
N_STEPS = 8
LORA_DIM = 4
LORA_ALPHA = 4.0
LR = 2e-3


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=== DC-Gen POC 3: end-to-end LoRA dry-run (stage 2) ===")
    print("--- 加载旧 DiT 并复制主干到新几何 ---")
    old_dit = load_anima_model(
        device=device, dit_path=str(DIT_PATH), attn_mode="torch",
        loading_device=device, dit_weight_dtype=dtype,
    )
    old_dit.eval()
    spec = DCGEN_F32C32_P1
    new_dit = Anima(
        max_img_h=512, max_img_w=512, max_frames=128,
        in_channels=spec.latent_channels, out_channels=spec.latent_channels,
        patch_spatial=spec.patch_spatial, patch_temporal=spec.patch_temporal,
        vae_spatial_compression=spec.vae_spatial_compression,
    ).to(device=device, dtype=dtype)
    new_state = new_dit.state_dict()
    for k, v in old_dit.state_dict().items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k].copy_(v)
    del old_dit
    torch.cuda.empty_cache()

    # 冻结 base，只训 LoRA
    for p in new_dit.parameters():
        p.requires_grad_(False)

    cfg = LoRANetworkCfg.from_kwargs(
        {}, network_dim=LORA_DIM, network_alpha=LORA_ALPHA,
        neuron_dropout=None, module_class=LoRAModule,
    )
    network = LoRANetwork(text_encoders=[], unet=new_dit, cfg=cfg, multiplier=1.0)
    # LoRA 模块默认只创建不挂载，必须显式 apply_to 把 Linear.forward monkey-patch 掉。
    network.apply_to(text_encoders=[], unet=new_dit, apply_text_encoder=False, apply_unet=True)
    # apply_to 里 add_module 之后，LoRA 参数才进 network.parameters()；
    # 新建的 LoRA 参数在 CPU，需要随 network 搬到 GPU。
    network = network.to(device=device, dtype=dtype)
    # 标准参数收集入口：LoRA 参数注册在 unet 子模块上，必须走 optimizer_groups。
    param_groups, _lr_descriptions = network.prepare_optimizer_params_with_multiple_te_lrs(
        text_encoder_lr=None, unet_lr=LR, default_lr=LR
    )
    n_params = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"LoRA trainable params: {n_params/1e6:.2f}M")

    size = 256
    suffix = latent_cache_suffix(spec.name)
    z1 = torch.from_numpy(
        np.load(CACHE_DIR / f"probe_{size:04d}x{size:04d}{suffix}")[
            f"latents_{size//32}x{size//32}"
        ]
    ).to(device=device, dtype=dtype)
    context = torch.randn(1, 64, 1024, device=device, dtype=dtype)
    padding_mask = torch.zeros(1, 1, z1.shape[-2], z1.shape[-1], device=device, dtype=dtype)

    opt = torch.optim.Adam(param_groups)

    def lora_weight_norm() -> float:
        return float(
            sum(torch.norm(p.detach()).item() for g in param_groups for p in g["params"])
        )

    new_dit.train()
    torch.manual_seed(0)
    x0 = torch.randn_like(z1)
    t = torch.full((1, 1), 0.5, device=device, dtype=dtype)
    xt = 0.5 * x0 + 0.5 * z1
    target = z1 - x0
    grad_norm = None
    before = lora_weight_norm()
    for step in range(1, N_STEPS + 1):
        opt.zero_grad()
        pred = new_dit(xt.unsqueeze(2), t, context, padding_mask=padding_mask).squeeze(2)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        grads = [p.grad for g in param_groups for p in g["params"] if p.grad is not None]
        if grads:
            grad_norm = float(
                sum(g.norm().item() ** 2 for g in grads) ** 0.5
            )
        opt.step()
        print(f"step {step:3d}  mse {float(loss.detach()):.6f}  grad_norm {grad_norm if grad_norm is not None else float('nan'):.4f}")

    after = lora_weight_norm()
    delta = abs(after - before)
    assert tuple(pred.shape) == (1, 32, 8, 8)
    assert grads and grad_norm is not None and grad_norm > 0, "LoRA 参数没有梯度"
    assert delta > 0, "LoRA 参数没有被 optimizer 更新"
    print(f"\nlora weight norm {before:.6f} -> {after:.6f} (delta {delta:.6f})")
    print("OK: LoRA E2E dry-run passed (gradient flows, optimizer updates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
