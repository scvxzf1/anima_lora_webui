"""R-verify: Krea-2-Raw LoRA attach + forward 真火测试 (阶段 3 出口之二).

target spec 匹配 (probe_lora_targets.py) 只验证 regex 命中正确; 本探针验证
LoRANetwork 真正能挂到 Krea-2 DiT 上, monkey-patch Linear.forward 后 DiT
forward 仍跑通且输出有限.

验证项:
1. LoRANetwork 构造 (cfg 用 krea2_target_kwargs + LoRAModule) 不抛异常.
2. apply_to 后 unet_loras 数 == 28×7 = 196.
3. DiT forward 输出 shape 对齐 (B, L_img, patch*patch*channels) + 有限.
4. LoRA 初始近零 (down kaiming, up zero-init) → 输出与无 LoRA 输出 delta < 1e-1.
5. 计 LoRA 参数量 + 显存峰值 (阶段 4 训练串通的基线参考).

PG199 bf16, 256×256 (与 probe_dit 一致, 避免 OOM).
不验证: 反向传播 (阶段 4), TE LoRA (首日不挂), 采样 (阶段 5).
"""
from __future__ import annotations

import os

# 必须在 import torch 之前设 (同 probe_dit.py): PCI_BUS_ID 让 index 与
# nvidia-smi 一致, CUDA_VISIBLE_DEVICES=1 选 PG199 (32GB).
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import time
from pathlib import Path

import torch
from einops import rearrange, repeat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402
from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402
from library.models.krea2_raw.lora_targets import krea2_target_kwargs  # noqa: E402
from networks.lora_anima.config import LoRANetworkCfg  # noqa: E402
from networks.lora_anima.network import LoRANetwork  # noqa: E402
from networks.lora_modules.lora import LoRAModule  # noqa: E402

KREA2_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"


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

    print(f"=== 阶段 3: Krea-2 LoRA attach + forward 真火测试 (PG199, {dtype}) ===")

    # 1. 加载 DiT
    print("\n--- 1. 加载 Krea-2 DiT ---")
    t0 = time.time()
    model = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=True)
    print(f"DiT 加载耗时: {time.time()-t0:.2f}s")
    model = model.to(device)
    torch.cuda.reset_peak_memory_stats()

    # 2. 基线 forward (无 LoRA)
    print("\n--- 2. 基线 forward (无 LoRA) ---")
    patch = model.config.patch
    channels = model.config.channels
    txtdim = model.config.txtdim
    txtlayers = model.config.txtlayers
    h = w = 256
    latent_h, latent_w = h // 8, w // 8
    torch.manual_seed(0)
    noise = torch.randn(1, channels, latent_h, latent_w, device=device, dtype=dtype)
    txtlen = 77
    txtmask = torch.ones(1, txtlen, device=device, dtype=torch.bool)
    context = torch.randn(1, txtlen, txtlayers, txtdim, device=device, dtype=dtype) * 0.02
    img_tokens, pos, mask = prepare(noise, txtlen, patch, txtmask)
    t = torch.tensor([0.5], device=device, dtype=dtype)

    with torch.inference_mode():
        base_out = model(img=img_tokens, context=context, t=t, pos=pos, mask=mask)
    base_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"基线输出 shape: {tuple(base_out.shape)}, peak {base_peak:.2f}GB")

    # 3. 构造 LoRANetwork (Krea-2 target kwargs + plain LoRAModule)
    print("\n--- 3. 构造 LoRANetwork (krea2_target_kwargs, lora_dim=16) ---")
    kwargs = {
        **krea2_target_kwargs(),
        "lora_dim": 16,
        "alpha": 8.0,
    }
    cfg = LoRANetworkCfg.from_kwargs(
        kwargs,
        network_dim=16,
        network_alpha=8.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    print(f"cfg.unet_target_replace_modules: {cfg.unet_target_replace_modules}")
    print(f"cfg.exclude_patterns: {len(cfg.exclude_patterns)} 条")

    torch.cuda.reset_peak_memory_stats()
    network = LoRANetwork(
        text_encoders=[],
        unet=model,
        cfg=cfg,
        multiplier=1.0,
    )
    print(f"LoRA 模块数: {len(network.unet_loras)} (期望 28×7=196)")

    # 4. apply_to (monkey-patch Linear.forward)
    print("\n--- 4. apply_to (monkey-patch DiT Linear.forward) ---")
    network.apply_to(
        text_encoders=[],
        unet=model,
        apply_text_encoder=False,
        apply_unet=True,
    )
    print(f"apply_to 后 unet_loras: {len(network.unet_loras)} (期望 196)")
    # LoRA module 加进 network (独立 Module), 没随 model.to(device) 上 GPU.
    network = network.to(device).to(dtype)
    # add_module 在 apply_to 里 (application.py:60); 之前 network.parameters() 不含 LoRA.
    n_lora_params = sum(p.numel() for p in network.parameters())
    print(f"LoRA 参数量: {n_lora_params/1e6:.2f}M (dim=16, alpha=8)")

    # 5. attach 后 forward
    print("\n--- 5. attach 后 forward ---")
    t1 = time.time()
    with torch.inference_mode():
        lora_out = model(img=img_tokens, context=context, t=t, pos=pos, mask=mask)
    fwd_t = time.time() - t1
    attach_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"LoRA forward 输出: {tuple(lora_out.shape)}, {fwd_t*1000:.0f}ms, peak {attach_peak:.2f}GB")

    shape_ok = tuple(lora_out.shape) == tuple(base_out.shape)
    finite = torch.isfinite(lora_out).all().item()

    # 6. delta (LoRA 初始近零: down kaiming, up zero-init → delta 应很小)
    delta = (lora_out - base_out).abs()
    print(f"\n--- 6. delta vs 基线 (LoRA 初始应近零) ---")
    print(f"base_out 范围: [{base_out.min().item():.4f}, {base_out.max().item():.4f}]")
    print(f"lora_out 范围: [{lora_out.min().item():.4f}, {lora_out.max().item():.4f}]")
    print(f"delta max: {delta.max().item():.6f}, mean: {delta.mean().item():.6f}")
    delta_small = delta.max().item() < 0.1

    # 结论
    print("\n=== 结论 ===")
    ok = (
        len(network.unet_loras) == 196
        and shape_ok
        and finite
        and delta_small
    )
    print(f"阶段 3 attach + forward 通过: {ok}")
    print(f"  模块数 196: {len(network.unet_loras) == 196}")
    print(f"  shape 对齐: {shape_ok}")
    print(f"  数值有限:   {finite}")
    print(f"  delta 小:   {delta_small}")
    print(f"\n基线 (PG199 bf16, 256×256):")
    print(f"  DiT 显存 peak: {base_peak:.2f}GB")
    print(f"  attach 后 peak: {attach_peak:.2f}GB (+{attach_peak-base_peak:.2f}GB LoRA)")
    print(f"  LoRA 参数: {n_lora_params/1e6:.2f}M (dim=16, alpha=8)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
