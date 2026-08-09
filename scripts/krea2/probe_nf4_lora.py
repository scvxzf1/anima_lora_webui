"""R-verify: Krea-2-Raw NF4 x LoRA 兼容性 (NF4 落地第 3 层前置).

在 NF4 量化的 DiT 上构造 LoRA network + apply_to, 验证 plumbing 不破:
  V1 selection: Linear4bit (继承 nn.Linear) 被 LoRA 扫描命中, target 数=196.
  V2 反量化保住: patch 后 org_forward 仍指向 Linear4bit.forward (bnb.matmul_4bit).
  V3 delta 叠加: 注入非零 lora_up 后, apply 前后 forward 输出有可见差异.
  V4 weight_svd 不触发: 默认 down_init=kaiming, monkeypatch 验构造不读 Params4bit.
  V5 梯度流向: backward 后 LoRA lora_up/down.grad 非零, DiT Linear4bit 无 grad.
  V6 merge/bake: 训练 forward 不调 (仅 save 路径触发, 本探针不调, 风险标注).

用随机 hiddens (shape 对齐 Qwen3-VL 12 层 stack (B,L,12,2560)) 免加载 TE,
纯验 plumbing 不验语义. 非目标: 端到端 loss 收敛 (层 3), 跨机复现 (层 4).

PG199 bf16, 256x256 (latent 32x32, patch=2 -> 256 img tokens + 64 txt = 320 seq).
"""
from __future__ import annotations

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 同目录 import probe_nf4

from bitsandbytes.nn import Linear4bit  # noqa: E402
from library.models.krea2_raw.family import (  # noqa: E402
    Krea2TextEmbedding,
    forward_for_loss,
)
from library.models.krea2_raw.lora_targets import krea2_target_kwargs  # noqa: E402
from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402
from networks.lora_anima.config import LoRANetworkCfg  # noqa: E402
from networks.lora_anima.network import LoRANetwork  # noqa: E402
from networks.lora_modules.lora import LoRAModule  # noqa: E402

from probe_nf4 import quantize_model  # noqa: E402  复用在线 NF4 量化 (同 env)

KREA2_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"

LORA_DIM = 16
LORA_ALPHA = 8.0
L_TXT = 64
IMG_SIZE = 256  # latent 32x32, patch=2


def _gpu_mem(label: str) -> None:
    print(f"  [{label}] alloc={torch.cuda.memory_allocated() / 1e9:.2f}GB")


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"=== NF4 x LoRA 兼容性探针 (PG199, {dtype}) ===")

    # --- A. 加载 bf16 DiT (CPU) + NF4 量化 (复用 probe_nf4, 内部 .to 触发量化) ---
    print("\n--- A. 加载 bf16 DiT + NF4 量化 ---")
    t0 = time.time()
    dit = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=False)
    print(f"  DiT 加载 (CPU): {time.time() - t0:.1f}s")
    quantize_model(dit, device)  # 打印 264 层 / 6.6GB / 量化耗时.
    for p in dit.parameters():
        p.requires_grad_(False)
    _gpu_mem("NF4 DiT on GPU")

    # --- B. 构造 LoRA + apply_to ---
    # V4: monkeypatch _init_down_weight_svd, 验默认 kaiming 路径不调它 (不读 Params4bit).
    svd_calls = {"n": 0}
    orig_svd = LoRAModule._init_down_weight_svd

    def _spy(self, org_module):
        svd_calls["n"] += 1
        return orig_svd(self, org_module)

    LoRAModule._init_down_weight_svd = _spy

    kwargs = {**krea2_target_kwargs(), "lora_dim": LORA_DIM, "alpha": LORA_ALPHA}
    cfg = LoRANetworkCfg.from_kwargs(
        kwargs,
        network_dim=LORA_DIM,
        network_alpha=LORA_ALPHA,
        neuron_dropout=None,
        module_class=LoRAModule,
    )
    network = LoRANetwork(text_encoders=[], unet=dit, cfg=cfg, multiplier=1.0)
    network.apply_to(
        text_encoders=[], unet=dit, apply_text_encoder=False, apply_unet=True
    )
    network = network.to(device).to(dtype)
    for p in network.parameters():
        p.requires_grad_(True)
    LoRAModule._init_down_weight_svd = orig_svd  # 还原

    n_lora = len(network.unet_loras)
    n_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"  LoRA 模块: {n_lora}, 可训参数: {n_train / 1e6:.2f}M")
    _gpu_mem("LoRA apply 后")

    # --- 验证 ---
    print("\n=== 验证 ===")
    # V1: target 数 (28 block x 7 = 196).
    v1 = n_lora == 196
    names = [m.lora_name for m in network.unet_loras[:2]]
    names.append(network.unet_loras[-1].lora_name)
    print(f"  V1 selection: {n_lora}==196? {v1}; 样例: {names}")
    print(f"  V4 weight_svd 调用: {svd_calls['n']} (默认 kaiming 应为 0)")

    # V2: patch 后 org_forward 仍指向 Linear4bit.forward (反量化路径保住).
    v2_ok = 0
    for m in network.unet_loras:
        fn = getattr(m.org_forward, "__func__", None)
        if fn is not None and getattr(fn, "__qualname__", "") == "Linear4bit.forward":
            v2_ok += 1
    v2 = v2_ok == n_lora
    print(f"  V2 org_forward=Linear4bit.forward: {v2_ok}/{n_lora} (全命中: {v2})")

    # forward 输入.
    latents_5d = torch.randn(
        1, 16, 1, IMG_SIZE // 8, IMG_SIZE // 8, device=device, dtype=dtype
    )
    t = torch.full((1,), 0.5, device=device, dtype=dtype)
    hiddens = torch.randn(1, L_TXT, 12, 2560, device=device, dtype=dtype)
    mask = torch.ones(1, L_TXT, device=device, dtype=torch.bool)
    text_emb = Krea2TextEmbedding(hiddens, mask)

    # V3: delta=0 (up 零初始化) 时 forward finite; 注入非零后输出有可见变化.
    with torch.no_grad():
        out_zero = forward_for_loss(dit, latents_5d, text_emb, t)
    v3a = torch.isfinite(out_zero).all().item()
    with torch.no_grad():
        for m in network.unet_loras:
            torch.nn.init.normal_(m.lora_up.weight, std=0.02)
        out_delta = forward_for_loss(dit, latents_5d, text_emb, t)
    v3b = torch.isfinite(out_delta).all().item()
    diff_max = (out_delta - out_zero).abs().max().item()
    v3c = diff_max > 1e-3
    print(
        f"  V3 forward finite (delta=0): {v3a}; (注入非零) finite: {v3b}; "
        f"输出变化 max={diff_max:.4f} (>1e-3: {v3c})"
    )

    # V5: 梯度流到 LoRA, DiT Linear4bit 无 grad (V3 已注入非零 up, 直接 backward).
    target = torch.randn_like(latents_5d)
    vel = forward_for_loss(dit, latents_5d, text_emb, t)
    loss = torch.nn.functional.mse_loss(vel, target)
    loss.backward()
    up_nz = sum(
        1 for m in network.unet_loras
        if m.lora_up.weight.grad is not None
        and m.lora_up.weight.grad.abs().sum().item() > 0
    )
    dn_nz = sum(
        1 for m in network.unet_loras
        if m.lora_down.weight.grad is not None
        and m.lora_down.weight.grad.abs().sum().item() > 0
    )
    v5a = up_nz == n_lora and dn_nz == n_lora
    l4 = next(m for _, m in dit.named_modules() if isinstance(m, Linear4bit))
    v5b = l4.weight.grad is None
    print(
        f"  V5 LoRA grad 非零: up {up_nz}/{n_lora}, down {dn_nz}/{n_lora} "
        f"(全通: {v5a}); DiT Linear4bit 无 grad: {v5b}"
    )
    print(f"  V6 merge/bake: 训练 forward 不调 (仅 save 路径, 风险已标注, 不实测)")
    _gpu_mem("backward 后")

    ok = v1 and v2 and v3a and v3b and v3c and v5a and v5b and svd_calls["n"] == 0
    print("\n=== 汇总 ===")
    print(f"  V1 selection(196):    {v1}")
    print(f"  V2 反量化保住:        {v2}")
    print(f"  V3 delta 叠加:        {v3a and v3b and v3c}")
    print(f"  V4 weight_svd 不触发: {svd_calls['n'] == 0}")
    print(f"  V5 梯度流向 LoRA:     {v5a and v5b}")
    print(f"  NF4 x LoRA 兼容性: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
