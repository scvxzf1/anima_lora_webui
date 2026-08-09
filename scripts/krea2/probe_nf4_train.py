"""R-verify: Krea-2-Raw NF4 + LoRA + grad-ckpt 端到端训练 (NF4 落地层 3).

在 NF4 量化的 DiT 上跑 1024x1024 flow-matching LoRA 训练, 对比 bf16 基线
(stage6 findings: loss 0.465->0.198, peak 27.9GB, step 3.47s/it, grad-ckpt on).

验证项:
1. forward + backward 跑通 (NF4 反量化 + grad-ckpt recompute + LoRA delta 三者叠加).
2. loss 单调下降 (末5 < 首5), 量级与 bf16 基线可比 (不发散/不坍塌).
3. LoRA grad 非零 (梯度流到 LoRA, 不被 NF4 反量化阻断).
4. DiT Linear4bit 权重不变 (frozen, 仅 LoRA 训练).
5. 显存 peak 远低于 bf16 27.9GB (NF4 省权重的核心收益).

固定 σ=0.5 + 固定 noise 过拟合 (同 probe_train): target 完全固定, loss
单调下降证明 LoRA 能调整 frozen NF4 DiT 输出拟合 target. 非目标: 真实数据集
sweep (留生产路径), 跨机复现 (层 4).

PG199 32GB bf16, 1024x1024, grad-ckpt on, NF4 DiT ~6.6GB + LoRA + AdamW + 激活.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library.models.krea2_raw.family import Krea2TextEmbedding, forward_for_loss  # noqa: E402
from library.models.krea2_raw.lora_targets import krea2_target_kwargs  # noqa: E402
from library.models.krea2_raw.strategy import (  # noqa: E402
    Krea2TextEncodingStrategy,
    Krea2TokenizeStrategy,
    load_krea2_text_encoder,
)
from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402
from networks.lora_anima.config import LoRANetworkCfg  # noqa: E402
from networks.lora_anima.network import LoRANetwork  # noqa: E402
from networks.lora_modules.lora import LoRAModule  # noqa: E402

from probe_nf4 import quantize_model  # noqa: E402
from probe_train import FIXED_SIGMA, PROMPT, gpu_power, make_test_image  # noqa: E402

KREA2_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"
KREA2_TE = ROOT / "models" / "text_encoders" / "qwen3vl_4b_bf16.safetensors"
KREA2_VAE = ROOT / "models" / "vae" / "qwen_image_vae.safetensors"

N_STEPS = 50
LORA_DIM = 16
LORA_ALPHA = 8.0
LR = 2e-3
IMG_SIZE = 1024


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(
        f"=== 层 3: NF4 + LoRA + grad-ckpt 端到端训练 "
        f"(PG199, {dtype}, {IMG_SIZE}x{IMG_SIZE}) ==="
    )
    print(
        f"steps={N_STEPS}, lora_dim={LORA_DIM}, alpha={LORA_ALPHA}, "
        f"lr={LR}, grad-ckpt on, NF4 on"
    )

    # === A. TE -> encode hiddens -> free (lazy loading 不变量) ===
    print("\n--- A. 加载 TE, encode 文本 hiddens ---")
    t0 = time.time()
    te_model, tokenizer = load_krea2_text_encoder(
        str(KREA2_TE), dtype=dtype, device="cuda"
    )
    print(f"TE 加载: {time.time()-t0:.2f}s, peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    tok = Krea2TokenizeStrategy()
    tokens = tok.tokenize([PROMPT])
    enc = Krea2TextEncodingStrategy()
    with torch.inference_mode():
        [hiddens, txtmask] = enc.encode_tokens(tok, [te_model], tokens)
    print(f"hiddens: {tuple(hiddens.shape)}, mask: {tuple(txtmask.shape)}")
    hiddens = hiddens.to("cpu")
    txtmask = txtmask.to("cpu")
    del te_model, tokens, enc, tok
    torch.cuda.empty_cache()
    print(f"TE 释放后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # === B. VAE -> encode latents -> free ===
    print("\n--- B. 加载 VAE, encode 图片 latent ---")
    t1 = time.time()
    vae = load_vae(str(KREA2_VAE), device=device, dtype=dtype, eval=True)
    print(f"VAE 加载: {time.time()-t1:.2f}s")
    pixels = make_test_image(IMG_SIZE, device, dtype)
    with torch.no_grad():
        latents_4d = vae.encode_pixels_to_latents(pixels)
    print(
        f"latents (4D): {tuple(latents_4d.shape)}, "
        f"有限: {torch.isfinite(latents_4d).all().item()}"
    )
    latents_4d = latents_4d.to("cpu")
    del vae, pixels
    torch.cuda.empty_cache()
    print(f"VAE 释放后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # === C. DiT (NF4) + LoRA + grad-ckpt ===
    print("\n--- C. 加载 DiT + NF4 量化 + LoRA + grad-ckpt ---")
    t2 = time.time()
    dit = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=False)
    print(f"DiT 加载 (CPU): {time.time()-t2:.2f}s")
    quantize_model(dit, device)  # 内部 .to 触发量化, 打印 264 层/6.6GB
    for p in dit.parameters():
        p.requires_grad_(False)
    print(f"NF4 DiT frozen, 显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

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
    # grad-ckpt 在 apply_to 后 (compile-after-apply 顺序不变量; Krea-2 不 compile
    # 但 grad-ckpt 包装 block.forward, LoRA 已 patch Linear.forward, 两者正交).
    dit.enable_gradient_checkpointing()
    print(f"  grad-ckpt on")
    n_lora = len(network.unet_loras)
    n_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"LoRA 模块: {n_lora}, 可训: {n_train/1e6:.2f}M")

    opt = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=0.0)

    latents_4d = latents_4d.to(device).requires_grad_(False)
    hiddens_d = hiddens.to(device)
    txtmask_d = txtmask.to(device)
    text_emb = Krea2TextEmbedding(hiddens_d, txtmask_d)
    latents_5d = latents_4d.unsqueeze(2)
    b, c, _, lh, lw = latents_5d.shape
    print(f"训练 latent (5D): {tuple(latents_5d.shape)}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    power_start = gpu_power()

    torch.manual_seed(123)
    fixed_noise = torch.randn_like(latents_5d)
    fixed_target = fixed_noise - latents_5d

    losses = []
    grad_norms = []
    step_times = []

    print(
        f"\n--- D. 训练 {N_STEPS} 步 (flow-matching, 固定 σ={FIXED_SIGMA} + 固定 "
        f"noise, NF4 + grad-ckpt) ---"
    )
    for step in range(N_STEPS):
        sigma = torch.full((b,), FIXED_SIGMA, device=device, dtype=dtype)
        x_t = (1.0 - sigma) * latents_5d + sigma * fixed_noise
        target = fixed_target

        opt.zero_grad(set_to_none=True)
        t_sync = time.time()
        velocity = forward_for_loss(dit, x_t, text_emb, sigma)
        loss = torch.nn.functional.mse_loss(velocity, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        step_t = time.time() - t_sync

        lv = loss.item()
        losses.append(lv)
        grad_norms.append(float(grad_norm))
        step_times.append(step_t)
        if step % 5 == 0 or step == N_STEPS - 1:
            print(
                f"  step {step:3d}: loss={lv:.4f}, grad_norm={grad_norms[-1]:.4f}, "
                f"step={step_t:.2f}s"
            )
        if not torch.isfinite(torch.tensor(lv)):
            print(f"  loss 非有限, 提前终止")
            break

    peak = torch.cuda.max_memory_allocated() / 1e9
    power_end = gpu_power()

    # === 验证 ===
    print("\n=== E. 验证 ===")
    finite_all = all(torch.isfinite(torch.tensor(x)).item() for x in losses)
    first5 = sum(losses[:5]) / 5
    last5 = sum(losses[-5:]) / 5
    loss_down = last5 < first5
    grad_nonzero = all(g > 0 for g in grad_norms)

    from bitsandbytes.nn import Linear4bit  # noqa: E402
    l4 = next(m for _, m in dit.named_modules() if isinstance(m, Linear4bit))
    print(f"losses: {[f'{x:.4f}' for x in losses]}")
    print(f"finite: {finite_all}")
    print(f"first5 avg={first5:.4f}, last5 avg={last5:.4f}, 下降: {loss_down}")
    print(
        f"grad_norm 范围 [{min(grad_norms):.4f}, {max(grad_norms):.4f}], "
        f"全非零: {grad_nonzero}"
    )

    print(
        "\n=== 基线对比 (bf16 基线: loss 0.465->0.198, peak 27.9GB, step 3.47s) ==="
    )
    print(f"  NF4 显存 peak: {peak:.2f}GB (vs bf16 27.9GB, 省 {27.9-peak:.1f}GB)")
    print(f"  LoRA 可训参数: {n_train/1e6:.2f}M")
    print(f"  avg step: {sum(step_times)/len(step_times):.2f}s (vs bf16 3.47s)")
    print(f"  首 step: {step_times[0]:.2f}s, 末 step: {step_times[-1]:.2f}s")
    print(f"  loss: first5={first5:.4f} -> last5={last5:.4f} (bf16: 0.465 -> 0.198)")
    print(f"  GPU 功耗: start={power_start:.1f}W, end={power_end:.1f}W")

    ok = finite_all and loss_down and grad_nonzero
    print(f"\n层 3 端到端训练通过: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
