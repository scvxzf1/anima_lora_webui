"""R-verify: Krea-2-Raw 块交换 稳态对照探针 (审计 5× 开销归因).

对照 scripts/krea2/probe_blockswap.py 的三个放大源, 测真实训练稳态路径:
  1. prepare_block_swap_before_forward(free_cache=False) — 对齐
     library/training/unet_prepare.py:36 的真实训练路径 (默认 free_cache=True
     多测每步 gc.collect()+empty_cache(), offloading.py:1718-1720).
  2. enable_block_swap(..., profile_jsonl=None) — 确保 profiler 关闭
     (offloading.py:_resolve_profiler None 时无 profile poller 线程/GIL 争用).
  3. 预热 WARMUP 步后只测稳态 N_STEPS 步 avg step, 排除 cudnn autotune 首步.

假设: 探针报告的 5× (1965ms vs 400ms) 含 free_cache=True 放大; 真实稳态路径
增量预计 1.5-2× (H2D 串行化 + depth=1 + 单 worker, 256×256 forward 太快重叠窗口小).

验证项:
1. 稳态 avg step < 探针 1965ms (证 free_cache=True 是放大源之一).
2. 显存 peak 与探针同量级 (free_cache=False 不改 peak, 只改 caching 抖动).
3. loss 下降 (训练语义未变, free_cache 不影响数值).
4. 打印 host_wait_ms/transfer_ms (若 offloader 暴露) 做归因佐证.

PG199 bf16, 256×256, swap 4 blocks (同 probe_blockswap.py, 控制变量).
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

from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402
from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402
from library.models.krea2_raw.strategy import (  # noqa: E402
    Krea2TextEncodingStrategy,
    Krea2TokenizeStrategy,
    load_krea2_text_encoder,
)
from library.models.krea2_raw.family import Krea2TextEmbedding, forward_for_loss  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402
from library.models.krea2_raw.lora_targets import krea2_target_kwargs  # noqa: E402
from networks.lora_anima.config import LoRANetworkCfg  # noqa: E402
from networks.lora_anima.network import LoRANetwork  # noqa: E402
from networks.lora_modules.lora import LoRAModule  # noqa: E402

KREA2_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"
KREA2_TE = ROOT / "models" / "text_encoders" / "qwen3vl_4b_bf16.safetensors"
KREA2_VAE = ROOT / "models" / "vae" / "qwen_image_vae.safetensors"

WARMUP = 3
N_STEPS = 15
LORA_DIM = 16
LORA_ALPHA = 8.0
LR = 2e-3
IMG_SIZE = 256
PROMPT = "a red circle on blue background"
BLOCKS_TO_SWAP = 4
FIXED_SIGMA = 0.5


def gpu_power() -> float:
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-i", "1"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
        return float(out.strip())
    except Exception:
        return -1.0


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"=== Krea-2 block swap 稳态对照 (PG199, {dtype}, free_cache=False) ===")
    print(f"warmup={WARMUP}, steady={N_STEPS}, swap={BLOCKS_TO_SWAP} blocks, img={IMG_SIZE}")

    # === A. TE -> encode hiddens -> free ===
    print("\n--- A. 加载 TE, encode 文本 hiddens ---")
    t0 = time.time()
    te_model, tokenizer = load_krea2_text_encoder(str(KREA2_TE), dtype=dtype, device="cuda")
    tok = Krea2TokenizeStrategy()
    tokens = tok.tokenize([PROMPT])
    enc = Krea2TextEncodingStrategy()
    with torch.inference_mode():
        [hiddens, txtmask] = enc.encode_tokens(tok, [te_model], tokens)
    hiddens = hiddens.to("cpu"); txtmask = txtmask.to("cpu")
    del te_model, tokens, enc, tok
    torch.cuda.empty_cache()
    print(f"TE: {time.time()-t0:.2f}s, hiddens {tuple(hiddens.shape)}")

    # === B. VAE -> encode latents -> free ===
    print("\n--- B. 加载 VAE, encode 图片 latent ---")
    vae = load_vae(str(KREA2_VAE), device=device, dtype=dtype, eval=True)
    ys = torch.linspace(0, 1, IMG_SIZE, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, IMG_SIZE, device=device, dtype=dtype)
    pixels = torch.stack([ys.view(-1,1).expand(IMG_SIZE,IMG_SIZE),
                          xs.view(1,-1).expand(IMG_SIZE,IMG_SIZE),
                          ((xs.view(1,-1).expand(IMG_SIZE,IMG_SIZE)*8).int().float()%2)], dim=0).unsqueeze(0)
    pixels = (pixels * 0.8 + 0.1).clamp(0, 1)
    with torch.no_grad():
        latents_4d = vae.encode_pixels_to_latents(pixels)
    latents_4d = latents_4d.to("cpu")
    del vae, pixels
    torch.cuda.empty_cache()
    print(f"latents: {tuple(latents_4d.shape)}")

    # === C. DiT + LoRA + block swap (free_cache=False 路径, profile 关) ===
    print(f"\n--- C. 加载 DiT (CPU) + LoRA + enable_block_swap({BLOCKS_TO_SWAP}, free_cache=False) ---")
    dit = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=False)

    kwargs = {**krea2_target_kwargs(), "lora_dim": LORA_DIM, "alpha": LORA_ALPHA}
    cfg = LoRANetworkCfg.from_kwargs(
        kwargs, network_dim=LORA_DIM, network_alpha=LORA_ALPHA,
        neuron_dropout=None, module_class=LoRAModule,
    )
    network = LoRANetwork(text_encoders=[], unet=dit, cfg=cfg, multiplier=1.0)
    network.apply_to(text_encoders=[], unet=dit, apply_text_encoder=False, apply_unet=True)

    for p in dit.parameters():
        p.requires_grad_(False)

    # free_cache=False 对照: 显式 profile_jsonl=None (profiler 关)
    dit.enable_block_swap(BLOCKS_TO_SWAP, device, profile_jsonl=None)
    dit.move_to_device_except_swap_blocks(device)
    network = network.to(device).to(dtype)
    dit.switch_block_swap_for_training()

    n_lora = sum(p.numel() for p in network.parameters())
    print(f"LoRA 模块: {len(network.unet_loras)}, 参数 {n_lora/1e6:.2f}M")
    print(f"DiT num_blocks: {dit.num_blocks}, blocks_to_swap: {dit.blocks_to_swap}")
    print(f"offloader forward_only: {dit.offloader.forward_only}, supports_backward: {dit.offloader.supports_backward}")

    opt = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=0.0)

    latents_4d = latents_4d.to(device)
    hiddens_d = hiddens.to(device)
    txtmask_d = txtmask.to(device)
    text_emb = Krea2TextEmbedding(hiddens_d, txtmask_d)
    latents_5d = latents_4d.unsqueeze(2)
    b = latents_5d.shape[0]

    torch.manual_seed(123)
    fixed_noise = torch.randn_like(latents_5d)
    fixed_target = fixed_noise - latents_5d

    # === D1. 预热 (cudnn autotune + block swap 首步建图) ===
    print(f"\n--- D1. 预热 {WARMUP} 步 (不计时) ---")
    for step in range(WARMUP):
        sigma = torch.full((b,), FIXED_SIGMA, device=device, dtype=dtype)
        x_t = (1.0 - sigma) * latents_5d + sigma * fixed_noise
        opt.zero_grad(set_to_none=True)
        dit.prepare_block_swap_before_forward(free_cache=False)
        velocity = forward_for_loss(dit, x_t, text_emb, sigma)
        loss = torch.nn.functional.mse_loss(velocity, fixed_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        print(f"  warmup {step}: loss={loss.item():.4f}")

    # 稳态测量窗口起点: reset peak
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    power_start = gpu_power()

    # === D2. 稳态计时 (free_cache=False) ===
    print(f"\n--- D2. 稳态 {N_STEPS} 步 (free_cache=False, 计时) ---")
    losses = []
    step_times = []
    for step in range(N_STEPS):
        sigma = torch.full((b,), FIXED_SIGMA, device=device, dtype=dtype)
        x_t = (1.0 - sigma) * latents_5d + sigma * fixed_noise

        opt.zero_grad(set_to_none=True)
        t_sync = time.time()
        dit.prepare_block_swap_before_forward(free_cache=False)
        velocity = forward_for_loss(dit, x_t, text_emb, sigma)
        loss = torch.nn.functional.mse_loss(velocity, fixed_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        step_t = time.time() - t_sync

        losses.append(loss.item())
        step_times.append(step_t)
        if step % 3 == 0 or step == N_STEPS - 1:
            print(f"  step {step:3d}: loss={losses[-1]:.4f}, step={step_t*1000:.0f}ms")

    peak = torch.cuda.max_memory_allocated() / 1e9
    power_end = gpu_power()

    # === E. 验证 ===
    print("\n=== E. 验证 ===")
    finite_all = all(torch.isfinite(torch.tensor(x)).item() for x in losses)
    first5 = sum(losses[:5]) / min(5, len(losses))
    last5 = sum(losses[-5:]) / min(5, len(losses))
    loss_down = last5 < first5

    print(f"losses: {[f'{x:.4f}' for x in losses]}")
    print(f"finite: {finite_all}")
    print(f"first5={first5:.4f}, last5={last5:.4f}, 下降: {loss_down}")

    avg_steady = sum(step_times) / len(step_times) * 1000
    min_steady = min(step_times) * 1000
    max_steady = max(step_times) * 1000

    print(f"\n=== 稳态对照基线 (PG199 bf16, 256×256, swap={BLOCKS_TO_SWAP}, free_cache=False) ===")
    print(f"  稳态 avg step: {avg_steady:.0f}ms (探针 free_cache=True 1965ms, 无 swap 400ms)")
    print(f"  稳态 min/max: {min_steady:.0f}ms / {max_steady:.0f}ms")
    print(f"  vs 探针 free_cache=True: {avg_steady/1965:.2f}× (探针)")
    print(f"  vs 无 swap 400ms: {avg_steady/400:.2f}× (无 swap)")
    print(f"  显存 peak: {peak:.2f}GB (探针 32.24GB, 无 swap 32.62GB)")
    print(f"  loss: first5={first5:.4f} -> last5={last5:.4f}")
    print(f"  GPU 功耗: {power_start:.1f}W -> {power_end:.1f}W")

    ok = finite_all and loss_down
    print(f"\n稳态对照通过 (训练语义正确): {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
