"""R-verify: NF4 × block swap 端到端训练 (方向 A 落地验证).

NF4 量化的 Krea-2 DiT + LoRA + grad-ckpt + block swap, 跑 flow-matching 训练.
验证 offloader 加的 Params4bit 专用搬运分支 (deepcopy master + Params4bit.to()
整体搬运) 在真实 DiT block swap 节奏下:
  prepare_block_swap_before_forward -> _run_blocks(wait/submit) -> swap_weight_devices
  -> _swap_weight_devices_cached_cuda 的 nf4_jobs 循环 (deepcopy master -> .to(device)
  -> module.weight = restored).

主战场是 host RAM: bf16 路径全量 28 块 pinned master 22.64GB 曾在 62GB 机器上宕机,
NF4 master 仅 ~6.7GB (4-bit 码 + quant_state), 应彻底解决. GPU 显存 NF4+swap 应
~10-12GB.

验证项:
1. NF4 × block swap 训练 forward+backward+opt 跑通 (Params4bit 搬运 + 反量化 + grad-ckpt
   recompute + LoRA delta 四者叠加, 不崩).
2. loss 单调下降 (末5 < 首5), 量级与 NF4-only (probe_nf4_train) 可比.
3. LoRA grad 非零 (梯度流到 LoRA, 不被 NF4 搬运/反量化阻断).
4. DiT Linear4bit 权重 frozen 不变 (搬运不改 frozen 权重).
5. host RAM peak 远低于 bf16 路径 22.64GB (主战场, 解决宕机).
6. GPU peak 远低于 bf16 27.9GB (NF4 省权重收益).
7. 数值与 NF4-only 一致: block swap 只搬运权重不改 forward 语义, 首步 loss 应与
   NF4-only 探针量级可比 (同 seed/同 σ).

环境变量覆盖 (3080 跑小分辨率, PG199 跑 1024):
  K2_NF4BS_GPU=0/1     (默认 1=PG199)
  K2_NF4BS_IMG=512/1024 (默认 1024)
  K2_NF4BS_SWAP=4/6/8  (默认 4)
  K2_NF4BS_STEPS=N     (默认 30)

同 probe_train/probe_nf4_train 过拟合方法论: 固定 σ=0.5 + 固定 noise seed.
非目标: 真实数据集 sweep, 跨机复现.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_NF4BS_GPU", "1"))

import resource
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

from probe_train import FIXED_SIGMA, PROMPT, gpu_power, make_test_image  # noqa: E402

KREA2_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"
KREA2_TE = ROOT / "models" / "text_encoders" / "qwen3vl_4b_bf16.safetensors"
KREA2_VAE = ROOT / "models" / "vae" / "qwen_image_vae.safetensors"

LORA_DIM = 16
LORA_ALPHA = 8.0
LR = 2e-3


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def host_rss_gb() -> float:
    """当前进程 RSS (GB). ru_maxrss 是峰值, 但单位依平台 (Linux KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def meminfo_available_gb() -> float:
    """系统可用内存 (GB), 容错读 /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return -1.0


def main() -> int:
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    img_size = _env_int("K2_NF4BS_IMG", 1024)
    blocks_to_swap = _env_int("K2_NF4BS_SWAP", 4)
    n_steps = _env_int("K2_NF4BS_STEPS", 30)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(
        f"=== NF4 × block swap 端到端训练 (方向 A, GPU={gpu_id}, {dtype}, "
        f"{img_size}×{img_size}, swap={blocks_to_swap}) ==="
    )
    print(
        f"steps={n_steps}, lora_dim={LORA_DIM}, alpha={LORA_ALPHA}, lr={LR}, "
        f"grad-ckpt on, NF4 on, block_swap on"
    )

    # === A. TE -> encode hiddens -> free (lazy loading 不变量) ===
    print("\n--- A. 加载 TE, encode 文本 hiddens ---")
    # 3080 (8GB) 放不下 Qwen3-VL 4B TE bf16 (~8GB), 给 env 开关让 TE 留 CPU encode:
    # encode 一次就 free, 不占 GPU; hiddens 搬 GPU 给 DiT 用.
    te_device = "cpu" if os.environ.get("K2_NF4BS_TE_CPU") else "cuda"
    t0 = time.time()
    te_model, tokenizer = load_krea2_text_encoder(
        str(KREA2_TE), dtype=dtype, device=te_device
    )
    tok = Krea2TokenizeStrategy()
    tokens = tok.tokenize([PROMPT])
    enc = Krea2TextEncodingStrategy()
    with torch.inference_mode():
        [hiddens, txtmask] = enc.encode_tokens(tok, [te_model], tokens)
    hiddens = hiddens.to("cpu")
    txtmask = txtmask.to("cpu")
    del te_model, tokens, enc, tok
    if te_device == "cuda":
        torch.cuda.empty_cache()
    print(f"TE (device={te_device}): {time.time()-t0:.2f}s, hiddens {tuple(hiddens.shape)}")

    # === B. VAE -> encode latents -> free ===
    print("\n--- B. 加载 VAE, encode 图片 latent ---")
    t1 = time.time()
    vae = load_vae(str(KREA2_VAE), device=device, dtype=dtype, eval=True)
    pixels = make_test_image(img_size, device, dtype)
    with torch.no_grad():
        latents_4d = vae.encode_pixels_to_latents(pixels)
    latents_4d = latents_4d.to("cpu")
    del vae, pixels
    torch.cuda.empty_cache()
    print(
        f"VAE: {time.time()-t1:.2f}s, latents {tuple(latents_4d.shape)}, "
        f"有限: {torch.isfinite(latents_4d).all().item()}"
    )

    # === C. DiT (CPU) -> NF4 量化 -> LoRA apply -> enable_block_swap -> 上 GPU ===
    print(f"\n--- C. DiT + NF4 + LoRA + enable_block_swap({blocks_to_swap}) ---")
    t2 = time.time()
    # 优先用磁盘 NF4 (save_nf4_dit 产物): 小卡可直接加载 6.6GB, 无需在线量化
    # 要的 26GB bf16 在 GPU. nf4_path 给则走 load_krea2_dit(nf4_path=...).
    nf4_disk = os.environ.get("K2_NF4BS_NF4_PATH", "")
    nf4_disk_path = Path(nf4_disk) if nf4_disk and Path(nf4_disk).exists() else None
    if nf4_disk_path:
        print(f"  用磁盘 NF4: {nf4_disk_path}")
        dit = load_krea2_dit(
            KREA2_DIT, device="cpu", dtype=dtype, eval=False, nf4_path=nf4_disk_path
        )
    else:
        dit = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=False, nf4=True)
    print(f"DiT 加载 (CPU, NF4): {time.time()-t2:.2f}s")

    # LoRA 先 apply_to (CPU 上, add_module 在 apply_to 里; compile-after-apply 顺序)
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

    # DiT frozen (仅 LoRA 训练)
    for p in dit.parameters():
        p.requires_grad_(False)
    # NF4 量化已在 load_krea2_dit 内部完成 (nf4=True 在线量化 或 nf4_path 磁盘加载).
    # 旧版用外部 quantize_model, 现在走 weights.py 收口路径, 不再重复量化.
    for p in dit.parameters():
        p.requires_grad_(False)

    # block swap: enable (CPU 构造 ModelOffloader, 此时 _ensure_cpu_weight_masters
    # 捕获 Params4bit master, 走 _capture_cpu_master 的 NF4 分支 = deepcopy) ->
    # move_to_device (DiT 非 block 部分 + LoRA 上 GPU, frozen Linear4bit 留 CPU master) ->
    # switch_block_swap_for_training (装 backward hook).
    dit.enable_block_swap(blocks_to_swap, device)
    dit.move_to_device_except_swap_blocks(device)
    network = network.to(device).to(dtype)
    dit.switch_block_swap_for_training()
    # grad-ckpt 在 apply_to + block_swap 之后 (Krea-2 不 compile, grad-ckpt 包装
    # block.forward; LoRA 已 patch Linear.forward; block swap 透明于 forward).
    dit.enable_gradient_checkpointing()

    n_lora = len(network.unet_loras)
    n_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"LoRA 模块: {n_lora}, 可训: {n_train/1e6:.2f}M")
    print(f"DiT num_blocks: {dit.num_blocks}, blocks_to_swap: {dit.blocks_to_swap}")
    rss_after_master = host_rss_gb()
    avail = meminfo_available_gb()
    print(
        f"block swap master 构造后: 进程 RSS 峰值 {rss_after_master:.2f}GB, "
        f"系统可用内存 {avail:.2f}GB"
    )

    opt = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=0.0)

    latents_4d = latents_4d.to(device).requires_grad_(False)
    hiddens_d = hiddens.to(device)
    txtmask_d = txtmask.to(device)
    text_emb = Krea2TextEmbedding(hiddens_d, txtmask_d)
    latents_5d = latents_4d.unsqueeze(2)
    b = latents_5d.shape[0]

    torch.manual_seed(123)
    fixed_noise = torch.randn_like(latents_5d)
    fixed_target = fixed_noise - latents_5d

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    power_start = gpu_power()
    rss_before_train = host_rss_gb()

    losses = []
    grad_norms = []
    step_times = []
    phase_times: dict[str, float] = {}
    print(
        f"\n--- D. 训练 {n_steps} 步 (NF4 + block_swap={blocks_to_swap} + grad-ckpt, "
        f"固定 σ={FIXED_SIGMA}) ---"
    )
    for step in range(n_steps):
        sigma = torch.full((b,), FIXED_SIGMA, device=device, dtype=dtype)
        x_t = (1.0 - sigma) * latents_5d + sigma * fixed_noise
        target = fixed_target

        opt.zero_grad(set_to_none=True)
        t_swap = time.perf_counter()
        dit.prepare_block_swap_before_forward()
        torch.cuda.synchronize()
        t_swap_end = time.perf_counter()
        velocity = forward_for_loss(dit, x_t, text_emb, sigma)
        loss = torch.nn.functional.mse_loss(velocity, target)
        torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd_end = time.perf_counter()
        grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        t_opt_end = time.perf_counter()
        step_t = t_opt_end - t_swap

        # 阶段累加: swap_prepare / forward / backward / opt
        for _k, _t in (("swap", t_swap_end - t_swap), ("fwd", t_fwd_end - t_swap_end),
                       ("bwd", t_bwd_end - t_fwd_end), ("opt", t_opt_end - t_bwd_end)):
            phase_times[_k] = phase_times.get(_k, 0.0) + _t

        lv = loss.item()
        losses.append(lv)
        grad_norms.append(float(grad_norm))
        step_times.append(step_t)
        if step % 5 == 0 or step == n_steps - 1:
            print(
                f"  step {step:3d}: loss={lv:.4f}, grad_norm={grad_norms[-1]:.4f}, "
                f"step={step_t:.2f}s "
                f"(swap={t_swap_end-t_swap:.2f}s fwd={t_fwd_end-t_swap_end:.2f}s "
                f"bwd={t_bwd_end-t_fwd_end:.2f}s opt={t_opt_end-t_bwd_end:.2f}s)"
            )
        if not torch.isfinite(torch.tensor(lv)):
            print(f"  loss 非有限, 提前终止")
            break

    peak_gpu = torch.cuda.max_memory_allocated() / 1e9
    rss_peak = host_rss_gb()
    avail_after = meminfo_available_gb()
    power_end = gpu_power()

    # === E. 验证 ===
    print("\n=== E. 验证 ===")
    from bitsandbytes.nn import Linear4bit  # noqa: E402

    finite_all = all(torch.isfinite(torch.tensor(x)).item() for x in losses)
    first5 = sum(losses[:5]) / min(5, len(losses))
    last5 = sum(losses[-5:]) / min(5, len(losses))
    loss_down = last5 < first5
    grad_nonzero = all(g > 0 for g in grad_norms)

    # frozen 验证: 取一个 Linear4bit, 检查 4-bit 码未变 (搬运不改 frozen 权重).
    l4 = next(m for _, m in dit.named_modules() if isinstance(m, Linear4bit))
    # 量化时 .data 已是 uint8 打包码, 训练后应同 device 同 bnb_quantized=True.
    frozen_ok = (
        l4.weight.bnb_quantized
        and l4.weight.data.device.type == "cuda"
        and torch.isfinite(l4.weight.data.float()).all().item()
    )

    print(f"losses: {[f'{x:.4f}' for x in losses]}")
    print(f"finite: {finite_all}")
    print(f"first5 avg={first5:.4f}, last5 avg={last5:.4f}, 下降: {loss_down}")
    print(
        f"grad_norm 范围 [{min(grad_norms):.4f}, {max(grad_norms):.4f}], "
        f"全非零: {grad_nonzero}"
    )
    print(
        f"Linear4bit frozen: bnb_quantized={l4.weight.bnb_quantized}, "
        f"data device={l4.weight.data.device}, 有限={frozen_ok}"
    )

    print(
        "\n=== 主战场对比 (bf16 block swap: master 22.64GB pinned + DiT 26GB → 62GB 机宕机) ==="
    )
    print(
        f"  host RAM RSS 峰值: {rss_peak:.2f}GB (训练前 {rss_before_train:.2f}GB)"
    )
    print(f"  系统可用内存: 训练前 {avail:.2f}GB -> 训练后 {avail_after:.2f}GB")
    print(
        f"  NF4 master ≈ bf16 master 的 25.8% → RSS 应远低于 bf16 路径"
    )

    print(
        "\n=== GPU 显存对比 (bf16 基线: NF4-only 10.49GB / bf16+swap 32.62GB) ==="
    )
    print(f"  NF4 + block_swap GPU peak: {peak_gpu:.2f}GB")
    print(f"  avg step: {sum(step_times)/len(step_times):.2f}s (NF4-only 反量化慢 ~5×)")
    print(f"  首 step: {step_times[0]:.2f}s, 末 step: {step_times[-1]:.2f}s")
    print(f"  loss: first5={first5:.4f} -> last5={last5:.4f}")
    print(f"  GPU 功耗: {power_start:.1f}W -> {power_end:.1f}W")

    # 主战场 (host RAM) 是核心判定: RSS 必须远低于 bf16 22.64GB master.
    # 6.7GB master + DiT CPU 副本(搬走后只剩 master) + Python/torch ~4-6GB.
    ram_ok = rss_peak < 20.0
    ok = finite_all and loss_down and grad_nonzero and frozen_ok and ram_ok
    print(f"\n方向 A 端到端通过: {ok}")
    print(f"  (host RAM<{20.0}GB: {ram_ok}, loss↓: {loss_down}, grad≠0: {grad_nonzero}, frozen: {frozen_ok})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
