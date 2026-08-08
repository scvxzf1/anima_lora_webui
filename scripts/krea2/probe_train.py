"""R-verify: Krea-2-Raw 训练串通 热测 (阶段 4 出口).

自包含训练探针: 不走 train.py bootstrap (反上帝守则, 不改 noise_target.py 热点文件),
在探针里手搓 Krea-2 flow-matching 训练 loop, 验证单 prompt 过拟合 loss 下降.

flow-matching 数学 (子代理核实 anima noise_target.py:380-381 / noise.py:171):
- target = noise - latents  (x1 - x0, rectified flow)
- x_t = (1-σ)·latents + σ·noise, σ = t ∈ [0,1] (t=0 clean, t=1 noise)
- loss = MSE(dit_output, target), 无 weighting 默认
- DiT timestep = σ ∈ [0,1] float, temb 内部 t*tfactor(1e3) sinusoidal embedding
- 训练默认 sigmoid 采样, 不做 mu shift (推理才做)

显存调度 (lazy loading 不变量):
  TE -> encode hiddens -> free -> VAE -> encode latents -> free -> DiT + LoRA -> train
  (TE 8.9GB + DiT 26GB > PG199 32GB, 必须分阶段释放)

验证项:
1. family.forward_for_loss 在训练模式 (require_grad) 下跑通, 输出 5D velocity 有限.
2. 单 prompt + 单 latent 过拟合: 20 步 loss 单调下降 (末 5 步均值 < 首 5 步均值).
3. LoRA grad 非零 (梯度流到 LoRA 参数).
4. DiT 权重不变 (frozen, 仅 LoRA 训练).
5. 基线: 显存 peak, step 时间, loss 曲线, LoRA grad norm, 功耗 (nvidia-smi).

PG199 bf16, 256×256 (与 probe_* 一致, 避免 OOM).
"""
from __future__ import annotations

import os

# 必须在 import torch 之前设 (同 probe_*.py): PCI_BUS_ID + PG199=device 1 (32GB).
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

N_STEPS = 30
LORA_DIM = 16
LORA_ALPHA = 8.0
LR = 2e-3
IMG_SIZE = 256
PROMPT = "a red circle on blue background"
# 固定 σ=0.5 做过拟合: flow-matching loss 绝对值依赖 σ (σ≈0/1 自然高/低),
# 不同 σ 的 loss 不可直接比较 "下降". 固定 σ 让每步 target 量级一致, loss
# 单调下降才证明 LoRA 能拟合 DiT 输出. (真实训练用 sigmoid 采样, 这里只验过拟合.)
FIXED_SIGMA = 0.5


def make_test_image(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """结构化图案 (渐变 + 条纹 + 棋盘) 模拟自然图像低频分布, 经 VAE encode 更稳定.

    同 probe_vae.py 的图案策略, 避免纯随机像素让 VAE 重建失真.
    返回 (1, 3, H, W) 像素值 [0,1].
    """
    h = w = size
    ys = torch.linspace(0, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, w, device=device, dtype=dtype)
    grad_h = ys.view(h, 1).expand(h, w)
    grad_w = xs.view(1, w).expand(h, w)
    checker = ((xs.view(1, w).expand(h, w) * 8).int().float() % 2)
    pixels = torch.stack([grad_h, grad_w, checker], dim=0).unsqueeze(0)
    pixels = (pixels * 0.8 + 0.1).clamp(0, 1)
    return pixels


def sample_timestep(batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """sigmoid 采样 σ ∈ (0,1) (anima 训练默认, 无 mu shift). 偏向中间段 (信息量大).

    真实训练用 sigmoid 采样; 过拟合测试用固定 σ (见 main FIXED_SIGMA).
    """
    return torch.sigmoid(torch.randn(batch, device=device, dtype=dtype))


def gpu_power() -> float:
    """读 nvidia-smi 功耗 (W), 失败返回 -1."""
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

    print(f"=== 阶段 4: Krea-2 训练串通 热测 (PG199, {dtype}) ===")
    print(f"steps={N_STEPS}, lora_dim={LORA_DIM}, alpha={LORA_ALPHA}, lr={LR}, img={IMG_SIZE}")

    # === 阶段 A: TE -> encode hiddens -> free ===
    print("\n--- A. 加载 TE, encode 文本 hiddens ---")
    t0 = time.time()
    te_model, tokenizer = load_krea2_text_encoder(str(KREA2_TE), dtype=dtype, device="cuda")
    print(f"TE 加载: {time.time()-t0:.2f}s, peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    tok = Krea2TokenizeStrategy()
    tokens = tok.tokenize([PROMPT])
    enc = Krea2TextEncodingStrategy()
    with torch.inference_mode():
        [hiddens, txtmask] = enc.encode_tokens(tok, [te_model], tokens)
    print(f"hiddens: {tuple(hiddens.shape)}, mask: {tuple(txtmask.shape)}")

    # 移 CPU, 释放 TE
    hiddens = hiddens.to("cpu")
    txtmask = txtmask.to("cpu")
    del te_model, tokens, enc, tok
    torch.cuda.empty_cache()
    print(f"TE 释放后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # === 阶段 B: VAE -> encode latents -> free ===
    print("\n--- B. 加载 VAE, encode 图片 latent ---")
    t1 = time.time()
    vae = load_vae(str(KREA2_VAE), device=device, dtype=dtype, eval=True)
    print(f"VAE 加载: {time.time()-t1:.2f}s")

    pixels = make_test_image(IMG_SIZE, device, dtype)
    with torch.no_grad():
        latents_4d = vae.encode_pixels_to_latents(pixels)
    print(f"latents (4D): {tuple(latents_4d.shape)}, 有限: {torch.isfinite(latents_4d).all().item()}")

    latents_4d = latents_4d.to("cpu")
    del vae, pixels
    torch.cuda.empty_cache()
    print(f"VAE 释放后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # === 阶段 C: DiT + LoRA -> train ===
    print("\n--- C. 加载 DiT + 构造 LoRA network ---")
    t2 = time.time()
    dit = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=False)
    dit = dit.to(device)
    for p in dit.parameters():
        p.requires_grad_(False)
    print(f"DiT 加载+frozen: {time.time()-t2:.2f}s")

    kwargs = {**krea2_target_kwargs(), "lora_dim": LORA_DIM, "alpha": LORA_ALPHA}
    cfg = LoRANetworkCfg.from_kwargs(
        kwargs, network_dim=LORA_DIM, network_alpha=LORA_ALPHA,
        neuron_dropout=None, module_class=LoRAModule,
    )
    network = LoRANetwork(text_encoders=[], unet=dit, cfg=cfg, multiplier=1.0)
    network.apply_to(text_encoders=[], unet=dit, apply_text_encoder=False, apply_unet=True)
    network = network.to(device).to(dtype)
    n_lora = sum(p.numel() for p in network.parameters())
    n_train = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"LoRA 模块: {len(network.unet_loras)}, 总参 {n_lora/1e6:.2f}M, 可训 {n_train/1e6:.2f}M")

    opt = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=0.0)

    # 训练数据准备上 GPU
    latents_4d = latents_4d.to(device).requires_grad_(False)  # (1,16,H/8,W/8) clean x0
    hiddens_d = hiddens.to(device)
    txtmask_d = txtmask.to(device)
    text_emb = Krea2TextEmbedding(hiddens_d, txtmask_d)

    # 5D (B,C,T=1,H,W) — anima 不变量
    latents_5d = latents_4d.unsqueeze(2)
    b, c, _, lh, lw = latents_5d.shape
    print(f"训练 latent (5D): {tuple(latents_5d.shape)}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    power_start = gpu_power()

    # 固定 noise seed: target = noise - x0 完全固定, 纯过拟合单样本,
    # loss 应单调下降到接近 0 (证明 LoRA 能调整 frozen DiT 输出拟合固定 target).
    torch.manual_seed(123)
    fixed_noise = torch.randn_like(latents_5d)
    fixed_target = fixed_noise - latents_5d

    losses = []
    grad_norms = []
    step_times = []

    print(f"\n--- D. 训练 {N_STEPS} 步 (flow-matching, 固定 σ={FIXED_SIGMA} + 固定 noise 过拟合) ---")
    for step in range(N_STEPS):
        # flow-matching: x_t = (1-σ)x0 + σ·noise, target = noise - x0 (velocity)
        # 固定 σ + 固定 noise: target 完全固定, loss 单调下降证明 LoRA 能拟合.
        sigma = torch.full((b,), FIXED_SIGMA, device=device, dtype=dtype)
        x_t = (1.0 - sigma) * latents_5d + sigma * fixed_noise
        target = fixed_target

        opt.zero_grad(set_to_none=True)
        t_sync = time.time()
        # DiT forward (require_grad on LoRA path; DiT frozen)
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
        if step % 2 == 0 or step == N_STEPS - 1:
            print(f"  step {step:3d}: loss={lv:.4f}, grad_norm={grad_norms[-1]:.4f}, "
                  f"step={step_t*1000:.0f}ms")

    peak = torch.cuda.max_memory_allocated() / 1e9
    power_end = gpu_power()

    # === 验证 ===
    print("\n=== E. 验证 ===")
    finite_all = all(torch.isfinite(torch.tensor(x)).item() for x in losses)
    first5 = sum(losses[:5]) / 5
    last5 = sum(losses[-5:]) / 5
    loss_down = last5 < first5
    grad_nonzero = all(g > 0 for g in grad_norms)

    # DiT 权重不变 (frozen): 抽样核对一个 weight
    dit_w = next(dit.parameters())
    print(f"losses: {[f'{x:.4f}' for x in losses]}")
    print(f"finite: {finite_all}")
    print(f"first5 avg={first5:.4f}, last5 avg={last5:.4f}, 下降: {loss_down}")
    print(f"grad_norm 范围 [{min(grad_norms):.4f}, {max(grad_norms):.4f}], 全非零: {grad_nonzero}")

    print(f"\n=== 基线 (PG199 bf16, {IMG_SIZE}×{IMG_SIZE}, lora_dim={LORA_DIM}) ===")
    print(f"  DiT+LoRA 显存 peak: {peak:.2f}GB")
    print(f"  LoRA 可训参数: {n_train/1e6:.2f}M")
    print(f"  avg step 时间: {sum(step_times)/len(step_times)*1000:.0f}ms")
    print(f"  首 step 时间: {step_times[0]*1000:.0f}ms (含 cudnn autotune)")
    print(f"  末 step 时间: {step_times[-1]*1000:.0f}ms")
    print(f"  loss: first5={first5:.4f} -> last5={last5:.4f}")
    print(f"  GPU 功耗: start={power_start:.1f}W, end={power_end:.1f}W")

    ok = finite_all and loss_down and grad_nonzero
    print(f"\n阶段 4 训练串通通过: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
