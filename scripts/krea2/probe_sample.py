"""R-verify: Krea-2-Raw 推理串通 热测 (阶段 5 出口).

自包含推理探针: 不走 generation.py (反上帝守则, 热点文件), 在探针里手搓
Krea-2 flow-matching Euler ODE + mu shift + CFG 采样, 验证能出图.

采样数学 (子代理核实 krea-ai/krea-2 sampling.py):
- Euler ODE, 28 步, img = img + (tprev - tcurr) * v
- mu shift: (x1=256,y1=0.5)/(x2=6400,y2=1.15), seq_len=纯图像 token 数
- CFG: v = cond + guidance*(cond - uncond), uncond=空字符串, cfg=4.5
- 初始 latent = randn, σ=1.0 隐含

显存调度 (lazy loading 不变量, 同 probe_train.py):
  TE -> encode cond + uncond hiddens -> free -> DiT (+LoRA) -> sample -> VAE decode -> 存图

验证项:
1. 28 步采样跑通, 输出 latent 有限.
2. VAE decode 出像素 (1,3,H,W) 范围 [0,1], 有限.
3. 存 PNG 到 output/tests/ (受 resolve_output_root 边界约束, 用 output/tests/ 固定).
4. CFG 对比: cfg=4.5 vs cfg=0 (无 CFG), 输出应有可见差异 (证明 CFG 起作用).
5. 基线: 显存 peak, 采样总耗时, 单 step 时间, 功耗.

PG199 bf16, 256×256 (无 block swap; 1024×1024 + block swap 留阶段 6).
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
from library.models.krea2_raw.sampling import sample  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402

KREA2_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"
KREA2_TE = ROOT / "models" / "text_encoders" / "qwen3vl_4b_bf16.safetensors"
KREA2_VAE = ROOT / "models" / "vae" / "qwen_image_vae.safetensors"

STEPS = 28
CFG = 4.5
IMG_SIZE = 256
PROMPT = "a red circle on blue background"
OUT_DIR = ROOT / "output" / "tests" / "krea2_stage5"


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


def encode_prompt(te_model, tokenizer, prompt: str):
    """encode 单 prompt -> (hiddens, mask) on GPU. 复用 strategy."""
    tok = Krea2TokenizeStrategy()
    tokens = tok.tokenize([prompt])
    enc = Krea2TextEncodingStrategy()
    with torch.inference_mode():
        [hiddens, mask] = enc.encode_tokens(tok, [te_model], tokens)
    return hiddens, mask


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"=== 阶段 5: Krea-2 推理串通 热测 (PG199, {dtype}) ===")
    print(f"steps={STEPS}, cfg={CFG}, img={IMG_SIZE}, prompt={PROMPT!r}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # === A. TE -> encode cond + uncond -> free ===
    print("\n--- A. 加载 TE, encode cond + uncond(空串) hiddens ---")
    t0 = time.time()
    te_model, tokenizer = load_krea2_text_encoder(str(KREA2_TE), dtype=dtype, device="cuda")
    print(f"TE 加载: {time.time()-t0:.2f}s, peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

    cond_h, cond_m = encode_prompt(te_model, tokenizer, PROMPT)
    uncond_h, uncond_m = encode_prompt(te_model, tokenizer, "")
    print(f"cond hiddens: {tuple(cond_h.shape)}, mask: {tuple(cond_m.shape)}")
    print(f"uncond hiddens: {tuple(uncond_h.shape)}, mask: {tuple(uncond_m.shape)}")

    cond_h = cond_h.to("cpu"); cond_m = cond_m.to("cpu")
    uncond_h = uncond_h.to("cpu"); uncond_m = uncond_m.to("cpu")
    del te_model, tokenizer
    torch.cuda.empty_cache()
    print(f"TE 释放后显存: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # === B. VAE (保留到最后 decode) + DiT ===
    print("\n--- B. 加载 VAE + DiT ---")
    t1 = time.time()
    vae = load_vae(str(KREA2_VAE), device=device, dtype=dtype, eval=True)
    print(f"VAE 加载: {time.time()-t1:.2f}s")

    t2 = time.time()
    dit = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=True)
    dit = dit.to(device).eval()
    for p in dit.parameters():
        p.requires_grad_(False)
    print(f"DiT 加载: {time.time()-t2:.2f}s")

    patch = dit.config.patch
    channels = dit.config.channels
    latent_h, latent_w = IMG_SIZE // 8, IMG_SIZE // 8
    img_seq_len = (latent_h // patch) * (latent_w // patch)  # 纯图像 token 数, mu shift 用
    print(f"latent: (1,{channels},{latent_h},{latent_w}), img_seq_len={img_seq_len} (mu shift 用)")

    cond_emb = Krea2TextEmbedding(cond_h.to(device), cond_m.to(device))
    uncond_emb = Krea2TextEmbedding(uncond_h.to(device), uncond_m.to(device))

    # dit_forward: (latents_5d, text_emb, t) -> velocity_5d (forward_for_loss 签名)
    def dit_forward(latents_5d, text_emb, t):
        return forward_for_loss(dit, latents_5d, text_emb, t)

    # === C. 采样 (cfg=4.5) ===
    print(f"\n--- C. 采样 {STEPS} 步 (cfg={CFG}, mu shift 自动) ---")
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    power_start = gpu_power()
    init_latent = torch.randn(1, channels, 1, latent_h, latent_w, device=device, dtype=dtype)

    t3 = time.time()
    final_latent = sample(
        dit_forward, init_latent, cond_emb, uncond_emb, img_seq_len,
        steps=STEPS, cfg=CFG, device=device, dtype=dtype,
    )
    torch.cuda.synchronize()
    sample_t = time.time() - t3
    peak = torch.cuda.max_memory_allocated() / 1e9
    power_end = gpu_power()
    print(f"采样耗时: {sample_t:.2f}s ({sample_t/STEPS*1000:.0f}ms/step), peak {peak:.2f}GB")
    print(f"final latent: {tuple(final_latent.shape)}, 有限: {torch.isfinite(final_latent).all().item()}")
    print(f"final latent 范围: [{final_latent.min().item():.3f}, {final_latent.max().item():.3f}]")

    # === D. VAE decode -> 存图 ===
    print("\n--- D. VAE decode -> 存 PNG ---")
    latent_4d = final_latent.squeeze(2)  # 5D -> 4D
    with torch.inference_mode():
        pixels = vae.decode_to_pixels(latent_4d)
    print(f"pixels: {tuple(pixels.shape)}, 范围 [{pixels.min().item():.3f}, {pixels.max().item():.3f}]")
    finite_px = torch.isfinite(pixels).all().item()

    # 存图 (VAE decode 输出 [-1,1], anima 约定; 转 [0,1] 再 uint8)
    from torchvision.utils import save_image
    px = (pixels.clamp(-1, 1).to(torch.float32) + 1.0) / 2.0  # [-1,1] -> [0,1]
    out_path = OUT_DIR / f"sample_cfg{CFG}.png"
    save_image(px[0], str(out_path))
    print(f"存图: {out_path}")

    # === E. CFG=0 对比 (无 CFG) ===
    print(f"\n--- E. CFG=0 对比 (无 CFG, 同 seed) ---")
    torch.manual_seed(0)
    init_latent0 = torch.randn(1, channels, 1, latent_h, latent_w, device=device, dtype=dtype)
    t4 = time.time()
    final_latent0 = sample(
        dit_forward, init_latent0, cond_emb, uncond_emb, img_seq_len,
        steps=STEPS, cfg=0.0, device=device, dtype=dtype,
    )
    sample_t0 = time.time() - t4
    with torch.inference_mode():
        pixels0 = vae.decode_to_pixels(final_latent0.squeeze(2))
    px0 = (pixels0.clamp(-1, 1).to(torch.float32) + 1.0) / 2.0
    out_path0 = OUT_DIR / "sample_cfg0.png"
    save_image(px0[0], str(out_path0))
    print(f"CFG=0 采样耗时: {sample_t0:.2f}s, 存图: {out_path0}")

    # CFG 差异 (证明 CFG 起作用)
    cfg_diff = (final_latent - final_latent0).abs().mean().item()
    print(f"cfg vs no-cfg latent abs mean diff: {cfg_diff:.4f} (应 > 0 证明 CFG 起作用)")

    # === 验证 ===
    print("\n=== F. 验证 ===")
    latent_finite = torch.isfinite(final_latent).all().item()
    # VAE decode 输出 [-1,1] (anima 约定), 转 [0,1] 后存图; 检查范围在 [-1.1,1.1]
    px_in_range = pixels.min().item() >= -1.1 and pixels.max().item() <= 1.1
    cfg_active = cfg_diff > 0.01
    print(f"final latent 有限: {latent_finite}")
    print(f"pixels 范围合理 [-1,1]: {px_in_range}")
    print(f"CFG 起作用 (diff>0.01): {cfg_active}")

    print(f"\n=== 基线 (PG199 bf16, {IMG_SIZE}×{IMG_SIZE}, {STEPS} steps, cfg={CFG}) ===")
    print(f"  DiT+VAE 显存 peak: {peak:.2f}GB")
    print(f"  采样总耗时: {sample_t:.2f}s ({sample_t/STEPS*1000:.0f}ms/step)")
    print(f"  CFG=0 采样耗时: {sample_t0:.2f}s ({sample_t0/STEPS*1000:.0f}ms/step)")
    print(f"  GPU 功耗: start={power_start:.1f}W, end={power_end:.1f}W")
    print(f"  输出图: {out_path}, {out_path0}")

    ok = latent_finite and px_in_range and cfg_active and out_path.exists() and out_path0.exists()
    print(f"\n阶段 5 推理串通通过: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
