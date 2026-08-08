"""R-verify: Krea-2-Raw LoRA 检查点 save/load 热测 (阶段 6 检查点出口).

验证 LoRA checkpoint 保存/加载 round-trip:
  训练几步 (LoRA 权重离开 zero-init) -> save checkpoint
  -> 新建 Krea-2 network (krea2_target_kwargs, 不走 create_network_from_weights
  的 family gap) -> load_weights -> attach -> forward
  -> 验证 (1) LoRA 权重逐键数值 round-trip 不变 (2) forward 输出 delta < 1e-4

子代理结论: 保存开箱即用 (save_lora_network_weights -> network.state_dict() 纯
state_dict, 经 lora_save.save_network_weights, 无 anima 硬编码); 加载有 family
gap (create_network_from_weights -> from_weights 不恢复 unet_target_replace_modules,
回退 anima 默认 ["Block",...] 不匹配 SingleStreamBlock). 本探针绕过 gap: 显式用
krea2_target_kwargs 构造新 network 再 load_weights (load_state_dict non-strict,
模块名一致即匹配). formal 的 family dispatch (metadata stamp + from_weights 读回)
留阶段 6 配置收口.

验证项:
1. save_weights 出 safetensors, checkpoint LoRA 键数 392 (196 模块 × down/up).
2. load_weights 到新 network, 逐键 LoRA 权重 delta < 1e-6 (round-trip 数值不变).
3. 加载后的 network attach 到干净 DiT, forward 输出与保存前 delta < 1e-4
   (bf16 round-trip 容差).
4. LoRA 权重离开 zero-init (训练过, down/up 非零).
5. checkpoint 文件可被 safetensors.load_file 读回 (格式合法).

PG199 bf16, 256×256, 不用 block swap (本就 fit).
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

N_STEPS = 8
LORA_DIM = 16
LORA_ALPHA = 8.0
LR = 2e-3
IMG_SIZE = 256
PROMPT = "a red circle on blue background"
FIXED_SIGMA = 0.5
CKPT_PATH = ROOT / "output" / "tests" / "krea2_stage6" / "lora_checkpoint.safetensors"


def make_network(dit):
    """构造 Krea-2 LoRANetwork (krea2_target_kwargs, 绕过 family gap)."""
    kwargs = {**krea2_target_kwargs(), "lora_dim": LORA_DIM, "alpha": LORA_ALPHA}
    cfg = LoRANetworkCfg.from_kwargs(
        kwargs, network_dim=LORA_DIM, network_alpha=LORA_ALPHA,
        neuron_dropout=None, module_class=LoRAModule,
    )
    net = LoRANetwork(text_encoders=[], unet=dit, cfg=cfg, multiplier=1.0)
    net.apply_to(text_encoders=[], unet=dit, apply_text_encoder=False, apply_unet=True)
    return net


def main() -> int:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== 阶段 6 检查点: Krea-2 LoRA save/load round-trip (PG199, {dtype}) ===")

    # === A. TE + VAE encode (同 probe_train) ===
    print("\n--- A. TE + VAE encode ---")
    te_model, tokenizer = load_krea2_text_encoder(str(KREA2_TE), dtype=dtype, device="cuda")
    tok = Krea2TokenizeStrategy()
    tokens = tok.tokenize([PROMPT])
    enc = Krea2TextEncodingStrategy()
    with torch.inference_mode():
        [hiddens, txtmask] = enc.encode_tokens(tok, [te_model], tokens)
    hiddens = hiddens.to("cpu"); txtmask = txtmask.to("cpu")
    del te_model, tokens, enc, tok
    torch.cuda.empty_cache()

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
    print(f"hiddens {tuple(hiddens.shape)}, latents {tuple(latents_4d.shape)}")

    # === B. DiT + LoRA, 训练几步 ===
    print(f"\n--- B. 加载 DiT + LoRA, 训练 {N_STEPS} 步 ---")
    dit = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=False)
    dit = dit.to(device)
    for p in dit.parameters():
        p.requires_grad_(False)

    network = make_network(dit)
    network = network.to(device).to(dtype)
    n_lora = sum(p.numel() for p in network.parameters())
    print(f"LoRA 模块: {len(network.unet_loras)}, 参数 {n_lora/1e6:.2f}M")

    opt = torch.optim.AdamW(network.parameters(), lr=LR, weight_decay=0.0)
    latents_4d = latents_4d.to(device)
    text_emb = Krea2TextEmbedding(hiddens.to(device), txtmask.to(device))
    latents_5d = latents_4d.unsqueeze(2)
    b = latents_5d.shape[0]
    torch.manual_seed(123)
    fixed_noise = torch.randn_like(latents_5d)
    fixed_target = fixed_noise - latents_5d

    for step in range(N_STEPS):
        sigma = torch.full((b,), FIXED_SIGMA, device=device, dtype=dtype)
        x_t = (1.0 - sigma) * latents_5d + sigma * fixed_noise
        opt.zero_grad(set_to_none=True)
        velocity = forward_for_loss(dit, x_t, text_emb, sigma)
        loss = torch.nn.functional.mse_loss(velocity, fixed_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        opt.step()
        print(f"  step {step}: loss={loss.item():.4f}")

    # 保存前: 捕获完整 LoRA state_dict (CPU clone) + forward 基准
    pre_save_sd = {k: v.detach().to("cpu", copy=True) for k, v in network.state_dict().items()}
    sigma_fixed = torch.full((b,), FIXED_SIGMA, device=device, dtype=dtype)
    x_t_fixed = (1.0 - sigma_fixed) * latents_5d + sigma_fixed * fixed_noise
    with torch.inference_mode():
        pre_save_out = forward_for_loss(dit, x_t_fixed, text_emb, sigma_fixed)
    pre_save_out = pre_save_out.detach().to("cpu", copy=True)
    print(f"保存前 forward: shape {tuple(pre_save_out.shape)}, 有限 {torch.isfinite(pre_save_out).all().item()}")

    # 抽查 LoRA 权重离开 zero-init
    down_nonzero = any("lora_down" in k and v.abs().max().item() > 1e-6 for k, v in pre_save_sd.items())
    up_nonzero = any("lora_up" in k and v.abs().max().item() > 1e-6 for k, v in pre_save_sd.items())
    print(f"LoRA down 非零 (训练过): {down_nonzero}, up 非零 (训练后梯度流入): {up_nonzero}")

    # === C. save checkpoint ===
    print(f"\n--- C. save checkpoint -> {CKPT_PATH} ---")
    metadata = {"ss_network_spec": "lora", "ss_base_model_version": "krea2_raw"}
    network.save_weights(str(CKPT_PATH), dtype=dtype, metadata=metadata)
    file_size = CKPT_PATH.stat().st_size / 1e6
    print(f"保存: {file_size:.1f}MB, 存在: {CKPT_PATH.exists()}")

    # checkpoint 文件键数 (直接读文件)
    from safetensors.torch import load_file
    saved_sd = load_file(str(CKPT_PATH))
    lora_keys = [k for k in saved_sd if "lora_up" in k or "lora_down" in k]
    print(f"checkpoint LoRA 键数: {len(lora_keys)} (期望 196×2=392)")

    # === D. 释放旧 network, 加载到新 network ===
    print("\n--- D. 释放旧 network, 新建 network, load_weights ---")
    # LoRANetwork 无 detach; monkey-patch 的 forward 引用还在 Linear 上.
    # 释放旧 DiT (26GB 参数驻留 GPU) + network + optimizer state, GC 后再加载干净 DiT.
    # pre_save_sd / pre_save_out 是 CPU clone, 不占 GPU, 保留到 E 段验证.
    import gc
    del network, opt, dit
    gc.collect()
    torch.cuda.empty_cache()
    print(f"释放后 GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    dit2 = load_krea2_dit(KREA2_DIT, device="cpu", dtype=dtype, eval=False)
    dit2 = dit2.to(device)
    for p in dit2.parameters():
        p.requires_grad_(False)

    network2 = make_network(dit2)
    network2 = network2.to(device).to(dtype)
    print(f"新 network LoRA 模块: {len(network2.unet_loras)}")

    # load_weights (从 checkpoint)
    network2.load_weights(str(CKPT_PATH))
    print(f"load_weights 完成")

    # === E. 验证 round-trip ===
    print("\n--- E. 验证 round-trip ---")
    post_sd = network2.state_dict()

    # 1. 逐键数值比对 (pre_save_sd vs post_sd)
    shared_keys = [k for k in pre_save_sd if k in post_sd]
    print(f"共享键: {len(shared_keys)} (期望 {len(pre_save_sd)})")
    deltas = []
    for k in shared_keys:
        a, b_ = pre_save_sd[k].float(), post_sd[k].to("cpu").float()
        if a.shape == b_.shape:
            deltas.append((k, (a - b_).abs().max().item()))
    max_delta = max(d for _, d in deltas) if deltas else float("inf")
    keys_all_match = len(shared_keys) == len(pre_save_sd) == len(post_sd)
    print(f"LoRA 权重逐键 max delta: {max_delta:.2e} (容差 1e-6, bf16 round-trip)")

    # 2. 加载后 forward, 与保存前比对
    with torch.inference_mode():
        post_load_out = forward_for_loss(dit2, x_t_fixed, text_emb, sigma_fixed).to("cpu")
    fwd_delta = (pre_save_out.float() - post_load_out.float()).abs().max().item()
    print(f"forward max delta: {fwd_delta:.2e} (容差 1e-4)")
    shape_ok = post_load_out.shape == (1, 16, 1, IMG_SIZE//8, IMG_SIZE//8)
    finite = torch.isfinite(post_load_out).all().item()

    # checkpoint 文件权重非零 (训练过)
    ckpt_lora_nonzero = any(saved_sd[k].abs().max().item() > 1e-6 for k in lora_keys)
    n_lora_loaded = len(network2.unet_loras)
    keys_ok = len(lora_keys) == 392

    print(f"\n=== 验证 ===")
    print(f"checkpoint 存在: {CKPT_PATH.exists()}")
    print(f"checkpoint LoRA 键数 392: {keys_ok}")
    print(f"checkpoint 权重非零 (训练过): {ckpt_lora_nonzero}")
    print(f"新 network 模块 196: {n_lora_loaded == 196}")
    print(f"LoRA 权重 round-trip 键全匹配: {keys_all_match}")
    print(f"LoRA 权重 max delta < 1e-6: {max_delta < 1e-6}")
    print(f"load 后 forward shape 对齐: {shape_ok}")
    print(f"load 后 forward 有限: {finite}")
    print(f"forward delta < 1e-4: {fwd_delta < 1e-4}")

    ok = (CKPT_PATH.exists() and keys_ok and ckpt_lora_nonzero
          and n_lora_loaded == 196 and keys_all_match and max_delta < 1e-6
          and shape_ok and finite and fwd_delta < 1e-4)
    print(f"\n阶段 6 检查点 save/load 通过: {ok}")
    if ok:
        print(f"checkpoint: {CKPT_PATH} ({file_size:.1f}MB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
