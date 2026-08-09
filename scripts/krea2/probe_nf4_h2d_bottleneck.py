#!/usr/bin/env python3
"""方向 B 前置诊断: NF4+blockswap 训练单步里 H2D 搬运占比.

proposal krea2_nf4_blockswap.md 方向 B 的存在前提是"大分辨率 H2D 串行成瓶颈".
本探针复用 probe_nf4_ablation.py 的 encode_te_vae/load_dit/make_network 全套
已验证加载链路, 只重写计时逻辑: CUDA event 测 forward/backward 段 + 开
profile_jsonl 拿 nf4_ms (搬运+同步墙钟).

测量:
  1. swap=0 单步时间 (纯计算基线, 含 NF4 反量化 + grad-ckpt recompute + backward)
  2. swap=4 单步时间 + profile_jsonl 的 nf4_ms (搬运+同步墙钟)
  3. CUDA event 测 forward 段 (含 block swap 等待) 和 backward 段
  4. H2D 占比 = (swap4_avg - swap0_avg) / swap4_avg  (搬运引入的串行开销占比)

判断:
  - H2D 占比 < 15% → 反量化主导, H2D 藏在反量化窗口, 方向 B 不值得 (归档)
  - H2D 占比 > 30% → H2D 串行成瓶颈, 方向 B 重叠有收益 (实现)

PG199 (device 1, 32GB), 1024×1024, NF4 on, grad-ckpt on.

用法:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/krea2/probe_nf4_h2d_bottleneck.py
  # 可选: K2_H2D_SWAP=4 K2_H2D_STEPS=20 K2_H2D_IMG=1024
"""
import os
import sys
import time
import json
import resource
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from library.models.krea2_raw.family import Krea2TextEmbedding, forward_for_loss
from probe_nf4_ablation import (
    encode_te_vae as _abl_encode_te_vae,
    load_dit as _abl_load_dit,
    make_network as _abl_make_network,
    KREA2_DIT, KREA2_TE, KREA2_VAE, LORA_DIM, LORA_ALPHA, LR,
)
from probe_train import FIXED_SIGMA, PROMPT

PROFILE_PATH = "/tmp/krea2_h2d_profile.jsonl"


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v else default


def _env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def host_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def encode_te_vae(img_size, device, dtype, te_device):
    """复用 ablation 探针的 TE+VAE 加载, 返回 (hiddens, txtmask, latents_5d)."""
    hiddens, txtmask, latents_4d = _abl_encode_te_vae(img_size, device, dtype, te_device)
    latents_5d = latents_4d.unsqueeze(2)
    return hiddens, txtmask, latents_5d


def load_dit(nf4, nf4_path, dtype):
    return _abl_load_dit(nf4, nf4_path, torch.device("cpu"), dtype)


def make_network(dit):
    return _abl_make_network(dit)


def timed_train_steps(dit, network, opt, latents_5d, text_emb, fixed_noise, fixed_target,
                      n_steps, device, dtype, with_profile=False):
    """跑 n_steps, CUDA event 测 forward/backward 段 + 墙钟单步.

    with_profile=True 时每步调 flush_block_swap_profile(blocking=False) 刷 nf4_ms.
    """
    b = latents_5d.shape[0]
    losses, step_times, fwd_evt_ms, bwd_evt_ms = [], [], [], []

    for step in range(n_steps):
        sigma = torch.full((b,), FIXED_SIGMA, device=device, dtype=dtype)
        x_t = (1.0 - sigma) * latents_5d + sigma * fixed_noise

        opt.zero_grad(set_to_none=True)
        if dit.blocks_to_swap and dit.blocks_to_swap > 0:
            dit.prepare_block_swap_before_forward()

        t_sync = time.time()
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        fwd_start.record()
        velocity = forward_for_loss(dit, x_t, text_emb, sigma)
        loss = torch.nn.functional.mse_loss(velocity, fixed_target)
        fwd_end.record()

        bwd_start.record()
        loss.backward()
        bwd_end.record()

        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        step_t = time.time() - t_sync

        if with_profile and dit.blocks_to_swap and dit.blocks_to_swap > 0:
            dit.flush_block_swap_profile(blocking=False)

        losses.append(loss.item())
        step_times.append(step_t)
        fwd_evt_ms.append(fwd_start.elapsed_time(fwd_end))
        bwd_evt_ms.append(bwd_start.elapsed_time(bwd_end))

        if step % 5 == 0 or step == n_steps - 1:
            print(f"  step {step:3d}: loss={losses[-1]:.4f} step={step_t:.2f}s "
                  f"fwd={fwd_evt_ms[-1]:.0f}ms bwd={bwd_evt_ms[-1]:.0f}ms")

    return {
        "losses": losses,
        "step_times": step_times,
        "fwd_evt_ms": fwd_evt_ms,
        "bwd_evt_ms": bwd_evt_ms,
    }


def setup_dit(dit, network, swap, device, profile_path=None):
    """配置 DiT+network: block_swap (可选 profile) + grad-ckpt."""
    for p in dit.parameters():
        p.requires_grad_(False)
    network = make_network(dit) if network is None else network

    if swap > 0:
        dit.enable_block_swap(swap, device, profile_jsonl=profile_path)
        dit.move_to_device_except_swap_blocks(device)
        network = network.to(device).to(torch.bfloat16)
        dit.switch_block_swap_for_training()
    else:
        dit = dit.to(device)
        network = network.to(device).to(torch.bfloat16)
    dit.enable_gradient_checkpointing()
    return dit, network


def read_profile_nf4_ms(profile_path):
    """从 block_swap profile jsonl 提取 nf4_ms (搬运+同步墙钟, ms)."""
    if not profile_path or not os.path.exists(profile_path):
        return []
    nf4_ms_list = []
    with open(profile_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("nf4_ms") is not None:
                nf4_ms_list.append(float(ev["nf4_ms"]))
    return nf4_ms_list


def main() -> int:
    swap = _env_int("K2_H2D_SWAP", 4)
    img_size = _env_int("K2_H2D_IMG", 1024)
    n_steps = _env_int("K2_H2D_STEPS", 20)
    nf4_path = os.environ.get("K2_ABL_NF4_PATH", "") or str(
        ROOT / "models" / "diffusion_models" / "krea2_raw_nf4.safetensors"
    )
    te_device = "cpu" if _env_bool("K2_H2D_TE_CPU", True) else "cuda"
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"=== 方向B 前置诊断: NF4+swap H2D 占比 ===")
    print(f"  swap={swap} {img_size}×{img_size} steps={n_steps} GPU={gpu_id} te={te_device}")
    print(f"  nf4_path={nf4_path}")

    # A. TE + VAE
    print(f"\n--- A. TE (device={te_device}) + VAE encode ---")
    t0 = time.time()
    hiddens, txtmask, latents_5d = encode_te_vae(img_size, device, dtype, te_device)
    print(f"  encode: {time.time()-t0:.1f}s, hiddens {tuple(hiddens.shape)}, latents {tuple(latents_5d.shape)}")

    text_emb = Krea2TextEmbedding(hiddens.to(device), txtmask.to(device))
    latents_5d = latents_5d.to(device)
    fixed_noise = torch.randn_like(latents_5d)
    fixed_target = torch.randn_like(latents_5d)

    # B. swap=N 训练 (含搬运+同步)
    print(f"\n--- B. DiT (NF4) + LoRA + block_swap={swap} + grad-ckpt ---")
    t1 = time.time()
    dit, nf4_source = load_dit(True, nf4_path, dtype)
    print(f"  DiT 加载 ({nf4_source}): {time.time()-t1:.1f}s")

    network = make_network(dit)
    if os.path.exists(PROFILE_PATH):
        os.remove(PROFILE_PATH)
    dit, network = setup_dit(dit, network, swap, device, profile_path=PROFILE_PATH)
    print(f"  DiT blocks={dit.num_blocks} swap={getattr(dit,'blocks_to_swap',0)}")
    opt = torch.optim.Adam(network.parameters(), lr=LR)

    print(f"\n--- C. swap={swap} 训练 {n_steps} 步 (含 H2D 搬运+同步) ---")
    rss_before = host_rss_gb()
    m_swap = timed_train_steps(
        dit, network, opt, latents_5d, text_emb, fixed_noise, fixed_target,
        n_steps, device, dtype, with_profile=True,
    )
    if swap > 0:
        dit.flush_block_swap_profile(blocking=True)
    rss_after = host_rss_gb()
    torch.cuda.reset_peak_memory_stats()

    # D. swap=0 基线 (纯计算, 无搬运) — 重建 DiT 不带 swap
    print(f"\n--- D. swap=0 基线 (纯计算, 无搬运) ---")
    del dit, network, opt
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    dit0, _ = load_dit(True, nf4_path, dtype)
    network0 = make_network(dit0)
    dit0, network0 = setup_dit(dit0, network0, 0, device)
    opt0 = torch.optim.Adam(network0.parameters(), lr=LR)
    m_base = timed_train_steps(
        dit0, network0, opt0, latents_5d, text_emb, fixed_noise, fixed_target,
        n_steps, device, dtype, with_profile=False,
    )

    # E. 分析 (跳过前 3 步预热)
    def avg(lst):
        s = lst[3:] if len(lst) > 3 else lst
        return sum(s) / max(1, len(s))

    swap_avg = avg(m_swap["step_times"])
    base_avg = avg(m_base["step_times"])
    swap_fwd_avg = avg(m_swap["fwd_evt_ms"])
    base_fwd_avg = avg(m_base["fwd_evt_ms"])
    swap_bwd_avg = avg(m_swap["bwd_evt_ms"])
    base_bwd_avg = avg(m_base["bwd_evt_ms"])
    nf4_ms_list = read_profile_nf4_ms(PROFILE_PATH)
    nf4_ms_avg = sum(nf4_ms_list) / len(nf4_ms_list) if nf4_ms_list else 0.0

    # 双口径交叉核验 (避免 recompute 稀释误判):
    #   口径 A (step, 下界): (swap_avg-base_avg)/swap_avg — 分母含 recompute+opt+sync,
    #     recompute 是 swap/base 共同开销做差抵消, 差=纯 H2D 串行开销, 但分母偏大 → 下界.
    #   口径 B (segment, 精准): H2D 段差 / 计算段和 — forward+backward 都搬 block
    #     (backward_hook 触发 backward_prefetch, recompute 时也搬, 见 offloading.py:1734),
    #     分母只含 fwd+bwd 计算段 (不含 opt/clip/sync), 更准.
    h2d_overhead_step = swap_avg - base_avg
    ratio_step = h2d_overhead_step / swap_avg if swap_avg > 0 else 0.0
    fwd_diff_ms = swap_fwd_avg - base_fwd_avg
    bwd_diff_ms = swap_bwd_avg - base_bwd_avg
    seg_swap_ms = swap_fwd_avg + swap_bwd_avg
    seg_base_ms = base_fwd_avg + base_bwd_avg
    h2d_seg_ms = fwd_diff_ms + bwd_diff_ms
    ratio_seg = h2d_seg_ms / seg_swap_ms if seg_swap_ms > 0 else 0.0
    # profile nf4_ms 独立佐证 (搬运墙钟, 不含计算), 期望 ≈ h2d_seg_ms 的量级
    profile_h2d_per_step_ms = nf4_ms_avg  # 单次搬运事件; swap=N 每步约 2N 次 (fwd+bwd)

    print(f"\n=== 方向B 前置诊断结果 ===")
    print(f"  swap=0 基线:  step={base_avg:.3f}s  (fwd {base_fwd_avg:.0f}ms + bwd {base_bwd_avg:.0f}ms = {seg_base_ms:.0f}ms)")
    print(f"  swap={swap}:     step={swap_avg:.3f}s  (fwd {swap_fwd_avg:.0f}ms + bwd {swap_bwd_avg:.0f}ms = {seg_swap_ms:.0f}ms)")
    print(f"  --- 双口径 H2D 占比 ---")
    print(f"  口径A (step, 含recompute分母, 下界): {h2d_overhead_step*1000:.0f}ms / {swap_avg*1000:.0f}ms = {ratio_step*100:.1f}%")
    print(f"  口径B (segment, 只看fwd+bwd):        {h2d_seg_ms:.0f}ms / {seg_swap_ms:.0f}ms = {ratio_seg*100:.1f}%")
    print(f"    其中 fwd 段 H2D: {fwd_diff_ms:.0f}ms (swap {swap_fwd_avg:.0f} vs base {base_fwd_avg:.0f})")
    print(f"    其中 bwd 段 H2D: {bwd_diff_ms:.0f}ms (swap {swap_bwd_avg:.0f} vs base {base_bwd_avg:.0f}, 含 recompute 搬运)")
    print(f"  profile nf4_ms 佐证: avg {nf4_ms_avg:.1f}ms/次 × {len(nf4_ms_list)} 次 = {nf4_ms_avg*len(nf4_ms_list):.0f}ms 总搬运墙钟")
    print(f"  host RSS: {rss_before:.2f} → {rss_after:.2f} GB")

    # 判断: 两口径交叉. 分歧时说明 H2D 被非计算段稀释, 按口径 B (精准) 判.
    ratios = [ratio_step, ratio_seg]
    both_high = all(r > 0.30 for r in ratios)
    both_low = all(r < 0.15 for r in ratios)
    if both_high:
        verdict = "BOTTLENECK"
        advice = (f"两口径 H2D 占比都 >30% (A={ratio_step*100:.0f}% B={ratio_seg*100:.0f}%), "
                  f"H2D 串行成瓶颈, 方向 B 重叠有收益 → 实现 (task #15)")
    elif both_low:
        verdict = "NOT_WORTH"
        advice = (f"两口径 H2D 占比都 <15% (A={ratio_step*100:.0f}% B={ratio_seg*100:.0f}%), "
                  f"反量化主导, H2D 藏在反量化窗口, 方向 B 不值得 → 归档")
    else:
        verdict = "MARGINAL"
        advice = (f"口径分歧 (A={ratio_step*100:.0f}% B={ratio_seg*100:.0f}%), "
                  f"H2D 被非计算段稀释或集中在单段, 收益存疑 → 谨慎评估, 按口径 B 判")
    print(f"\n  判断: {verdict}")
    print(f"  {advice}")

    result = {
        "config": {
            "nf4": True, "swap": swap, "img_size": img_size, "n_steps": n_steps,
            "nf4_source": nf4_source, "te_device": te_device, "gpu": gpu_id,
            "lora_dim": LORA_DIM, "grad_ckpt": True, "fixed_sigma": FIXED_SIGMA,
            "warmup_skip": 3,
        },
        "metrics": {
            "base_avg_step_s": base_avg,
            "swap_avg_step_s": swap_avg,
            "base_seg_ms": seg_base_ms,
            "swap_seg_ms": seg_swap_ms,
            # 口径 A (step, 下界): 分母含 recompute+opt+sync
            "ratio_step": ratio_step,
            "h2d_overhead_step_s": h2d_overhead_step,
            # 口径 B (segment, 精准): 分母只含 fwd+bwd 计算段
            "ratio_seg": ratio_seg,
            "h2d_seg_ms": h2d_seg_ms,
            "fwd_diff_ms": fwd_diff_ms,
            "bwd_diff_ms": bwd_diff_ms,
            "base_fwd_evt_ms": base_fwd_avg,
            "swap_fwd_evt_ms": swap_fwd_avg,
            "base_bwd_evt_ms": base_bwd_avg,
            "swap_bwd_evt_ms": swap_bwd_avg,
            # profile 独立佐证 (搬运墙钟, 不含计算)
            "nf4_ms_avg": nf4_ms_avg,
            "nf4_events": len(nf4_ms_list),
            "nf4_ms_total": nf4_ms_avg * len(nf4_ms_list),
            "rss_before_gb": rss_before,
            "rss_after_gb": rss_after,
        },
        "verdict": verdict,
        "verdict_rule": "两口径交叉: both>30%=BOTTLENECK, both<15%=NOT_WORTH, 分歧=MARGINAL(按B判)",
        "verdict_advice": advice,
        "losses_swap": m_swap["losses"],
        "losses_base": m_base["losses"],
    }
    out_path = os.environ.get("K2_H2D_OUT", "docs/findings/krea2_nf4_h2d_bottleneck.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  结果写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
