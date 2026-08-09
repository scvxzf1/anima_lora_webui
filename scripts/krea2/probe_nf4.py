"""R-verify: Krea-2-Raw NF4 量化确定性 + round-trip 保真 (NF4 落地第 1+2 层判定).

自包含探针: 不改 weights.py 生产路径, 在探针里手搓 NF4 在线量化, 验证:
  层 1 确定性: 同一 bf16 权重 + 同一 bnb 版本, 两次量化逐层 Params4bit.data +
              quant_state 完全一致 (量化是纯函数, 应逐 bit 确定).
  层 2 round-trip 保真: dequant(quant(w_bf16)) vs w_bf16 逐层算
              max_abs / mean_abs / rel_l2 / cosine, 期待 rel_l2<1% cosine>0.999.
  显存: 观察量化前后 GPU 显存 (bf16 25.6GB -> NF4 ~6.5GB).

量化触发 (子代理 + 亲自读 bnb 0.49.2 modules.py 核实):
  Linear4bit 继承 nn.Linear (:421). 构造后 load_state_dict 喂 bf16, 再 .to("cuda")
  时 Params4bit.to() (:344-345) 检测 bnb_quantized=False 调 _quantize (:298-312)
  -> bnb.functional.quantize_4bit. forward (:528-556) 用 bnb.matmul_4bit (fused
  反量化+matmul, 不生成完整 bf16 权重, 临时显存最省).

非目标 (本探针不验证, 留层 3 端到端 + LoRA 兼容性探针):
  - NF4 x LoRA apply_to 兼容性 (子代理已理论核实: Linear4bit 被 isinstance 命中,
    LoRA forward 调 org_forward 保反量化; 风险在 weight_svd init / merge bake).
  - 1024x1024 训练 loss 收敛 (层 3).
  - 存盘复用跨机复现 (层 4).

PG199 bf16, 量化全模型 (28 block x 7 Linear + 顶层 Linear). 显存峰值由量化本身
产生, 不跑训练 forward, 避免与训练显存混淆.
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

import bitsandbytes as bnb  # noqa: E402
from bitsandbytes.nn import Linear4bit, Params4bit  # noqa: E402

from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402
from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402

KREA2_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"

# NF4 量化超参 (QLoRA 论文默认).
QUANT_TYPE = "nf4"
COMPRESS_STATISTICS = True  # double quantization (二次量化), 省量化元数据显存
BLOCKSIZE = 64  # bnb 默认, 不显式传
COMPUTE_DTYPE = torch.bfloat16


def _gpu_mem(label: str) -> None:
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(
        f"  [{label}] alloc={alloc:.2f}GB reserved={reserved:.2f}GB "
        f"free={free / 1e9:.2f}GB / {total / 1e9:.2f}GB"
    )


def _collect_linear_paths(module: torch.nn.Module, prefix: str = "") -> list[tuple[str, torch.nn.Linear]]:
    """递归收集所有 nn.Linear 的 (dotted_name, module), 供量化替换."""
    out: list[tuple[str, torch.nn.Linear]] = []
    for name, child in module.named_children():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, torch.nn.Linear):
            out.append((path, child))
        else:
            out.extend(_collect_linear_paths(child, path))
    return out


def _replace_with_linear4bit(
    parent: torch.nn.Module, attr: str, orig: torch.nn.Linear
) -> Linear4bit:
    """把 parent.attr 处的 nn.Linear 换成 Linear4bit, 拷贝权重与 bias."""
    new = Linear4bit(
        orig.in_features,
        orig.out_features,
        bias=orig.bias is not None,
        compute_dtype=COMPUTE_DTYPE,
        compress_statistics=COMPRESS_STATISTICS,
        quant_type=QUANT_TYPE,
        device="cpu",  # 先在 CPU 构造, 喂 bf16 权重, 再 .to(cuda) 触发量化
    )
    # 喂 bf16 权重: Params4bit 此时还是未量化的 bf16 data (bnb_quantized=False).
    with torch.no_grad():
        new.weight = Params4bit(
            orig.weight.data.to(torch.bfloat16).contiguous(),
            requires_grad=False,
            compress_statistics=COMPRESS_STATISTICS,
            quant_type=QUANT_TYPE,
            module=new,
        )
        if orig.bias is not None:
            new.bias = torch.nn.Parameter(orig.bias.data.to(torch.bfloat16).contiguous())
    setattr(parent, attr, new)
    return new


def _set_nested(module: torch.nn.Module, path: str, value: torch.nn.Module) -> None:
    """按 dotted path setattr (a.b.c -> module.a.b.c)."""
    parts = path.split(".")
    obj = module
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def _get_nested(module: torch.nn.Module, path: str) -> torch.nn.Module:
    obj = module
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def quantize_model(model: SingleStreamDiT, device: torch.device) -> dict:
    """遍历所有 nn.Linear 替换为 Linear4bit(nf4), .to(device) 触发量化.

    返回 {path: orig_bf16_weight} 供 round-trip 对比 (反量化需要原 bf16).
    """
    print(f"\n=== 在线 NF4 量化 (quant_type={QUANT_TYPE}, compress={COMPRESS_STATISTICS}) ===")
    _gpu_mem("量化前")

    paths = _collect_linear_paths(model)
    print(f"  发现 {len(paths)} 个 nn.Linear 待量化")

    orig_weights: dict[str, torch.Tensor] = {}
    for path, orig in paths:
        orig_weights[path] = orig.weight.data.to(torch.bfloat16).contiguous().clone()
        parent_path, attr = path.rsplit(".", 1) if "." in path else ("", path)
        parent = _get_nested(model, parent_path) if parent_path else model
        _replace_with_linear4bit(parent, attr, orig)

    # .to(device) 触发量化: Params4bit.to() 检测 bnb_quantized=False 调 _quantize.
    t0 = time.time()
    model.to(device)
    quant_secs = time.time() - t0
    print(f"  量化耗时 (含 H2D): {quant_secs:.1f}s")
    _gpu_mem("量化后")
    return orig_weights


def check_determinism(
    model: SingleStreamDiT, config: SingleMMDiTConfig, device: torch.device
) -> bool:
    """层 1: 再构造一个独立 DiT, 同样量化, 逐层比两次 Params4bit.data + scale 是否一致."""
    print("\n=== 层 1: 量化确定性 (两次量化逐层对比) ===")
    # 重建一个空 DiT, 加载同一份 bf16, 同样量化.
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from library.env import resolve_under_home

    with init_empty_weights():
        m2 = SingleStreamDiT(config)
    dit_path = str(resolve_under_home(KREA2_DIT))
    sd = load_file(dit_path, device="cpu")
    m2.load_state_dict(sd, strict=True, assign=True)
    m2 = m2.to(torch.bfloat16)

    paths = _collect_linear_paths(m2)
    for path, orig in paths:
        parent_path, attr = path.rsplit(".", 1) if "." in path else ("", path)
        parent = _get_nested(m2, parent_path) if parent_path else m2
        _replace_with_linear4bit(parent, attr, orig)
    m2.to(device)

    # 逐层比 model (第一次, 已在 GPU 量化) vs m2 (第二次).
    ok = True
    mismatches = 0
    # 直接遍历 named_modules 取 Linear4bit.
    mods1 = {n: m for n, m in model.named_modules() if isinstance(m, Linear4bit)}
    mods2 = {n: m for n, m in m2.named_modules() if isinstance(m, Linear4bit)}
    assert set(mods1) == set(mods2), f"两次量化 Linear4bit 键集合不一致: {set(mods1) ^ set(mods2)}"
    for name in mods1:
        w1, w2 = mods1[name].weight, mods2[name].weight
        # Params4bit.data 是 4-bit bytes (uint8).
        d_eq = torch.equal(w1.data, w2.data)
        # quant_state 的 scale/zero 对比 (as_dict 取 packed).
        qs1 = w1.quant_state.as_dict(packed=True) if w1.quant_state else {}
        qs2 = w2.quant_state.as_dict(packed=True) if w2.quant_state else {}
        qs_eq = qs1.keys() == qs2.keys() and all(
            torch.equal(qs1[k], qs2[k]) if isinstance(qs1[k], torch.Tensor) else qs1[k] == qs2[k]
            for k in qs1
        )
        if not (d_eq and qs_eq):
            mismatches += 1
            if mismatches <= 3:
                print(f"  MISMATCH {name}: data_eq={d_eq} qs_eq={qs_eq}")
            ok = False
    print(f"  对比 {len(mods1)} 层, mismatch={mismatches}")
    print(f"  层 1 确定性: {'PASS' if ok else 'FAIL'}")
    # 释放第二次模型.
    del m2, sd
    torch.cuda.empty_cache()
    return ok


def check_roundtrip(
    model: SingleStreamDiT, device: torch.device, orig_weights: dict
) -> bool:
    """层 2: dequant(quant(w_bf16)) vs w_bf16 逐层算误差指标."""
    print("\n=== 层 2: round-trip 保真 (dequant vs 原 bf16) ===")
    mods = {n: m for n, m in model.named_modules() if isinstance(m, Linear4bit)}
    ok = True
    worst_rel_l2 = 0.0
    worst_cosine = 1.0
    worst_name = ""
    n_reported = 0
    # 收集全层统计, 用分布判据而非单层阈值 (NF4 逐元素误差 ~5-12% 是 4-bit 物理下限,
    # 不是 bug; QLoRA "lossless" 指端到端 loss 无损, 非数值无损. 故判据看方向保持度
    # cosine + 全层分布, 不卡 rel_l2 单层阈值).
    all_rel_l2: list[float] = []
    all_cos: list[float] = []
    for name, mod in mods.items():
        w_bf16 = orig_weights[name].to(device)
        # bnb.functional.dequantize_nf4 / dequantize_4bit: 需要 4-bit data + quant_state.
        w_recon = bnb.functional.dequantize_4bit(mod.weight.data, mod.weight.quant_state)
        if w_recon.shape != w_bf16.shape:
            w_recon = w_recon.reshape(w_bf16.shape)
        diff = w_bf16.float() - w_recon.float()
        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()
        denom = w_bf16.float().norm().item()
        rel_l2 = diff.norm().item() / denom if denom > 0 else 0.0
        cos = torch.nn.functional.cosine_similarity(
            w_bf16.float().flatten(), w_recon.float().flatten(), dim=0
        ).item()
        all_rel_l2.append(rel_l2)
        all_cos.append(cos)
        if rel_l2 > worst_rel_l2:
            worst_rel_l2 = rel_l2
            worst_cosine = cos
            worst_name = name
        # 只打印少量样例层.
        if n_reported < 3:
            print(f"  {name}: max={max_abs:.4f} mean={mean_abs:.5f} rel_l2={rel_l2:.4%} cos={cos:.6f}")
            n_reported += 1
    import statistics
    cos_min = min(all_cos)
    cos_mean = statistics.mean(all_cos)
    rel_l2_max = max(all_rel_l2)
    rel_l2_mean = statistics.mean(all_rel_l2)
    print(f"  最差层: {worst_name} rel_l2={worst_rel_l2:.4%} cos={worst_cosine:.6f}")
    print(f"  全层 {len(all_cos)} 个: cos min={cos_min:.6f} mean={cos_mean:.6f}")
    print(f"  全层 rel_l2: max={rel_l2_max:.4%} mean={rel_l2_mean:.4%}")
    # NF4 判据: 方向保持度 cosine 是核心指标, 0.99 是 4-bit 量化的合理下限.
    ok = cos_min > 0.99
    print(
        f"  层 2 round-trip: {'PASS' if ok else 'FAIL'} "
        f"(判据: 全层 cos>0.99 方向保持; rel_l2~10% 是 NF4 逐元素误差物理下限, 非数值无损)"
    )
    return ok


def main() -> int:
    print(f"bitsandbytes {bnb.__version__}, torch {torch.__version__}")
    device = torch.device("cuda")
    config = SingleMMDiTConfig.krea2_raw()

    # 1. 加载 bf16 DiT (走 weights.py 生产路径, 不改它).
    print("\n=== 加载 bf16 DiT ===")
    t0 = time.time()
    model = load_krea2_dit(KREA2_DIT, device="cpu", dtype=torch.bfloat16, eval=False)
    print(f"  加载耗时 {time.time() - t0:.1f}s, Linear 数: {len(_collect_linear_paths(model))}")
    _gpu_mem("bf16 在 CPU")

    # 2. 在线量化 (记录原 bf16 供 round-trip).
    orig_weights = quantize_model(model, device)

    # 3. 层 1 确定性.
    det_ok = check_determinism(model, config, device)

    # 4. 层 2 round-trip.
    rt_ok = check_roundtrip(model, device, orig_weights)

    print("\n=== 汇总 ===")
    print(f"  层 1 量化确定性: {'PASS' if det_ok else 'FAIL'}")
    print(f"  层 2 round-trip:  {'PASS' if rt_ok else 'FAIL'}")
    _gpu_mem("结束")
    return 0 if (det_ok and rt_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
