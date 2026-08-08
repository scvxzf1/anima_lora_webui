"""R-verify: Krea-2-Raw LoRA target spec 匹配验证 (阶段 3 出口之一).

验证 Krea2LoRATargetSpec 的 include/exclude regex 对真实 Krea-2 DiT 的
named_modules 正确分类:
1. 每个 block 命中 7 (attn.wq/wk/wv/wo + mlp.up/down/gate), 推广到 28 block = 196.
2. 排除 mod.lin / attn.gate / norms / first/tmlp/tproj/txtmlp/txtfusion/last.
3. family-aware cfg 字段:Krea-2 cfg 用 Krea2LoRATargetSpec,anima cfg 用默认.

不跑完整 LoRA attach + forward (那需要完整 network 构造, 阶段 4 训练串通做).
只验证 target 发现逻辑正确.

注:用 layers=2 迷你 config (而非 krea2_raw() 28 层) 避免 CPU 实际分配 12.8B
参数超时; regex 匹配逻辑与层数无关, 2 block × 7 = 14 命中可外推到 28 × 7 = 196.
features 保持 6144 以维持 GQA 头数整除关系 (48 头 × 128 headdim).
用 init_empty_weights (meta 设备) 构造, 不实际分配内存.
"""
from __future__ import annotations

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU 即可, 不需 GPU

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402

from library.models.krea2_raw.dit import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402
from library.models.krea2_raw.lora_targets import (  # noqa: E402
    KREA2_LORA_INJECTION_POINTS,
    KREA2_LORA_TARGET_SPEC,
    krea2_target_kwargs,
)
from networks.lora_anima.targeting import (  # noqa: E402
    LoRATargetCandidate,
    collect_lora_target_candidates,
    compile_lora_target_patterns,
)
from networks.lora_anima.config import LoRANetworkCfg  # noqa: E402


def main() -> int:
    print("=== 阶段 3: Krea-2 LoRA target spec 匹配验证 (CPU, 迷你 config) ===")

    # 1. 构造迷你 Krea-2 DiT (CPU, meta 设备, 不实际分配内存)
    # 用 layers=2 而非 28 加速; features=6144 保持 GQA 整除关系.
    print("\n--- 1. 构造迷你 Krea-2 DiT 结构 (layers=2, meta 设备) ---")
    full = SingleMMDiTConfig.krea2_raw()
    config = replace(full, layers=2)
    with init_empty_weights():
        model = SingleStreamDiT(config)
    # 统计所有 Linear
    all_linears = [
        (n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    ]
    print(f"DiT 总 Linear 数: {len(all_linears)}")

    # 2. 用 Krea2LoRATargetSpec 跑 target 发现
    print("\n--- 2. Krea2LoRATargetSpec target 发现 ---")
    spec = KREA2_LORA_TARGET_SPEC
    exclude_patterns = compile_lora_target_patterns(spec.exclude_patterns)
    include_patterns = compile_lora_target_patterns(spec.include_patterns)

    candidates = collect_lora_target_candidates(
        root_module=model,
        prefix="lora_unet",
        target_replace_modules=spec.unet_target_replace_modules,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        is_unet=True,
        layer_start=None,
        layer_end=None,
        modules_dim=None,
        modules_alpha=None,
        reg_dims=None,
        reg_alphas=None,
        default_dim=16,
        lora_dim=16,
        alpha=8.0,
        verbose=False,
    )
    matched = [c for c in candidates if not c.skipped]
    skipped = [c for c in candidates if c.skipped]
    print(f"target 命中: {len(matched)} (期望 2×7=14, 外推 28×7=196)")
    print(f"skipped (dim=None/0): {len(skipped)}")

    # 3. 分类命中: 按类型
    type_counter = Counter()
    for c in matched:
        # c.original_name 如 "blocks.0.attn.wq"
        parts = c.original_name.split(".")
        if len(parts) >= 3 and parts[0] == "blocks":
            type_counter[f"{parts[2]}"] += 1  # attn / mlp
    print(f"按类型: {dict(type_counter)} (期望 attn=8, mlp=6)")

    # 4. 验证每个 block 都有 7 个
    per_block = Counter()
    for c in matched:
        parts = c.original_name.split(".")
        if parts[0] == "blocks" and len(parts) >= 3:
            per_block[parts[1]] += 1
    blocks_with_7 = sum(1 for v in per_block.values() if v == 7)
    print(f"有 7 target 的 block 数: {blocks_with_7}/2 (外推 28/28)")

    # 5. 验证排除点不在命中里
    excluded_names = {c.original_name for c in matched}
    must_exclude = [
        "blocks.0.attn.gate",        # sigmoid 门控
        "blocks.0.attn.qknorm.qnorm.scale",  # RMSNorm (非 Linear, 本来就不在)
        "blocks.0.mod.lin",          # Parameter (非 Linear)
        "first.weight",              # 首日不挂
    ]
    leak = [n for n in must_exclude if n in excluded_names]
    print(f"应排除但命中: {leak} (期望 [])")

    # 6. 验证 family-aware cfg 字段
    print("\n--- 3. family-aware LoRANetworkCfg 字段 ---")
    krea2_kwargs = krea2_target_kwargs()
    # 模拟 from_kwargs: 用 kwargs["exclude_patterns"] (from_kwargs 会再 append
    # _DEFAULT_EXCLUDE), 这里直接传给 from_kwargs 走真实路径.
    krea2_cfg_kwargs = {
        "lora_dim": 16,
        "alpha": 8.0,
        "unet_target_replace_modules": krea2_kwargs["unet_target_replace_modules"],
        "text_encoder_target_replace_modules": krea2_kwargs[
            "text_encoder_target_replace_modules"
        ],
        "include_patterns": krea2_kwargs["include_patterns"],
        "exclude_patterns": krea2_kwargs["exclude_patterns"],
    }
    krea2_cfg = LoRANetworkCfg.from_kwargs(
        krea2_cfg_kwargs,
        network_dim=16,
        network_alpha=8.0,
        neuron_dropout=None,
        module_class=torch.nn.Module,  # 探针只验 cfg 字段, 不实例化 module
    )
    print(f"Krea-2 cfg.unet_target_replace_modules: {krea2_cfg.unet_target_replace_modules}")
    print(f"Krea-2 cfg.text_encoder_target_replace_modules: {krea2_cfg.text_encoder_target_replace_modules}")
    inc = krea2_cfg.include_patterns
    print(f"Krea-2 cfg.include_patterns ({len(inc) if inc else 0}): {(inc or [])[:3] or '[] (空, 不用 adaln 豁免)'}")

    # anima 回归: 不设 family 字段 → None → 用类属性默认
    anima_cfg = LoRANetworkCfg.from_kwargs(
        {"lora_dim": 16, "alpha": 8.0},
        network_dim=16,
        network_alpha=8.0,
        neuron_dropout=None,
        module_class=torch.nn.Module,
    )
    print(f"\nAnima cfg.unet_target_replace_modules: {anima_cfg.unet_target_replace_modules} (期望 None)")
    print(f"Anima cfg.text_encoder_target_replace_modules: {anima_cfg.text_encoder_target_replace_modules} (期望 None)")
    anima_unchanged = (
        anima_cfg.unet_target_replace_modules is None
        and anima_cfg.text_encoder_target_replace_modules is None
    )
    print(f"Anima 路径 cfg 字段未设 (回归不变): {anima_unchanged}")

    # 结论
    print("\n=== 结论 ===")
    ok = (
        len(matched) == 14
        and blocks_with_7 == 2
        and not leak
        and krea2_cfg.unet_target_replace_modules == ["SingleStreamBlock"]
        and anima_unchanged
    )
    print(f"阶段 3 target spec 匹配通过: {ok}")
    print(f"注入点清单: {KREA2_LORA_INJECTION_POINTS}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
