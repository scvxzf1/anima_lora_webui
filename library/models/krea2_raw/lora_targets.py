# Krea-2-Raw LoRA target spec (阶段 3)
#
# Krea-2 是 single-stream MMDiT,block 命名 `blocks.N.attn.{wq,wk,wv,wo,gate}` +
# `mlp.{up,down,gate}` + `mod.lin`(Parameter)+ norms。与 anima 的 dual-stream
# Cosmos DiT(blocks.N.self_attn.qkv_proj / cross_attn.kv_proj / adaln_modulation)
# 命名完全不同。
#
# 注入点定论见 docs/findings/krea2_raw_migration_stage0_findings.md §R4:
#   - q/k/v 不 fused(独立 wq/wk/wv),anima 的 qkv fuse spec 不适用。
#   - 无独立 cross_attn(single-stream),CROSSATTN_EMB_DIM 不适用。
#   - 保守注入:mlp.{up,down,gate} + attn.{wq,wk,wv,wo}。
#   - 首日不挂 mod.lin(Parameter,非 Linear)、attn.gate(sigmoid 门控,语义敏感)。
#
# target 机制复用 anima 的 targeting.py(容器类白名单 + fullmatch 正则 + block
# 索引正则 blocks\.(\d+)\.),只是参数化不同。关键语义(targeting.py L86-97):
#   include_patterns 是 exclude-override(豁免),不是白名单。
#   `excluded and not included` 才跳过 → 要排除某 Linear,必须把它写进
#   exclude_patterns(include 不命中)。所以 Krea-2 用 exclude 精确排除 attn.gate,
#   include 留空(不需要 adaln 那套豁免)。anima 路径行为不变(cfg 字段默认
#   None → 用 LoRANetwork.ANIMA_TARGET_REPLACE_MODULE + _DEFAULT_EXCLUDE)。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Krea2LoRATargetSpec:
    """Krea-2-Raw 的 LoRA target 参数。

    通过 LoRANetworkCfg 的 unet_target_replace_modules / exclude_patterns 字段
    注入,覆盖 anima 默认。anima 路径不设这些字段 → 回退到
    ANIMA_TARGET_REPLACE_MODULE + _DEFAULT_EXCLUDE,行为不变。
    """

    # 容器类白名单:Krea-2 DiT block 类名是 SingleStreamBlock。
    # 不含 PatchEmbed/TimestepEmbedding/FinalLayer(anima 的对应物是
    # SingleStreamDiT.first/tmlp/tproj/last,首日保守不挂)。
    unet_target_replace_modules: Sequence[str] = ("SingleStreamBlock",)

    # include_patterns:Krea-2 不用 anima 的 adaln 豁免那套,留空。
    # 注入控制完全由 exclude_patterns 精确排除实现(见下)。
    include_patterns: List[str] = field(default_factory=list)

    # exclude_patterns:精确排除首日不挂的 Linear。
    # targeting.py 语义:excluded and not included 才跳过。
    # attn.gate 是 nn.Linear(dim,dim),会被容器扫描命中,这里精确排除。
    # mod.* / qknorm / prenorm / postnorm 是 Parameter(非 Linear),本不命中,
    # 这里兜底防御(若未来 Krea-2 改成 Linear 也不误挂)。
    # first/last/tmlp/tproj/txtfusion/txtmlp 不在 SingleStreamBlock 容器内,
    # 本不会被扫到,这里同样兜底防御。
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            r"blocks\.\d+\.attn\.gate",   # sigmoid 门控,语义敏感,首日不挂
            r".*\.mod\..*",               # DoubleSharedModulation.lin 是 Parameter
            r".*\.qknorm\..*",            # RMSNorm
            r".*\.prenorm\..*",           # RMSNorm
            r".*\.postnorm\..*",          # RMSNorm
            r".*\.first\..*",             # patch embed / 首 Linear,首日不挂
            r".*\.last\..*",              # final layer,首日不挂
            r".*\.tmlp\..*",              # text fusion MLP,首日不挂
            r".*\.tproj\..*",             # text projection,首日不挂
            r".*\.txtfusion\..*",         # text fusion block,首日不挂
            r".*\.txtmlp\..*",            # text MLP,首日不挂
        ]
    )

    # Krea-2 无 q/k/v fuse(独立权重),fuse spec 为空。
    # attn_fuse.py 的 AttnFuseSpec 列表——空 tuple 表示不 fuse。
    fuse_specs: tuple = ()

    # 文本编码器:Krea-2 用 Qwen3-VL-4B(anima 用 Qwen3-0.6B/3.5)。
    # 首日不挂 TE LoRA(Conservative,TE 权重 8.9GB + 12 层 MFA,先稳住 DiT)。
    # 阶段 4 训练串通后再考虑 TE 注入。
    text_encoder_target_replace_modules: Sequence[str] = ()
    train_text_encoder: bool = False


# 单例:Krea-2-Raw 默认 target spec。
KREA2_LORA_TARGET_SPEC = Krea2LoRATargetSpec()


def krea2_target_kwargs() -> dict:
    """返回注入 LoRANetworkCfg 的 kwargs(覆盖 anima 默认 target)。

    用法(在 family-aware network 构造处):
        cfg = LoRANetworkCfg.from_kwargs({
            **krea2_target_kwargs(),
            "lora_dim": 16, "alpha": 8, ...
        }, network_dim=..., network_alpha=..., neuron_dropout=...,
           module_class=LoRAModule)

    注:from_kwargs 读 kwargs["exclude_patterns"] 并 append _DEFAULT_EXCLUDE
    在后;这里传的 exclude 会与之共存(anima 那条匹配 _modulation 等后缀,
    Krea-2 用 .mod. 子路径 + blocks\\.\\d+\\.attn\\.gate, 两者不冲突)。
    include_patterns 传 None → from_kwargs 的 _as_str_list(None) → 不豁免
    (Krea-2 不用 anima 的 adaln 豁免那套)。
    """
    spec = KREA2_LORA_TARGET_SPEC
    return {
        "unet_target_replace_modules": list(spec.unet_target_replace_modules),
        "text_encoder_target_replace_modules": list(
            spec.text_encoder_target_replace_modules
        ),
        "include_patterns": list(spec.include_patterns) or None,
        # from_kwargs 读的是 "exclude_patterns" 键 (会再 append _DEFAULT_EXCLUDE).
        "exclude_patterns": list(spec.exclude_patterns),
        "train_text_encoder": spec.train_text_encoder,
    }


# 注入点的人类可读清单(调试/文档用)。
# 每 block 7 个:attn.{wq,wk,wv,wo} + mlp.{up,down,gate}。28 block = 196 target。
KREA2_LORA_INJECTION_POINTS = {
    "blocks.N.attn.wq": "query proj (GQA 48 头, 6144→6144)",
    "blocks.N.attn.wk": "key proj (GQA 12 头, 6144→1536)",
    "blocks.N.attn.wv": "value proj (GQA 12 头, 6144→1536)",
    "blocks.N.attn.wo": "output proj (6144→6144)",
    "blocks.N.mlp.up": "SwiGLU up (6144→16384)",
    "blocks.N.mlp.down": "SwiGLU down (16384→6144)",
    "blocks.N.mlp.gate": "SwiGLU gate (6144→16384)",
}

# 首日明确不挂的点(及原因)。
KREA2_LORA_EXCLUDED_POINTS = {
    "blocks.N.attn.gate": "sigmoid 门控,语义敏感(乘性作用于 attention 输出)",
    "blocks.N.mod.lin": "DoubleSharedModulation.lin 是 Parameter(6*dim),非 Linear",
    "blocks.N.prenorm.scale": "RMSNorm Parameter,非 Linear",
    "blocks.N.postnorm.scale": "RMSNorm Parameter,非 Linear",
    "blocks.N.attn.qknorm.{q,k}norm.scale": "RMSNorm Parameter,非 Linear",
    "first/tmlp/tproj/txtmlp/txtfusion/last": "首日保守,只挂 block 内标准 Linear",
}
