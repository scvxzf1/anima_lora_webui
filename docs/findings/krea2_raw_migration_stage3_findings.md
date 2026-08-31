# Krea-2-Raw 迁移 阶段 3: LoRA 注入点 (findings)

状态：历史阶段快照 / 阶段 3 已完成
适用版本：阶段 3 LoRA 注入落地时点；不作为当前完整能力说明
入口命令：`.venv/bin/python scripts/krea2/probe_lora_targets.py` + `.venv/bin/python scripts/krea2/probe_lora_attach.py`
相关代码：`library/models/krea2_raw/lora_targets.py`、`networks/lora_anima/config.py`、`networks/lora_anima/network.py`、`scripts/krea2/probe_lora_targets.py`、`scripts/krea2/probe_lora_attach.py`

> “已知限制 / 后续”保留阶段 3 当时状态；block swap 已落地，attention 当前使用
> Krea-2 专用 backend 而非 Anima 通用 dispatch。当前边界见
> [`../multi_model_support.md`](../multi_model_support.md)。

## 目标

复用 anima 的 LoRA network (`networks.lora_anima`) 通过 family-aware target
参数化挂到 Krea-2 DiT, 不新增 `networks/lora_krea2` 模块。anima 路径行为不变
(cfg 字段默认 None → 用类属性默认)。

## 设计定论

### 复用 anima LoRA network 的可行性

`networks/lora_anima/network.py::LoRANetwork.__init__` 改为 family-aware:

```python
unet_targets = (
    cfg.unet_target_replace_modules
    if cfg.unet_target_replace_modules is not None
    else LoRANetwork.ANIMA_TARGET_REPLACE_MODULE
)
text_encoder_targets = (
    cfg.text_encoder_target_replace_modules
    if cfg.text_encoder_target_replace_modules is not None
    else LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
)
```

anima 路径 (cfg 字段 None) → 用 `ANIMA_TARGET_REPLACE_MODULE`, 行为不变。
Krea-2 路径 → `["SingleStreamBlock"]`。

### targeting.py 的 include/exclude 语义 (关键)

`collect_lora_target_candidates` 的过滤逻辑 (`targeting.py` L86-97):

```python
excluded = any(pattern.fullmatch(original_name) for pattern in exclude_patterns)
included = any(pattern.fullmatch(original_name) for pattern in include_patterns)
if excluded and not included:
    continue  # 跳过
```

**include_patterns 是 exclude-override (豁免), 不是白名单**。
`excluded and not included` 才跳过。所以:
- 要排除某 Linear, 必须把它写进 `exclude_patterns` (include 不命中)。
- 留空的 include + 精确的 exclude = 期望的注入集。

### Krea-2 LoRA target spec (`lora_targets.py`)

`Krea2LoRATargetSpec`:
- `unet_target_replace_modules = ("SingleStreamBlock",)` — 容器类白名单。
- `include_patterns = []` — 不用 anima 的 adaln 豁免那套。
- `exclude_patterns`:
  - `blocks\.\d+\.attn\.gate` — sigmoid 门控, 语义敏感, 首日不挂。
  - `.*\.mod\..*` / `.*\.qknorm\..*` / `.*\.prenorm\..*` / `.*\.postnorm\..*` —
    Parameter (非 Linear), 兜底防御。
  - `.*\.first\..*` / `.*\.last\..*` / `.*\.tmlp\..*` / `.*\.tproj\..*` /
    `.*\.txtfusion\..*` / `.*\.txtmlp\..*` — 首日不挂, 兜底防御。
- `fuse_specs = ()` — Krea-2 q/k/v 独立权重, 不 fuse。
- `text_encoder_target_replace_modules = ()` — 首日不挂 TE LoRA。

### `krea2_target_kwargs()` kwargs 键名

`LoRANetworkCfg.from_kwargs` 读 `kwargs["exclude_patterns"]` (会再 append
`_DEFAULT_EXCLUDE`), 不是 `exclude_patterns_override`。所以 `krea2_target_kwargs()`
返回的 dict 用 `exclude_patterns` 键。anima 的 `_DEFAULT_EXCLUDE` (匹配
`_modulation` 等后缀) 与 Krea-2 的 `blocks\.\\d+\\.attn\\.gate` 等共存, 不冲突。

## 注入点 (28 block × 7 = 196 target)

每 block 7 个 Linear:

| 注入点 | 语义 | shape |
|---|---|---|
| `blocks.N.attn.wq` | query proj | 6144→6144 (GQA 48 头 × 128 headdim) |
| `blocks.N.attn.wk` | key proj | 6144→1536 (GQA 12 头 × 128 headdim) |
| `blocks.N.attn.wv` | value proj | 6144→1536 |
| `blocks.N.attn.wo` | output proj | 6144→6144 |
| `blocks.N.mlp.up` | SwiGLU up | 6144→16384 |
| `blocks.N.mlp.down` | SwiGLU down | 16384→6144 |
| `blocks.N.mlp.gate` | SwiGLU gate | 6144→16384 |

### 首日明确不挂

| 点 | 原因 |
|---|---|
| `blocks.N.attn.gate` | sigmoid 门控, 乘性作用于 attention 输出, 语义敏感 |
| `blocks.N.mod.lin` | DoubleSharedModulation.lin 是 Parameter(6*dim), 非 Linear |
| `blocks.N.prenorm/postnorm/qknorm.scale` | RMSNorm Parameter, 非 Linear |
| `first/tmlp/tproj/txtmlp/txtfusion/last` | 首日保守, 只挂 block 内标准 Linear |

## 出口验证

### 验证 1: target spec 匹配 (CPU)

`scripts/krea2/probe_lora_targets.py`, 用 layers=2 迷你 config (meta 设备,
不实际分配 12.8B) 验证 regex 逻辑 (与层数无关, 2×7=14 外推 28×7=196):

```
target 命中: 14 (期望 2×7=14, 外推 28×7=196)
skipped (dim=None/0): 0
按类型: {'attn': 8, 'mlp': 6} (期望 attn=8, mlp=6)
有 7 target 的 block 数: 2/2 (外推 28/28)
应排除但命中: [] (期望 [])
Krea-2 cfg.unet_target_replace_modules: ['SingleStreamBlock']
Anima cfg.unet_target_replace_modules: None (期望 None)
Anima 路径 cfg 字段未设 (回归不变): True
阶段 3 target spec 匹配通过: True
```

### 验证 2: attach + forward 真火测试 (PG199)

`scripts/krea2/probe_lora_attach.py`, 加载真实 12.8B DiT, 构造 LoRANetwork
(lora_dim=16, alpha=8, LoRAModule), apply_to 后 forward:

```
LoRA 模块数: 196 (期望 28×7=196)
LoRA 参数量: 48.17M (dim=16, alpha=8)
apply_to 后 unet_loras: 196
LoRA forward 输出: (1, 256, 64)
delta max: 0.000000, mean: 0.000000  (up zero-init → 无 delta)
阶段 3 attach + forward 通过: True
```

### 回归测试 (anima 路径)

改了共享热点文件 `config.py` + `network.py`, 跑 anima 路径测试:

```
tests/test_network_cfg.py + tests/test_factory_metadata_flow.py: 57 passed
tests/test_method_network_lifecycle.py: 9 passed
```

## 基线 (PG199 bf16, 256×256, lora_dim=16/alpha=8)

| 指标 | 值 |
|---|---|
| DiT 权重显存 (bf16) | 25.74GB peak |
| attach 后 peak | 25.86GB (+0.12GB LoRA) |
| LoRA 参数 | 48.17M |
| attach 后 forward | 133ms (vs 基线 ~90ms, monkey-patch 开销) |
| LoRA 初始 delta | 0 (up zero-init, 验证 hook 不破坏 forward) |

## 阶段 3 截止时的已知限制 / 后续（历史快照）

- **未验证反向传播**: 阶段 4 训练串通做 (loss.backward + optimizer.step)。
- **未验证 TE LoRA**: 首日不挂, 阶段 4 后再考虑 Qwen3-VL 注入。
- **未验证采样**: 阶段 5 推理串通做。
- **阶段 3 当时未集成 block swap**：后续已在
  `library/models/krea2_raw/dit.py` 接入共享 offloader，当前不再是能力缺口。
- **阶段 3 当时未集成 Anima `attention_dispatch`**：Krea-2 仍保留专用
  `library/models/krea2_raw/attention_backend.py`，不直接复用 Anima 的通用 layout
  dispatch；这是当前的架构边界，而不是遗漏。
- **`_DEFAULT_EXCLUDE` 兜底**: anima 那条 `.*(_modulation|_norm|...).*` 对 Krea-2
  的 `mod.lin` (Parameter) 本不命中 (非 Linear), 共存无害。
