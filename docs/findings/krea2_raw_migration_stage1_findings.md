# Krea-2-Raw 迁移 阶段 1: 文本链路 (findings)

状态：稳定
适用版本：当前 main
入口命令：`.venv/bin/python scripts/krea2/probe_text.py`
相关代码：`library/models/krea2_raw/strategy.py`、`library/models/krea2_raw/configs/qwen3vl_4b/`、`scripts/krea2/probe_text.py`

## 目标

Qwen3-VL-4B 文本编码 + 12 层 MFA + 单 prompt encode 出 context,
padding 行为确认 (R1: mask 屏蔽 vs anima zero-sink)。

## 设计定论 (子代理核实 krea-ai/krea-2 encoder.py, commit db3984f)

### Qwen3-VL 加载

- 类: `Qwen3VLForConditionalGeneration` (不是 `Qwen3VLModel`)。
- model_id: `Qwen/Qwen3-VL-4B-Instruct` (单文件 safetensors 在
  `models/text_encoders/qwen3vl_4b_bf16.safetensors`, 8.4GB, 713 keys)。
- 单文件加载模式 (同 anima `load_qwen3_text_encoder`): bundled config 目录
  `library/models/krea2_raw/configs/qwen3vl_4b/` (从 HF 下载的小文件, ~11.5MB) +
  单 safetensors 权重。
- 权重 key: `model.language_model.*` (396 keys = 36 层 × 11 + embed + norm) +
  `model.visual.*` (visual encoder, 不用但加载)。`embed_tokens (151936, 2560)`
  → hidden_dim=2560 = DiT txtdim, 无需投影。
- 加载用 `strict=False` (visual 部分可能 missing/unexpected, 阶段 1 只用 LM)。

### tokenize (ChatML 模板)

- system prompt 固定: `"<|im_start|>system\nDescribe the image by detailing
  the color, shape, size, texture, quantity, text, spatial relationships of
  the objects and background:<|im_end|>\n<|im_start|>user\n"`
- suffix: `"<|im_end|>\n<|im_start|>assistant\n"`
- padding 长度公式: `max_length(512) + prefix_idx(34) - suffix_start_idx(5) = 541`
- prompt (system + user) 一起 tokenize, pad 到 541; suffix 单独 tokenize (不 pad)
  cat 到右侧。最终 seq_len = 541 + suffix_len。

### select_layers + MFA

- `select_layers = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)` — 12 层, stack dim=2。
- `output_hidden_states=True` → `hidden_states` tuple 长 37 (embedding + 36 层),
  选 12 层 stack。
- `prompt_template_encode_start_idx = 34` — 切掉 system prompt 前 34 token。
- MFA 的带权重 projector (`Linear(12, 1, bias=False)`, 权重名
  `txtfusion.projector.weight`) 和 2560→6144 投影 (`txtmlp`) 都在 DiT 内部
  (`mmdit.py` TextFusionTransformer), **encoder 零可训练参数**。

### R1: padding 契约 (关键差异)

| | anima | Krea-2 |
|---|---|---|
| 文本编码器 | Qwen3 纯文本 + LLM Adapter 桥 T5 space | Qwen3-VL-4B 直接, 2560=txtdim |
| padding | max-pad 到 512 | ChatML pad 到 541 + cat suffix |
| pad 位处理 | `prompt_embeds[~mask]=0` 置零, DiT 内 `[~mask]=0` 二次置零 (zero-sink) | **不置零**, 用 attention mask 屏蔽 |
| DiT 内部 | `context[~crossattn_mask.bool()] = 0` (models.py:2736) | `_mask(mask)` 扩展成 `(B,1,L,L)` 屏蔽, 不二次置零 |

迁移结论: strategy.py 只负责 Qwen3-VL 选 12 层 + stack + 切 prefix + 返回
`(hiddens, mask)`; **不能套 anima 的 `[~mask]=0` 置零逻辑**。

## 出口验证

### 验证: 单 prompt encode + DiT forward (PG199 bf16)

`scripts/krea2/probe_text.py`:

```
--- 2. tokenize 单 prompt ---
input_ids: (1, 546) (期望 (1, 541+suffix_len))    # 541 pad + 5 suffix
attn_mask: (1, 546)
真实 token 数 (含 suffix): 45                    # 34 system + 11 user + suffix
prefix (前 34) 全 True: True

--- 3. encode -> hiddens + mask ---
hiddens: (1, 512, 12, 2560) (期望 (1, L-34, 12, 2560))   # 546-34=512
txtmask: (1, 512)
hiddens 有限: True
切 prefix 后真实 token 数: 11 (期望 real_len-34=11)

R1: padding 位 hiddens (不置零, 应有值):
  padding token 数: 501
  padding hiddens abs max: 438.000000
  padding hiddens abs mean: 1.265625
  Krea-2 不二次置零 padding (R1 契约): True

--- 5. DiT forward (真实 Qwen3-VL context) ---
DiT forward 输出: (1, 256, 64) (期望 (1, 256, 64))
forward 耗时: 482ms, DiT peak 26.13GB

阶段 1 文本链路通过: True
```

验证项全绿:
- tokenize ChatML pad 到 541 + cat suffix = 546 ✓
- system prefix 前 34 token 全 True (都是真实 system prompt token) ✓
- hiddens (1, 512, 12, 2560), 12 层 stack dim=2, 切 34 prefix ✓
- hiddens 有限 ✓
- **R1 契约确认**: padding 位 (501 个) hiddens abs max=438 (非零) → Krea-2 不二次置零, 用 mask 屏蔽 ✓
- DiT forward (真实 Qwen3-VL context) 输出 shape (1, 256, 64) 对齐 + 有限 ✓

## 基线 (PG199 bf16)

| 指标 | 值 |
|---|---|
| TE (Qwen3-VL-4B) 显存 | 8.93GB |
| DiT (256×256, 真实 context) peak | 26.13GB |
| context shape | (1, 512, 12, 2560) |
| tokenize + encode 耗时 | ~0.6s (单 prompt) |
| DiT forward (真实 context) | 482ms |
| DiT forward (合成 context, 阶段 2) | 90ms |

DiT forward 从 90ms (合成 context, probe_dit) 增到 482ms (真实 context), 因为
真实 context 激活了 txtfusion 全路径 (2 layerwise + 2 refiner TextFusionBlock +
12 层加权 projector + txtmlp), 合成 context 的随机值同样走这条路径, 但 forward
整体耗时差异可能来自 context 长度 (512 vs 77) 和 attention 规模。

## 已知限制 / 后续

- **caching 未完整实现**: `Krea2TextEncoderOutputsCachingStrategy.cache_batch_outputs`
  留 NotImplementedError, 阶段 4 训练串通补 multi-variant + per-sample 拆分。
  suffix 隔离 `_krea2_te.safetensors` (不污染 anima `_anima_te`)。
- **未实现 weighted prompt**: `tokenize_with_weights` 直走 tokenize (Krea-2 无 T5
  weighted tokenize 需求)。
- **TE LoRA 首日不挂**: Krea2LoRATargetSpec.text_encoder_target_replace_modules=()。
  阶段 4 后再考虑 Qwen3-VL 注入。
- **TE+DiT 同卡显存**: TE 8.93GB + DiT 26.13GB = ~35GB 超 PG199 32GB, 探针分两段
  (TE 跑完 → 移 CPU → 释放 → DiT 上 GPU)。训练时 TE cache 后释放 (lazy loading 不变量)。
- **suffix_len 实测 5**: ChatML suffix `<|im_end|>\n<|im_start|>assistant\n`
  tokenize 成 5 token, 与 official 一致。
