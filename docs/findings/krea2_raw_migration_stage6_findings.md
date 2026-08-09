# Krea-2-Raw 迁移 阶段 6: 块交换 + 检查点 (findings)

状态：稳定
适用版本：当前 main
入口命令：
- `.venv/bin/python scripts/krea2/probe_blockswap.py`
- `.venv/bin/python scripts/krea2/probe_checkpoint.py`
相关代码：`library/models/krea2_raw/dit.py`、`scripts/krea2/probe_blockswap.py`、`scripts/krea2/probe_checkpoint.py`

## 目标

验证 Krea-2 训练管线的两个出口能力:
1. **块交换 (block swap)**: 把 `SingleStreamDiT` 接上 anima 的 `ModelOffloader`,
   让 28 个 transformer block 的权重在 CPU↔GPU 间动态搬运, 腾出显存.
2. **检查点 (checkpoint)**: LoRA 训练后 `save_weights` 出 safetensors,
   重新构造 network 后 `load_weights` 回来, 数值 round-trip 一致 + attach 后
   forward 与保存前一致.

两者都是 goal 验收 "lora训练+检查点+块交换+采样 真实可用" 的硬要求.

## 设计定论

### 块交换: 复用 anima ModelOffloader, 零 forward 假设

子代理核实: `library/runtime/offloading.py::ModelOffloader` (anima models.py:2291-2387
调用) 对 block forward **签名零假设** — 它只遍历 `block.named_modules()` 取 `.weight`
+ `.to(device)` + `register_full_backward_hook`, 对 block forward 怎么调、传什么参数
完全透明. `SingleStreamBlock.forward(x, vec, freqs, mask)` 与 anima `Block.forward`
签名不同, 但 swap 钩子 (`wait_for_block` / `submit_move_blocks`) 只夹在 block 调用
前后, 不读 forward 内部.

落地 (dit.py):
- `__init__` 加 block swap 状态: `blocks_to_swap` / `offloader` / `_paused_blocks_to_swap`
- `num_blocks` property (返回 `len(self.blocks)`)
- 5 个方法移植自 anima models.py:2291-2387:
  `enable_block_swap` / `move_to_device_except_swap_blocks` /
  `switch_block_swap_for_inference` / `switch_block_swap_for_training` /
  `prepare_block_swap_before_forward` / `flush_block_swap_profile`
- `_run_blocks(combined, tvec, freqs, mask)`: 替换 `forward` 里原来的
  `for block in self.blocks: combined = block(...)` 循环, 加 swap 钩子
  (`if self.blocks_to_swap: offloader.wait_for_block(i)` 前置 +
  `offloader.submit_move_blocks(...)` 后置)

**块交换只搬运权重, 不搬激活**. 这决定了它的适用场景 (见下 "256×256 净负" 发现).

### 检查点: save 开箱即用, load 有 family gap

子代理核实 `networks/lora_anima/persistence.py`:
- **save**: `save_lora_network_weights` → `network.state_dict()` 纯 state_dict →
  `lora_save.save_network_weights`, **无 anima 硬编码**, 对 Krea-2 plain LoRA 开箱即用.
  键名 `lora_unet_blocks.{i}.attn.{wq|wk|wv|wo|gate}.lora_{down|up}.weight` +
  `...mlp.{up|down|gate}...` (196 模块 × 2 = 392 键).
- **load**: `load_lora_network_weights` → `network.load_state_dict(weights_sd, False)`
  non-strict. 模块名一致即匹配 (新建 network 用相同 `krea2_target_kwargs()` + 相同
  dit 结构 → 模块名一致 → round-trip 成立).
- **family gap**: `create_network_from_weights` → `from_weights` 不恢复
  `unet_target_replace_modules`, 回退 anima 默认 `["Block", ...]` 不匹配
  Krea-2 的 `SingleStreamBlock`. 本探针**绕过 gap**: 显式用
  `krea2_target_kwargs()` 构造新 network 再 `load_weights`. formal 的 family
  dispatch (metadata stamp `ss_unet_target_replace_modules` + `from_weights`
  读回) 留阶段 6 配置收口 (train.py / generation.py 正式串通).

## 出口验证

### 验证 1: 块交换训练 (PG199 bf16, 256×256, swap=4, lora_dim=16)

`scripts/krea2/probe_blockswap.py`:

```
--- C. 加载 DiT (CPU) + LoRA + enable_block_swap(4) ---
Block swap frozen CPU masters prepared: 22.64 GiB across 28 blocks (transfer_dtype=bf16)
LoRA 模块: 196, 参数 48.17M
DiT num_blocks: 28, blocks_to_swap: 4

--- D. 训练 15 步 (block swap=4, 固定 σ=0.5) ---
  step   0: loss=0.0125, step=2939ms
  step   3: loss=0.0044, step=1885ms
  step   6: loss=0.0019, step=1935ms
  step   9: loss=0.0013, step=1895ms
  step  12: loss=0.0009, step=1871ms
  step  14: loss=0.0007, step=1823ms

=== E. 验证 ===
losses: [0.0125, 0.0117, 0.0121, 0.0044, 0.0031, 0.0022, 0.0019, 0.0016,
         0.0012, 0.0013, 0.0010, 0.0009, 0.0009, 0.0008, 0.0007]
finite: True
first5=0.0087, last5=0.0009, 下降: True

=== 基线 ===
  block swap 训练 显存 peak: 32.24GB (无 swap 时 32.62GB)
  节省: 0.38GB (swap 4 块)
  avg step: 1965ms (无 swap 400ms)
  loss: first5=0.0087 -> last5=0.0009
  GPU 功耗: 43.8W -> 46.7W

阶段 6 块交换训练通过: True
```

验证项全绿:
- block swap 启用不抛异常, ModelOffloader 复用成功 (对 SingleStreamBlock 透明) ✓
- 训练 forward+backward+optimizer 跑通, loss 下降 (first5=0.0087 → last5=0.0009,
  9.7× 下降, 与无 swap 同款过拟合) ✓
- 数值与无 swap 一致 (step 0 loss=0.0125, 与无 swap probe_train 完全相同 —
  block swap 只搬权重不改 forward 语义) ✓
- 显存 peak 32.24GB (无 swap 32.62GB, 省 0.38GB) ✓

### 验证 2: 检查点 save/load round-trip (PG199 bf16, 256×256, lora_dim=16)

`scripts/krea2/probe_checkpoint.py`:

```
--- B. 训练 8 步 ---
  step 0: loss=0.0125  ...  step 7: loss=0.0016
保存前 forward: shape (1, 16, 1, 32, 32), 有限 True
LoRA down 非零 (训练过): True, up 非零: True

--- C. save checkpoint ---
保存: 96.4MB, 存在: True
checkpoint LoRA 键数: 392 (期望 196×2=392)

--- D. 释放旧 network, 新建 network, load_weights ---
释放后 GPU allocated: 0.24GB   ← gc + empty_cache 把 26GB DiT 释放干净
新 network LoRA 模块: 196
load_weights 完成

--- E. 验证 round-trip ---
共享键: 588 (期望 588)
LoRA 权重逐键 max delta: 0.00e+00 (容差 1e-6)
forward max delta: 0.00e+00 (容差 1e-4)

=== 验证 ===
checkpoint 存在: True
checkpoint LoRA 键数 392: True
checkpoint 权重非零 (训练过): True
新 network 模块 196: True
LoRA 权重 round-trip 键全匹配: True
LoRA 权重 max delta < 1e-6: True
load 后 forward shape 对齐: True
load 后 forward 有限: True
forward delta < 1e-4: True

阶段 6 检查点 save/load 通过: True
checkpoint: output/tests/krea2_stage6/lora_checkpoint.safetensors (96.4MB)
```

验证项全绿:
- save_weights 出 safetensors (96.4MB, 392 键) ✓
- load_weights 到新 network, 逐键 LoRA 权重 max delta=0 (bf16 round-trip 完全一致) ✓
- 加载后 attach 到干净 DiT, forward 输出与保存前 max delta=0 ✓
- LoRA 权重离开 zero-init (训练过, down/up 非零) ✓
- checkpoint 文件可被 `safetensors.load_file` 读回 ✓

**delta=0 的合理性**: save 用 bf16 `state_dict()`, load 回 bf16, 中间无 fp32 转换;
forward 用相同权重 + 相同输入 (固定 σ + 固定 noise seed), 确定性计算 → delta=0.

## 基线 (PG199 bf16, 256×256, lora_dim=16/alpha=8)

| 指标 | 块交换 探针 (swap=4, free_cache=True) | 块交换 稳态 (swap=4, free_cache=False) | 无 swap (probe_train) | 检查点 (无 swap) |
|---|---|---|---|---|
| 显存 peak | 32.24GB | 29.15GB | 32.62GB | 32.62GB (训练段) |
| 节省显存 | 0.38GB | 3.47GB | — | — |
| avg step | 1965ms | 1404ms | 400ms | 400ms |
| step 稳态区间 | — | 1393-1433ms | 363-411ms | — |
| loss (first5→last5) | 0.0087→0.0009 | 0.0028→0.0007 | 0.0125→0.0003 | 0.0125→0.0016 (8步) |
| GPU 功耗 | 43.8→46.7W | 47.5→48.5W | 44.8→221.9W | — |
| TE 加载 | 148.83s (冷启) | 162.96s | 148s | — |
| block swap CPU masters | 22.64 GiB / 28 blocks | 22.64 GiB / 28 blocks | — | — |
| checkpoint | — | — | — | 96.4MB, 392 键 |

> **探针路径 vs 稳态路径区分**: `probe_blockswap.py` 每步调
> `prepare_block_swap_before_forward()` 默认 `free_cache=True`
> (`dit.py:549`), 触发 `gc.collect()`+`torch.cuda.empty_cache()`
> (`offloading.py:1718-1720`), 放大 step 并推高显存 peak 统计.
> 真实训练路径 `library/training/unet_prepare.py:36` 显式 `free_cache=False`
> 跳过此开销. 对照探针 `scripts/krea2/probe_blockswap_steady.py` 复刻真实
> 路径 (free_cache=False + profile_jsonl=None + 3 步预热后计时稳态 15 步):
> step 1965→1404ms (省 561ms/步, 28%), 显存 peak 32.24→29.15GB (省 3GB).
> `free_cache` 放大占探针开销的 28%, 不是主因; 主开销是 H2D 串行化
> (1404-400=1004ms/步, depth=1 硬钉 `offloading.py:1763` + 单 worker
> `offloading.py:306` + 256×256 forward 太快无重叠窗口).

## 关键发现

### 块交换在 256×256 是净负 (但机制正确)

256×256 训练无 swap 时 32.62GB 已紧贴 PG199 32GB 上限但能 fit. swap 4 块稳态
(`free_cache=False`) 省显存 3.47GB (peak 29.15GB), 但 step 从 400ms 涨到 1404ms
(3.5× 慢). 功耗从 221.9W 降到 48.5W (近乎 idle, 计算间隙 GPU 在等搬运).

**step 开销归因** (稳态 1404ms = 400ms 计算 + 1004ms 块交换):
- H2D 串行化 1004ms/步是主开销, 非 `free_cache`. 根因: depth=1 硬钉
  (`offloading.py:1763`, >1 会退役未执行块致 `mat2 is on cpu`) +
  `ThreadPoolExecutor(max_workers=1)` (`offloading.py:306`) + 256×256 forward
  单块计算太短, H2D 拷贝 (`offloading.py:330-331` 自承 bf16 transfer 略超
  单块计算) 无法重叠 → `wait_for_block` 的 `future.result()` host 侧串行等待.
- 纯 H2D 拷贝仅 ~26ms/步 (4 块 × 132MiB ÷ 20GB/s), 串行化开销是纯拷贝的 38×,
  即"调度结构在小分辨率被放大", 非带宽瓶颈.
- `free_cache=True` 额外加 561ms/步 (探针 1965ms vs 稳态 1404ms), 是次要放大源.

**块交换的真实价值**:
- 大分辨率训练 (512×512 / 1024×1024) 权重部分仍占 ~26GB, swap N 块腾出 N×0.8GB
  给激活. 但 512×512 激活约 26GB, 32GB 卡 swap 完 28 块也救不回 (swap 28 块省
  ~22GB 权重, 留 ~10GB 给 26GB 激活仍 OOM).
- **结论**: PG199 32GB 的 block swap 救不了大分辨率训练 (激活才是瓶颈);
  它的真实用武之地是**权重显存不够**的场景 — 比如把 Krea-2 (12.82B bf16 = 26GB)
  和另一个大模型 (TE/VAE) 同时驻留时腾权重. 或者在 16GB 卡上跑 256×256 训练
  (权重 26GB > 16GB, 必须 swap).

这不是代码错误, 是块交换机制的本质 (搬权重不搬激活). 作为 finding 诚实记录.
阶段 6 的 block swap 接口**机制可用** (loss 下降 + 数值一致), 但 256×256 不是
它的受益场景.

### 检查点 family gap 留阶段 6 配置收口

`create_network_from_weights` 的 family dispatch 没做 Krea-2 分支. 探针绕过 gap
(显式构造 + load_weights), 证明**保存和加载的核心机制可用**. 正式串通
train.py resume / ComfyUI loader 需要:
- `stamp_lora_save_metadata` 加 `ss_unet_target_replace_modules` =
  `["SingleStreamBlock"]` (保存时盖章)
- `from_weights` / `create_network_from_weights` 读回该 metadata, 注入
  `unet_target_replace_modules` (加载时恢复)

这是阶段 6 配置收口 (train.py / generation.py 正式串通) 的工作, 不在本轮探针范围.

## 1024×1024 梯度检查点正式训练 (PG199 bf16, lora_dim=16/alpha=8)

阶段 6 配置收口里程碑: `forward_for_loss` + `model_family` 正式串通 train.py 路径,
首个真实 1024×1024 LoRA 训练完成. 前置修复 + grad-ckpt 移植如下.

### 前置修复: cached TE outputs 末位 rate 约定 (mask 丢失 bug)

**症状**: Krea-2 训练首步 `text_encoder_conds[1] out of range` — mask 丢失,
`text_encoder_conds` 只剩 len=1 (仅 hiddens).

**根因**: `library/training/batch_preprocess.py::split_cached_text_encoder_outputs`
假设 cached TE list 末位是 `caption_dropout_rates` (anima cache 约定, strategy.py:416).
Krea-2 `Krea2TextEncoderOutputsCachingStrategy.load_outputs_npz` 原返回 `[hiddens, mask]`,
末位 mask 被错误拆成 `caption_dropout_rates`, teo_list 只剩 `[hiddens]`.

**修复**: `load_outputs_npz` 改返回 `[hiddens, mask, caption_dropout_rate]` —
与 anima cache 布局对齐 (rate 作末位 aux, split 拆到 `batch["caption_dropout_rates"]`,
teo_list 留 `[hiddens, mask]`). caption_dropout_rate 本就由 `cache_batch_outputs`
写盘 (strategy.py:239), 只是 load 时没读出来. Krea-2 caption_dropout_rate=0.0,
rate 不参与 forward (family.compute_noise_pred_and_target 只 unpack hiddens/mask),
`split` 的 by-product 对 Krea-2 是 no-op.

### grad-ckpt 移植 (SingleStreamBlock)

Krea-2 `SingleStreamBlock` 原只有 `forward`. 移植 anima models.py:1691-1740 的标准
grad-ckpt 模式 (`library/models/krea2_raw/dit.py`):
- `__init__` 加 `self.gradient_checkpointing = False`
- 原计算逻辑挪到 `_forward` (纯计算, 无 checkpoint)
- `forward` 改为: `torch.is_grad_enabled() and self.training and
  self.gradient_checkpointing` 时 `torch_checkpoint(self._forward, ...,
  use_reentrant=False)`, 否则直调 `_forward`
- `SingleStreamDiT.enable_gradient_checkpointing` / `disable` 遍历 blocks 调子方法
  (移植自 anima models.py:1942-1946)

Krea-2 首日只支持标准 grad-ckpt (无 cpu_offload / unsloth / adapter-aware 变体,
那些是 anima 专属优化路径). `bootstrap.py:564` 已通过鸭子类型调
`unet.enable_gradient_checkpointing()` (无参数), 签名兼容, 无需改 bootstrap.

### 基线 (PG199 bf16, 1024×1024, lora_dim=16/alpha=8, grad-ckpt on, swap off)

| 指标 | 值 |
|---|---|
| 显存 peak (allocated) | 27.9 GB |
| 显存 peak (reserved) | 28.1 GB |
| 显存 allocated (稳态) | 24.5 GB |
| step 时间 (稳态) | 3.47 s/it |
| 50 步总耗时 | ~173 s |
| loss (step 2 → 50) | 0.465 → 0.198 (下降 57%) |
| loss (末 10 步) | 稳定在 0.199–0.214 |
| checkpoint | 92 MB (`output/ckpt/krea2_lora_test.safetensors`) |

**对比 256×256 无 grad-ckpt (probe_train 基线)**:
- 显存: 27.9 GB (1024+grad-ckpt) vs 32.62 GB (256 无 grad-ckpt) — grad-ckpt 把
  1024 激活压到比 256 无 grad-ckpt 还低, 留 ~4 GB 余量.
- 速度: 3.47 s/it (1024+grad-ckpt) vs 0.4 s/it (256 无 grad-ckpt) — 8.7× 慢,
  符合 grad-ckpt 重算 + 1024 计算量 16× 的预期.
- loss 量级: 0.2 (1024 5.6B DiT, 60 图) vs 0.001 (256 探针固定 σ=0.5 玩具) —
  不可直接比, 1024 是真实 flow-matching 全 σ 范围训练.

**结论**: grad-ckpt 是 1024×1024 训练在 PG199 32GB 上 fit 的正解 (block swap 救不了
激活瓶颈, 见上). 训练 loss 稳定下降, checkpoint 落盘, 机制真实可用.

## metadata stamp family dispatch + WebUI model_family 表单 (阶段 6 配置收口)

阶段 6 配置收口的两个未闭合合取项: LoRA checkpoint metadata 不携带 family 标识
(加载侧 `create_network_from_weights` 无 `args`, 无法 family-dispatch), WebUI 全局
设置无 family 选择器。本轮闭合。

### A. ss_model_family stamp 闭环 (networks 层)

**定论**: target container stamp (`ss_unet_target_replace_modules`) 在阶段 6 探针
阶段已闭环 (`persistence.py:162` 写 / `factory.py:808` 读回), 探针靠它绕过 family
gap。但缺 family 标识本身 — metadata 无任何 `ss_model_family` 键, 加载侧拿不到 args,
只能从 `ss_unet_target_replace_modules` 间接推断, 无法支撑推理侧 DiT/TE/forward
family dispatch。

**落地**:
- `LoRANetworkCfg` 加 `model_family: str = "anima"` 字段 (`config.py:236`)。`from_kwargs`
  读 `kwargs["model_family"]` (bootstrap 注入, 见 B), 未知值回退 anima (不抛 — 错 TOML
  应训练 anima 路径而非中途 abort)。`from_weights` 加 `model_family` 形参, None/absent
  → "anima" 默认。
- `persistence.stamp_lora_save_metadata` 写 `ss_model_family` (`persistence.py:171-178`),
  **仅非 anima 时盖** — anima 省略 key 保 checkpoint 字节不变 (anima 是 load 默认,
  缺失即 anima, 见 `factory.py` 读回逻辑)。这是探针绕过 gap 的同款"省略即默认"约定。
- `factory.create_network_from_weights` 从 `file_metadata["ss_model_family"]` 读回
  (`factory.py:824-834`), 未知值 warn + 回退 anima, 传入 `from_weights(model_family=)`。

**smoke 验证** (无 GPU, 纯 metadata 路径):
- Krea-2 cfg stamp → `ss_model_family=krea2_raw` + `ss_unet_target_replace_modules=["SingleStreamBlock"]` ✓
- Anima cfg stamp → 空 metadata dict (无 `ss_model_family` 键, 字节不变) ✓

**测试** (`tests/test_factory_metadata_flow.py` 加 4 项):
- `test_ss_model_family_krea2_read_into_cfg` — Krea-2 stamp 进 cfg ✓
- `test_ss_model_family_absent_defaults_to_anima` — 缺失 = anima ✓
- `test_ss_model_family_unknown_falls_back_to_anima` — 未知值不抛 ✓
- `test_ss_model_family_round_trip_through_stamp` — stamp save 侧 anima 省略 key / Krea-2 写 key ✓

### B. bootstrap 注入 model_family (训练侧)

`library/training/bootstrap.py::build_net_kwargs` 在 net_kwargs 注入
`model_family = resolve_model_family(args)` (bootstrap.py:148-157)。**显式注入**而非
走 `NETWORK_KWARG_ALLOWLIST`, 因为 `resolve_model_family` 的 env 兜底 (`ANIMA_MODEL_FAMILY`)
不进 `args.model_family` — allowlist 循环只在 `args.model_family` 非 None 时注入,
会漏掉 env-only 的 family 选择。`from_kwargs` 见到该键即落到 cfg, save 时 stamp。
显式 `--network_args model_family=...` / TOML `model_family` 仍优先 (`if k not in net_kwargs`)。

### C. WebUI model_family 表单 (web 层)

**后端** (`web/services/settings_service.py`):
- 加 `GLOBAL_FAMILY_KEYS = ("model_family",)` + `_KNOWN_MODEL_FAMILIES = ("anima", "krea2_raw")`
  (镜像 `library/env.py::_KNOWN_FAMILIES`)。
- `_normalize_model_family`: 已知值返回小写, 未知/空 → `""` (不抛, 防 typo 砸面板)。
- save: `model_family` 存为 `""` (anima 默认) — 写显式 `model_family = "anima"` 会
  mask env 兜底链, 故 anima 选项存空, 面板选 Anima 即"留空走默认"。非 anima 写键,
  空/未知/anima → 删键。
- load: on-disk `anima` 也读回 `""` (env 兜底不被 mask)。`_default_global_settings` 含
  `model_family: ""`。

**前端** (`web/static/`):
- `index.html` 基础模型路径卡片 (02) 加 `<select id="global-model-family">` 两选项:
  `""` = Anima(默认) / `"krea2_raw"` = Krea-2-Raw。放 02 卡片避免动卡片 01-04 编号
  (`test_global_settings_cards_follow_requested_numbering_order` 守的契约)。
- `defaults.js` 加 `GLOBAL_FAMILY_FIELDS = [['model_family', 'global-model-family']]`,
  并入 `GLOBAL_SETTING_INPUTS`。settings.js 的 apply/collect 循环用 `.value` 对
  `<select>` 同样适用, 无需改 settings.js。

**测试** (`tests/test_settings_model_family.py`, 5 项):
- 默认空 / krea2 round-trip (save 返回 + on-disk toml + get 重读) / anima 存空 / 未知值
  存空 / on-disk 未知值读空。全过。

### 后续

- **推理侧 family dispatch 未串通**: `library/inference/{generation,models,text}.py`
  全部硬编码 anima (DiT 加载走 `anima_utils.load_anima_model`, denoise loop 调 anima
  forward 签名, 文本走 `AnimaTokenizeStrategy`)。子代理核实 generation.py 零
  `resolve_model_family` / 零 krea2 import。`ss_model_family` stamp 已让加载侧能识别
  family, 但 generation.py 的 DiT 加载 + 文本链路 + denoise loop family fork 是独立工作
  (非本轮 goal 五条件), 留阶段 6 后续。
- **WebUI 预设生成未接 model_family**: 当前新建空白预设从方法 TOML 继承
  `model_family` (base.toml=anima / krea2_lora.toml=krea2_raw)。全局设置的 `model_family`
  选择器是面板默认值, 尚未流入新预设生成的 TOML — 该连接是增强项, 非五条件要求。

## 已知限制 / 后续

- **未串通 train.py / generation.py**: 块交换和检查点都仅探针验证 (反上帝守则,
  不动 train.py / generation.py 热点文件). 正式串通是阶段 6 配置收口.
  - **create_network_from_weights family gap 已闭合**: `ss_model_family` stamp 闭环
    (见上 "metadata stamp family dispatch" 章节), 加载侧可从 checkpoint 识别 family。
    train.py/generation.py 的 DiT/TE/forward family fork 仍待续 (推理侧)。
- **block swap 大分辨率训练受限**: 32GB 卡靠 block swap 救不了 512×512+ 训练
  (激活瓶颈); 1024×1024 已通过 grad-ckpt 移植解决 (见上"1024×1024 梯度检查点
  正式训练"章节), block swap 仍只在权重显存不够的场景有用.
- **未测 train.py resume**: checkpoint save/load round-trip 验证了数值一致性,
  但没测 train.py 从 checkpoint resume 训练 (optimizer state / scheduler state
  未在本探针范围 — 探针只存 LoRA 权重, 不存 optimizer state). 1024 正式训练的
  checkpoint (92 MB) 已落盘, 可作为 resume 测试的输入.
- **未挂 LoRA 推理对比**: 阶段 5 推理探针只测 base model; 阶段 6 配置收口应验证
  加载 checkpoint → attach → sample → 与 base 对比风格可控. 推理侧 family dispatch
  (generation.py fork) 是前置 (见上 "后续").
