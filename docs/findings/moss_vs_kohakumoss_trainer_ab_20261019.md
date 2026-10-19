# MOSS-Image A/B 对比与合并建议（krea2-webui vs KohakuMossTrainer）

状态：实验 / 审计报告（汇总 t1–t4，含性能与功能两条主线；**不含任何画质、收敛结论**）
适用版本：本项目 `krea2-webui` `dev` HEAD `9ef94b1a`；A 侧全部代码事实与本机实测采集自工作树
`krea2-moss-ab`（branch `codex/moss-ab-compare`，HEAD `b696ff46841d6fa25c20af6528b655635b377aaa`，与 dev 的共同祖先 `0d294920`）。
对比对象：KohakuMossTrainer — <https://github.com/KohakuBlueleaf/KohakuMossTrainer>（本地 `/home/scv/nvme0n1p1/训练器相关/KohakuMossTrainer`），branch `main`，HEAD `c73be898daab565dbefca94afbccdc45a34dae4e`。
相关代码（A）：`library/models/moss_image/`、`library/training/{moss_compat,compat_matrix,precision_policy}.py`、`library/models/family_registry.py`、
`library/runtime/{offloading,block_swap_config}.py`、`networks/`、`library/config/`。
相关代码（B）：`src/kohakumoss/`、`src/kohakumoss_kernels/`、`src/kohakumoss_data/`、`configs/moss/`、`docs/`。
相关配置（A）：`configs/base.toml`、`configs/presets.toml`、`configs/methods/moss_image_lora{,_t768,_t512,_tecap1024}.toml`。
上游盘点：`output/analysis/moss-ab/{current-baseline,target-baseline,perf-compare,feature-compare}.md`（t1–t4，本文只做汇总与判定，不重复两侧实现细节）。
相关证据（A，本机）：`docs/findings/moss_image_bf16_baseline10_50hx_20261017.md`、`docs/findings/moss_image_token_budget_tiers_20261017.md`、
`docs/findings/moss_image_tier_hot_50hx_20261018.md`、`docs/findings/moss_image_int8_blockswap_diagnosis_20261018.md`、
`output/runs/moss-image-ab-pcie-x16-20261017-151444/compare.md`（该运行目录在 `krea2-webui` 主工作树）。

## 0. 证据分级与硬规则

| 标记 | 含义 | 使用规则 |
| --- | --- | --- |
| `[代码事实]` | 源码可定位的 `路径:行号` 事实 | 本文结论的主依据。**双根约定**：A 侧路径相对工作树 `krea2-moss-ab`，`output/runs/*` 则相对 `krea2-webui` 主工作树（`krea2-moss-ab` 的 `output/` 只有 `analysis/`，见 §8.2）；B 侧路径前缀 `B ` 表示相对目标仓库根 `KohakuMossTrainer` |
| `[本项目实测]` | 本机 CMP 50HX 亲跑，或有落盘产出可回查 | 只在本文标注的硬件条件内成立 |
| `[对方自述]` | B 仓库文档/脚本的声明，**未在本机执行** | 一律不外推为本机结论，只做机制级引用 |
| `[算术推导]` | 由上述两类数字推算，本身不含独立实测 | 必须给出算式 |

统一硬件口径（全文默认）：**A = NVIDIA CMP 50HX，sm75/TU102，10240 MiB 标称 / 9646 MiB 空闲，225 W 上限，
PCIe Gen1 ×8，pinned H2D 实测 1.58 GB/s、D2H 1.67 GB/s `[本项目实测]`**；
**B 的实测数字来自 RTX 5090 32 GB / RTX 4090 24 GB / RTX 3090 24 GB / L4 / V100 32 GB / Tesla T10 / 2080 Ti，一律 `[对方自述]` 或 `[代码事实]`+其文档声明，不得跨硬件外推。**

三条贯穿全文的更正（写在这里，避免被历史段落带偏）：

1. **不存在"bf16 环境退化"**。`library/training/precision_policy.py:19-53` 的 `resolve_mixed_precision` 在 `GPU major < 8`
   且 `mixed_precision=="bf16"` 时**就地改写成 `fp16`**，日志措辞即 "bf16 autocast runs the slower fp32 emulation"。
   sm75 生产路径**从来只跑 fp16**，历史步时数字有效。bf16 的 7.12–7.14 TFLOP/s 只出现在绕过该策略的 bench 探针里
   （对比 fp16 81.70–84.67 TFLOP/s，11.5× 结构性慢路径，`[本项目实测]`）——**bf16 是策略已规避的死路，不是当前问题**。
2. **PCIe ×8 的代价是档位相关的，不是统一 +52%**。步时 ≈ `max(总 DMA, 总计算)` 而非相加：默认 1024 档短字幕
   5.666 → 5.73 s（**+1.2%**）、长字幕档 **0%**；但 768-short +67%、512-short +89%、256×256 **+96%**
   （`[算术推导]`，依据见 §2 与 perf-compare §4.2 的五个交叉验证点）。**×8 只打击"计算轻 + swap 深"的小分辨率档。**
3. **fp16 数值稳定性尚未通过**，这是"能否长训练"的前提，不是性能问题：`output/runs/moss-image-ab-pcie-x16-20261017-151444/compare.md:24-26` 记录
   FP16 臂 5 步全 NaN、scaler 65536→2048、第 6 步 CUDA 失败，内核日志 Xid 31 MMU Fault（ENGINE CE2），**根因未定位**；
   而同机 10 步 tier-hot 干净，两者矛盾未解释。另：`library/training/moss_compat.py:84-97` 的必须关闭项经逐项计数为 **39 项**
   （t1 记 35，以 39 为准）；A 的 `--ema*` 参数是**死代码**：`_save_ema_model` 确有调用点，但全部包在
   `if ema is not None` 分支内，而 `ema` 在整仓没有任何构造/传参路径（`EMAModel` 零实例化），条件恒假 —— 属"现状"而非"能力"。

## 1. 结论摘要（7 条，一行一条）

1. **采纳**：`P-C9` 误差统计落盘 + 阈值判据，以及 `F-E6`/`F-E5`/`F-B1` 三条测试与文档动作——零风险、零性能假设，且是后续所有优化的验收前提。
2. **采纳**：`P-C3` PCIe ×8→×16 纯诊断（含 7 步判定法）——但只能买到小分辨率档与"抬 swap 换显存"的余量，默认 1024 档仅 +1.2%。
3. **试点（最高优先）**：`P-C1` fp16 数值稳定性治理——不提速（本来就在 fp16），但它是长训练能否成立的前提；NaN/Xid 31 根因未定位前，一切加速都建在未验证地基上。
4. **试点**：`P-C2` elementwise 融合 kernel（situ_glu / rms_norm_mod / qk_norm_rope / gated_residual），预计 42 块 4.78→≈3.8 s/步（−20% `[算术推导]`），配套 `P-C4` 把 swap 从 20 重标定到 ~14.6。
5. **试点**：`F-A1` 把 A 已有的 LoHa/LoKr/GLoRA 放到 MOSS（按 Qwen 的 allow-list 形态，不动共享常量），配合 `F-C2` 放开 `loss_type`（一处白名单）。
6. **观察**：B 的显存体系（4 profile / 240 条 catalog / `estimate` 预估 / ring 流式）与调优表机制——机制值得记录，但在 10 GB + 1.58 GB/s 这条曲线上方向相反，可抄的是**激活侧**与**规划侧**，不是权重流式侧。
7. **拒绝**：fp8/MXFP8、权重全流式、在线 latent+TE 编码、全量微调 + Muon、直接引入 `lycoris-lora`、kogine"脚本即配置"、merged 导出。

## 2. 两侧基线对照

| 维度 | A 侧：krea2-webui（`b696ff46`） | B 侧：KohakuMossTrainer（`c73be89`） | 证据 |
| --- | --- | --- | --- |
| 模型实现 | 自研 `MOSSImageTransformer`：42 层 / 3072 宽 / 24 头 / 128 latent 通道，torch+sdpa，**逐样本真实文本长度、batch 恒 1** | 自研 `MossBackbone`：模块名等于官方 state-dict key，**packed varlen**（`SeqInfo`/`cu_seqlens`）一条 launch 覆盖多段 | `library/models/moss_image/dit.py:13-24`、`:67-70`；`B src/kohakumoss/model/backbone.py:27`、`B src/kohakumoss/model/opset.py:69-73` |
| 内核 | **无自研 kernel**：attention 无条件 `F.scaled_dot_product_attention`，`attn_mode` 只做合法性校验（torch 与 sdpa 实际同一条路径）；norm/situ 为 eager 组合 | 四后端 `tilelang/triton/compile/eager`，按 arch 定优先级；逐 `(kernel, device, key)` 计时落表，结果在**用户 cache 目录**而非仓库 | `library/models/moss_image/math.py:44`、`library/models/moss_image/dit.py:49-51`；`B src/kohakumoss_kernels/registry.py:26-40`、`B docs/internals/tuning.md:160-164` |
| 显存 | 共享 `ModelOffloader` 块换页：MOSS 只提供 42 块封装，上界 `len(layers)-2=40`，`swap>0` 时所有层级 batch 必须 1 | 4 个具名 profile + 240 条 catalog + `MEMORY_OVERRIDES` 全字段覆盖（checkpoint / stream / act_offload / optim / grad 五轴） | `library/models/moss_image/block_swap.py:64-77`、`library/training/moss_compat.py:121-125`；`B src/kohakumoss/memory/profiles.py:24-61` |
| 精度 | 配置写 `bf16`，sm75 上被 `resolve_mixed_precision` **就地改成 `fp16`**；`_RMSNorm`/time-modulation/head 内部升 fp32 再 cast，**残差流仍是 fp16**；loss scaler 由 accelerate 提供 | bf16 默认；fp16 溢出有 4 模式 fold（`none/weights/online/fp32_residual` + `clamp`），fold 可写进权重并用 safetensors metadata `kohakumoss_fp16_fix` 传递 | `library/training/precision_policy.py:19-53`、`library/models/moss_image/math.py:16`；`B src/kohakumoss/model/numerics/fp16fix.py:1-12`、`B src/kohakumoss/model/io/keymap.py:10` |
| 适配器 | **plain LoRA only**：registry `supported_network_specs={"lora"}`，门禁 39 项必须关闭；A 侧 LoHa/LoKr/GLoRA/VeRA/DoRA/T-LoRA/Hydra 等实现都在，只是被 MOSS 门禁挡住 | LyCORIS 全算法：`lora/locon/loha/lokr/oft/boft/glora/ia3/dylora/tlora` + `ALGO="full"` 全量微调 | `library/models/family_registry.py:210-212`、`library/training/moss_compat.py:84-101`；`B docs/concepts/adapters.md:26-38` |
| 数据 | `.txt`/`.caption`/`captions.json` + 每图 sidecar；TE 缓存 schema `moss_image_te_v2` **把 `max_text_tokens` 纳入身份**；resume 靠 `skip_first_batches` 重放 | 6 种源（folder/kohya/jsonl/parquet/webdataset/moss_shard）+ catalog 增量表；`latents.db`/`text.db` 三态缓存；规划是"数据+seed+旋钮"的纯函数（`seed_of=blake2b`） | `library/models/moss_image/cache_policy.py:37-64`、`library/training/loop.py:665`；`B src/kohakumoss_data/seeds.py:1-20`、`B docs/concepts/data.md:164-172` |
| 多卡 | MOSS `pipeline_parallel` 直接 `NotImplementedError`，registry `pipeline_parallel=None`；未读到 MOSS 专属多卡路径（`[推测]` 部分见 t1 §7） | `ddp`（可与 backward 重叠）/ `fsdp`（FSDP2）/ `pipeline` / `threaded`（单进程多设备、逐 stage 绑后端），**仅 Linux** | `library/models/moss_image/integration.py:70-71`、`library/models/family_registry.py:221`；`B docs/guides/multi-gpu.md:10-16` |
| 导出 | 只有 LoRA safetensors：`lora_unet_layers_{i}_proj_{q,k,v,o}`；`merged`/`diffusers` 导出无消费方（MOSS 无独立推理） | `FORMATS=(lycoris, comfy, merged, diffusers)`；`FUSED_QKV=True` 写 `layers.N.proj.qkv` 并能把 fused 拆回 q/k/v | `networks/lora_anima/network.py:57`、`tests/test_moss_image_family.py:220`；`B src/kohakumoss/adapters/export/fused.py:1-16`、`B docs/guides/export-to-comfy.md:85-95` |
| 配置 | TOML 四层合并链 + schema 校验 + `ConfigLayer` 来源追踪与合并快照落盘 | kogine"脚本即配置"：设置项是模块级大写常量，`use_config` 整值替换，后调用覆盖先调用 | `library/config/provenance.py:1-14`、`library/config/io.py:585-636`；`B docs/guides/configs.md:3-14` |

**一句话定位**：A 是"训练 + 训练预览专用族"（registry 的 inference mode/sampler 均为空集合，独立推理入口直接抛错，
`library/models/family_registry.py:213-214`、`library/inference/family_runtime.py:18-19`）；B 是"完整自研 MOSS-Image 训练栈 + 单卡到多卡的部署面"。
两者的交集只有"训 5.08B MOSS LoRA"这一件事，因此可借鉴项的判定标准是**能否在 10 GB sm75 上兑现**，不是功能多寡。

## 3. 性能与显存候选（P 组，来自 perf-compare）

字段：名称 / 收益 / 代价 / 风险 / 难度 / 建议 / 证据。编号前缀 `P-` 与功能组 `F-` 消歧（两侧源码文档里都出现过 "C1/C2"）。
所有收益数字都标注来源与硬件；B 侧倍数一律 `[对方自述]`，不得当作本机预期。

| # | 名称 | 收益 | 代价 | 风险 | 难度 | 建议 | 证据（路径:行号） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `P-C1` | fp16 数值稳定性治理（借鉴 B 的 4 模式残差 fold / fp32 residual / clamp） | 不提速（生产本来就在 fp16）；是"能否长训练"的前提 | 先定位 NaN/Xid 31 根因 + 三臂数值对照；B 的 fold 数学能否移植未验证 | 高 | M | **试点（最高优先）** | `output/runs/moss-image-ab-pcie-x16-20261017-151444/compare.md:24-26`；`B src/kohakumoss/model/numerics/fp16fix.py:1-12` |
| `P-C2` | elementwise 融合 kernel：`situ_glu`/`rms_norm_mod`/`qk_norm_rope`/`gated_residual` | fp16 下单块 114 ms 中 GEMM 仅 ≈41 ms，norm+situ ≈22–28 ms；融合后单块 ≈90 ms，42 块 4.78→≈3.8 s/步（−20%）`[算术推导]`；B 在 sm75 报 16–22× `[对方自述]` | 自研 Triton/TileLang kernel + 数值验收；bf16 下只有 Triton elementwise 可用 | 中 | L | **试点** | `library/models/moss_image/math.py:16`、`:20-21`；`B docs/performance/block-ops.md:14-25` |
| `P-C3` | PCIe ×8→×16 恢复（**纯诊断**，含 7 步判定法：nvidia-smi / lspci LnkCap-LnkSta / 父桥 / dmesg retrain / Xid / 反复采样 / H2D 实测） | 256×256 −49%、512-short −47%、768-short −40%，**1024-short 仅 −1.2%**、长字幕 0% `[算术推导]`；真正的价值是把 swap 抬到 28–36 换显存余量 | 只做诊断，不改 BIOS/硬件；无代码改动 | 低 | S | **采纳（先诊断）** | `output/runs/moss-image-ab-pcie-x16-20261017-151444/compare.md:18`；`library/runtime/offloading.py:2317-2344` |
| `P-C4` | swap 深度按膝点重标定 `S_cross = 42·C_b·BW/453MB` | ×8 + 现算力下膝点 19.8，现配 `blocks_to_swap=20` 恰压膝点；`P-C2` 落地使 `C_b`→100 ms 后膝点降到 14.6，可再吃 ≈0.5 s/步 `[算术推导]` | 改配置 + 一次档位热测 | 低 | S | **采纳** | `configs/methods/moss_image_lora.toml:26-27`、`configs/presets.toml:22-23` |
| `P-C5` | 激活 offload（把 Anima 的成对实现移植到 MOSS `_Block`） | B 在 5090 上 12.02→10.94 GiB、代价 +13 ms/步 `[对方自述]`；本机等价于"抬 4.7 块 swap"（≈1.34 s/步换 ≈1.08 GiB）但不占常驻块槽 | 端口 ≈30 行 + 数值验收 | 中 | S/M | **观察**（仅在撞 OOM 边界时启用） | `library/anima/models.py:93-137`、`library/models/moss_image/math.py:100-102`；`B src/kohakumoss/memory/profiles.py:24-61` |
| `P-C6` | 缓存存储 fp8（latent / TE cache 存 e4m3） | 字节减半，受益于磁盘与读取；**不改步时** | 缓存 schema 变更（A 的 TE 校验器强制 bf16）+ 约 7% 最大相对误差需验收 | 中 | M | **观察** | `library/models/moss_image/strategy.py:56-67`；`B docs/concepts/data.md:170-172` |
| `P-C7` | 显存预估器（借鉴 `estimate(work, plan)`） | 消除"首步 OOM"试错（1024-long 现状就是首步 OOM） | 自研估算表 + 本机校准 | 低 | M | **观察** | `docs/findings/moss_image_tier_hot_50hx_20261018.md:63`；`B src/kohakumoss/memory/estimate.py:227` |
| `P-C8` | 调优表机制（按卡/按 shape 落盘 op 选择） | 一次调优长期命中，是 `P-C2` 的天然配套 | 需稳定 problem key 与原子写；A 无该层 | 低 | M | **观察**（`P-C2` 落地后再谈） | `B src/kohakumoss_kernels/tuning/tuner.py:207`、`B docs/internals/tuning.md:79-101` |
| `P-C9` | 误差统计落盘 + 阈值判据（`MAX_ERROR_RATIO` 形式） | 不是性能项，但是 `P-C1`/`P-C2` 能否被接受的必要条件 | 流程成本 | 低 | S | **采纳（配套）** | `B docs/internals/fp16.md:36-46` |
| `P-C10` | packed varlen 批处理 | 多段场景 1.08–2.21× `[对方自述]`；A 单段恒在 1.08× 以下 | 需同时放开 batch>1，与块换页 batch=1 硬约束冲突 | 高 | L | **观察** | `library/models/moss_image/dit.py:67-68`、`B docs/performance/attention.md:40-43` |
| `P-C11` | 在线 latent+TE 编码 | 24 GB 卡上可省预处理阶段 | DiT 权重 9.467 GiB 已占 9646 MiB 的 98%，TE/VAE 无处常驻 | 高 | L | **拒绝** | `B docs/concepts/data.md:164-172`；`docs/configuration/moss-image-training-preflight.md:23` |
| `P-C12` | 权重全流式（ring, `stream_resident=2`） | B 在 5090 上 3.27 GiB 峰值 `[对方自述]` | 本机 ×8：18.12 GB/步 → **11.47 s 纯 DMA** `[算术推导]` | 高 | L | **拒绝** | `B docs/internals/offload.md:700-717`；`library/runtime/offloading.py:2317-2344` |
| `P-C13` | fp8 / MXFP8 / 权重-only fp8 | sm89+ 才有；A 侧 DiT 无量化分支 | 本机 sm75 无 fp8 指令 | 高 | L | **拒绝** | `library/models/moss_image/weights.py:6-8`；`B src/kohakumoss/memory/profiles.py:97-105` |
| `P-C14` | host optimizer / grad offload / selective & partial checkpoint | 面向 full FT | A 是 LoRA-only（无基座梯度）；B 自己数据显示 selective/partial 更费显存 | 高 | M | **拒绝** | `library/training/moss_compat.py:98-99`；`B docs/internals/offload.md:712-728` |
| `P-C15` | FSDP2 / pipeline / threaded 多卡 | 面向多卡节点 | 单卡定位无收益；A 侧 MOSS pipeline 未实现 | 中 | L | **拒绝** | `library/models/moss_image/integration.py:70-71`；`B docs/guides/multi-gpu.md:10-16` |
| `P-C16` | `base_compute` 与 `mixed_precision` 语义对齐（**文档级**） | 消除"配置写 bf16、实际跑 fp16"的认知偏差；直接关系到历史/WebUI 显示与可复现性 | 改文档与展示文案；**不要**改语义（会破历史配置） | 低 | S | **采纳（文档级）** | `library/training/moss_compat.py:45-46`、`library/training/precision_policy.py:19-53` |

### 3.1 性能面 Top 5（收益 × 可行性）

| 排名 | 候选 | 风险 | 一句话理由 |
| --- | --- | --- | --- |
| 1 | `P-C2` elementwise 融合 kernel | 中 | fp16 已就位后，剩余时间里 elementwise+attention 是主体（单块 114 ms 里 GEMM 只占 ≈41 ms） |
| 2 | `P-C1` fp16 数值稳定性治理 | 高 | 不提速，但 NaN/Xid 31 未定位，"fp16 能否长训练"是先决问题 |
| 3 | `P-C3` 链路 ×8→×16 诊断 | 低 | 零代码；价值集中在小分辨率与"抬 swap 换显存"，默认档仅 +1.2% |
| 4 | `P-C4` swap 膝点重标定 | 低 | 一改配置一测即得，且防住 `P-C2` 落地后继续用错 swap |
| 5 | `P-C9` 误差统计门禁 | 低 | 不是性能项，但它是 `P-C1`/`P-C2` 被接受的前提 |

### 3.2 两条必须记住的算术

- **步时模型**：`step ≈ max(2·S·226.5MB / BW, 42·C_b) + 未隐藏的还原开销`，由 5 个独立数据点交叉验证
  （256×256 ×16 2.93 s、256×256 int8 1.97 s、1024-short 5.666 s、1024-short int8 5.822 s、256×256 bf16 臂 3.0564 s，
  见 perf-compare §4.2）。它同时解释了两份 A 侧 findings 里"int8 在 256×256 有效、1024 档被否证"的现象。
- **可行性下限**：DiT 全部权重 9693.7 MiB = 9.467 GiB > 可用 9646 MiB `[算术推导]`+ `[本项目实测]`
  ⇒ `swap=0` 连权重都装不下，**块换页是结构性必需，不是优化项**。这条决定了 §6 的整张不可行清单。

## 4. 功能候选（F 组，来自 feature-compare）

编号沿用 t4 的分组字母（A=适配器、B=导出、C=训练目标、D=微调/数据、E=工程化）加 `F-` 前缀，与原文档一一对应，便于回查。
"需先改门禁"指必须先改 `library/training/moss_compat.py`；除 `library/training/compat_matrix.py:419-425` 这一处共享 allow-list 外，
其余门禁改动**只对 MOSS 生效**（`check_moss_training_contract` 仅在 `family_spec.name=="moss_image"` 时调用，
`library/training/compat_matrix.py:463-466`）。

| # | 名称 | 收益 | 代价 | 风险 | 难度 | 建议 | 证据（路径:行号） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `F-A1` | A 已有的 LoHa / LoKr / GLoRA 上 MOSS（需先改门禁） | 同一显存档位多一种表达力选择；三个算法在 A 里都已实现且只接 `nn.Linear`，MOSS 的 `proj` 正是 `nn.Linear` | 三处改动：`moss_compat` 白名单、registry `supported_network_specs`、`compat_matrix` allow-list 分支；4 条 family 测试 + 20 步真机冒烟 | 低（Qwen 先例已证形态；原估中） | S-M | **试点** | `library/training/compat_matrix.py:419-425`、`library/models/family_registry.py:210-212`、`library/models/moss_image/math.py:94-97` |
| `F-A2` | DoRA 上 MOSS（需先改门禁） | 同 rank 下更接近全量微调 | 构造期要读 base 行范数，而 `swap>0` 时 base 停在 CPU：数值 + 设备放置 + 换页三变量叠加 | 中高 | M | **观察**（先做 `F-A1`） | `networks/lora_modules/dora.py:43-45`、`library/models/moss_image/integration.py:22-51`、`B docs/concepts/adapters.md:46-49` |
| `F-A3` | 借鉴 B 的 `Kind` 记录抽象重整 save/load/merge 四面对齐 | 新增算法从"动四处"变成"填一条记录" | 引入第三套抽象；A 的 `attn_fuse` 真相源已部分收敛 | 低 | L | **观察** | `B docs/internals/adapter-kinds.md:11-32`、`networks/lora_save.py:204-232` |
| `F-A3'` | 直接引入 `lycoris-lora` 依赖 | 一次拿到 B 的全部算法 | 两套 `ss_network_module` 语义与两套保存格式；`sshs_model_hash` 元数据要重新对齐 | 高 | L | **拒绝** | `B pyproject.toml:24`、`networks/lora_save.py:280-291` |
| `F-B1` | 断言并文档化"MOSS LoRA 键布局与 ComfyUI MOSS-Image loader 一致" | 这是一条"其实已经对齐、但从未被验证"的资产 | 只补一条 key 布局契约测试 + 文档说明 | 低 | S | **采纳** | `tests/test_moss_image_family.py:220`、`B docs/guides/export-to-comfy.md:85-95` |
| `F-B2` | fused `proj.qkv` 文件读取（split-on-load）+ 未知键硬校验 | 修掉当前最危险的行为：`strict=False` 会把 `..._proj_qkv.*` **静默忽略**，adapter 变空操作 | 两步走：先"unexpected 计数 > 0 即报错/告警"，再 split-on-load；必须只对 MOSS 生效 | 中 | M | **试点** | `networks/lora_anima/persistence.py:258`、`B src/kohakumoss/adapters/export/fused.py:114-140` |
| `F-B3` | fused `proj.qkv` 写出（`FUSED_QKV=True` 等价） | 只在目标 ComfyUI 装 fused repack 模型时有用 | 需处理"非 plain LoRA 只能写精确 diff"分支；A 无 ComfyUI MOSS 消费方 | 低 | M | **观察** | `B src/kohakumoss/adapters/export/fused.py:47-78`、`library/models/moss_image/math.py:94-97` |
| `F-B4` | merged transformer 导出 | B 的单卡部署路径之一 | A 的 MOSS 无独立推理，merged 无消费方；为它开推理等于改掉族定位 | 中 | L | **拒绝** | `library/models/family_registry.py:213-214`、`B docs/guides/export-to-comfy.md:22-25` |
| `F-B5` | LyCORIS 原生格式 / diffusers 目录导出 | 与 LyCORIS/Comfy 生态互通 | A 无 LyCORIS 消费者；A 的 MOSS 本身就只吃 diffusers 目录 | 低 | L | **观察**（前者）/ **拒绝**（后者） | `B src/kohakumoss/adapters/export/convert.py:15-17`、`library/models/moss_image/weights.py:9-23` |
| `F-C1` | 新增可选 `moss` timestep 采样器（**默认不变**，需先改门禁） | 给 A 补一个"σ 分布"实验维度（B 的离散 logit-normal + uniform mix + σ=1 质量 + 静态 shift） | 三层改动：白名单、`family.py` 去掉 `uniform_grid_sigmas` 硬编码、新增纯函数模块；**`discrete_flow_shift` 门禁不要动** | 中 | M | **试点** | `library/models/moss_image/family.py:58`、`B src/kohakumoss/diffusion/timestep/moss.py:36-64` |
| `F-C2` | 放开 `loss_type`（l1/huber/smooth_l1/pseudo_huber，需先改门禁） | 通用损失已支持 5 种并在 `_flow_match_loss` 生效，MOSS 只被单值白名单挡在 `l2` | 一处白名单 + 一条定向测试；只影响 MOSS | 低 | S | **试点（性价比最高）** | `library/training/moss_compat.py:51-53`、`library/training/losses.py:310-326` |
| `F-C3` | B 的分阶段表 `TECH_REPORT_STAGE3`（shift 3.0→1.0） | 复现技术报告训练阶段切换 | 切换点是 140k/207k/274k 步，A 的 MOSS LoRA 是 3 epoch / batch 1，差 3~4 个数量级 | 低 | S | **拒绝**（作用域不匹配） | `B src/kohakumoss/diffusion/timestep/moss.py:14-19`、`configs/methods/moss_image_lora.toml:12-13` |
| `F-C4` | B 的 `WEIGHTING=x0`/`eps`（换目标空间） | 在 x0 或 ε 空间算损失 | 与 A 的 σ 空间权重不是同一维度；改目标空间易引入静默数值偏差 | 中 | M | **观察** | `library/runtime/noise.py:85-93`、`B scripts/train/moss.py:109` |
| `F-D1` | MOSS 全量微调（含 FSDP/流式） | 能训基座 | 16 B/参数 × 5.1B ≈ 81 GB 优化器态 `[算术推导]`；A 无 FSDP/pipeline；推翻"plain LoRA only"族定位 | 高 | L | **拒绝** | `library/models/moss_image/integration.py:70-71`、`B docs/guides/full-finetune.md:15-25` |
| `F-D2` | adapter 权重 EMA（**先把死参数改成显式报错**） | EMA 成本低（只有 adapter 大小）；长训练可能受益 | A 的 `--ema*` 参数存在但无实现：`_save_ema_model` 仅在 `if ema is not None` 下被调用，而 `ema` 从未实例化/传入，直接启用会**静默无效果**；若做需接 resume | 中 | M | **试点** | `library/anima/training.py:478-517`、`:742-758`；`B src/kohakumoss/training/ema/averagers.py:1-8` |
| `F-D3` | Muon 优化器 | B 的全量微调配方之一 | LoRA 场景下作用在 rank-32 A/B 上，Newton–Schulz 算力换不到收益；B 自己的单卡路径也退回 AdamW | 中 | L | **拒绝** | `library/training/optimizers.py:41-79`、`B src/kohakumoss/training/optim/optimizers/muon.py:1-8` |
| `F-D4` | counter-based 确定性采样（`seed_of`/`rng_of` 式） | 让"任意 step ↔ 任意样本"成为可计算坐标，resume 不再顺序重放 `skip_first_batches`（当前成本随已完成步数线性增长） | 只对 MOSS 或新增数据层落地，不动共享 sampler/test 面 | 中 | M | **试点** | `library/training/loop.py:665`、`B src/kohakumoss_data/seeds.py:1-20` |
| `F-D5` | 多源数据（jsonl/parquet/webdataset/shard） | 接大数据集与远端 shard | 要改数据层、预处理、WebUI 三处；A 的定位不是大规模数据训练 | 中 | L | **拒绝**（确有需求先只做 `jsonl` 单源） | `library/config/loader.py:193-240`、`B docs/guides/datasets.md:51-54` |
| `F-D6` | `CacheMiss` 式严格缓存语义 | 把"缓存没准备好就报错"变成可选行为（现在缺文件会静默重算） | 新增 `cache_miss_policy=recompute\|fail` 配置项 + 门禁联动 | 低 | S | **观察** | `library/training/moss_compat.py:70-74`、`B docs/concepts/data.md:164-178` |
| `F-E1` | 整体改用 kogine"脚本即配置" | 写新脚本零样板 | Python 源码即配置会绕过 schema 校验/WebUI 表单/队列存档三层 | 高 | L | **拒绝** | `library/config/schema.py:1-13`、`B docs/guides/configs.md:3-14` |
| `F-E2` | 按硬件生成起点配置（`kohakumoss-config new` 式） | 用户不必自己判断 swap 与分辨率档 | 与 WebUI 草稿/保存流程重复；建议先只做 CLI 侧 | 低 | M | **观察** | `configs/presets.toml:1-35`、`B docs/guides/configs.md:17-46` |
| `F-E3` | `doctor` 式环境自检 CLI（复用 preflight） | 纯 CLI/无界面部署也能拿到同一套诊断（nvcc/triton/显存） | 必须复用现有 preflight，否则出现两套检查 | 低 | S-M | **试点** | `web/services/config/preflight.py:329-330`、`B pyproject.toml:53-56` |
| `F-E4` | 把"不支持的并行组合"写成可执行门禁而非文档 | 用户不会跑到一半才炸 | A 对 MOSS 已等价（`NotImplementedError`），其它族仍缺显式校验 | 低 | S | **观察** | `library/models/moss_image/integration.py:70-71`、`B src/kohakumoss/training/checks/parallel.py:35` |
| `F-E5` | 细分硬件 pytest marker（sm 代数 / multi_gpu / real_weights） | `-m "cuda and not sm70"` 成为常规操作；A 现在只有笼统 `hardware` | 新增而非替换，改名风险低；需同步 `test-hardware` 命令与文档 | 低 | S | **采纳** | `pyproject.toml:95-103`、`B pyproject.toml:86-100` |
| `F-E6` | `tests/rules/` 式"仓库约定即测试" | A 有 432 个测试文件却没有一条守卫 AGENTS.md 里的硬约定（vendor 不许分叉、文档状态块必备字段） | 需要 `vendor-sync --check` 之类支撑；新增 3~5 条规则测试成本可忽略 | 低 | S | **采纳**（先做 2 条） | `B tests/rules/test_ascii.py:1-25`、`AGENTS.md`（ComfyUI Vendor 树） |

### 4.1 功能面 Top 5 与两条零风险项

1. `F-E6` tests/rules 仓库约定测试 —— 风险低 / S / 采纳。A 的硬约定目前只能靠人守。
2. `F-C2` 放开 MOSS 的 `loss_type` —— 风险低 / S / 试点，一处处白名单即生效，且只影响 MOSS。
3. `F-B2` fused `proj.qkv` 读取 + 未知键硬校验 —— 风险中 / M / 试点；"静默空载"是当前实际存在的严格性缺口。
4. `F-D4` counter-based 确定性采样 —— 风险中 / M / 试点；数据面里性价比最高的纯函数工具。
5. `F-A1` LoHa/LoKr/GLoRA 上 MOSS —— 风险低（按 t4 §7.0 的 Qwen allow-list 先例修正）/ S-M / 试点 / 需先改门禁。

两条顺手就能做的零风险项：`F-B1`（ComfyUI 键布局契约测试）、`F-E5`（细分硬件 marker）。

> **真正缺的不是能力，是对照**：A 侧 MOSS **从未做过画质/收敛对照**（`docs/configuration/moss-image-training-preflight.md:124`），
> 所以本表所有"放开后更好"都只是**机制收益**而非实测收益；这也是 `F-C1`/`F-C2`/`F-A1` 只能定级"试点"的原因。

## 5. 推荐合并顺序（P0 / P1 / P2）

三条排序原则：①**先判可用性再要速度**——`P-C1` 的 NaN 根因不结论，任何提速都建在未验证地基上；
②**一次只动一个归因面**——数值治理与 kernel 融合不得并行，否则失败无法归因；
③**门禁改动按"最小步进"放行**——每放一项都要有一条只对 MOSS 生效的证据。

### P0（前置地基，无此则后续全部不作数）

| 序 | 工作项 | 前置依赖 | 验收方式 |
| --- | --- | --- | --- |
| P0-1 | `P-C9` 误差统计落盘 + `MAX_ERROR_RATIO` 式判据 | 无；可立即做 | 一条命令产出 jsonl（cosine/abs/rel + worse-list），阈值判据可自动 pass/fail；落盘路径进文档 |
| P0-2 | `P-C1` fp16 数值稳定性治理：NaN / Xid 31 根因定位 → 三臂（fp16 现状 / fp16+fold / fp32_residual）对照 | 需 1024 档 + **显式 bypass** `train_session.resolve_mixed_precision` 的对照臂（256 档 DMA 主导，分辨不出计算差异） | 复现脚本 + 根因结论**无论有无都落盘**；三臂 loss 有限性与误差统计；不得只看 10 步 tier-hot 就判定通过 |
| P0-3 | `P-C16` 精度语义文档化（`base_compute=bf16` 与实际 `fp16` 的关系、`ss_mixed_precision` 记录时机核对） | 无 | `ss_mixed_precision` 元数据与实跑 dtype 一致；文档状态块与实际一致；不改语义 |
| P0-4 | `F-E6` tests/rules 两条（vendor 一致性、文档状态块必备字段）+ `F-E5` 细分 marker + `F-B1` 键布局契约测试 | 无；与 P0-1..3 可并行 | 新增测试在 `test-fast` 内可跑；`tasks.py test-hardware` 仍通过 |

### P1（性能主线，前置 = P0-2 有结论）

| 序 | 工作项 | 前置依赖 | 验收方式 |
| --- | --- | --- | --- |
| P1-1 | `P-C2` elementwise 融合 kernel（situ_glu / rms_norm_mod / qk_norm_rope / gated_residual） | P0-1 的误差判据 + P0-2 判定 fp16 可用；**不得与 P0-2 并行** | 同档位 10 步 ×2 组（eager/kernel）步时与显存；误差统计在阈值内；目标单块 114→≈90 ms |
| P1-2 | `P-C4` swap 深度按膝点重标定（`S_cross = 42·C_b·BW/453MB`） | P1-1 的实测 `C_b`（未做 P1-1 时保持 `blocks_to_swap=20`，膝点 19.8 已压线） | 改配置 + 一次档位热测；给出膝点算式与实测点的一致性；同时采集 `block_swap_profile_jsonl` 逐块时序 |
| P1-3 | `P-C3` PCIe ×8→×16 纯诊断（7 步判定法） | 无；可与 P1-1 并行（零代码） | 7 步判据逐条给出结论（物理 ×8 槽 / 协商掉级 / 训练期降级）；H2D 实测复核 1.58 vs 2.8–2.9 GB/s |

### P2（功能面，前置 = 门禁改动设计 + 质量对照方案）

| 序 | 工作项 | 前置依赖 | 验收方式 |
| --- | --- | --- | --- |
| P2-1 | `F-C2` 放开 `loss_type`（l1/huber/smooth_l1/pseudo_huber） | 仅需 `library/training/moss_compat.py:51-53` 白名单；`P0-1` 的误差判据 | MOSS 一份配置 + 一条定向测试；其它三族 family 测试全绿（证明只影响 MOSS） |
| P2-2 | `F-B2` 未知键硬校验（先做这一步，再做 split-on-load） | 无（不依赖门禁） | 构造 fused `proj.qkv` 文件加载时**报错**而非静默空操作；Anima/Krea-2 加载路径行为不变 |
| P2-3 | `F-A1` LoHa / LoKr / GLoRA 上 MOSS（复制 Qwen 的 allow-list 形态，不动共享常量） | `F-C2` 的门禁改法先落地（同一文件、同一评审面） | 三处改动 + 4 条 family 测试；20 步真机冒烟 loss 有限；**必须给出"Krea-2/Z-Image/Qwen 行为不变"的测试证据** |
| P2-4 | `F-D4` counter-based 确定性采样（限 MOSS 或新增数据层） | 无（纯函数工具） | "任意 step → 样本集合"可复算；resume 不再依赖 `skip_first_batches` 的内存态；四个族 resume 回归 |
| P2-5 | `F-D2` EMA：先把 `--ema*` 死参数改成"未实现即报错"，再评估实现 | P2-4（若要从 resume 恢复 EMA 状态） | 传 `--ema` 时明确报错（当前 `if ema is not None` 恒假，实际表现为静默无效果）；若实现则覆盖 save/resume 两条路径 |
| P2-6 | `F-E3` `doctor` 式自检 CLI（复用 preflight） | 无 | 复用同一套检查逻辑（不产生第二套）；无界面环境下可跑出与 WebUI preflight 同结论 |

> **不要做的事**：在 P0-2 有结论前做 `F-C1`（换训练目标分布）——那会在"数值不稳定 + 目标分布变更"两个变量叠加时彻底失去归因能力。

## 6. 10 GB sm75（CMP 50HX）上的不可行清单

判据统一：**DiT 权重 9.467 GiB 已占可用 9646 MiB 的 98%，且链路只有 ×8（pinned H2D 1.58 GB/s）`[本项目实测]`**。

| 项 | 为什么不可行或需要先解决前提 |
| --- | --- |
| fp8 / MXFP8 compute、fp8 权重、fp8 激活 | B 自己写死 sm89+（`B src/kohakumoss/memory/profiles.py:97-105`：`arch_of(device).has_fp8` + 文案 "needs fp8 GEMMs (sm89+)"）；sm75 无 fp8 指令，A 侧 DiT 也无量化分支（`library/models/moss_image/weights.py:6-8`） |
| `checkpoint=none`（权重常驻） | 权重 9.467 GiB > 9646 MiB 可用，**连一次前向都起不来** `[算术推导]` |
| selective / `partial:0.5` checkpoint | B 在 24 GB 卡上反而把峰值推到 23.28 / 25.14 GiB `[对方自述]`；A 门禁直接拒绝（`library/training/moss_compat.py:49`） |
| ring 全流式权重（`stream=ring, stream_resident=2`） | 本机 ×8 下 18.12 GB/步 = **11.47 s 纯 DMA**，已超当前整步时间 `[算术推导]` |
| packed varlen 批处理 | batch=1 是块换页的结构性约束（`library/models/moss_image/dit.py:67-68`、`library/training/moss_compat.py:121-125`）；不解除则段数恒为 1，varlen 收益 ≤1.08× |
| 在线 latent+TE 编码 | DiT 已占 98% 显存，TE/VAE 无处置放；B 的 12–13 GB 是 24 GB 卡上的结果 `[对方自述]` |
| `torch.compile` | A 门禁全禁（`library/training/moss_compat.py:85-97`）；B 的 block-ops 表显示 sm75 上 compile 只在小 N 有优势 `[对方自述]` |
| host optimizer / grad offload | LoRA-only 没有基座梯度可 offload（`library/training/moss_compat.py:98-99`） |
| 全量微调（含 Muon） | 16 B/参数 × 5.08B ≈ 81 GB 优化器态 `[算术推导]`；A 无 FSDP/pipeline（`library/models/moss_image/integration.py:70-71`） |
| FSDP2 / pipeline / threaded 多卡 | 单卡定位；本机第二张卡是 cc 6.1 / 4 GB 的 GTX 1050，异构且显存不足 |
| 文本编码器 fp8 训练 / TE 训练 | A 侧 TE 永久冻结，`requires_grad_(True)` 直接抛错（`library/models/moss_image/text_encoder_runtime.py:44-47`） |
| **绕过 `resolve_mixed_precision` 去跑 bf16** | 方向上仍是死路，但代价必须分档说：bf16 在 sm75 只有 fp32 仿真路径，**计算主导档（1024）**会退化到 ≈26 s/步（块级 fwd+bwd 4.6×）`[算术推导]`；而 **256×256 这类 DMA 主导档实测只差 1.04×**（BF16 3.0564 vs FP16 2.9399 s/步，`output/runs/moss-image-ab-pcie-x16-20261017-151444/compare.md:16`）。因此 4.6× 只能引作"计算主导档的块级上界"，不是本机当前代价（`library/training/precision_policy.py:19-53`） |

## 7. 未验证与风险声明

**只有文档级证据、没有真机验证的结论（本报告一律按 `[对方自述]` 或 `[代码事实]` 处理，不得当成本机结论）：**

1. **B 侧全部性能数字**均未在本机执行（无 GPU 运行、无 `kohakumoss-tune`）。其 sm75 block-ops 表只有 **Tesla T10**（`B docs/performance/block-ops.md:23` 的 "sm75 (Tesla T10), fp16"）——
   同属 TU102 系 sm75 但**是另一张卡**，本文只做机制级引用（如"eager→kernel 有 16–22× 空间"），倍数不可移植。
   本机能做的交叉校验只有一处：同一 `_situ` 组合在 CMP 50HX / 4235 token 上 fwd+bwd 为 11.95 ms（原生累加）/
   20.52 ms（fp32 累加，即 B 的 eager 参考口径）`[本项目实测]`，与 T10 的 25.89 ms 同量级。
2. **fp16 数值失败根因未定位**：`output/runs/moss-image-ab-pcie-x16-20261017-151444/compare.md:24-26` 记录
   FP16 臂 5 步全 NaN、scaler 65536→2048、`effective=5/skipped=0`（说明 **scaler 拦不住这类 NaN**，NaN 不是 overflow 产生的）、
   第 6 步 CUDA 失败于 MOSS `q_norm` 调用处（异步错误栈不证明该处是根因），内核日志 Xid 31 MMU Fault（ENGINE CE2）。
   **与生产 10 步 tier-hot 干净之间存在未解释的矛盾**——在矛盾解释之前，"fp16 可长训练"不能算已证。
3. **质量对照从未做过**：A 侧 MOSS 没有画质/收敛对照基线（`docs/configuration/moss-image-training-preflight.md:124`），
   所以本文所有"放开后更好"都是机制判断，不是实测收益。
4. **256×256 两处 A 侧记录不可比（最可能为另一块物理卡 + ×4 链路，属推断；不是口径差异）**：2026-10-17 基线的 **11.3386 s/步**
   （`docs/findings/moss_image_bf16_baseline10_50hx_20261017.md:38`）与 2026-10-18 传输诊断的 **2.9258–2.9312 s/步**
   （`docs/findings/moss_image_int8_blockswap_diagnosis_20261018.md:292`）差 3.87×，三条例证说明两者不是同一条件：
   ①基线 run 的 `output/runs/moss-image-bf16-baseline10-20261017-085341/environment.json` 记 `gpu_uuid = GPU-bfda6d91-…`，
   与当前在机 GPU（`GPU-46c7d0db-…`）是**两块不同的物理卡**；②同仓原文已写明"历史x4约11.3秒/步属于另一批运行。
   此次重插后链路/UUID/硬件状态改变"（`output/runs/moss-image-ab-pcie-x16-20261017-151444/compare.md:30`）；
   ③算术自洽：9.06 GB / 11.3386 s = **0.799 GB/s**，正是 Gen1 **×4** 的量级（×16 实测 3.09 GB/s 的 1/4 = 0.773 GB/s，差 3.4%）。
   第三点对照：同日重插后 **×16** 的同配置 BF16 臂为 **3.0564 s/步**（`output/runs/moss-image-ab-pcie-x16-20261017-151444/compare.md:16`）。
   ⇒ 11.3386 s/步 **最可能属"另一块卡 + ×4 链路"的运行**（属推断而非实测：证据②是仓库内陈述而非独立测量、证据③只是量级吻合、基线那次运行的链路宽度无直接记录），与 1024 档数据不可比；这**与 checkpoint 保存无关**
   （该 run `output/runs/moss-image-bf16-baseline10-20261017-085341/config.toml:40` `checkpointing_epochs=0`、`:52` `save_state=false`，且被引 findings 本身写明中途保存关闭）。
   本文的档位结论因此只用 1024 档与"步时 ≈ max(DMA, 计算)"模型，不用 256 档的绝对秒数做任何推断。
5. **步时模型是 5 点拟合的 `max()` 近似**，偏粗。A 已有 `block_swap_profile_jsonl`（`configs/base.toml:102`）
   可导出逐块等待/提交时序，建议在 P1-2 一并采集，把模型从"两点夹逼"变成"逐块可解释"。
6. **B 的两条等价性数字属纯自述**：fp32 前向相对误差 4.7e-6 与 50 步 LPIPS 2.3e-6 全仓唯一出现在
   `B README.md:21-23`，无结果文件；复现需真实权重 + 2 GPU + `lpips`。`约 0.85 images/s` 演示
   （`B README.md:10-12`：**RTX 4090**、rank 8、fp16、fp8 base GEMMs、batch ~4、在线 TE+VAE 编码）
   没有命令与树 id → 仅 `[对方自述]`，不得外推。
7. **适配器冒烟未做**：`F-A1`/`F-A2` 的"只接 `nn.Linear` 就能套到 MOSS `proj` 上"是代码契约推断
   （`library/models/moss_image/math.py:94-97`），未真机跑通；`F-A2` 还多一层"换页时 base 在 CPU"的设备放置问题。
8. **ComfyUI 真实消费未验证**：B 文档要求"ComfyUI MOSS-Image loader"，该节点不在 B 仓库内，A 的 `custom_nodes/` 也没有 MOSS 节点。
9. **resume 成本未计时**：`F-D4` 的收益（"A 的 resume 成本随已完成步数线性增长"）是从 `skip_first_batches`
   （`library/training/loop.py:665`）推断的机制结论，未做计时对比。
10. **A 侧 MOSS 的多卡行为未验证**：代码中未见 MOSS 专属多卡路径，也未读到显式 world_size 校验（t1 标 `[推测]`）。
11. **链路 ×8 与 Xid 31 的因果关系未证实**：内核日志只把排查指向"拷贝引擎的非法地址读取"，
    不能单凭该日志区分软件异步生命周期问题与硬件/驱动问题。
12. **本报告未执行任何训练或推理**；A 侧数字全部转引自已落盘的 findings 与本轮 perf-compare 的探针脚本
    （脚本放在 `/tmp`，不属交付物）。

## 8. 可追溯性

### 8.1 上游输入与本仓 MOSS 线索

两侧实现盘点与对比分析的原始材料（本文只做汇总与判定）：

- A 侧基线：`output/analysis/moss-ab/current-baseline.md`（t1，298 行，249 条 `路径:行号` 全部校验有效）
- B 侧基线：`output/analysis/moss-ab/target-baseline.md`（t2，173 行，104+ 条引用）
- 性能/显存对比：`output/analysis/moss-ab/perf-compare.md`（t3，459 行，72 条引用）
- 功能面对比：`output/analysis/moss-ab/feature-compare.md`（t4，406 行，272 条引用）

以上四份在 `output/` 下（该目录由 `.gitignore` 忽略，不随版本发布）；可长期引用的实体证据是同目录的本仓 findings：

- [MOSS-Image BF16 配置 CMP 50HX 十步基准](moss_image_bf16_baseline10_50hx_20261017.md)：256×256 / 22-token / plain LoRA rank32 / swap20 的十步基线。
- [MOSS-Image 分辨率与图像 token 预算档位](moss_image_token_budget_tiers_20261017.md)：1024/768/512 三档的 bucket 表、塌缩陷阱与 attention 成本模型。
- [MOSS-Image 档位热测试：CMP 50HX 十二行实测](moss_image_tier_hot_50hx_20261018.md)：本文全部 1024 档步时与显存数字的来源。
- [MOSS-Image 块交换传输诊断：int8 的 0.94 s 与 3.5 GiB 归属](moss_image_int8_blockswap_diagnosis_20261018.md)：int8 传输收益、链路下限与"1024 档被否证"的来源。

### 8.2 机械校验（本次实际执行）

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 空白/冲突标记 | `git diff --check -- docs/findings/moss_vs_kohakumoss_trainer_ab_20261019.md docs/findings/README.md` | **clean（exit 0）**；新文件另用 `git diff --no-index --check /dev/null` 复核，0 处空白错误（exit 1 仅代表"与 /dev/null 有差异"） |
| 相对链接存在性 | 一次性 Python 脚本（`/tmp`，不新增常驻文件）剥离行内代码后逐条解析相对链接并检查文件存在 | **4 条相对链接全部存在，0 坏链**（脚本先对植入坏链的副本做负向自测，能正确报错） |
| `路径:行号` 可定位性 | 同一脚本：正则抽取全部 `路径:行号`（含 `B ` 前缀与行号区间）。**计数规则**：每个 `path:N` 或 `path:a-b` 记 1 条；裸续写 `:N` / `:a-b` 归属同一行或上方最近的前置路径，也各记 1 条（行号区间算 1 条、不拆分） | **151 条全部有效，0 条无效**。**双根**：A 侧 = `krea2-moss-ab` 工作树 + `krea2-webui` 主工作树（`output/runs/*` 的引用只存在于后者）；B 侧 = `KohakuMossTrainer`。只用 `krea2-moss-ab` 一根复算，那几条会被误判为 file-missing。负向自测植入的假引用被正确判为 file-missing |
| 表格完整性 | 同一脚本：逐张表格统计每行未转义竖线数，要求同表一致 | **10 张表格列数全部一致，0 处断表** |
| 行数约束 | `wc -l` | **285 行**（约束 250–400） |
