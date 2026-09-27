# 多模型支持现状

状态：稳定架构说明
适用版本：当前 `main` 工作树
相关代码：`library/models/family_registry.py`、`library/training/`、`library/models/<family>/`

本文记录当前已经落地的多模型边界及仍然存在的 Anima 耦合。它不是新增模型的未来方案，
也不把训练预览等同于通用推理能力。2026-08-10 的阶段性风险快照仍保留在
[后端多模型审计](findings/backend_multi_model_audit_20260810.md)；其中部分问题已经由后续
registry 和 fail-closed dispatch 解决，应以本文和实时源码为准。

## 当前模型族

`library/models/family_registry.py::MODEL_FAMILY_REGISTRY` 是模型族名称、别名和能力边界的
单一事实源。所有 family operation 必须为已注册模型族提供完整 handler；缺失 handler、未知
family、未注册推理模式或 sampler 都应失败，不得静默回退 Anima。

| 模型族 | 当前定位 | Adapter | 通用推理 | Attention |
| --- | --- | --- | --- | --- |
| `anima` | 默认生产模型族 | 完整方法面 | `single` / `batch` / `interactive`；`euler` / `er_sde` / `lcm` | registry 中声明的 Anima 后端 |
| `krea2_raw` | 第二生产模型族 | 开放所有网络变体选择 | `single` + `euler` | `torch` / `sdpa` / `flash` |
| `z_image` | 训练与训练预览 v1 | 开放所有网络变体选择 | 尚未注册；mode/sampler 集合为空 | `torch` / `sdpa` / `flash` (`flash_varlen`) |
| `qwen_image_2_1` | T2I LoRA 首阶段；CMP 90HX torch/Flash 指定组合各 60 步短测通过 | 仅 plain LoRA | 尚未注册 | `torch` / `sdpa` / `flash`（BF16） |

WebUI 能选择四个模型族，但这不扩大 registry 的能力集合。尤其是 Z-Image 的训练 sample
preview 只服务训练流程，不代表生图测试、独立 CLI 推理、batch 或 interactive 已可用。

Anima、Krea-2 与 Z-Image 开放网络变体选择，配置预检与 network 创建/加载不再套用
“非 Anima 仅 plain LoRA”的统一白名单。`supported_network_specs=None`、
`plain_lora_only=False` 表示允许尝试，不代表所有变体组合均已通过训练验证。
不兼容的权重、精度、目标层结构与 forward 契约仍在对应实现处报错，不静默替换成 LoRA。
Krea-2 与 Z-Image 的 `supports_method_adapters=False` 保留：IP-Adapter、EasyControl、BYG
等依赖独立训练 forward hooks 的方法不因开放网络变体而自动适配。
Qwen Image 2.1 的 `plain_lora_only=True`，不适用上述开放策略。

## Pipeline Parallel 规划能力

Anima、Krea-2 与 Z-Image 声明了双卡 PP 拓扑，并通过公共
`library/models/pipeline_parallel.py` 完成配置归一化、互斥校验和连续 block 范围规划。
Anima 使用 `blocks` 并均分 28/40 block；Krea-2 使用 `blocks` 并保留 28 block
的 13/15 启发式；Z-Image 只分配 `layers` 中的 30 个 main layers，不混入 refiner。

这些是 planner/probe 能力，不是生产 runtime。当前三族的 registry capability 都显式标记
`runtime_available=false`；CLI、WebUI、续训和队列均在合法规划后 fail closed，不会回退到
普通 DDP。完整设计和当前边界见 [双卡 Pipeline Parallel 阶段 1](proposal/krea2_pipeline_parallel.md)。

## 当前路由

### 训练

- `train.py` 仍保留 `AnimaTrainer` 类名作为兼容 facade；统一训练编排位于
  `library/training/`。
- `library/training/model_loading.py` 按 `model_family` 分派 text encoder、VAE 和 DiT 加载。
- `library/training/batch_step.py::_get_noise_pred_and_target` 使用完整 handler 表分派每个 batch；
  Anima、Krea-2 和 Z-Image 各自拥有训练 forward/target 契约。
- 非 Anima family 不通过 metadata fallback 猜测行为；配置、network metadata 和运行 family
  发生冲突时应由 compatibility/registry 层拒绝。

### 缓存和模型实现

- Anima 实现在 `library/anima/`，Krea-2 和 Z-Image 分别位于
  `library/models/krea2_raw/`、`library/models/z_image/`。
- text cache 通过 registry 隔离：Anima 使用 `_anima_te.safetensors`，Krea-2 使用
  `_krea2_te.safetensors`，Z-Image 使用 `_z_image_te.safetensors`。
- latent/cache schema 由各 family 的 strategy/latent 模块维护；不要通过改名让不同模型族
  共享不兼容 sidecar。
- 所有 family 继续遵守 lazy loading 顺序：text encoder 缓存并释放，VAE 缓存并释放，最后
  加载 DiT、apply adapter 并进入训练。

### 推理

- Anima 推理主链位于 `library/inference/`。
- Krea-2 通过 `library/models/krea2_raw/inference_runner.py` 走独立 single/euler 路径；
  Anima-only extras 必须显式拒绝。
- Z-Image 目前只有 `library/models/z_image/training_preview.py`。通用 image-test 和独立推理
  没有注册 handler，请勿从 WebUI family 下拉框推断其已支持。
- Qwen Image 2.1 的推理、image-test 和训练 sample preview 均未实现，入口显式拒绝。

## Qwen Image 2.1 首阶段边界

`configs/methods/qwen_image_2_1_lora.toml` 是 T2I plain LoRA 起始配置。三个组件使用
ComfyUI 单文件 safetensors：32-block DiT、Qwen3-VL 8B 和独立的 RGBA/64-channel VAE。
加载器在装载权重前核对完整参数名与形状；文本编码器的 ComfyUI `model.*` 命名映射到
Transformers `model.language_model.*`，VAE 转换卷积的 singleton 时间核，DiT 拆分
`img_mlp.gate_up`。文本以官方 T2I 模板左填充，取最终 decoder 层 RMSNorm 前 hidden，
使用独立 `_qwen_image_2_1_te.safetensors` 缓存。RGB 数据在 VAE 前补不透明 alpha，
64-channel latent 使用 VAE 的逐通道 mean/std，并隔离到 `_qwen_image_2_1.npz`。

训练 forward 使用未 patch 的 latent token、2x2 image slot mask、动态指数 shift 的
flow-matching sigma 和 attention-only LoRA。当前仅允许 BF16、`torch`/`sdpa` 原生 attention
或 opt-in `flash`，以及 full checkpoint。Qwen 专属 `transformer_blocks` block swap 和 per-block `torch.compile`
已接入，方法配置初始启用 `blocks_to_swap=24`、`torch_compile=true`、dynamic sequence；
编译发生在 LoRA apply/load 后，换块调度留在编译图外。Next 数据集预设现支持 Qwen Image 2.1
编辑 LoRA：每组使用同一配对名称的“编辑前”和“编辑后”两个 UI 子集，可有多组；保存时只有
“编辑后”进入训练蓝图，其 `reference_image_dir` 指向对应“编辑前”的图片目录。图片仍按
相对路径与 case-fold 文件名 stem 配对，缓存 Qwen3-VL image-slot 条件及参考图 VAE latent；
首版要求 batch size 1，不支持混用普通/正则子集或分阶段调度，
随机裁剪、翻转和颜色增强。Apply 会校验 Qwen family、plain LoRA、缓存开关及图像配对，并写入
`qwen_image_2_1_task="edit"`，不会启动训练。

Edit 已通过配对、缓存 schema/网格、DiT 序列形状、Apply 和 UI 单元测试，并以本地 tokenizer 和真实
`Qwen3VLProcessor` 对 64×32 合成参考图完成 CPU processor smoke（生成 2 个 image slots）；尚未加载
Qwen3-VL 模型权重执行文本/图像条件编码，也未完成 GPU 编辑训练热测。因此 swap/compile 虽被兼容层允许，
仍不代表已在 Edit 任务上完成硬件验收。`auto_block_swap`、CUDA Graph compile preset、NF4、
文本编码器训练、caption dropout、sample preview、通用推理和 PP 仍未接入。依赖 Diffusers 尚未发版的 Qwen 2.1 实现，
`pyproject.toml` 暂固定到包含官方 PR 的上游提交。Flash 使用 Qwen 专属 packed
FlashAttention 2 varlen：按真实 key-valid mask gather K/V，避免把 tokenizer 左填充误作有效前缀；
带因果三角 mask 的文本段仍走原生 SDPA，图像查询走 Flash varlen。`configs/methods/`
继续默认 `attn_mode="torch"`。CPU 小模型 forward/backward、外置 adapter 梯度与路由契约测试
通过；CMP 90HX 的 torch 和 Flash 各完成 60-step 单图运行及 LoRA 保存，报告分别见
[torch 热测](findings/qwen_image_2_1_90hx_hot_test_20260926.md)和
[Flash 热测](findings/qwen_image_2_1_flash_90hx_20260927.md)。当前 90HX 单次对照中 Flash
中位步时约 10.16 秒，torch 约 7.94 秒，因此 Flash 保持 opt-in，不作为速度推荐。
这些短测不代表训练收敛、样图质量、保存后重载/续训、其他 swap 数或其他硬件；`24` 是通过
测试的点位，不是跨硬件默认建议。

## Z-Image v1 边界

Z-Image 使用官方 Diffusers transformer、Qwen3 text encoder 和 Flux VAE 组件。当前加载器支持
Diffusers 目录及经过组件校验的单文件路径，并在训练前校验 latent geometry、affine factor、
transformer input/caption width 和 Qwen3 hidden width。

已经接通的契约：

- Qwen3 ChatML、倒数第二层 hidden state、固定 512-token hidden/mask cache；
- 独立 `_z_image_te.safetensors` 和 `_z_image.npz` sidecar；
- `(latent - 0.1159) * 0.3611` 归一化和 shift `6` 的 1000-step flow grid；
- BF16 SDPA/FlashAttention varlen、full gradient checkpointing、attention-only plain LoRA；
- 训练 sample preview；
- `library/models/z_image/block_swap.py` 对 30 个 main layers 的 block swap，refiner 与
  input/output 模块常驻。

`blocks_to_swap=0` 表示关闭；启用时实现要求
`1 <= blocks_to_swap <= len(model.layers) - 2`，当前 30-layer 模型即 `1..28`。adapter 复用
共享 `ModelOffloader`，并保留 checkpoint、training/inference switch、pause/resume 和 profile
协议；不要把它当作 Krea-2 裸 DiT 的复制实现。

以下能力在 v1 明确不可用或由兼容层关闭：

- 通用推理/image-test、batch/interactive mode 和任意推理 sampler；
- NF4、`torch.compile`、per-band dynamic-seq 和 selective/offloaded checkpointing；
- 依赖独立训练 forward hooks 的 MethodAdapter（如 IP-Adapter、EasyControl、BYG）；
- weighted caption、layer range、alternate loss/timestep scheme；
- 任意大于 0 的 dataset/subset caption dropout。

起始配置为 `configs/methods/z_image_lora.toml`。不要把“理论上可移植”的 adapter 写成当前
支持；能力扩展必须先更新 registry、compatibility、训练/推理 handler 和定向测试。

## 仍存在的耦合

多模型主链已经穿过 trainer forward，但尚未变成动态插件系统，以下耦合仍是维护边界：

| 区域 | 当前事实 | 维护要求 |
| --- | --- | --- |
| family registry | 显式静态注册，handler 必须覆盖全部 family | 新 family 同步补全所有 operation handler 和 fail-closed 测试 |
| 训练加载 | `model_loading.py` 已分派，但仍直接导入 Anima loader/compat | 不要把 Anima fallback 当成通用协议 |
| 训练辅助 | `noise_target.py`、`train_session.py`、`trainer_network_mixin.py` 仍有 Anima import | 新 family 先走显式分支，不要伪装成 Anima tensor/layout |
| Adapter | 部分高级方法仍绑定具体 block 名称、cross-attention 和 monkey patch | 所有模型开放网络变体选择，具体实现校验兼容性；独立 forward hooks 仍需逐项适配 |
| 推理 | Anima 与 Krea-2 有独立路径，Z-Image 尚无通用路径 | registry 集合为空时必须拒绝，不得借用 Anima sampler |
| Compile/bucket | Anima 支持 dynamic-seq；Krea-2 使用固定 token family；Z-Image 未验证 compile | family compatibility 必须关闭不适用参数 |

## 修改检查清单

新增或扩展模型族时至少同步检查：

1. `MODEL_FAMILY_REGISTRY` 的别名、cache、adapter、推理和 attention 能力。
2. `library/training/model_loading.py`、`batch_step.py` 及 preprocessing strategy 的完整 handler。
3. network target、metadata stamp、保存/加载和 resume round-trip。
4. WebUI family 过滤、preflight 文案、训练预览和 image-test 的真实能力差异。
5. compatibility matrix 是否对无效配置 fail closed，且没有“接受但忽略”。
6. family-specific cache suffix、latent normalization、text mask 和 4D/5D latent 边界测试。

相关回归入口包括 `tests/test_model_family_registry.py`、`tests/test_z_image_family.py`、
`tests/test_z_image_block_swap.py`、`tests/test_krea2_*` 和 WebUI family/compatibility 测试；实际文件名
以当前 `tests/` 为准。
