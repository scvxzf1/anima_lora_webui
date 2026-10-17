# MOSS-Image 文生图 LoRA 首版

状态：基础实现入口。真机训练、质量和性能证据需另附验收报告；配置中
`blocks_to_swap=20` 只是调参起点，不能据此推断任意硬件能运行。

使用 `configs/methods/moss_image_lora.toml`，模型族为 `moss_image`（别名 `moss`），
WebUI 显示为 MOSS-Image。组件路径相对项目根目录：

- DiT：`models/moss_image/transformer`
- 文本编码器：`models/moss_image/text_encoder`
- VAE：`models/moss_image/vae`

先准备本地组件权重和数据，再使用现有预处理入口建立磁盘 latent 与 text 缓存。
训练加载顺序保持文本编码器 → 缓存 → 释放、VAE → 缓存 → 释放，最后加载 DiT。
首版只支持普通文生图 plain LoRA，文本编码器始终冻结。

## 精度与缓存

`moss_te_quantization="nf4"` 是默认文本编码器路径：从已下载的 BF16 权重按需量化，
使用 BF16 compute。量化或依赖失败直接报错，不会自动切换 BF16。
保留 `moss_te_quantization="bf16"` 作为未量化对照。该字段已进入 CLI 和配置 schema。

DiT 要求 `mixed_precision="bf16"`、`base_compute="bf16"`；VAE 推荐 FP32
（方法配置 `no_half_vae=true`）。`base_compute="nf4"` 不代表文本量化，会被拒绝。

必须开启 `use_text_cache=true` 与 `use_vae_cache=true`，两者都使用磁盘缓存。
不同模型族的缓存不得混用。NF4 与 BF16 文本缓存也必须隔离：切换精度时选择新的缓存目录
或重新预处理；不要仅改精度开关后复用旧缓存。文本长度上限为 4096 token。

## 训练与预览

训练使用完整 gradient checkpointing、`torch` 或 `sdpa` attention、均匀采样的
1000-step flow grid，training shift 固定为 1。`blocks_to_swap` 允许整数 0..40，
0 关闭交换；启用交换时训练及每个 dataset 的 batch size 都必须为 1。

MOSS 不使用 P2 加权，方法配置固定 `p2_gamma=0`。默认每轮保存一个可恢复训练状态，
保留最近两个状态；状态包含 optimizer、scheduler 和训练步数，可用于中断后续训。

训练预览使用 Euler，默认 50 步、CFG 3、shift 5。训练 shift 与预览 shift 分开处理。
预览默认值由 MOSS 专属预览实现提供；普通推理/image-test 的支持以 registry 为准。

配置预检明确拒绝 compile、DiT NF4、其他 adapter、流水线并行、选择性/卸载 checkpoint、
自适应精度、AUTO block swap、weighted caption、TE 训练和高级损失。
训练只接受 L2 flow loss、`timestep_sampling="uniform"`、`weighting_scheme="none"`。
caption dropout 在顶层、general、dataset 和 subset 都必须为 0。
不要把其他模型的编译、路由或损失配置直接带入 MOSS；这些配置会报错。

## 验收范围

配置可解析、兼容规则与前端能力入口属于基础实现证据。
完整权重的 forward/backward、LoRA 保存重载、预览和恢复，以及 NF4/BF16 对照、
显存与性能，需要分别记录真机环境、命令和结果后再作为推荐依据。
