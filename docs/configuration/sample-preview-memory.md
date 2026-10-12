# 训练预览 VAE 解码策略

适用于 Anima、Krea-2、Z-Image 与 Qwen Image 2.1 的训练内预览和训练停止后的延迟解码。
前端 `/next/training` 的「训练计划 / 训练预览」提供通用字段「预览 VAE 解码」。
这是 VAE 解码策略；各模型族的 DiT 块交换仍按现有训练/预览契约执行。

```toml
sample_vae_decode = "tiled"
```

| 值 | 行为 |
| --- | --- |
| `tiled`（默认） | 主动在目标设备分块解码；不自动转 CPU |
| `auto` | 沿用 VAE 原 tiling 状态进行目标设备解码；仅 CUDA OOM 时尝试分块 GPU，再尝试分块 CPU |
| `cpu` | 分块 CPU 解码，保留 VAE 原精度 |

正常训练目标设备为 GPU；CPU 验证环境仍可调用相同接口。
分块可能增加耗时并改变局部像素。默认值用于减少预览解码的显存峰值，不保证峰值低于训练。
策略不改变训练 latent 缓存、模型权重或预处理编码策略。

## 模型差异

- Anima / Krea-2 共用本地 Qwen VAE；保留 `decode_to_pixels()` 内部的 mean/std 还原和 4D/5D 约定。
- Z-Image 使用 Diffusers Flux `AutoencoderKL`；保留 `latent / scaling_factor + shift_factor`。
  其默认 tiling 阈值等于配置中的 1024，开启 tiling 后解码 1024 图仍可能走整图。
  预览策略因此临时将像素 tile 上限设为 512，并按 VAE 空间倍率同步 latent tile，退出恢复原阈值。
- Qwen Image 2.1 保留专属 64 通道 latent、mean/std 和编辑参考图结果保存逻辑。

## 恢复与失败

解码前将 latent 保持在 CPU；具体解码回调按 VAE 当前设备搬运，CPU 分支不会提前搬 latent 到 GPU。
退出恢复 VAE 原设备、精度、tiling 开关和临时 tile 阈值。恢复失败为致命错误，各 rank 在尾部 barrier
前协调失败，不能让其他 rank 无限等待或继续使用不确定的训练状态。

CUDA OOM 回退会先清理异常帧与 VAE feature cache；普通错误不触发自动回退。
保存 PNG（以及已有的元数据/日志）成功后才删除对应 latent；普通单图失败保留 latent 供重试。

`torch.OutOfMemoryError` 与 `torch.cuda.OutOfMemoryError` 在当前 PyTorch 中可能是同一类型。
因此回退还要求错误明确包含 CUDA 与 out-of-memory 信息；CPU/host OOM、来源不明的 OOM、
NaN、超时和非法内存访问都不会降级重试。预览失败不发送训练 OOM 通知，也不修改训练精度、
batch、optimizer 或 `adaptive_oom_retry`。

进入 CPU 阶段前，先检查主机可用 RAM（尊重 cgroup v2 限制，不把 swap 当作 RAM）。
额外内存估算包含需要从设备复制的 VAE 参数/buffer、latent 转换、分块工作区及整图像素转换；
再保留至少 2 GiB 或可见总 RAM 的 10%。估算超出余量时，拒绝 CPU 迁移和解码，保留 latent。
分块工作区使用保守启发式，检查是一次性预算而非操作系统硬配额；其他进程、模型内部临时分配
和实际分块实现仍可能导致 CPU OOM。CPU OOM 只结束该图解码并保留 latent，不触发训练自适应重试。

预览事件独立追加到 `output_dir/sample/vae-decode-events.jsonl`，记录 latent 文件名、策略阶段、
attempt、明确 OOM、fallback 的下一阶段、CPU estimated/available/reserve 字节数、success、
reject 和 restoration_fatal。事件日志写入失败仅 warning，不把解码或训练变成失败。
success 表示解码返回图像；若随后的状态恢复失败，还会记录 restoration_fatal 并停止训练。

## 旧配置

`qwen_sample_vae_decode` 是 `sample_vae_decode` 的兼容别名；旧 TOML/CLI 仍可使用。
同一配置层同时包含新旧键时，新键优先，与 TOML 顺序无关。配置层间仍遵守正常合并顺序，
因此方法配置中的旧键会覆盖基础配置默认值，显式 `auto` / `cpu` 不会被新默认改写。
新界面只显示和写入通用键；读取旧配置时也能显示原策略。

## 验证边界

单元测试覆盖所有族的策略校验、别名合并、新键优先、latent 归一化、PNG 成功后删除、失败保留和恢复错误传播。
Qwen 1024 双图的 tiled/normal 实测见本地测试目录 `output/tests/qwen-vae-tiled-ab-20261004-r1/REPORT.md`。

2026-10-04 已使用四族真实 VAE，在 CMP 90HX 上验证 1024×1024、BF16 解码。
每策略预热一次、正式三次；下表为解码耗时中位数和 PyTorch allocated 峰值，
计时包含设备迁移、PIL 转换与状态恢复，不包含权重加载和 PNG 保存。
`auto` 在这些测量中均未触发 OOM，实际使用整图 GPU 解码。

| 模型族 | 整图 → 分块耗时 | 整图 → 分块峰值 GiB |
| --- | --- | --- |
| Anima | 0.71 → 1.37 秒 | 4.37 → 0.51 |
| Krea-2 | 0.70 → 1.36 秒 | 4.37 → 0.51 |
| Z-Image | 0.63 → 0.87 秒 | 2.55 → 0.77 |
| Qwen Image 2.1 | 1.68 → 2.40 秒 | 7.21 → 1.07 |

每次分块解码均确认进入真实 tiled 路径，设备、精度、tiling 与 tile 阈值恢复通过。
同一 latent 的整图/分块 PNG 对比中，Anima/Krea-2 差异较小；Z-Image 有轻微区域亮度变化，
当前样本未见明显硬接缝。需要优先保持整图解码结果时可选择 `auto`。
这是单一内容样本的解码验证，不代表任意内容、硬件、尺寸均有相同表现。

原始记录、像素指标和对比图见本地 `output/tests/all-family-vae-preview-20261004-r2/REPORT.md`。
Anima/Krea-2/Z-Image 使用原图重建 latent；Qwen 使用此前 LoRA 生成 latent，
不据此比较各模型的生成质量。该 1024 矩阵尚未覆盖真实 CPU 解码完成、CUDA OOM 后回退、
训练内完整采样后继续反向传播、续训或在线浏览器操作。

同日另补 Anima 真实 Qwen VAE 的 CPU BF16 小图验收：固定 `[1,16,1,8,8]` latent，
CPU 直接解码与 `auto` 的 GPU → tiled GPU → CPU 链均生成有限像素的 64×64 PNG，
保存后删除 latent，恢复 VAE 原设备/精度/tiling/tile 参数，训练配置未变。
两个 GPU 阶段各请求 2×目标总显存并产生真实 CUDA OOM；这是容量故障注入，
不能作为自然 1024 解码溢出的证据。CPU 阶段使用真实主机预算，未 mock 放行。

完整复跑 exit 0、无超时、无项目训练 supervisor 通知；CPU / auto 耗时 87.89 / 80.95 秒，
每秒采样 GPU 峰值 480 MiB、进程树 RSS 峰值约 1.20 GiB。CPU BF16 回退只是保底路径，
这个小图测试也较慢，不承诺大尺寸实用吞吐。其他族 CPU 解码及完整训练采样后的 backward
未据此验收。脚本、逐阶段事件、监控和 PNG 在本地
`output/tests/preview-cpu-fallback-90hx-20261004-r2/`；首轮脚本误断言记录保留。
完整恢复与预览证据见 [交付报告](../findings/anima_committed_oom_preview_delivery_20261004.md)。
