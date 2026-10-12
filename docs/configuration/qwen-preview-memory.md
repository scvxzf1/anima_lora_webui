# Qwen Image 2.1 训练预览显存策略

训练内采样默认复用当前训练 DiT 的块交换数量、传输精度与恢复方式。
VAE 解码使用独立策略；继承 DiT 配置不代表解码峰值不会超过训练。
块交换覆盖项仅用于 Qwen Image 2.1；VAE 解码策略已扩展到全部模型族，见 [通用预览解码](sample-preview-memory.md)。

## 配置和界面

Next 工作台的「训练计划 / 训练预览」提供以下字段：

| 配置键 | 默认值 | 行为 |
| --- | --- | --- |
| `qwen_sample_swap_mode` | `"inherit"` | 继承训练配置；`"custom"` 为自定义数量 |
| `qwen_sample_blocks_to_swap` | `0` | 自定义交换块数，0–30；0 表示全部驻留 GPU |
| `sample_vae_decode` | `"tiled"` | 默认分块 GPU；另可选 `"auto"` OOM 回退或 `"cpu"` CPU 解码 |

VAE 解码字段在所有模型族下显示；块交换覆盖字段仅在 Qwen 模型族下显示，交换数量在选择自定义后可编辑。
切回继承会忽略已保存的合法自定义数量，便于之后切回自定义。
CLI 可使用相同名称的参数；TOML、CLI 和 Web 预检共享配置校验。

例如训练仍为 swap24，采样临时增加至 swap28：

```toml
blocks_to_swap = 24
block_swap_transfer_dtype = "bf16"
qwen_sample_swap_mode = "custom"
qwen_sample_blocks_to_swap = 28
sample_vae_decode = "tiled"
```

自定义数量发生变化时，只允许 BF16 无损传输，并要求已结束当前训练更新、清空梯度。
它不改变训练配置文件，也不替换 adapter/optimizer 对象。训练关闭块交换而采样需要开启时，
加载阶段会在 apply/compile 前安装 count=0 的调度入口，避免采样时重新包装已编译 forward。
退出恢复原训练数量、模式和 RNG；恢复失败必须停止，不能继续使用不确定的驻留状态。
入口、采样和解码的致命恢复错误都会汇总到其他 rank，避免直接进入尾部 barrier。

不做去噪 OOM 后自动增加交换数量：当前 Qwen 没有动态 AUTO 训练契约，
增加交换数量也不保证解决参考图、CFG 或高分辨率激活导致的 OOM。
采样默认仍为 B，不启用 DiT K/V 缓存。

## VAE 解码

旧键 `qwen_sample_vae_decode` 作为 `sample_vae_decode` 的兼容别名；新旧键同层并存时新键优先。

默认 `tiled` 主动降低解码显存峰值；配置缺少此字段时也使用 `tiled`。
已有配置显式选择 `auto` 或 `cpu` 时继续按所选策略执行，不自动改写用户配置。

`auto` 先按 VAE 原有 tiling 状态在目标设备解码；仅遇到 CUDA OOM 时，
依次尝试分块 GPU、分块 CPU。`tiled` 强制在目标设备分块，不自动转 CPU；
`cpu` 强制分块 CPU。CPU 保留 VAE 原精度，可能明显变慢并消耗主机内存。
分块解码可能与整图解码有浮点或拼接差异。

VAE 搬入 GPU 本身的 OOM 也在降级范围内。失败会清理 VAE 中间 feature cache 和异常帧，
避免普通 Python 列表中的 GPU tensor 残留；非 OOM 错误不触发降级。
最终尝试恢复 VAE 原设备、精度和 tiling 设置，恢复失败明确向上抛出。
PNG/元数据保存成功才删除 latent；普通单图失败保留 latent 供之后重试。

现有 live decode 仍先将 DiT/adapter 暂存 CPU，再解码，最后恢复训练。
本次没有改训练用 VAE latent 缓存或文本编码器 offload 策略。

## 独立测试与验收

`scripts/qwen_image_2_1/sampling_benchmark.py` 读取训练配置的交换数量、传输精度和恢复方式，
同时使用上述采样覆盖与解码策略。日志记录训练/采样有效数量与实际解码分支。
此前用于固定条件 A/B 的 `sampling_prepare_ab.py` 默认仍要求 swap24/BF16/slab，
不会静默改变历史实验契约。
新版 benchmark 的逐图 `decode_s` 包含策略切换/设备迁移，与旧报告的纯解码计时口径不同。

2026-10-03 验证：

- 170 项后端定向检查通过，4 项依赖 CUDA 的检查跳过；包含 OOM 降级、状态恢复、
  latent 保留、配置预检、跨 rank 失败汇总和原有 Qwen 预览回归。
- benchmark/共享动态重配定向检查 21 项通过、6 项硬件检查跳过。
- 真实小型 Diffusers Qwen 模型与真实 ModelOffloader 的 CPU 检查 2 项通过：
  `0→1→0`、`1→0→1`，与重配前基线相等、权重不变、adapter 梯度有限且非零。
- 前端字段/表单 29 项通过；工作台 23 项通过。工作台首次默认 5 秒测试限时出现 4 项超时，
  单文件以 15 秒限时重跑全部通过；未修改测试断言或项目默认限时。
- 前端 typecheck 与隔离构建通过，构建位置 `/tmp/qwen-preview-memory-build-20261003`。

实施时 20203 和 CMP 90HX 有正在运行的 Qwen 编辑训练，因此没有重启服务、
发布新的线上静态入口或执行完整权重 GPU 热测。正在运行的进程不会自动采用本次修改；
待任务结束后重启 WebUI 并正常构建发布前端，方可在线使用新控件。
