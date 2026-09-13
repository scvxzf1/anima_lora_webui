# 预处理 Auto Batch 实测与消融

状态：实现与定向验证完成；完整模型族硬件验证受本机资源限制。
日期：2026-09-06
配置说明：[preprocess-auto-batch](../configuration/preprocess-auto-batch.md)
原始记录与聚合：[JSON 证据](assets/preprocess_auto_batch_20260906.json)

## 环境与保护边界

- RTX 3080 10GB，PyTorch 2.12.0+cu130，BF16，4 CPU threads。
- 桌面/远程桌面常驻约 1.6-1.8GB；未终止任何服务、未启动训练、未改功耗/风扇。
- GPU 探针设 allocator fraction=0.75；Krea 模型加载探针为 0.80。
- 使用现有 Qwen Image VAE（chunk=64、disable_cache=True）及 Anima Qwen3-0.6B 权重。
- 基线 VAE 为确定性 synthetic pixels；真实图片另做缓存 smoke，所有生成缓存只写
  `/tmp/preprocess-auto-20260906/` 下的新目录，不覆盖用户缓存。
- 测量范围为 H2D + 实际编码 + CPU 输出传输；编码消融不包含模型加载或磁盘保存。
  完整缓存 smoke 包含图片 decode、保存、变体生成等端到端开销。

## 先测量，再决定算法

固定批次各测 3 次，第一次保留但不当稳态。部分代表值：

| 工作负载 | batch | allocated peak (GiB) | 稳态吞吐范围 |
| --- | ---: | ---: | ---: |
| VAE 512² | 1 | 0.578 | 11.92-11.97 image/s |
| VAE 512² | 4 | 1.590 | 12.43-12.61 image/s |
| VAE 512² | 16 | 5.642 | 12.73-12.74 image/s |
| VAE 1024² | 1 | 1.479 | 2.95-2.96 image/s |
| VAE 1024² | 2 | 2.720 | 2.99-3.00 image/s |
| VAE 1024² | 4 | 5.201 | 2.80-2.97 image/s |
| VAE 1024² | 8 | OOM | 后续 batch 1 成功恢复 |
| Anima text 短 caption | 1 | 1.132 | 14.47-14.85 caption/s |
| Anima text 短 caption | 4 | 1.172 | 40.21-40.41 caption/s |
| Anima text 短 caption | 16 | 1.333 | 43.51-43.58 caption/s |
| Anima text 短 caption | 64 | 1.978 | 43.73-43.79 caption/s |

结论：VAE 显存增量近似线性，适合保守预测；吞吐却不随 batch 线性增加。
文本从 1 到 4 有明显收益，8 之后逐渐平台化。当前 max-padding 使长短 caption
峰值接近，不能用可见字符数粗暴推算显存，也不能裁掉 padding。

据此实现 `1 -> 2 -> 4 -> 预测容量内至多4倍上拉`，显存余量、两次有效测量、
吞吐平台收敛、失败折半与失败上界。上限为 32，CPU auto 为 1。
每个 resolution 用独立策略，策略不改变 precision、模型加载、缓存 schema 或训练配置。

## VAE 消融

每个模式处理 96 张，2 次重复，第二次反向执行模式顺序。表内时间为均值，
peak 为成功批次的最大 allocated peak，包含探测期，而非收敛后常驻显存。

| 分辨率 | 模式 | 秒/96张 | peak GiB | 最终 batch | 两次合计 OOM |
| --- | --- | ---: | ---: | --- | ---: |
| 512² | 固定 2 | 7.891 | 0.914 | 2 / 2 | 0 |
| 512² | 完整 auto | 7.806 | 5.642 | 2 / 2 | 0 |
| 512² | 去预测，仅倍增 | 7.838 | 2.942 | 2 / 2 | 0 |
| 512² | 去吞吐收敛 | 7.620 | 5.642 | 16 / 16 | 0 |
| 1024² | 固定 2 | 32.156 | 2.720 | 2 / 2 | 0 |
| 1024² | 完整 auto | 32.696 | 5.202 | 1 / 1 | 0 |
| 1024² | 去预测，仅倍增 | 33.744 | 6.443 | 1 / 1 | 4 |
| 1024² | 去吞吐收敛 | 32.182 | 5.202 | 4 / 4 | 0 |
| 1024² | 去预测且去退避 | 中止 | 不报告 | 8 失败 | 2 |

注意：`no_backoff` 特意同时关闭预测以制造相同 OOM 路径，**应与 no_prediction
比较退避效果，不能与完整 auto 当作单因素比较**。去退避的两次运行均只完成 15/96 张。
初版探针对中止运行汇报 peak=0 是缺失值，不是零显存；报告表不采用此值，脚本已修正。

- 预测将 1024² 的探测 OOM 从每次 2 个降到 0，均完成全部数据。
- 退避让同一无预测路径从中止变成完整处理，无漏项；成功恢复后还能在安全边界内继续试探。
- 吞吐收敛使稳态 batch 更小，但不减少已经发生的探测峰值。512² 的大 batch 仅快几个百分点，
  不值得宣称为显著加速；1024² 完整 auto 比已知适用的固定 2 多耗时约 1.7%。

## 文本消融

Anima Qwen3-0.6B，每模式 192 条，短 caption 为同一句，长 caption 重复该句32次，
经原 tokenizer 截断/pad。每个模式 2 次、顺序反转；未加载可选 LLM adapter。

| caption | 模式 | 秒/192条 | peak GiB | 最终 batch |
| --- | --- | ---: | ---: | --- |
| 短 | 固定 16 | 4.875 | 1.333 | 16 / 16 |
| 短 | 完整 auto | 5.279 | 1.548 | 16 / 4 |
| 短 | 去预测，仅倍增 | 5.366 | 1.333 | 8 / 4 |
| 短 | 去吞吐收敛 | 5.100 | 1.548 | 32 / 32 |
| 长 | 固定 16 | 5.065 | 1.333 | 16 / 16 |
| 长 | 完整 auto | 5.376 | 1.548 | 16 / 16 |
| 长 | 去预测，仅倍增 | 5.506 | 1.333 | 8 / 8 |
| 长 | 去吞吐收敛 | 5.347 | 1.548 | 32 / 32 |

全部成功，无 OOM。短任务 auto 探测比已知好用的固定 16 多耗时约 6-8%，
预测上拉比仅倍增略快，但两次重复不足以证明普遍性能优势。短 caption 最终档位有波动，
这是桌面负载、计时噪声和平台区间共同作用的实测限制，不应宣传自动找到严格最优 batch。

## 数值与持久化验收

BF16 改 batch 后可能选择不同 kernel。基线 VAE 512² 相对 L2 约 0.29%，
1024² 在此输入上为 0；Anima prompt embeddings 相对 batch 1 约 1.5-1.9%。
这些是数值差异观测，不是训练质量等价证明。没有改变 token、mask、padding 或 latent affine。

真实 cache smoke 用4张现有图片构造 64 张混合512²/1024²输入，输出隔离，
比较固定2与auto的全部缓存字段、dtype、shape、metadata和有限值；非浮点字段必须精确一致。
文本为每图3变体，float 相对 L2 门限为3%，仅用于本次数值 smoke，不作为质量指标。

验收曾发现 Rich 日志渲染消耗全局 Python RNG，使新增调档日志改变后续 shuffle。
独立实际 encoder RNG 探针证明编码本身不改变 Python RNG，CPU 日志探针证实日志会改变。
已在 auto 遥测边界保存/恢复 RNG，补回归测试，未放宽 token ID 或 mask 的比较要求。
首次失败缓存保留在隔离实验目录供诊断，不覆盖为通过结果。

修复后完整重跑通过，记录见 [缓存 smoke JSON](assets/preprocess_auto_cache_smoke_20260906.json)：

| 路径 | 文件数 | 逐bit一致文件 | 最大相对L2 | 二次跳过 |
| --- | ---: | ---: | ---: | ---: |
| VAE | 64 | 34 | 0.003410 | 64 |
| Anima text，3变体 | 64 | 57 | 0.015239 | 64 |

全部非浮点 token/mask 精确一致，metadata/shape/dtype 一致且无非有限值。
二次运行没有任何缓存文件 mtime 变化。该轮固定/auto端到端时间分别为
VAE 15.933/15.410秒、文本6.231/5.884秒；仅单次 smoke，且固定模式先跑、auto后跑，
不能拿这组时间作为速度提升结论。
[encoder RNG 探针](assets/preprocess_encoder_rng_20260906.json)保留4次真实编码的随机状态检查。

## 未验证范围

- Krea-2 Qwen3-VL BF16 在本机桌面显存占用下，0.80 allocator 限额的模型 `.to(cuda)`
  已 OOM，未进入 batch 1 forward。不是 auto 回退失败，不通过关闭桌面或改模型精度掩盖。
- Z-Image 配置的外置模型文件在本机缺失，未下载。Krea/ZImage 通过实际缓存策略 +
  小型替身编码器验证变体顺序、shape/schema、跳过及注入 OOM；这不等于真实模型硬件消融。
- Anima 可选 LLM adapter / DOP 在小型替身测试覆盖；本次无真实 adapter GPU 验证。
- 无跨GPU、长训质量或多进程竞争测试，不把 RTX 3080 的结果外推为所有硬件最优。

## 回归检查

- 最终定向集 **146 passed**：auto policy/cache、preprocess paths/reuse/dataset、
  Krea text cache、ZImage family、runtime CLI，以及3项预处理/帮助前端契约。
- 新增/修改 Python 的定向 Ruff 检查通过，新增文档相对链接与 `git diff --check` 通过。
- 全仓 `timeout 300 python tasks.py test-core` 在约75%进度超时，期间出现失败，
  未完成逐项定位，**不能宣称全仓回归通过**。工作树本来已有大量其他功能未提交改动。
- 单独前端回归确认旧测试仍要求已被其他改动替换的底模帮助文案；全帮助摘要一致性测试的
  差异为 `dim_from_weights`、`max_train_epochs`、`max_train_steps`、
  `save_last_n_epochs`、`checkpointing_last_n_epochs`、`sample_ratio`，不涉及这三个auto字段。
  本次保留这些用户改动，不跨范围修订对应断言或文案。

## 复现入口

```bash
# 固定batch数据；输出文件要求不存在，防止覆盖旧证据。
python -m scripts.experiments.preprocess_batch_probe --kind vae \
  --weights models/vae/qwen_image_vae.safetensors \
  --output /tmp/new-vae-baseline.jsonl --sizes 512 1024 --repeats 3

# VAE消融；no_backoff与no_prediction组成退避单因素对照。
python -m scripts.experiments.preprocess_batch_probe --kind vae \
  --weights models/vae/qwen_image_vae.safetensors \
  --output /tmp/new-vae-ablation.jsonl --ablate --sizes 512 1024 \
  --items 96 --repeats 2 --modes fixed2 auto no_prediction no_throughput no_backoff

# 文本测量同一脚本：--kind anima_text --weights <local-qwen3-0.6b> --sizes 1 32。
# 真正缓存端到端验收，输出目录必须是新目录。
python -m scripts.experiments.preprocess_cache_smoke \
  --source-dir post_image_dataset/resized --output /tmp/new-cache-smoke \
  --vae models/vae/qwen_image_vae.safetensors --text <local-qwen3-0.6b>

python -m pytest -q tests/test_preprocess_auto_batch.py tests/test_preprocess_auto_cache.py
```
