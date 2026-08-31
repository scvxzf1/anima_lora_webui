# Anima per-band dynamic-seq 验证记录（2026-08-30）

状态：实验验证完成，默认关闭
适用版本：提交 `a726cefa` 的 dirty 工作树；性能数字仅适用于下述环境和 workload

## 结论

已将上游 `7e2728c7` 的 Anima per-band dynamic-seq 能力移植到当前工作树，
并接通 `--compile_seq_bands`、TOML 默认值和 WebUI 配置字段。功能在 64GB
GPU 上完成 60-step 训练 A/B：两臂均通过，无 OOM、无
`ConstraintViolationError`，loss 轨迹最大绝对差 `0.0001`。

当前合成的多 band workload 没有带来显存收益，且 per-band 稳态步时约比 union
慢 43--44%。因此 `compile_seq_bands` 保持默认关闭；它只在
`compile_dynamic_seq=true` 时生效，作为 Anima-only 的显式实验开关保留。

## 上游与移植边界

上游同步报告对应仓库为 `sorryhyun/anima_lora`，per-band 功能提交为
`7e2728c7`。本仓的固定 native bucket 表仍是 4032/4200 两个 canonical
family，未直接替换为上游后来扩展的 tier 表；这是当前 DCW 顺序、缓存和已发布
配置的不变量。

本次代码入口：

- `library/datasets/buckets.py`：`cluster_token_bands`、`band_for_seq`、
  `widen_bands`。初始 bands 只按当前训练实际使用的 dataset bucket 聚类；
  register-token tail 只扩大上界，碰到下一 band 时拒绝静默合并。
- `library/anima/models.py`：`compile_blocks(..., seq_bands=...)` 和
  checkpoint-safe per-band dispatch。编译发生在 adapter apply/load 之后；
  重编译复用原始 `_anima_compile_base_forward`，不会嵌套 compiled wrapper。
- `library/runtime/harness.py::ensure_training_compile_seq_range`：把 bands 纳入
  compile signature/cache 隔离。训练采样事件重新读取 prompt 后，如发现新 token count，
  保留原 bands、加入 singleton range 并触发一次重编译；sample prompt 不参与 bootstrap
  的初始聚类。
- `library/training/bootstrap.py`：训练入口派生实际 token budget；没有显式
  bucket 时与 `compile_blocks` 使用同一 canonical fallback，避免遗漏
  register-token 上界。
- `library/training/compat_matrix.py`、`library/models/krea2_raw/dit.py`：
  Krea-2 固定 padded token-family 图，显式关闭/拒绝该 Anima-only 参数；
  Z-Image 同样自动关闭。
- `configs/base.toml`、`library/training/cli_args.py` 和 WebUI catalog：
  `compile_seq_bands = false` 为默认值，并提供 live/preflight 提示。

`library/runtime/harness.py::compile_dit_blocks_for_pool()`（distillation
pool）仍使用 union range，这是当前明确边界，不把训练侧开关误用于另一套
编排路径。

## 64GB GPU 基准

硬件与运行环境：

- GPU：`NVIDIA CMP 170HX`，65,536 MiB，driver `610.43.02`
- 设备选择：`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`
- Python 3.13.11，PyTorch `2.12.0+cu130`
- 当前 `HEAD=a726cefa83efb32a5e1e2fba6292906ba00017f2`，工作树含未提交的
  其他用户改动；结果只用于本功能验证

为避免 canonical 4032/4200 的间距不足以形成两个 band，基准使用显式临时
dataset 配置和 `bucket_no_upscale=true` 的三个真实尺寸（宽高均为 16 的倍数）。
其中 3024-token bucket 不属于 `CONSTANT_TOKEN_BUCKETS`；这是专门构造的 synthetic
多 band workload，不是 canonical bucket 表的第三个 family：

| 像素 bucket | DiT token 数 |
| --- | ---: |
| `768x1008` | 3024 |
| `1008x1024` | 4032 |
| `960x1120` | 4200 |

因此：

- union：`seq_range=(3024, 4200)`，一张动态序列图
- per-band：`[(3024, 3024), (4032, 4200)]`，两张紧 band 图

两臂均为 seed `42`、60 steps、同一数据和 `attn_mode=torch`，每臂独立进程并
设置 `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`。稳态口径取第 30--60 步之间的
step 时间；compile wall 是进程启动到第一训练 step，包含首次模型/图编译成本。

### 当前合并工作树（union -> perband）

| arm | compile wall | steady s/it | peak VRAM | total wall | outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| union | 49.5 s | 0.330 | 15,984 MiB | 72.0 s | ok |
| per-band | 48.8 s | 0.476 | 15,984 MiB | 90.5 s | ok |

共同 56 个 loss 点的最大绝对差为 `0.000100`，平均绝对差为 `0.000027`；
per-band/union 稳态比值为 `1.4424`。

### 反向顺序复测（perband -> union）

| arm | compile wall | steady s/it | peak VRAM | total wall | outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| per-band | 49.9 s | 0.476 | 15,984 MiB | 92.0 s | ok |
| union | 49.4 s | 0.332 | 15,984 MiB | 72.0 s | ok |

共同 56 个 loss 点的最大绝对差仍为 `0.000100`，平均绝对差 `0.000020`；
稳态比值为 `1.4337`。顺序交换后方向不变，说明差异不是臂顺序造成的。

per-band 日志中可见第二 band 的预期 lazy specialization；未出现越界或编译
失败。`library/anima/models.py::forward_mini_train_dit` 在 strict band 且
`attn_mode=torch` 时显式限定 CUDNN/efficient/math SDPA backend，以避开任意 native
token count 的 alignment guard；在这张卡和该 workload 上，这个兼容代价大于 tight
guard 的收益。

### 上游 CMP 170HX 对照

上游同类探针的记录为：union compile wall `54.9 s`、steady `0.459 s/it`、
peak `16,072 MiB`；per-band 分别为 `68.6 s`、`0.469 s/it`、`16,072 MiB`。
当前移植结果保持“功能通过、显存不变、per-band 没有速度收益”的结论。

### 可复核产物

- `bench/perband_seq/results/20260830-2048-final-cmp64-multiband-20260830/`
- `bench/perband_seq/results/20260830-2055-final-cmp64-multiband-reverse-20260830/`
- 每个目录含 `result.json`、`losses.json`、两臂 stdout log 和 VRAM CSV
- 使用的临时 dataset/config：`/tmp/anima-perband-data-current-multiband/`、
  `/tmp/anima-perband-data/current-bench.toml`

## 验证与决策

原移植执行记录中的定向回归覆盖 helper、compile signature/forwarding、native flatten、
bootstrap fallback、Krea/Z-Image compatibility 和 WebUI catalog，按当时命令分组为：

```text
核心/runtime/compat：70 passed
前端/WebUI：100 passed
bootstrap + harness/per-band：29 passed
```

这些计数是 2026-08-30 dirty 工作树的执行快照，不应替代当前 HEAD 的复测。对应 GPU
基准证据以两个 `result.json` 及同目录日志/VRAM CSV 为准。

`git diff --check` 当时通过。现阶段不要把 per-band 宣称为通用加速或显存优化；
生产建议继续使用现有 `torch_compile=true` + union dynamic-seq 默认路径，只有
在数据确实跨越多个 token tier 且能接受额外首 band 编译和 backend 约束时，才显式
尝试 `--compile_seq_bands`。
