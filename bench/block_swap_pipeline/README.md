# Block Swap 宏观流水线探针

独立于 `bench/cache_identity_hot/` 的性能/调度测量工具：它**只测量，不改变**任何训练行为。

## 为什么需要它

微观层面（单张量的 fused 恢复 + 环形缓冲 + 双流）已接近物理极限，下一步的杠杆在**宏观调度**：
每一步的 retire(D2H) + restore(H2D) 任务如何排队、被计算掩盖了多少、折返点前后有没有可用的空窗。
这些问题的答案只能在真机事件流里看，不能靠静态阅读推断。

## 载体

| 项 | 值 |
| --- | --- |
| 模型 | Anima（真实 `anima-base-v1.0-sha.safetensors` + 真实 Qwen3 TE + 真实 VAE） |
| 数据 | 2 张 512×512 程序生成图 + 短 caption |
| 配置 | plain LoRA rank8/alpha4、bf16、gradient_checkpointing、`attn_mode="torch"`（50HX 是 Turing，flash 不可用）、`torch_compile=false` |
| profiler | `block_swap_profile_jsonl = "auto"` → `<run>/logs/block_swap_pipeline.block_swap_profile.jsonl`，每个完成的 retire+restore 任务写一条事件 |

工作区（`output/bench/block_swap_pipeline/workspace/`）持久复用：图像、权重副本、latent/TE 缓存只建一次，
之后同一分辨率下的 A/B 只花一次 DiT 加载 + N 步训练。

## 命令

```bash
.venv/bin/python -m bench.block_swap_pipeline.run_probe --gpu 0 --steps 4 --blocks-to-swap 12 --label swap12
```

产物：`<run>/result.json`（原始 metrics）、`<run>/train.log`、`<run>/config.runtime.toml`、profile JSONL。

## 指标口径

事件字段来自 `library/runtime/offloading.py::_build_profile_wait_event`。分析器（`analyze_profile`）按 step 聚合：

| 指标 | 含义 |
| --- | --- |
| `jobs` / `jobs_forward` / `jobs_backward` | 每步 retire+restore 任务数，分相位 |
| `dma_busy_ms` | 每步 DMA 占用时间下界 = Σ(`d2h_ms` + `h2d_ms`) |
| `exposed_wait_ms` | **未被掩盖**的等待 = Σ(`wait_ms`)，即主线程/GPU 真被卡住的时间 |
| `worker_queue_ms` | Σ(`submit_to_enqueue_ms`)：任务在单线程拷贝池里排队的时间，直接反映 `max_workers=1` 的串行化 |
| `step_span_ms` | 该步从第一个任务入队到最后一个任务等待返回的墙钟跨度 |
| `forward_tail_idle_ms` | 最后一个前向任务 ready → 第一个反向任务入队之间的空窗（跨折返点的可预取窗口） |
| `duplicate_restores` | 同一步内同一个 block 被恢复两次的目标块号（折返点重复搬运的直接证据） |
| `h2d_bytes` / `gb_per_s` | 每步 H2D 字节数与有效带宽（`Offloader._estimate_swap_bytes()` 从交换计划的 `stored_nbytes()` 推算） |

## 纯拷贝微基准

`copy_bench.py` 不加载任何模型，直接测链路上限与逐张量开销：

```bash
.venv/bin/python -m bench.block_swap_pipeline.copy_bench --gpu 0 --sizes-mb 100,50 --chunks 1,8,24
```

输出每档 `median_ms` / `GB/s` / `us/chunk`，写入 `<root>/<stamp>-copy-bench.json`；它是判断训练内搬运
是链路受限还是开销受限的参照系。

## 注意

- `exposed_wait_ms` 只统计"等待返回"的耗时；任务在等待点之前就 ready 的部分不计入，因此它是
  **暴露停顿的下界**，`dma_busy_ms` 是 DMA 占用的下界。两者都用于相对比较，不当作绝对物理量。
- 探针默认 `--blocks-to-swap` 由调用方给定；不要用它做显存压力测试（2 图 / 512×512 只用于调度观察）。
- 真机运行时不要与用户训练抢 GPU；本工具不检查 GPU 占用，调用方负责确认。
