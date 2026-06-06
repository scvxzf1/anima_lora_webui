# lokr-anima-shaojianV1 任务推进记录

日期：2026-06-04

## 当前结论

`lokr-anima-shaojianV1` 已从 step 0 OOM / optimizer 配置错误推进到可正常训练。最终 3-step smoke 以 `status=ok` 结束，`final_step=3`。当前瓶颈不再是 block swap 失效或显存不足；在 `gradient_checkpointing=true + blocks_to_swap=23 + lokr_factor=8 + lokr_factor_group_size=8` 下，3-step 最高显存约 `2.39 GiB allocated / 2.52 GiB reserved`，block swap 等待 p95 约 `0.44 ms`。

这不是完整 1140-step 训练结论，只是低风险短周期验证。完整 WebUI 训练可以基于修复后的导入配置重新启动。

## 背景

用户任务来自导入配置：

- 项目配置：`configs/imported/lokr-anima-shaojianV1.toml`
- 最近失败 WebUI 训练：`configs/web-training-history/20260604-014154-training-imported-lokr-anima-shaojianV1`
- WebUI runtime：`/home/scv/nvme0n1p1/训练器相关/anima缓存/lokr-anima-shaojianV1-20260604-014042/config.runtime.toml`
- 数据缓存：`/home/scv/nvme0n1p1/训练器相关/anima缓存/lokr-anima-shaojianV1-20260604-014042/dataset_cache/dataset-01`

第一次真实失败分两层：

- `lokr_factor=1` 全矩阵 LoKr：step 0 backward OOM，adapter 参数约 `1.76B`，16G 显存不可行。
- 改为 `lokr_factor=8` 后：forward/backward 已通过，但 CAME 在 optimizer step 抛 `ValueError: not enough values to unpack (expected 3, got 2)`。

## 已修复配置

已更新 `configs/imported/lokr-anima-shaojianV1.toml`：

```toml
lokr_factor = 8
lokr_factor_group_size = 8
optimizer_type = "CAME"
optimizer_args = ["weight_decay=0.01", "eps=1e-8", "betas=0.9,0.999,0.9999"]
gradient_checkpointing = true
blocks_to_swap = 23
block_swap_transfer_dtype = "bf16"
torch_compile = true
compile_inductor_mode = "max-autotune-no-cudagraphs"
attn_mode = "flash"
```

关键判断：

- `lokr_factor=1` 是全矩阵 LoKr，当前 15.58 GiB GPU 不现实。
- `lokr_factor=8` 把 trainable adapter 降到 `27,537,664` 参数，约 `110 MiB` fp32 参数量。
- 当前 `pytorch_optimizer.CAME` 需要三元 `betas`，两元 `0.9,0.99` 会在 `step()` 解包失败。

## 短跑过程

本轮没有直接跑完整训练，而是复用最新 WebUI 预处理缓存，生成临时 runtime TOML 做短周期验证。临时短跑统一设置：

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
config_file=/tmp/.../config.runtime.toml
blocks_to_swap=23
gradient_checkpointing=true
lokr_factor=8
lokr_factor_group_size=8
memory_probe_jsonl=/tmp/.../memory_probe.jsonl
block_swap_profile_jsonl=/tmp/.../block_swap_profile.jsonl
```

| run | 结果 | 说明 |
| --- | --- | --- |
| `/tmp/anima-lokr-shaojian-smoke-20260604-021330` | error before step | 我传入 `max_train_epochs=0`，覆盖 `max_train_steps=2`，scheduler 正确拒绝 0 steps。 |
| `/tmp/anima-lokr-shaojian-smoke2-20260604-021607` | `final_step=2` 后 error | 两个 step 已完成，optimizer 正常；尾部因临时禁保存设置 `save_every_n_epochs=0` 触发取模错误。 |
| `/tmp/anima-lokr-shaojian-smoke3-20260604-021952` | ok | 3-step clean smoke，`status=ok`，`final_step=3`。 |

第二次短跑虽以尾部错误结束，但已经证明 forward/backward/optimizer step 全部通过。第三次短跑用于获得 clean `run_end=ok`。

## 最终 smoke 数据

最终数据来自：

- `/tmp/anima-lokr-shaojian-smoke3-20260604-021952/progress.jsonl`
- `/tmp/anima-lokr-shaojian-smoke3-20260604-021952/memory_probe.jsonl`
- `/tmp/anima-lokr-shaojian-smoke3-20260604-021952/block_swap_profile.jsonl`

训练结果：

| step | current loss | average loss | max allocated | max reserved |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0978546` | `0.0978546` | `2.288 GiB` | `2.510 GiB` |
| 2 | `0.0939903` | `0.0959225` | `2.394 GiB` | `2.516 GiB` |
| 3 | `0.0918843` | `0.0945764` | `2.394 GiB` | `2.516 GiB` |

显存：

- GPU：NVIDIA GeForce RTX 3080 Ti Laptop GPU，约 `15.58 GiB`
- 最高 allocated：`2.394 GiB`
- 最高 reserved：`2.516 GiB`
- after_forward 最低 free：`11.873 GiB`
- after_backward 最低 free：`11.874 GiB`

block swap：

- `blocks_to_swap=23`
- `transfer_dtype=bf16`
- profile event 数：`138`
- wait_ms：p50 `0.031 ms`，p90 `0.359 ms`，p95 `0.441 ms`，max `0.669 ms`
- forward_wait p95：`0.549 ms`
- backward_wait p95：`0.370 ms`
- h2d_ms：p50 `13.588 ms`，p90 `13.905 ms`，p95 `14.119 ms`，max `15.377 ms`

结论：block swap H2D 开销稳定，等待时间很低；当前配置的短跑显存余量非常宽。

## 遇到的问题和处理

1. `max_train_epochs=0` 会覆盖 `max_train_steps`，导致 scheduler 抛 `num_training_steps must be positive`。
   - 处理：临时 runtime 移除 `max_train_epochs`，直接使用 `max_train_steps`。

2. `save_every_n_epochs=0` 不等价于禁用保存，会在 epoch 尾部触发 `epoch_no % args.save_every_n_epochs` 的 `ZeroDivisionError`。
   - 处理：短跑使用正数 `save_every_n_epochs=999`；正式配置保持原本 `save_every_n_epochs=1`。

3. CAME 两元 betas 不兼容当前依赖。
   - 处理：改为三元 `betas=0.9,0.999,0.9999`。

## 验证

已通过：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_config.py -q
timeout 60 .venv/bin/python -m pytest tests/test_training_optimizers.py -q
timeout 60 .venv/bin/python -m pytest tests/test_training_resume.py -k "block_swap or progress_jsonl" -q
timeout 60 .venv/bin/python -m pytest tests/test_deferred_sample_cleanup.py -q
```

结果：

- `tests/test_config.py`：22 passed
- `tests/test_training_optimizers.py`：6 passed
- `tests/test_training_resume.py -k "block_swap or progress_jsonl"`：4 passed, 84 deselected
- `tests/test_deferred_sample_cleanup.py`：5 passed

## 下一步

建议从 WebUI 重新启动 `imported/lokr-anima-shaojianV1 / default`，让它生成新的 runtime 配置。不要恢复旧失败历史继续跑，因为旧 runtime 中仍保存了两元 CAME betas。

完整训练建议：

- 保留 `gradient_checkpointing=true + blocks_to_swap=23`，这是当前最稳显存配置。
- 保留 `lokr_factor=8 + lokr_factor_group_size=8`。
- 保留 `block_swap_profile_jsonl=auto` 和 `memory_probe_jsonl=auto` 跑前几步，确认长训无异常后可把 `memory_probe_jsonl` 关掉以减少日志。
- 长训练时继续使用 allocator fallback：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`。
- 若要做速度优化，下一轮可以消融 `gradient_checkpointing=false`，但这应作为单独实验，不要和完整训练混在一起。
