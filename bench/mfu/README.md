# MFU 基准

这套基准用于给当前训练器增加一套独立的 MFU（Model FLOPs Utilization）量化测试。

## 设计原则

- 独立目录：`bench/mfu/`
- 不改训练主链，只复用已有 `progress_jsonl` 与 `peak_probe_jsonl`
- 默认做短跑训练，适合比较不同配置的相对利用率
- MFU 是 Anima 专用估算值，不是硬件性能计数器
- 默认 baseline 使用仓库已跟踪的 `lora_signal_probe` smoke 配置，避免依赖本机私有 MFU 配置

## 计算口径

MFU 定义为：

```text
estimated_train_step_flops / (avg_step_sec * peak_hw_flops)
```

其中：

- `estimated_train_step_flops`：按当前 Anima DiT 结构估算的单步训练 FLOPs
- `avg_step_sec`：从 `progress_jsonl` 推导的平均步时
- `peak_hw_flops`：手动提供的硬件峰值 TFLOPS，默认 `181`

当前估算覆盖的主路径：

- patch embedding
- 28 个 DiT block 的 self-attn / cross-attn / MLP
- final projection

未单独精算的外围项：

- optimizer 内核
- layernorm / silu / rope / dropout 等小头算子
- 数据加载和 host 侧开销

因此它更适合：

- 同机比较不同训练配置
- 看 compile / rank / preset 的相对变化

不适合直接当成跨框架、跨模型、跨硬件的绝对真值。

## 快速开始

先获取当前 GPU 的理论基准：

```bash
.venv/bin/python -m bench.mfu.gpu_theoretical --gpu-index 1
```

这会输出：

- `fp32_peak_tflops`
- `bf16_peak_tflops`
- `memory_bandwidth_gbps`
- `recommended_peak_tflops`

先做一组最小跑法：

```bash
.venv/bin/python -m bench.mfu.run_training --suite baseline --steps 80 --peak-tflops 读取到的recommended_peak_tflops
```

对比 compile 开关：

```bash
.venv/bin/python -m bench.mfu.run_training --suite compile --steps 80
```

如果你要先检查 MFU runner 会生成什么命令，可以用 dry-run；它会写到 `tmp/bench-dry-runs/mfu`，
不会碰默认训练输出目录：

```bash
.venv/bin/python -m bench.mfu.run_training --dry-run --skip-preflight --suite baseline --steps 4
```

如果你要在本机缓存包上做更短、更稳的 smoke，对训练本体的观察优先用 direct 模式，
并显式传入本机数据集和 prompt：

```bash
.venv/bin/python -m bench.mfu.run_training \
  --launch-mode direct \
  --arms no_compile \
  --steps 4 \
  --allow-low-vram \
  --metric-step-window off \
  --dataset-config configs/bench/mfu_rokkotsu_cached_dataset_single.toml \
  --sample-prompts configs/sample-prompts/imported/rokkotsu_goddess_528_tag.txt \
  -- \
  --max_data_loader_n_workers 0
```

direct 模式会：

- 直接调用 `train.py`，绕过 `tasks.py lora-gui` wrapper
- 在 `no_compile` arm 下物化 merged config，并把 `torch_compile=false` 真正压进训练链
- 更适合作为 `timeout 60` 约束下的当前训练 smoke 起点；如果你要抠最紧的窗口，
  仍然优先直接调用 `train.py`

对比 rank：

```bash
.venv/bin/python -m bench.mfu.run_training --suite rank --steps 80
```

如果你的 GPU 峰值不是默认值，手动传入：

```bash
.venv/bin/python -m bench.mfu.run_training --steps 80 --peak-tflops 165
```

## 结果输出

每次 run 会输出到：

```text
output/bench/mfu/<run_name>/
```

关键文件：

- `summary.json`：单次 run 的完整结构化结果
- `logs/*.progress.jsonl`：步时与显存日志
- `logs/*.peak_probe.jsonl`：token 形状恢复用探针
- `gpu_theoretical.json`：GPU 理论峰值与推荐 `peak_tflops`
- `runs.csv`：汇总表
- `result.json`：沿用 bench 通用 envelope

## 本机实测记录

本目录只提交基准脚本和说明，不提交本机缓存包、运行结果或 `output/` 产物。

如果需要归档某次 MFU 对比，请把结论写到 `docs/findings/`，并在报告里说明：

- GPU 型号和 `recommended_peak_tflops`
- suite、arm、步数和统计窗口
- 使用的数据集配置和缓存来源
- `avg_step_sec`、`achieved_tflops`、MFU
- 关键 `progress_jsonl` / `peak_probe_jsonl` 路径

这样 README 保持可发布，具体机器结果留在 findings 报告里。

## 注意事项

- 默认 GUI 变体是 `configs/gui-methods/lora_signal_probe.toml`
- 默认 dataset / prompt 是 `configs/bench/signal_probe_dataset.toml` 和 `configs/bench/signal_probe_prompts.txt`
- `plain_lora` suite 对应已跟踪的 `configs/gui-methods/lora-8gb.toml`
- `--single-sample-smoke` 当前保持在默认 smoke 数据集上；本机 rokkotsu single-cache 需要显式传 `--dataset-config`
- rokkotsu 缓存包属于本机复现场景，不能作为可发布默认值
- 默认启用 `peak_probe_level=block`，只取最轻量的 block 边界形状
- 真实训练前要确认传入的数据集配置在当前机器可用；dry-run 只验证命令生成和输出隔离
- `gpu_theoretical.py` 是理论上限探针，不代表真实 sustained 吞吐
- 如果你要做“更严格的 MFU”，下一步建议接 profiler / CUDA counter 版
