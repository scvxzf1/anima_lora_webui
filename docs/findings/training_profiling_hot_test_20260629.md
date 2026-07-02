# 训练性能 Profiling 热测记录：Plain LoRA vs LoKr Block Swap

时间：`2026-06-29 17:42-17:58 +0800`；追加 nsys clean rerun：`2026-06-29 19:08-19:13 +0800`；
追加 GPU metrics rerun：`2026-06-29 21:45-21:51 +0800`

## 结论

- `plain LoRA + checkpointing + no block swap` 的稳定热窗口约 `1.21s/step`，MFU 约 `36.6%`。
- `LoKr + checkpointing + blocks_to_swap=26` 的短窗口约 `6.24s/step`，MFU 约 `7.1%`。
- LoKr block swap 把峰值显存压到约 `1.87 GiB allocated / 2.15 GiB reserved`，但单个训练 step 内观测到约 `5.5-5.7s` 的 H2D 传输累计量；低 MFU 主要是省显存策略的吞吐代价。
- 已修复仓库 nsys wrapper 的旧版兼容：`Nsight Systems 2022.4` 会自动跳过 Python sampling、使用旧式 GPU metrics 参数和旧式 stats report 名。
- 已补齐 nsys 后处理链路：wrapper 会在只生成 `.qdstrm` 时自动调用 `QdstrmImporter` 导入 `.nsys-rep`，并继续生成 stats 文本。
- 当前开发机已安装用户级 `Nsight Systems 2026.1.3`，`~/.local/bin/nsys` 优先于系统 apt 的 2022.4。新版 smoke 已直接生成 `.nsys-rep/.sqlite`。
- GPU metrics 驱动权限已放开；重启后 `/proc/driver/nvidia/params` 显示
  `RmProfilingAdminOnly: 0`。本机是 GTX 960 + RTX 3080 混架构，正式复测建议显式传
  `NSYS_GPU_METRICS_DEVICES=1 NSYS_GPU_METRICS_SET=ga10x`。
- GPU metrics LoKr rerun 已生成：`GPU_METRICS` 表 `8758771` 行。profile window 均值：
  `GPU Active 50.17%`、`SMs Active 42.82%`、`SM Issue 9.30%`、
  `Tensor Active 4.92%`、`DRAM Read 15.44%`、`DRAM Write 12.44%`。
- 已用 `Nsight Systems 2026.1.3` 重跑 LoKr `PROFILE_STEPS=3-5` clean timeline。需要设置
  `NSYS_PYTHON_SAMPLING=0`，否则本机该训练场景会在 capture range 附近触发
  multiprocessing `ConnectionResetError`，并生成缺 CUDA trace 的报告。
- 热测发现并修复了 MFU 配置问题：`max_train_epochs=6` 会覆盖 runner 传入的 `--max_train_steps`，导致 `--steps` 无法截断短跑。
- 修复后已用 `plain_lora --steps 2` 真实验证：`total_steps=2`，`run_end status=ok`，`summary.json` 正常生成。

## 环境

- GPU：`NVIDIA GeForce RTX 3080 10GB`
- PyTorch：`2.12.0+cu130`
- Python：`3.13.11`
- Nsight Systems：
  - 系统 apt：`2022.4.2.50-32196742v0`
  - 当前 PATH：`2026.1.3.243-261337792075v0`，来自用户级 `~/.local/opt/nvidia/nsight-systems/2026.1.3`
- 理论峰值口径：`recommended_peak_tflops = 119.07072`
- 数据与缓存：`bench/mfu/assets/rokkotsu_goddess_528_tag/`
- 训练 bucket：`896x1152`，`token_count = 4032`
- FLOPs 估算：
  - `forward_flops = 17583494397952`
  - `train_step_flops = 52750483193856`

## 产物

- Plain LoRA：
  - `output/bench/mfu/plain_lora_ckpt_s42_20step/logs/plain_lora_ckpt_s42_20step.progress.jsonl`
  - `output/bench/mfu/plain_lora_ckpt_s42_20step/logs/plain_lora_ckpt_s42_20step.peak_probe.jsonl`
- Plain LoRA 修复验证：
  - `output/bench/mfu/plain_lora_ckpt_s42_2step/summary.json`
  - `output/bench/mfu/plain_lora_ckpt_s42_2step/logs/plain_lora_ckpt_s42_2step.progress.jsonl`
- LoKr JSONL：
  - `output/bench/hot_profile/hot_profile_20260629_1752_lokr_blockswap_jsonl/logs/hot_profile_20260629_1752_lokr_blockswap_jsonl.progress.jsonl`
  - `output/bench/hot_profile/hot_profile_20260629_1752_lokr_blockswap_jsonl/logs/hot_profile_20260629_1752_lokr_blockswap_jsonl.memory_probe.jsonl`
  - `output/bench/hot_profile/hot_profile_20260629_1752_lokr_blockswap_jsonl/logs/hot_profile_20260629_1752_lokr_blockswap_jsonl.block_swap_profile.jsonl`
  - `output/bench/hot_profile/hot_profile_20260629_1752_lokr_blockswap_jsonl/logs/hot_profile_20260629_1752_lokr_blockswap_jsonl.peak_probe.jsonl`
- LoKr nsys 2022 尝试：
  - `output/bench/hot_profile/hot_profile_20260629_1756_lokr_blockswap_nsys2022/logs/hot_profile_20260629_1756_lokr_blockswap_nsys2022.progress.jsonl`
  - `output/bench/hot_profile/hot_profile_20260629_1756_lokr_blockswap_nsys2022/logs/hot_profile_20260629_1756_lokr_blockswap_nsys2022.block_swap_profile.jsonl`
  - `output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022.qdstrm`
  - `output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022.nsys-rep`
  - `output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022.sqlite`
  - `output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022_{gpukernsum,nvtxkernsum,gpumemtimesum,gpumemsizesum,cudaapisum,kernexecsum}.txt`
- LoKr nsys 2026 clean rerun：
  - `output/bench/hot_profile/hot_profile_20260629_lokr_blockswap_nsys2026_nopy/baseline_s42_8step/summary.json`
  - `output/bench/hot_profile/hot_profile_20260629_lokr_blockswap_nsys2026_nopy/baseline_s42_8step/logs/baseline_s42_8step.progress.jsonl`
  - `output/bench/hot_profile/hot_profile_20260629_lokr_blockswap_nsys2026_nopy/baseline_s42_8step/logs/baseline_s42_8step.block_swap_profile.jsonl`
  - `output/bench/hot_profile/hot_profile_20260629_lokr_blockswap_nsys2026_nopy/baseline_s42_8step/logs/baseline_s42_8step.memory_probe.jsonl`
  - `output/nsys/hot_profile_20260629_lokr_blockswap_nsys2026_nopy.nsys-rep`
  - `output/nsys/hot_profile_20260629_lokr_blockswap_nsys2026_nopy.sqlite`
  - `output/nsys/hot_profile_20260629_lokr_blockswap_nsys2026_nopy_{cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_kern_exec_sum,nvtx_kern_sum,nvtx_sum}.txt`
- nsys wrapper smoke：
  - `output/nsys/nsys_wrapper_smoke_2022_pci.qdstrm`
  - `output/nsys/nsys_wrapper_smoke_2022_pci.nsys-rep`
  - `output/nsys/nsys_wrapper_smoke_2022_pci.sqlite`
  - `output/nsys/nsys_wrapper_smoke_2026_nometrics.nsys-rep`
  - `output/nsys/nsys_wrapper_smoke_2026_nometrics.sqlite`

## MFU 结果

| 配置 | 统计窗口 | step events | avg_step_sec | median_step_sec | p90_step_sec | achieved_tflops | MFU | peak_allocated_gb | peak_reserved_gb |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| plain LoRA | `5-15` | 127 | `1.209` | `1.209` | `1.212` | `43.6315` | `36.64%` | `5.4501` | `5.6445` |
| plain LoRA tail | `108-127` | 127 | `1.23395` | `1.234` | `1.235` | `42.7493` | `35.90%` | `5.4501` | `5.6445` |
| plain LoRA fixed runner | `all` | 2 | `1.212` | `1.212` | `1.212` | `43.5235` | `36.55%` | `5.4501` | `5.6426` |
| LoKr block swap | `2-5` | 5 | `6.23675` | `6.223` | `6.562` | `8.4580` | `7.10%` | `1.8685` | `2.1484` |
| LoKr under nsys | `2-5` | 5 | `8.02875` | `7.8435` | `9.866` | `6.5702` | `5.52%` | `1.8685` | `2.1484` |
| LoKr nsys 2026 clean | `2-5` | 5 | `7.8685` | `7.785` | `9.587` | `6.7040` | `5.63%` | `1.8685` | `2.1484` |

`LoKr under nsys` 和 `LoKr nsys 2026 clean` 包含 profiler 开销，只作为 nsys 采样环境参考，
不作为真实性能口径。clean run 的第一步 compile 约 `107s`，统计窗口仍只取 `2-5`。

## Block Swap 证据

LoKr JSONL run 共记录 `313` 条 block-swap 事件，其中 `forward_wait=156`、
`backward_wait=156`，另有 1 条初始化/摘要事件。

| 指标 | sum_ms | avg_ms | median_ms | p90_ms | max_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| wait_ms | `156.381` | `0.501` | `0.555` | `0.851` | `2.210` |
| h2d_ms | `5698.696` | `18.265` | `13.899` | `23.349` | `99.866` |
| transfer_ms | `5834.099` | `18.699` | `14.188` | `24.190` | `100.595` |
| enqueue_ms | `3773.912` | `12.096` | `5.476` | `36.551` | `110.699` |
| submit_lag_ms | `115134.863` | `369.022` | `157.583` | `1495.746` | `2464.512` |

复跑的 nsys 采样 run 数值接近：`h2d_ms sum = 5553.604`，`transfer_ms sum = 5691.231`，
`wait_ms sum = 179.540`。两次都显示显式 wait 很小，但 H2D 传输总量接近单步耗时主项。

2026 clean run 的 block-swap JSONL 仍然一致：

| 指标 | sum_ms | avg_ms | median_ms | p90_ms | max_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| wait_ms | `175.241` | `0.562` | `0.647` | `0.796` | `1.867` |
| h2d_ms | `5465.260` | `17.517` | `13.975` | `16.227` | `119.821` |
| transfer_ms | `5596.000` | `17.936` | `14.326` | `17.041` | `119.824` |
| enqueue_ms | `3447.155` | `11.049` | `5.683` | `25.116` | `144.386` |
| submit_lag_ms | `122860.906` | `393.785` | `169.280` | `1566.602` | `2357.445` |

## Nsight 状态

热测时仓库 `scripts/tasks/_common.py` 的 nsys wrapper 还假设新版 nsys 参数：

- `--python-sampling=true`
- `--gpu-metrics-devices=cuda-visible`

本机 `Nsight Systems 2022.4` 不支持 `--python-sampling=true`，并且 GPU metrics 参数是旧式单数
`--gpu-metrics-device=1`。第一次 wrapper run 在训练启动前失败。之后已修复 wrapper：

- 探测 `nsys profile --help`，旧版自动跳过 Python sampling。
- 探测 GPU metrics 参数，旧版自动使用 `--gpu-metrics-device=<id>`。
- 探测 `nsys stats --help-reports`，旧版自动使用 `gpukernsum`、`cudaapisum`、
  `gpumemtimesum`、`gpumemsizesum`、`kernexecsum`、`nvtxkernsum`。
- 不再向 nsys 和 Python 命令之间插入旧版会误解析的独立 `--`。

手动改用 2022.4 兼容参数后，捕获窗口可以正常开始和结束，并生成：

```text
output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022.qdstrm
```

系统 apt 版本的 `/usr/bin/nsys` 指向 target 采集端，自动 finalize 时未生成 `.nsys-rep`。
本机实际存在 host importer：

```text
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
```

手动导入可以生成 `.nsys-rep`，但 2022.4 importer 对当前 CUDA/PyTorch 运行时过旧，会返回：

```text
Unknown runtime API function index: 461
Unknown driver API function index: 719
```

尽管 importer 返回码是 `3`，导出的 `.nsys-rep` 仍可被 `nsys stats` 消费。仓库 wrapper 已改为：

- 找到 `.qdstrm` 但缺 `.nsys-rep` 时，自动调用 `QdstrmImporter`。
- 支持 `NSYS_QDSTRM_IMPORTER=/path/to/QdstrmImporter` 覆盖 importer 位置。
- importer 非零退出但 `.nsys-rep` 存在时，继续生成 stats，并只输出简短诊断。

已用 LoKr 热测 trace 验证 wrapper 后处理可产出：

```text
output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022.nsys-rep
output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022.sqlite
output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022_gpukernsum.txt
output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022_nvtxkernsum.txt
output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022_gpumemtimesum.txt
output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022_gpumemsizesum.txt
output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022_cudaapisum.txt
output/nsys/hot_profile_20260629_1756_lokr_blockswap_nsys2022_kernexecsum.txt
```

随后安装用户级 Nsight Systems 2026.1.3：

```text
~/.local/bin/nsys -> ~/.local/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
~/.local/bin/QdstrmImporter -> ~/.local/opt/nvidia/nsight-systems/2026.1.3/host-linux-x64/QdstrmImporter
```

当前 `PATH` 会优先命中 `~/.local/bin/nsys`。新版 smoke 已验证能直接生成 `.nsys-rep`
和 `.sqlite`，不再出现旧 importer 的 unknown CUDA API 诊断。

修复后 smoke 命令已在 RTX 3080 上验证 CUDA workload 可以被 nsys 捕获：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 nsys profile \
  -o output/nsys/nsys_wrapper_smoke_2022_pci \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --wait=primary \
  --trace=cuda,nvtx,cudnn,cublas \
  --cuda-graph-trace=node \
  --cuda-memory-usage=true \
  --stats=true \
  --sample=none \
  --cpuctxsw=none \
  --cudabacktrace=none \
  --resolve-symbols=false \
  --gpu-metrics-device=1 \
  --gpu-metrics-frequency=10000 \
  .venv/bin/python -c "import torch; torch.cuda.profiler.start(); x=torch.randn((512,512),device='cuda'); y=x@x; torch.cuda.synchronize(); torch.cuda.profiler.stop(); print(float(y[0,0].detach().cpu()))"
```

结果：捕获窗口正常开始/结束并生成
`output/nsys/nsys_wrapper_smoke_2022_pci.qdstrm`；通过 wrapper 后处理可以导入为
`.nsys-rep` 并生成 stats。

新版 smoke 命令使用 2026.1.3 且不带 GPU metrics：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 nsys profile \
  -o output/nsys/nsys_wrapper_smoke_2026_nometrics \
  --force-overwrite=true \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --wait=primary \
  --trace=cuda,nvtx,cudnn,cublas \
  --cuda-graph-trace=node \
  --cuda-memory-usage=true \
  --stats=true \
  --sample=none \
  --cpuctxsw=none \
  --cudabacktrace=none \
  --resolve-symbols=false \
  .venv/bin/python -c "import torch; torch.cuda.profiler.start(); x=torch.randn((512,512),device='cuda'); y=x@x; torch.cuda.synchronize(); torch.cuda.profiler.stop(); print(float(y[0,0].detach().cpu()))"
```

结果：`output/nsys/nsys_wrapper_smoke_2026_nometrics.nsys-rep` 和
`output/nsys/nsys_wrapper_smoke_2026_nometrics.sqlite` 均生成成功。

`--gpu-metrics-devices=cuda-visible` 在当前驱动权限下仍失败：

```text
Illegal --gpu-metrics-devices argument: cuda-visible.
Insufficient privilege, see https://developer.nvidia.com/ERR_NVGPUCTRPERM.
```

仓库 `_nsys_wrapper()` 已验证会自动跳过 GPU metrics，同时保留新版
`--python-sampling=true`。

### LoKr 2026 Clean Timeline

直接用新版 `nsys 2026.1.3` 重跑 LoKr timeline 时，默认 Python sampling 会让本机
`mfu_rokkotsu_cached` 场景在 capture range 附近触发 multiprocessing
`ConnectionResetError`，最终 `.nsys-rep` 只有 NVTX、没有 CUDA trace。为此仓库 wrapper
新增 `NSYS_PYTHON_SAMPLING=0` 开关；关闭 Python sampling 后重跑成功。

命令：

```bash
PROFILE_STEPS=3-5 \
NSYS_PYTHON_SAMPLING=0 \
NSYS_OUT=output/nsys/hot_profile_20260629_lokr_blockswap_nsys2026_nopy.nsys-rep \
.venv/bin/python -m bench.mfu.run_training \
  --suite baseline \
  --steps 8 \
  --metric-step-window 2-5 \
  --peak-tflops 119.07072 \
  --allow-low-vram \
  --output-root output/bench/hot_profile/hot_profile_20260629_lokr_blockswap_nsys2026_nopy \
  -- \
  --block_swap_profile_jsonl auto \
  --memory_probe_jsonl auto \
  --memory_probe_max_steps 6
```

结果：

- `returncode = 0`
- `steps_completed = 5`，profile window `3-5`
- `avg_step_sec = 7.8685`，`median_step_sec = 7.785`，`p90_step_sec = 9.587`
- `achieved_tflops = 6.704008`，`MFU = 5.6303%`
- `peak_allocated_gb = 1.8685`，`peak_reserved_gb = 2.1484`

`nvtx_sum`：

| Range | Instances | Total ms | Avg ms | Median ms |
| --- | ---: | ---: | ---: | ---: |
| `:step=3` | 1 | `9107.294` | `9107.294` | `9107.294` |
| `:step=4` | 1 | `9136.188` | `9136.188` | `9136.188` |
| `:step=5` | 1 | `9625.400` | `9625.400` | `9625.400` |
| `:forward` | 3 | `6851.788` | `2283.929` | `2281.061` |
| `:backward` | 3 | `20961.775` | `6987.258` | `6840.294` |
| `:optimizer` | 3 | `21.209` | `7.070` | `7.039` |

`cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum`：

| Operation | Count | Total time ms | Total MB | Avg MB |
| --- | ---: | ---: | ---: | ---: |
| Host-to-Device | 2364 | `2144.410` | `21598.712` | `9.137` |
| Device-to-Device | 93 | `8.250` | `2776.177` | `29.851` |
| Device-to-Host | 5 | `0.008` | `0.000` | `0.000` |

`cuda_api_sum` 显示 host 侧主要是 launch 密度：

| API | Calls | Total ms | Avg ns |
| --- | ---: | ---: | ---: |
| `cudaLaunchKernel` | 378777 | `5358.405` | `14146.6` |
| `cuLaunchKernel` | 18747 | `302.317` | `16126.1` |
| `cudaLaunchKernelExC` | 25539 | `256.688` | `10050.8` |
| `cudaMemcpyAsync` | 2462 | `66.191` | `26885.1` |

`cuda_gpu_kern_sum` 前几项不是单个大 GEMM 统治，而是大量 elementwise / copy /
LoKr 相关小 kernel 与 GEMM、flash attention 交织：

- top elementwise/copy kernel 合计占比靠前，单类约 `8-13%`。
- 最大单类 GEMM：`ampere_bf16_s16816gemm_bf16_256x128...tn`，
  `504` 次，总 `1028.390ms`，约 `7.9%`。
- Flash Attention backward：`168` 次，总 `510.338ms`，约 `3.9%`。
- Flash Attention forward：`336` 次，总 `436.118ms`，约 `3.4%`。

### LoKr 2026 GPU Metrics Timeline

重启并放开 NVIDIA perf counter 后，使用显式 3080 metrics 设备重跑：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
PROFILE_STEPS=3-5 \
NSYS_PYTHON_SAMPLING=0 \
NSYS_GPU_METRICS_DEVICES=1 \
NSYS_GPU_METRICS_SET=ga10x \
NSYS_OUT=output/nsys/hot_profile_20260629_lokr_blockswap_nsys2026_metrics.nsys-rep \
timeout -s INT 900 \
.venv/bin/python -m bench.mfu.run_training \
  --suite baseline \
  --steps 8 \
  --metric-step-window 2-5 \
  --peak-tflops 119.07072 \
  --allow-low-vram \
  --output-root output/bench/hot_profile/hot_profile_20260629_lokr_blockswap_nsys2026_metrics \
  -- \
  --block_swap_profile_jsonl auto \
  --memory_probe_jsonl auto \
  --memory_probe_max_steps 6
```

结果：

- `returncode = 0`
- `steps_completed = 5`，profile window `3-5`
- `avg_step_sec = 8.168`，`median_step_sec = 7.871`，`p90_step_sec = 10.453`
- `achieved_tflops = 6.458188`，`MFU = 5.4238%`
- `peak_allocated_gb = 1.8683`，`peak_reserved_gb = 2.1621`

产物：

- `output/bench/hot_profile/hot_profile_20260629_lokr_blockswap_nsys2026_metrics/baseline_s42_8step/summary.json`
- `output/nsys/hot_profile_20260629_lokr_blockswap_nsys2026_metrics.nsys-rep`
- `output/nsys/hot_profile_20260629_lokr_blockswap_nsys2026_metrics.sqlite`
- `output/nsys/hot_profile_20260629_lokr_blockswap_nsys2026_metrics_{cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_kern_exec_sum,nvtx_kern_sum}.txt`
- `output/nsys/hot_profile_20260629_lokr_blockswap_nsys2026_metrics_nvtx_sum_nvtx_sum.txt`

GPU metrics：

| Metric | Samples | Avg % | Max % |
| --- | ---: | ---: | ---: |
| `GPU Active` | 282541 | `50.17` | `100` |
| `SMs Active` | 282541 | `42.82` | `100` |
| `SM Issue` | 282541 | `9.30` | `55` |
| `Tensor Active` | 282541 | `4.92` | `50` |
| `DRAM Read Bandwidth` | 282541 | `15.44` | `98` |
| `DRAM Write Bandwidth` | 282541 | `12.44` | `93` |

`nvtx_sum`：

| Range | Instances | Total ms | Avg ms | Median ms |
| --- | ---: | ---: | ---: | ---: |
| `:step=3` | 1 | `9378.182` | `9378.182` | `9378.182` |
| `:step=4` | 1 | `9257.285` | `9257.285` | `9257.285` |
| `:step=5` | 1 | `9460.154` | `9460.154` | `9460.154` |
| `:forward` | 3 | `7020.101` | `2340.034` | `2346.857` |
| `:backward` | 3 | `21013.319` | `7004.440` | `7006.484` |
| `:optimizer` | 3 | `22.373` | `7.458` | `7.510` |

`cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum`：

| Operation | Count | Total time ms | Total MB | Avg MB |
| --- | ---: | ---: | ---: | ---: |
| Host-to-Device | 2364 | `2130.797` | `21598.712` | `9.137` |
| Device-to-Device | 93 | `8.248` | `2776.177` | `29.851` |
| Device-to-Host | 5 | `0.009` | `0.000` | `0.000` |

block-swap JSONL：

- rows：`313`，其中 1 行元信息，`forward_wait=156`，`backward_wait=156`
- `wait_ms = 158.176`
- `h2d_ms = 5465.139`
- `transfer_ms = 5618.245`
- `enqueue_ms = 3543.265`

GPU metrics 结论：

- `GPU Active` 平均只有约 `50%`，`SM Issue` 平均约 `9%`，`Tensor Active` 平均约 `5%`；
  这不是 Tensor Core 饱和型瓶颈。
- DRAM read/write 平均也不高，但 H2D 仍有 `21.6GB / profile window`，说明主要问题不只是
  on-device DRAM 带宽，而是 host-device offload、同步/排队和大量小 kernel 调度共同放大。
- 与 no-metrics clean run 相比，带 metrics 的步时从 `7.8685s` 到 `8.168s`，额外开销约
  `3.8%`，可接受但正式性能对比仍应以 no-metrics run 为主，metrics run 用于归因。

### LoKr Grouped Delta Triton Clean Rerun（2026-07-02）

为了验证方案 5 的实际收益，使用同一条 LoKr block swap 热测命令分别重跑：

- `--network_args lokr_grouped_delta_backend=eager`
- `--network_args lokr_grouped_delta_backend=triton`

两次都使用 `PROFILE_STEPS=3-5`、`NSYS_PYTHON_SAMPLING=0`、`--gpu-index 1`，并输出到：

- `output/nsys/hot_profile_20260702_lokr_blockswap_eager_timeline.*`
- `output/nsys/hot_profile_20260702_lokr_blockswap_triton_timeline.*`

说明：本轮 3080 不是绝对真空卡，`nvidia-smi` 仍可见一个常驻 Python compute context，
约 `940MiB` 显存占用，但 `GPU util=0`；因此结果可用于组件级对比，不应解读成严格极限吞吐。

bench 汇总：

| Backend | Avg step s | TFLOPS | MFU | Peak GB |
| --- | ---: | ---: | ---: | ---: |
| `eager` | `7.93775` | `6.6455` | `5.58%` | `2.1621` |
| `triton` | `5.90550` | `8.9324` | `7.50%` | `2.0586` |

其中 `triton` 相比 `eager`：

- `avg_step_sec` 从 `7.93775s` 降到 `5.90550s`，下降约 `25.6%`
- `TFLOPS` 从 `6.65` 升到 `8.93`，提升约 `34.4%`
- 峰值显存没有上升，反而从 `2.162GiB` 降到 `2.059GiB`

为了避免把 backward / optimizer 噪声混进结论，进一步对 `.sqlite` 做 NVTX window 统计：

- `forward` 窗口：按 `NVTX_EVENTS.text='forward'` 的 3 个 range 统计
- launch：统计 `cudaLaunchKernel*`、`cuLaunchKernel*` runtime API
- 小 kernel：统计 `CUPTI_ACTIVITY_KIND_KERNEL` 中执行时长低于阈值的 kernel 数
- H2D：统计 `CUPTI_ACTIVITY_KIND_MEMCPY` 中 Host-to-Device 事件与 `forward` range 的重叠

前向窗口对比：

| 指标 | `eager` | `triton` | 变化 |
| --- | ---: | ---: | ---: |
| `forward` 总时长 | `6864.985 ms` | `4341.994 ms` | `-36.8%` |
| 前向 launch 总数 | `108723` | `3807` | `-96.5%` |
| 前向 launch / forward | `36241` | `1269` | `-96.5%` |
| 前向 launch 密度 | `15837 / s` | `877 / s` | `-94.5%` |
| 前向 kernel 总数 | `108723` | `3594` | `-96.7%` |
| `<10us` 小 kernel | `34521` | `1491` | `-95.7%` |
| `<20us` 小 kernel | `62750` | `1494` | `-97.6%` |
| `<50us` 小 kernel | `81596` | `1578` | `-98.1%` |
| `<100us` 小 kernel | `107711` | `2064` | `-98.1%` |
| 前向 elementwise-like kernel | `91719` | `270` | `-99.7%` |

这组数字说明这次收益不是“同样的碎 kernel 只是更快一点”，而是 LoKr forward 的执行形态已经
从海量小 kernel + 高频 host launch，切到了少量大 kernel 主导。

`triton` 侧最关键的新形态：

- `nvtx_kern_sum` 中 `:forward` 范围里，`_lokr_grouped_delta_forward_kernel` 共 `588` 次，
  总 `3503.215 ms`
- `cuda_gpu_kern_sum` 里该 kernel 占整段 GPU kernel time 约 `47.6%`
- `cuda_api_sum` 中新增 `cuLaunchKernelEx = 1176`，对应 Triton kernel launch

同时，H2D 基本没动：

| H2D 指标（仅 `forward` 重叠） | `eager` | `triton` | 变化 |
| --- | ---: | ---: | ---: |
| H2D count | `1170` | `1170` | `0` |
| H2D size | `10.796 GB` | `10.796 GB` | `0` |
| H2D total time | `1068.373 ms` | `1063.066 ms` | `-0.5%` |

所以这轮提升可以明确归因到 LoKr grouped delta 的 kernel / launch 压缩，而不是 block swap
H2D 行为发生了显著变化。

### 空卡 3080 复跑（2026-07-02）

随后又在 3080 绝对空卡条件下重跑同口径 8-step clean timeline：

- `nvidia-smi` 显示 `GPU 1 = 12MiB, util=0`
- `nvidia-smi --query-compute-apps` 中没有 3080 compute 进程
- 产物：
  - `output/nsys/hot_profile_20260702_lokr_blockswap_eager_empty3080_timeline.*`
  - `output/nsys/hot_profile_20260702_lokr_blockswap_triton_empty3080_timeline.*`

bench 汇总：

| Backend | 上一轮 | 空卡复跑 | 变化 |
| --- | ---: | ---: | ---: |
| `eager avg_step_sec` | `7.93775` | `8.38325` | `+5.6%` |
| `triton avg_step_sec` | `5.90550` | `6.02400` | `+2.0%` |
| `eager TFLOPS` | `6.6455` | `6.2924` | `-5.3%` |
| `triton TFLOPS` | `8.9324` | `8.7567` | `-2.0%` |

也就是说，3080 真空并没有把上一轮结果“洗白”成更快，反而两版都略慢一点；因此前一轮
“非绝对空卡”并不是主导因素。

更关键的是，空卡复跑后的结构统计几乎完全重合：

| 指标 | 上一轮 `eager` | 空卡 `eager` | 上一轮 `triton` | 空卡 `triton` |
| --- | ---: | ---: | ---: | ---: |
| 前向 launch 总数 | `108723` | `108723` | `3807` | `3807` |
| 前向 kernel 总数 | `108723` | `108723` | `3594` | `3594` |
| `<50us` 小 kernel | `81596` | `81505` | `1578` | `1576` |
| 前向 H2D size | `10.796 GB` | `10.796 GB` | `10.796 GB` | `10.796 GB` |
| 前向 H2D time | `1068.373 ms` | `1050.290 ms` | `1063.066 ms` | `1065.161 ms` |

因此可以把结论再收紧一层：

- `triton` 相比 `eager` 的收益是稳定存在的，且仍然主要来自 LoKr forward 的
  launch / kernel 压缩。
- “上一轮 3080 不是绝对空卡”这件事不会改变结论方向；在绝对空卡下，launch 密度、小 kernel
  数量、H2D 体量几乎原样复现。
- 当前总步时的小幅波动更像 run-to-run 噪声、频率/热状态或 profiler 环境波动，而不是
  另一条被遗漏的大瓶颈。

### LoKr Grouped Delta Microbench（2026-07-02）

为了把组件级收益和真实训练收益拆开看，新增了一个可复跑 microbench：

- 脚本：`bench/lokr_grouped_delta/run_microbench.py`
- 运行命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m bench.lokr_grouped_delta.run_microbench \
  --allow-low-vram \
  --label hot_mlp1_20260702_scaled \
  --output-root output/bench/lokr_grouped_delta_microbench
```

默认 shape 对应此前 probe 里最热的 `blocks.27.mlp.layer1`：

- `input = [2, 1, 72, 56, 2048]`
- `output = [2, 1, 72, 56, 8192]`
- `factor = 8`，即 `in_dim = 256`、`out_dim = 1024`
- `group_size = 8`

为了更贴近真实训练的 delta 量级，microbench 默认使用：

- `activation dtype = bf16`
- `weight dtype = fp32`
- `w1_scale = 1.0`
- `w2_scale = 1e-3`
- `gate = 0.75`

组件级结果：

| 指标 | `eager` | `triton` | 变化 |
| --- | ---: | ---: | ---: |
| forward avg latency | `23.4827 ms` | `22.9451 ms` | `-2.29%` |
| kernel count | `480` | `1` | `-99.79%` |
| launch count | `480` | `1` | `-99.79%` |
| launch density | `20.44 / ms` | `0.0436 / ms` | `-99.79%` |
| `<50us` 小 kernel | `168` | `0` | `-100%` |

trace 形态也很清楚：

- `eager`：`480` 次 `cudaLaunchKernel`，top kernel 以 elementwise / copy / `ampere_sgemm_*`
  交织为主
- `triton`：只剩 `1` 次 `cuLaunchKernelEx`，对应单个
  `_lokr_grouped_delta_forward_kernel`

数值对齐（microbench 默认权重缩放下）：

- `max_abs = 0.0625`
- `mean_abs = 0.00258`
- `rms_abs = 0.00465`

注意 `max_rel` / `mean_rel` 会被接近 0 的输出位置放大，因此这里以 `abs diff` 为主看。

这组 microbench 的含义很明确：单次 grouped delta forward 的纯 latency 下降不算特别大，
但 kernel / launch 密度几乎被清空，说明 `triton` 的主要价值是把执行形态从
“大量碎 kernel”切成“一个大 kernel”。

### LoKr 50-step 无 nsys 真实训练对比（2026-07-02）

随后又跑了不挂 nsys 的 50-step 真训练，口径如下：

```bash
timeout -s INT 3600 .venv/bin/python -m bench.mfu.run_training \
  --suite baseline \
  --steps 50 \
  --metric-step-window 10-45 \
  --peak-tflops 119.07072 \
  --peak-probe-level block \
  --gpu-index 1 \
  --allow-low-vram \
  --output-root output/bench/mfu/lokr_grouped_delta_<backend>_50step_nonsys \
  -- \
  --network_args lokr_grouped_delta_backend=<eager|triton>
```

结果：

| 指标 | `eager` | `triton` | 变化 |
| --- | ---: | ---: | ---: |
| avg step sec | `6.199972` | `5.047806` | `-18.58%` |
| median step sec | `6.1915` | `5.0440` | `-18.53%` |
| p90 step sec | `6.338` | `5.190` | `-18.12%` |
| achieved TFLOPS | `8.5082` | `10.4502` | `+22.82%` |
| MFU | `7.1455%` | `8.7764%` | `+1.63 pct` |
| peak allocated GB | `1.8685` | `1.8720` | `+0.0035 GB` |
| peak reserved GB | `2.1484` | `2.0586` | `-0.0898 GB` |
| final avr_loss | `0.120378` | `0.120316` | 基本重合 |

这一步很关键：虽然 microbench 里单次 grouped delta forward 只快了约 `2.3%`，但真实训练
50-step 热窗里，端到端 step time 稳定快了约 `18.6%`。这说明：

1. `forward fused` 的收益不止是“单个 kernel 快一点”，更重要的是它持续降低了整条训练链里的
   host launch / 小 kernel 压力。
2. 这条优化已经明显越过“只有 1~2%，不值得继续”的门槛，值得继续评估 backward fused。
3. 同时，`triton` 路径没有带来可见的 loss 漂移；`avr_loss` 基本重合，显存也没有恶化，
   `peak_reserved` 反而略低。

### LoKr 50-step 3-seed 稳定性矩阵（2026-07-02）

为了确认 `forward fused` 的收益不是单次好运气，又补了一轮 3-seed 小矩阵。口径保持不变：

- 同一配置
- 不挂 nsys
- `steps = 50`
- `metric-step-window = 10-45`
- 只切换 `seed` 与 `lokr_grouped_delta_backend`

使用的 seed：

- `42`
- `43`
- `44`

其中 `seed=42` 复用前面已经完成的无 nsys 50-step 结果，再补跑 `43/44`。

逐 seed 对比：

| Seed | `eager avg` | `triton avg` | `avg` 提升 | `eager median` | `triton median` | `median` 提升 | `eager p90` | `triton p90` | `p90` 提升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `42` | `6.199972` | `5.047806` | `18.58%` | `6.1915` | `5.0440` | `18.53%` | `6.338` | `5.190` | `18.11%` |
| `43` | `6.286000` | `5.065139` | `19.42%` | `6.2550` | `5.0515` | `19.24%` | `6.552` | `5.165` | `21.17%` |
| `44` | `6.612167` | `5.114250` | `22.65%` | `6.5770` | `5.1060` | `22.37%` | `6.949` | `5.239` | `24.61%` |

聚合结论：

- `avg step` 平均提升约 `20.22%`
- 三颗 seed 的 `avg step` 提升最小值也有 `18.58%`
- `median step` 三颗 seed 全部 `> 18%`
- `p90 step` 三颗 seed 全部 `> 18%`

也就是说，如果把通过线定义为“`triton` 在 2~3 次里都稳定快 `>5%`”，那么这轮不只是通过，
而是明显超线。

loss 与显存也保持稳定：

| Seed | `eager avr_loss` | `triton avr_loss` | loss delta | `alloc delta GB` | `reserved delta GB` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `42` | `0.120378` | `0.120316` | `-0.000062` | `+0.0035` | `-0.0898` |
| `43` | `0.117094` | `0.117018` | `-0.000076` | `+0.0035` | `-0.0898` |
| `44` | `0.109687` | `0.109627` | `-0.000060` | `+0.0035` | `-0.0898` |

这里可以直接读出三点：

1. `avr_loss` 没有漂，三颗 seed 的 `loss delta` 都只有 `6e-5 ~ 8e-5` 量级。
2. `peak_allocated_gb` 只增加约 `0.0035GB`，基本可视为不变。
3. `peak_reserved_gb` 三颗 seed 都稳定下降约 `0.0898GB`。

因此，“先把 forward fused 的收益做成稳定结论”这一关已经可以视为通过：

- `triton` 在 3/3 个 seed 上都稳定快于 `eager`
- 所有 seed 的 `avg/median/p90` 提升都远高于 `5%`
- loss 不漂
- 显存没有恶化

换句话说，`forward fused` 的真实训练收益已经足够稳，不需要再把它当成“可能只是单次好运气”的
可疑项。

### LoKr Backward Microbench 归因（2026-07-02）

为了判断下一步是否值得继续做 `backward fused`，又把 microbench 扩成了
`forward_only + forward_backward` 两套口径，并在 trace 里用
`record_function("lokr_forward")` / `record_function("lokr_backward")` 分开归因。

- 脚本：`bench/lokr_grouped_delta/run_microbench.py`
- 主产物：
  - `output/bench/lokr_grouped_delta_microbench/20260702-1711-hot_mlp1_backward_20260702/summary.json`
  - `output/bench/lokr_grouped_delta_microbench/20260702-1713-hot_mlp1_backward_recheck_20260702/summary.json`

两次都使用同一组热 shape：

- `outer_shape = [2, 1, 72, 56]`
- `rows = 8064`
- `factor = 8`
- `in_features = 2048`
- `out_features = 8192`
- `group_size = 8`

主跑结果：

| 指标 | `eager` | `triton` | 变化 |
| --- | ---: | ---: | ---: |
| forward-only avg | `23.5288 ms` | `22.9368 ms` | `-2.52%` |
| fwd+bwd forward avg | `28.1342 ms` | `23.2595 ms` | `-17.33%` |
| fwd+bwd backward avg | `58.5718 ms` | `36.0590 ms` | `-38.44%` |
| fwd+bwd total avg | `86.7325 ms` | `59.3207 ms` | `-31.61%` |
| backward share | `67.53%` | `60.79%` | `-` |

对应 trace 归因：

| 范围 | `eager` kernel/launch | `triton` kernel/launch | 备注 |
| --- | ---: | ---: | --- |
| `forward` | `480 / 480` | `1 / 1` | forward 形态已经被压平 |
| `backward` | `1151 / 1151` | `1151 / 1151` | backward 形态完全没变 |
| `backward <50us` | `971` | `971` | 小 kernel 仍然极密 |
| `all` | `1631 / 1631` | `1152 / 1152` | `triton` 几乎只剩 backward |

为了确认这不是单次偶然值，又做了一个短迭代复核（`warmup=3, iters=8`）：

| 指标 | `eager` | `triton` | 变化 |
| --- | ---: | ---: | ---: |
| forward-only avg | `23.9904 ms` | `22.9990 ms` | `-4.13%` |
| fwd+bwd forward avg | `25.2885 ms` | `23.2289 ms` | `-8.14%` |
| fwd+bwd backward avg | `55.9911 ms` | `41.5906 ms` | `-25.72%` |
| fwd+bwd total avg | `81.2972 ms` | `64.8211 ms` | `-20.27%` |

但这里要非常谨慎地解读：

1. 两次 run 都稳定表明：`backward` 仍然是主要剩余热点，单模块窗口里占总时长约
   `60% ~ 69%`。
2. `triton` 路径下，`forward` 已经几乎被压成 `1` 次 launch；总 trace 中剩余的
   `1151` 次 launch 基本全部来自 `backward`，`triton_backward_launch_share_pct`
   约 `99.9%`。
3. 真正决定“要不要继续做 backward fused”的关键证据不是 event timing，而是 trace：
   `backward` 范围里的 kernel 数、launch 数和 `<50us` 小 kernel 数完全没有下降，
   说明当前实现还没有实际优化 backward 的执行形态。

也就是说，`backward_ms` 在 event timing 上看起来有 `25% ~ 38%` 的下降，但这不能直接当成
“backward 已经自然变快”的结论。反过来看，trace 里：

- `eager backward kernel_total_us = 31150.111 / 31148.322`
- `triton backward kernel_total_us = 31254.376 / 31254.376`

两轮都几乎持平，甚至 `triton` 侧略高一点。因此当前更稳妥的判断是：

- `forward fused` 的收益已经坐实；
- `backward` 仍然保留了完整的碎 kernel / 高频 launch 形态；
- 当前 microbench 的 `backward_ms_drop_pct` 很可能混入了 stream 排布或 timing 口径差异，
  不能直接拿来当作 backward 现成收益；
- 但正因为 `backward` 剩余形态几乎原封未动，所以它仍然是最值得继续推进的下一热点。

这也把下一阶段的门槛说清楚了：

- 继续做 `backward fused` 是有价值的；
- 但在投入完整工程量前，应该先用当前归因结果定义目标：压的是 `backward` 的
  launch 密度、小 kernel 数和端到端 step time，而不是追逐一个口径可疑的
  `backward_ms_drop_pct`。

阶段决策：

- 本阶段目标“做 backward 归因 / backward microbench，评估是否值得继续做 backward fused”
  已完成。
- 结论是：值得继续，但下一阶段应以 `backward fused prototype` 为目标，优先验证
  `launch_count`、`<50us` 小 kernel 数和 50-step 真训练步时能否继续下降。

### LoKr Backward 子阶段归因 + `grad_x` Prototype Smoke（2026-07-02）

这一轮先把 `_lokr_add_grouped_delta_backward()` 内部显式拆成 4 个 range：

- `recompute_projected`
- `grad_w1_reduce`
- `grad_w2_reduce`
- `grad_x_writeback`

实现位置：

- `networks/plugins/lokr/autograd.py`
- `bench/lokr_grouped_delta/run_microbench.py`

microbench trace 现在会把 `backward_phases` 单独写进 `summary.json`。先用
`triton forward + eager backward` 跑一个短 smoke：

- `output/bench/lokr_grouped_delta_microbench/20260702-1734-triton_fwd_eager_bwd_smoke_20260702/summary.json`

按 `range_total_us` 看，当前 backward 内部热点排序是：

| phase | range_total_us | launch_count | `<50us` kernels | 说明 |
| --- | ---: | ---: | ---: | --- |
| `grad_w2_reduce` | `56.654 ms` | `400` | `320` | 当前最重的 reduction 段 |
| `grad_w1_reduce` | `41.880 ms` | `320` | `240` | 第二热点 |
| `grad_x_writeback` | `30.861 ms` | `248` | `248` | launch 很碎，但 GPU kernel time 不高 |
| `recompute_projected` | `11.382 ms` | `80` | `80` | 不是第一优先级 |

这里有个很关键的信号：如果只看 launch 数，`grad_x_writeback` 很显眼；但如果同时看
`kernel_total_us`，它并不是当前最贵的 GPU 计算段。真正最应该优先盯的是：

1. `grad_w2_reduce`
2. `grad_w1_reduce`
3. `grad_x_writeback`

随后又做了一个实验态 `grad_x` Triton 原型，只替换 `grad_x_writeback` 路径，`grad_w1/w2`
仍保留 eager：

- `output/bench/lokr_grouped_delta_microbench/20260702-1734-triton_fwd_tritongradx_bwd_smoke_20260702/summary.json`

与基线 `triton forward + eager backward` 对比：

| 指标 | eager backward | `triton_grad_x` | 变化 |
| --- | ---: | ---: | ---: |
| backward avg | `38.184 ms` | `52.568 ms` | `+37.7%` |
| backward launch_count | `1151` | `903` | `-21.5%` |
| backward `<50us` kernels | `971` | `722` | `-25.6%` |
| `grad_x_writeback` launch_count | `248` | `1` | `-99.6%` |
| `grad_x_writeback` kernel_total_us | `3.586 ms` | `24.239 ms` | `+575.8%` |

这说明第一版 `grad_x` fused prototype 虽然成功压掉了大量 launch / 小 kernel，但当前 kernel
本体太慢，吞掉了这部分收益，结果是：

- 组件形态更“干净”
- 但 backward latency 实际变差

所以现阶段的工程判断应该再收紧一层：

- `backward fused` 方向仍然值得做；
- 但不应该把 `grad_x` 当成唯一优先级，因为它更像“launch 很碎但 GPU 本体不算最贵”的段；
- 当前最值钱的下一步，应该优先评估 `grad_w2_reduce` / `grad_w1_reduce` 的
  tile partial + reduction 方案；
- 如果继续打 `grad_x`，需要先把 Triton kernel 的 tile / memory 行为调顺，否则只会得到
  “launch 更少、时间更长”的坏交换。

随后又把 `grad_x` prototype 改成第二版混合实现：

- Triton kernel 只做 factor mixing，写出 `mixed[row, in_factor, out_dim]`
- `grad_x` 主乘法改回 cuBLAS 大 GEMM
- 仍通过同一个隐藏 `backward_backend="triton_grad_x"` 入口启用

第二版 smoke：

- `output/bench/lokr_grouped_delta_microbench/20260702-1740-triton_fwd_tritongradx_v2_bwd_smoke_20260702/summary.json`

与当前 `triton forward + eager backward` 基线相比：

| 指标 | eager backward | `triton_grad_x v2` | 变化 |
| --- | ---: | ---: | ---: |
| backward avg | `38.184 ms` | `32.402 ms` | `-15.14%` |
| backward launch_count | `1151` | `932` | `-19.0%` |
| backward `<50us` kernels | `971` | `732` | `-24.6%` |
| `grad_x_writeback` range_total_us | `30.861 ms` | `11.822 ms` | `-61.7%` |
| `grad_x_writeback` launch_count | `248` | `30` | `-87.9%` |

这说明第二版 `grad_x` 原型终于把方向掰正了：

- launch / 小 kernel 数下降
- 组件级 backward latency 也一起下降

但这还不够说明它能过真实训练 gate，所以又补了训练链路验证。

### `triton_grad_x` 训练链路 Smoke + 50-step Gate（2026-07-02）

为了不接正式 config / WebUI，这一轮只把 hidden backend 透传到 LoKr plugin args：

- `lokr_grouped_delta_backward_backend=eager`
- `lokr_grouped_delta_backward_backend=triton_grad_x`

接线位置：

- `networks/plugins/lokr/module.py`
- `networks/plugins/lokr/__init__.py`

并先做了 2-step smoke，确认训练链路能实际走通：

| 配置 | avg_step | peak |
| --- | ---: | ---: |
| `triton forward + eager backward` | `5.068s` | `2.0586GB` |
| `triton forward + triton_grad_x` | `4.956s` | `2.1094GB` |

2-step 只说明“链路可跑且方向不是反的”，不能当热窗结论，因此又补了 current-worktree 下的
同口径 50-step 对照：

- eager:
  `output/bench/mfu/lokr_grouped_delta_tritoneager_50step_nonsys_20260702/runs.csv`
- `triton_grad_x`:
  `output/bench/mfu/lokr_grouped_delta_tritongradx_50step_nonsys_20260702/runs.csv`

50-step 热窗结果（`metric-step-window=10-45`）：

| 指标 | eager backward | `triton_grad_x v2` | 变化 |
| --- | ---: | ---: | ---: |
| avg step sec | `5.0125` | `4.9515` | `-1.22%` |
| TFLOPS | `10.5238` | `10.6534` | `+1.23%` |
| MFU | `8.8383%` | `8.9471%` | `+0.109 pct` |
| peak GB | `2.0586` | `2.1094` | `+0.0508GB` |

这组结果把 gate 结论说得很清楚：

1. `backward launch_count` 明显下降：通过。
2. `backward <50us` 小 kernel 数显著下降：通过。
3. 真训练 `step time >= 5%` 改善：未通过，当前只有约 `1.22%`。
4. 显存新增 `<= 0.05GB`：边界未通过，当前约 `+0.0508GB`。

因此当前阶段的最终结论是：

- `triton_grad_x v2` 已经证明“组件级上可以把 backward 形态压干净一些，并拿到正收益”；
- 但它还没有跨过真实训练 gate，不值得直接扩成正式方案；
- 下一轮不应继续只盯 `grad_x`，而应把主攻点转向 `grad_w2_reduce / grad_w1_reduce`，
  因为 current 50-step 结果已经说明，仅靠 `grad_x` 这一段还拿不到足够大的端到端收益。

### `triton_grad_w2_partial` Hidden Prototype Smoke（2026-07-02）

随后按同样的 hidden-backend 路线，先做了 `grad_w2_reduce` 的混合式 prototype：

- backend 名：`lokr_grouped_delta_backward_backend=triton_grad_w2_partial`
- 形态：
  - Triton kernel 先按 `out_factor group` 做 factor mixing，生成
    `mixed[row, in_factor, out_dim]`
  - `grad_w2` 主乘法改成 cuBLAS GEMM：
    `grad_w2 += mixed_2d.T @ x_2d`
- 其余 backward 段暂时保持 eager，方便隔离 `grad_w2_reduce` 的单段收益

第一版尝试把 row chunk 也按 `mixed + x + grad` 临时量重新收紧，产物：

- `output/bench/lokr_grouped_delta_microbench/20260702-1830-triton_fwd_tritongradw2partial_bwd_smoke_20260702/summary.json`

结果很差：`chunk_rows` 被压得过小，row chunk 数量暴涨，直接把 backward launch 数放大到
`15843`，`backward avg` 退化到 `754.346 ms`。这版说明“按显式临时量硬收 chunk”在当前 hot
shape 上不可取，因此没有继续保留。

随后把 `triton_grad_w2_partial` 改成第二版：继续沿用现有 `_projection_chunk_rows()`，
只替换 `grad_w2_reduce` 的内核形态，不再额外缩 row chunk。对照如下：

- eager baseline：
  `output/bench/lokr_grouped_delta_microbench/20260702-1829-triton_fwd_eager_bwd_w2partial_smoke_20260702/summary.json`
- `triton_grad_w2_partial v2`：
  `output/bench/lokr_grouped_delta_microbench/20260702-1831-triton_fwd_tritongradw2partial_v2_bwd_smoke_20260702/summary.json`

| 指标 | eager backward | `triton_grad_w2_partial v2` | 变化 |
| --- | ---: | ---: | ---: |
| backward avg | `37.464 ms` | `34.853 ms` | `-6.97%` |
| total avg | `60.585 ms` | `58.019 ms` | `-4.24%` |
| backward launch_count | `1151` | `1041` | `-9.56%` |
| backward `<50us` kernels | `971` | `841` | `-13.39%` |
| `grad_w2_reduce` range_total_us | `55.649 ms` | `9.622 ms` | `-82.71%` |
| `grad_w2_reduce` launch_count | `400` | `70` | `-82.5%` |
| `grad_x_writeback` range_total_us | `29.101 ms` | `68.259 ms` | `+134.6%` |

这组结果说明：

1. `grad_w2_reduce` 这个最大热点已经被真正打穿了；phase 级 launch 数和区间时长都出现了
   大幅下降。
2. 与 `grad_x v1` 不同，这次不是“形态更干净但 latency 更差”，而是组件级 backward
   latency 也同步下降，说明这条 mixed Triton + cuBLAS 路线方向是对的。
3. 收益并不是免费的：因为 `grad_x_writeback` 仍然保留 eager 路径，当前有明显“热点转移”
   到 `grad_x_writeback` 的现象。

所以到这一步，`triton_grad_w2_partial` 的结论可以先定成：

- 组件级：通过，值得保留为 hidden prototype；
- 训练级：本轮还没有新的热窗 gate 结论，不能直接外推成正式默认方案；
- 下一刀如果继续做 backward fused，优先级应该是
  `grad_w2_partial + grad_x_writeback` 组合，或者转向同类的 `grad_w1_reduce`。

### `triton_grad_w2_grad_x` 组合版 Hidden Prototype（2026-07-02）

接着把 `triton_grad_w2_partial` 和现有 `triton_grad_x v2` 合到了一起，做成新的 hidden backend：

- backend 名：`lokr_grouped_delta_backward_backend=triton_grad_w2_grad_x`
- 目标：
  - `grad_w2_reduce` 继续走 Triton factor-mix + cuBLAS GEMM
  - `grad_x_writeback` 不再回退到 eager per-factor `einsum + matmul`
  - 优先复用同一个 row chunk 生命周期里的 `mixed[row, in_factor, out_dim]`
- 当前实现：
  - 每个 row chunk 预分配一块 `mixed_buffer`
  - `grad_w2_reduce` 先填 `mixed`
  - 同一块 `mixed` 直接喂给 `grad_w2` GEMM 和 `grad_x` GEMM
  - 不额外再做一轮 factor-mix

本轮组件级对照：

- eager：
  `output/bench/lokr_grouped_delta_microbench/20260702-1854-triton_fwd_eager_bwd_comboeval_20260702/summary.json`
- `triton_grad_w2_partial`：
  `output/bench/lokr_grouped_delta_microbench/20260702-1855-triton_fwd_tritongradw2partial_comboeval_20260702/summary.json`
- `triton_grad_w2_grad_x`：
  `output/bench/lokr_grouped_delta_microbench/20260702-1855-triton_fwd_tritongradw2gradx_comboeval_20260702/summary.json`

#### eager -> `triton_grad_w2_grad_x`

| 指标 | eager backward | `triton_grad_w2_grad_x` | 变化 |
| --- | ---: | ---: | ---: |
| backward avg | `35.215 ms` | `22.712 ms` | `-35.50%` |
| total avg | `58.377 ms` | `45.881 ms` | `-21.41%` |
| backward launch_count | `1151` | `583` | `-49.35%` |
| backward `<50us` kernels | `971` | `453` | `-53.35%` |
| `grad_w2_reduce` range_total_us | `50.004 ms` | `10.336 ms` | `-79.33%` |
| `grad_x_writeback` range_total_us | `26.973 ms` | `5.639 ms` | `-79.10%` |
| `grad_x_writeback` launch_count | `248` | `39` | `-84.27%` |
| avg `peak_allocated_delta_gb` | `0.1288` | `0.1422` | `+0.0134 GB` |

#### `triton_grad_w2_partial` -> `triton_grad_w2_grad_x`

| 指标 | `triton_grad_w2_partial` | `triton_grad_w2_grad_x` | 变化 |
| --- | ---: | ---: | ---: |
| backward avg | `34.634 ms` | `22.712 ms` | `-34.42%` |
| total avg | `57.900 ms` | `45.881 ms` | `-20.76%` |
| backward launch_count | `1041` | `583` | `-44.00%` |
| backward `<50us` kernels | `841` | `453` | `-46.14%` |
| `grad_w2_reduce` range_total_us | `10.044 ms` | `10.336 ms` | 基本持平 |
| `grad_x_writeback` range_total_us | `67.864 ms` | `5.639 ms` | `-91.69%` |
| avg `peak_allocated_delta_gb` | `0.1453` | `0.1422` | `-0.0031 GB` |

这组结果基本把组件级 gate 说死了：

1. backward avg：明显通过。
2. backward launch_count：明显通过。
3. `<50us` 小 kernel：明显通过。
4. `grad_x_writeback` phase range：明显通过。
5. 显存增量：相对 eager 只增加约 `0.013GB`，相对 `triton_grad_w2_partial` 还略降。

也就是说，这版组合 backend 不是“把热点从 `grad_w2` 推给 `grad_x`”，而是真的把两段一起压了下去。

#### 训练链路 smoke 尝试

按约定只在组件级明显更好后再碰训练链路，于是补跑了：

```bash
timeout -s INT 60 .venv/bin/python -m bench.mfu.run_training \
  --suite baseline \
  --steps 2 \
  --metric-step-window off \
  --peak-tflops 119.07072 \
  --peak-probe-level block \
  --gpu-index 1 \
  --allow-low-vram \
  --output-root output/bench/mfu/lokr_grouped_delta_tritongradw2gradx_2step_smoke_20260702 \
  -- \
  --network_args lokr_grouped_delta_backend=triton \
  lokr_grouped_delta_backward_backend=triton_grad_w2_grad_x
```

这轮 `2-step` 没有在 `60s` 维护超时内收尾；当前产物：

- `output/bench/mfu/lokr_grouped_delta_tritongradw2gradx_2step_smoke_20260702/baseline_s42_2step/logs/train.stdout.log`
- `output/bench/mfu/lokr_grouped_delta_tritongradw2gradx_2step_smoke_20260702/baseline_s42_2step/logs/baseline_s42_2step.progress.jsonl`

从日志看，这次不是 backend 数值错误或 crash，而是 `torch.compile` 预热仍卡在 step 0 期间，
最终被 `timeout -s INT 60` 中断；`progress.jsonl` 里 `final_step = 0`，因此这轮还不能拿来做
训练侧速度判断。

因此当前阶段可以先落一个中间结论：

- `triton_grad_w2_grad_x` 已经通过组件级 gate；
- 下一步值得继续做训练侧 gate，但需要接受 `2-step smoke` 在 `torch.compile` 打热之前，
  很可能不适合作为 60s 维护超时内的稳定判据。

#### bench `no_compile` 修复后补跑的训练侧 smoke（2026-07-02）

继续追这个训练侧 gate 时，先发现 `bench.mfu.run_training` 里的 `no_compile` arm 实际上是失效的：

- arm 里虽然带了 `--no-torch_compile` 哨兵；
- 但实现只是“不要额外拼 `--torch_compile`”；
- `configs/base.toml` 和 `configs/gui-methods/mfu_rokkotsu_cached.toml` 仍然把
  `torch_compile = true` 写死在 merge 链里；
- 而训练 CLI 只有 `--torch_compile`，没有正式的 `--no-torch_compile`。

所以先把 bench 修到“真的能关 compile”：

- `bench/mfu/run_training.py` 现在会在 `no_compile` arm 下物化一份当前
  `base -> preset -> gui-method` merged config；
- 然后只把 `torch_compile = false` 覆盖进去，再通过 `--config_file` 喂给训练；
- 这样不需要新增正式 CLI/配置面，只修 bench harness 本身。

回归：

```bash
timeout 60 .venv/bin/python -m pytest tests/test_mfu_bench.py
```

结果：`12 passed`

修完之后，补跑了 `no_compile` 下的 eager / combo 对照。

2-step 完整产物：

- eager：
  `output/bench/mfu/lokr_grouped_delta_tritoneager_nocompilefix_2step_20260702/no_compile_s42_2step/`
- combo：
  `output/bench/mfu/lokr_grouped_delta_tritoncombo_nocompilefix_2step_20260702/no_compile_s42_2step/`

| 指标 | eager | `triton_grad_w2_grad_x` | 变化 |
| --- | ---: | ---: | ---: |
| first-step latency | `28.916 s` | `25.305 s` | `-12.49%` |
| step interval avg | `5.157 s` | `4.764 s` | `-7.62%` |
| active window (`run_end - run_start`) | `34.794 s` | `30.799 s` | `-11.48%` |
| peak allocated | `2.0145 GB` | `2.0213 GB` | `+0.0068 GB` |
| peak reserved | `2.2363 GB` | `2.2363 GB` | 持平 |

这里的 `2-step` 已经能说明两件事：

1. 训练链路在 `torch.compile` 关闭后可以真实走通，不再停在 step 0。
2. 组合版 backend 在真实训练里也没有翻方向，step interval 仍然稳定快于 eager。

考虑到 `2-step` 只有一个 interval，又补了 `4-step` smoke，想拿到更接近 warm 的区间。
但在当前机器和 `timeout 60` 维护约束下，两边都只稳定记录到了前 3 个 step，随后被
外层 `timeout` 中断；`run_end.status=error` 里的
`DataLoader worker ... killed by signal: Terminated` 是 timeout 引起的派生错误，
不是 backend 本身的数值或兼容性问题。

4-step 部分产物：

- eager：
  `output/bench/mfu/lokr_grouped_delta_tritoneager_nocompilefix_4step_20260702/no_compile_s42_4step/`
- combo：
  `output/bench/mfu/lokr_grouped_delta_tritoncombo_nocompilefix_4step_20260702/no_compile_s42_4step/`

在两边都已落盘的前 3 个 step 上比较：

| 指标 | eager | `triton_grad_w2_grad_x` | 变化 |
| --- | ---: | ---: | ---: |
| first-step latency | `25.378 s` | `25.507 s` | 基本持平 |
| step2-3 interval avg | `5.188 s` | `4.779 s` | `-7.88%` |
| active window to timeout/error | `40.965 s` | `39.817 s` | `-2.80%` |
| peak allocated | `2.0145 GB` | `2.0213 GB` | `+0.0068 GB` |
| peak reserved | `2.2402 GB` | `2.2402 GB` | 持平 |

所以训练侧目前能落下的最稳妥结论是：

- 方向是对的：真实训练 step interval 也能复现组件级的优势，当前大约是 `7%~8%`；
- 显存代价依旧很小：allocated 只多约 `0.0068 GB`，reserved 基本不动；
- 但在当前 10GB 3080 + `timeout 60` 的维护约束下，`50-step` 热窗暂时拿不到，
  端到端 wall-clock 也仍然会被固定启动成本显著稀释。

因此，这版 hidden backend 已经值得继续保留并向更长训练窗推进；只是下一步若要追
`50-step`，需要先接受“当前维护约束下它不是一次 60 秒后台命令就能稳定收出来的”这个现实。

#### direct train 窗口复核：bench wrapper 不是训练本体（2026-07-02）

为了确认 `bench.mfu` 外层 wrapper 是否还在吃掉维护超时窗口，又补了一条更直接的验证链：

- 继续使用 `torch_compile=false`
- 继续使用同一套 LoKr hidden backend 覆盖
- 但绕过 `bench.mfu.run_training` 的 GPU 检查 / 子进程包装，直接跑 `train.py`
- 数据集改成 bench-only 的单样本缓存版：
  `configs/bench/mfu_rokkotsu_cached_dataset_single.toml`
  - 同样是 `896x1152` / `4032 tokens`
  - 用 `path_pattern = "Floating_in_a_202604200120.png"` 只保留 1 张缓存图
  - `num_repeats = 64`，足够覆盖多步 smoke
  - `--max_data_loader_n_workers 0`，避免 worker 生命周期再引入额外噪音

这里有两个重要结论：

1. 把数据集缩成单样本，对 `run_start -> first step` 的改善很小，说明前面那段固定成本的主因
   不是 57 张图的 dataset scan。
2. 真正影响 `timeout 60` 的，是 `bench.mfu` 外层 wrapper 本身也在吃窗口；直接跑训练本体后，
   `4-step` 可以稳定完整落盘。

direct 4-step 产物：

- eager：
  `output/bench/direct_train/lokr_grouped_delta_tritoneager_single_nocompile_4step_20260702/`
- combo：
  `output/bench/direct_train/lokr_grouped_delta_tritoncombo_single_nocompile_4step_20260702/`

对比结果：

| 指标 | eager | `triton_grad_w2_grad_x` | 变化 |
| --- | ---: | ---: | ---: |
| first-step latency | `26.045 s` | `25.695 s` | `-1.34%` |
| step interval avg (steps 2-4) | `5.223 s` | `4.807 s` | `-7.95%` |
| last logged step ts | `53.648 s` | `51.743 s` | `-3.55%` |
| run_end status | `stopped` | `ok` | combo 更从容 |
| peak allocated | `2.0145 GB` | `2.0213 GB` | `+0.0068 GB` |
| peak reserved | `2.2402 GB` | `2.2402 GB` | 持平 |

这里的 `run_end status` 很关键：

- eager：4 个 step 已经全部完成，但在最终保存阶段被 `timeout 60` 截断，`run_end=stopped`
- combo：4 个 step 和最终保存都在超时前完成，`run_end=ok`

也就是说，在同样的 60 秒维护约束下，combo 不只是“interval 更快”，而是已经开始跨过
“这组 smoke 能不能自然收尾”的门槛。

又补跑了 direct 5-step，结果两边都没有完整收尾：

- eager：
  `output/bench/direct_train/lokr_grouped_delta_tritoneager_single_nocompile_5step_20260702/`
- combo：
  `output/bench/direct_train/lokr_grouped_delta_tritoncombo_single_nocompile_5step_20260702/`

两边都只稳定记录到了前 3 个 step，随后在第 4 步附近被 `timeout 60` 打断；但 interval 对照仍然一致：

- eager partial 5-step：`avg interval = 5.198 s`
- combo partial 5-step：`avg interval = 4.804 s`

因此当前训练侧最干净的判断可以再往前推进一格：

- `bench.mfu` wrapper 不适合作为当前机器上的最长 smoke 窗口载体；
- direct train + single-sample cached dataset 是现阶段更可靠的训练侧验证路径；
- 在这个路径下，combo 已经拿到了比 eager 更稳定、更完整的 `4-step` 收尾证据；
- 但 `5-step` 仍然顶到 60 秒天花板，所以当前机器上的稳定训练 gate 可以先定在
  direct `4-step`，而不是继续在 `5-step` / `50-step` 上硬撞。

#### 50-step 分段续训验证：resume slice 路线已打通（2026-07-02）

为了继续往 `50-step` 靠，但又不违反单条后台命令 `timeout 60` 的维护约束，又补做了
step-state 续训验证。

策略：

- 继续使用 direct train + single-sample cached dataset
- 目标仍设成 `--max_train_steps 50`
- 打开 step-cadence state save：
  - `--save_every_n_steps 4`
  - `--save_state`
  - `--save_last_n_steps_state 1`
- 每段都用 `timeout 60` 截住，让训练自然在 step-save 之后停下
- 下一段通过：
  - `--resume <...-step00000004-state>`
  - `--skip_until_initial_step`
  继续接着跑

当前已验证的产物：

- 第一段：
  - progress：
    `output/bench/direct_train/lokr_grouped_delta_tritoncombo_single_resume50_20260702/logs/combo_resume50.progress.jsonl`
  - state：
    `output/bench/direct_train/lokr_grouped_delta_tritoncombo_single_resume50_20260702/ckpt/combo_resume50-step00000004-state/`
- 第二段：
  - progress：
    `output/bench/direct_train/lokr_grouped_delta_tritoncombo_single_resume50_20260702/logs/combo_resume50_slice2.progress.jsonl`
  - state：
    `output/bench/direct_train/lokr_grouped_delta_tritoncombo_single_resume50_20260702/ckpt/combo_resume50-step00000008-state/`

`train_state.json` 已确认推进到：

```json
{"current_epoch": 1, "current_step": 8}
```

这说明两件事已经被当前工作树里的真实产物证明：

1. `step 4` state 可以在 60 秒窗内稳定落盘；
2. 从 `step 4` resume 后，训练会继续推进到 `step 8`，而不是卡在 resume / scheduler / dataloader 对齐上。

resume 后第二段的 `progress_jsonl` 里，`global_step` 已经直接从 `5` 继续到 `8`：

- `5 -> 6 -> 7 -> 8`

说明 `--resume + --skip_until_initial_step` 这条链路在当前 LoKr hidden backend 上是可用的。

还可以再细分一下 interval：

- 第一段（steps `1~4`）：
  - intervals: `4.778 / 4.830 / 5.900 s`
  - avg: `5.169 s`
- 第二段（steps `5~8`）：
  - intervals: `4.841 / 4.823 / 5.917 s`
  - avg: `5.194 s`

这里最后一个 interval 会把 `step 4 / step 8` 的 step-save 开销一起吞进去，所以会明显偏大。
如果只看每段前两个不带保存边界的 steady-state interval：

- `4.778 / 4.830 / 4.841 / 4.823 s`
- steady avg: `4.818 s`

这个数和前面的 direct combo `4-step` steady 区间是对齐的，说明：

- 切片续训不会把训练本体的 steady-state 步时明显打坏；
- 额外成本主要来自每个 slice 的 state save 边界，而不是 resume 后每一步都变慢。

因此，当前阶段已经可以把“50-step 在现有维护约束下如何推进”这件事定成：

- **不是不可做**
- 而是需要走 **分段 step-state resume** 路线

也就是说，`50-step` 现在的阻力主要是运行时长和切片次数，不再是技术路径是否成立。

#### 50-step 已完整收尾：4-step + 2-step 混合切片（2026-07-02）

继续把同一条 `seed=42` 路线往后推后，这组 `50-step` 训练现在已经完整收尾。

最终关键产物：

- progress 根目录：
  `output/bench/direct_train/lokr_grouped_delta_tritoncombo_single_resume50_20260702/logs/`
- 最终 rolling state：
  `output/bench/direct_train/lokr_grouped_delta_tritoncombo_single_resume50_20260702/ckpt/combo_resume50-state/train_state.json`
- 周期 state：
  同目录下保留了 `step00000004 / 08 / 12 / 16 / 20 / 24 / 28 / 32 / 36 / 40 / 44 / 48`
  这 `12` 个 step-state 目录

最终 `train_state.json`：

```json
{"current_epoch": 1, "current_step": 50}
```

推进过程分成两段：

- `steps 1~28`：继续使用 `4-step` step-state slice
- `steps 29~50`：改成 `2-step` end-of-run rolling state

触发切换的原因也很直接。`combo_resume50_slice8.progress.jsonl` 已经证明：

- 从 `step00000028-state` 出发，`4-step` slice 在当前机器上只能稳定推进到 `29 -> 30 -> 31`
- `run_end.status = stopped`
- 超时前没有落下新的 `step00000032-state`

也就是说，到了 `step 28 -> 32` 这一段，`4-step` slice 已经开始贴着 `timeout 60` 的天花板跑，
再强行坚持只会让续跑脚本在“推进了但没 state”这个尴尬区间里打滑。

后续改成：

- 先从 `step00000028-state` 跑一个 `--max_train_steps 30`，验证正常结束会落
  `combo_resume50-state`
- 再以 `combo_resume50-state` 为 rolling resume 点，按
  `32 / 34 / 36 / ... / 50` 的 2-step 节奏继续推进

这条 2-step 路线已经完整成功，`slice9 ~ slice19` 全部 `run_end.status = ok`。

把失败的 `slice8` 排除、只看最终成功串起来的整条 50-step 链：

- 不带保存边界的 steady-state interval 共 `20` 个：
  - `avg = 4.824 s`
  - `min = 4.778 s`
  - `max = 4.843 s`
- 带保存边界的 interval 共 `12` 个：
  - `avg = 5.919 s`
  - `min = 5.877 s`
  - `max = 5.952 s`

这个 `4.824 s` 和前面 direct combo `4-step` 的 `4.807 s` 基本重合，只差约 `+0.35%`。
也就是说：

- 把训练拆成很多个 resume slice 后，训练本体的 steady-state 步时没有明显漂移
- 额外成本依然主要集中在保存边界和反复启动，不是 hidden combo backend 自己在长窗口里变钝

显存也没有出现随 slice 累积抬升：

- `cuda/max_memory_allocated_gb` steady 峰值稳定在 `2.0213 GB`
- `cuda/max_memory_reserved_gb` 落在 `2.2324 ~ 2.2402 GB`

因此，这一轮可以把结论再往前推进一步：

- `triton_grad_w2_grad_x` 不只是通过了组件级 gate 和 direct `4-step` gate
- 它已经在当前维护约束下，真实完成了一条可复现的 `50-step` 训练链
- 当前剩下的主要问题是“怎样更省重启、更少 orchestration overhead 地拿长窗对照”，
  而不再是“这个 hidden backend 到了长窗口会不会自己失稳”

如果后面要补 eager 的 `50-step` 真训练对照，建议复用同样的混合切片口径：

- `1~28` 走 `4-step` state slice
- `29~50` 改 `2-step` rolling state

这样才能避免把“切片方法不同”混进 eager vs combo 的端到端对比里。

#### eager 长窗对照补齐：同一路线尝试后改成更细切片（2026-07-02）

随后把 eager 对照也按同一条 direct train + single-sample cached dataset 路线补了出来。

先说最重要的事实：**同样的切片口径，eager 比 combo 更早撞到 `timeout 60` 的天花板。**

eager 产物分成两段：

- 前半段仍在仓库输出目录：
  `output/bench/direct_train/lokr_grouped_delta_tritoneager_single_resume50_20260702/`
- 后半段 tail 改写到根分区，避免满盘挂载点继续写坏 state：
  `/home/scv/workspace/anima_lora_bench/lokr_grouped_delta_tritoneager_single_resume50_tail_20260702/`

最终 rolling state：

```json
{"current_epoch": 1, "current_step": 50}
```

但 eager 的切片轨迹和 combo 不一样：

- `steps 1~4`：`4-step` state slice 可行
- `steps 4~22`：必须退到 `2-step` rolling state
- `steps 23~50`：必须再退到 `1-step` rolling state

也就是说，combo 能撑住：

- `1~28` 用 `4-step`
- `29~50` 用 `2-step`

而 eager 实际上在两个地方更早碰墙：

1. 从 `step00000004-state` 出发，`4-step` resume 已经来不及落下 `step00000008-state`。
2. `2-step` rolling 虽然能稳定推进到 `step 23`，但 `22 -> 24` 在原输出挂载点上触发了真实的
   `accelerator.save_state()` 写入故障。

那次失败不是 backend 数值或训练逻辑问题，而是环境问题。`df -h` 当时显示：

```text
/dev/nvme1n1p1  836G  836G  632K  100%  /home/scv/nvme0n1p1
```

失败日志对应：

```text
RuntimeError: [enforce fail at inline_container.cc:672] . unexpected pos ...
```

因此没有去碰用户数据目录做清理，而是直接把 tail 切到根分区继续跑，并复用旧链里已经确认可读的：

- `.../ckpt/eager_resume50-state/train_state.json` at `current_step=23`

后半段 `1-step` tail 的真实情况也说明 eager 已经进入“靠近边界”的区间：

- `23 -> 24` 在新输出根下能稳定成功
- `37 -> 38` 第一次尝试仍然被 `timeout 60` 打断
- 同一步第二次重试后成功推进到 `38`
- 之后 `39 -> 50` 又恢复到基本一枪过

所以，当前这条 eager 长窗链的结论不是“它跑不满 50 step”，而是：

- 它能跑满
- 但同样的维护约束下，需要比 combo 更细的切片
- 而且对输出挂载点的可用空间更敏感

#### eager vs combo：长窗对照口径

需要先说明 interval 口径：

- combo 仍然可以从成功链里抽出 `4-step` / `2-step` slice 的稳定 interval
- eager 到 `step 23` 后必须改成 `1-step` slice，因此 `step 24~50` 不再产生 slice 内 interval
- 所以下面 eager 的 interval 指标，取的是 `step 1~22` 这段仍可稳定观测区间

对比结果：

| 指标 | eager | combo | 变化 |
| --- | ---: | ---: | ---: |
| steady interval avg（不带保存边界） | `5.271 s` | `4.824 s` | combo `-9.27%` |
| save-boundary interval avg | `6.552 s` | `5.919 s` | combo `-10.69%` |
| 成功收满 50 step 所需 successful slices | `39` | `18` | combo 少 `53.8%` |
| successful slice 数量比 | `2.17x` | `1.00x` | eager 重启负担更重 |
| peak allocated | `2.0145 GB` | `2.0213 GB` | eager 少 `0.0068 GB` |
| peak reserved | `2.2383 ~ 2.2402 GB` | `2.2324 ~ 2.2402 GB` | 基本同级 |

这里可以把结论收得很具体：

1. 只看训练本体的稳定步时，combo 相比 eager 仍然稳定快约一个 `9%~11%` 的量级。
2. 只看“把 50-step 真训练收完整”这件事，combo 的运维负担也明显更轻：
   eager 需要 `39` 个 successful slices，combo 只要 `18` 个。
3. eager 的显存并没有明显更差，甚至 `peak_allocated` 略低一点；因此长窗差距主要还是落在
   步时和切片/重启负担上，而不是显存峰值。
4. `/home/scv/nvme0n1p1` 满盘导致的 `step24` state 写坏，是一次环境事件，不应该被误读成
   eager backward 数值不稳定；把输出切到有空间的根分区后，tail 训练继续正常推进到了 `50`。

因此，到这一步可以把“eager 的同口径 50-step 长窗对照”定稿成：

- 已补齐
- 结果方向与前面的 `4-step` smoke 一致
- 而且在更接近真实维护场景的长窗口里，combo 的优势不只是 step 更快，
  还包括明显更低的 slice / restart 负担

换句话说，`triton_grad_w2_grad_x` 这条 hidden backend 的训练侧价值，现在不再只是
“短窗里快 `7%~8%`”，而是已经扩展成：

- 长窗 steady-state 也更快
- 同样的 60 秒维护约束下更容易把任务自然推进下去
- 对操作层面的切片复杂度也有实打实的改善

#### 同挂载点 clean rerun：去掉跨盘 tail 和满盘噪音（2026-07-02）

为了把上一轮 eager 长窗里“满盘导致 state 写坏、后半段不得不跨盘 tail”这层环境噪音彻底拿掉，
又在同一挂载点 `/home/scv/nvme0n1p1` 上，从头重跑了两条 fresh baseline：

- combo：
  `output/bench/direct_train/lokr_grouped_delta_tritoncombo_single_resume50_cleanrerun_20260702/`
- eager：
  `output/bench/direct_train/lokr_grouped_delta_tritoneager_single_resume50_cleanrerun_20260702/`

这次的 clean rerun 有一个实际变化需要先说清楚：

- 在当前机器的 fresh run 条件下，`step0 -> step4` 已经不再稳定落在 `timeout 60` 窗口里
- 因此没有再强行复刻旧的 `4-step + 2-step` 混合切片
- 为了保证两边都能在**同一挂载点**、**同一约束**下完整收满，统一改成：
  - 从 `step 0` 开始就走 `2-step rolling`
  - 只在必要时才允许降到 `1-step`

最终结果比预期更整齐：

- combo：`25` 个 `2-step` slice，全部 `run_end.status = ok`
- eager：`25` 个 `2-step` slice，全部 `run_end.status = ok`
- 两边最终 rolling state 都推进到：

```json
{"current_epoch": 1, "current_step": 50}
```

这很关键，因为它说明：

1. 上一轮 eager 长窗里“必须切到 `1-step` tail”的主要噪音，确实来自满盘环境；
2. 在 clean 同挂载点条件下，eager 并不需要跨盘 tail，也能完整收满 `50-step`；
3. 因此这一轮是后续 block swap / LoKr 优化更该优先引用的干净训练基线。

##### clean rerun 对比

| 指标 | eager | combo | 变化 |
| --- | ---: | ---: | ---: |
| successful slices | `25` | `25` | 持平 |
| run_end status | `25 ok` | `25 ok` | 持平 |
| first-step latency avg | `26.317 s` | `26.416 s` | 基本持平 |
| 2-step steady interval avg（不碰 step-save） | `5.291 s` | `4.843 s` | combo `-9.25%` |
| 2-step save-boundary interval avg（目标步是 4 的倍数） | `6.415 s` | `6.018 s` | combo `-6.60%` |
| overall interval avg | `5.831 s` | `5.407 s` | combo `-7.27%` |
| peak allocated | `2.0145 GB` | `2.0213 GB` | eager 少 `0.0068 GB` |
| peak reserved avg | `2.2382 GB` | `2.2363 GB` | 基本同级 |

这里可以把结论收得比上一轮更硬：

1. **启动成本几乎一样。**
   `first-step latency avg` 只有 `0.099s` 差异，说明 combo 的收益不来自“更快进第一步”。
2. **收益仍然稳定落在训练步本体。**
   同样的 `2-step rolling` 口径下，combo 在：
   - 非保存边界 interval 上快约 `9.25%`
   - 保存边界 interval 上快约 `6.60%`
3. **跨盘 tail 已经不再需要。**
   这意味着上一轮 eager 长窗里最容易让人纠结的环境噪音点，现在已经被 clean rerun 消掉了。
4. **显存差距依旧很小。**
   combo 仍然只多约 `0.0068 GB` allocated，reserved 也基本同级。

因此，后续如果要继续推进 block swap 调度层，或者继续评估 LoKr fused kernel 的收益，
更建议把这两条 clean rerun 当作新的训练侧基线，而不是继续引用那条带跨盘 tail 的旧 eager 长窗。

## 不改 blocks_to_swap 的优化方向

当前瓶颈拆成三类：block swap H2D、LoKr elementwise/copy 小 kernel、host launch/sync
排队。保持 `blocks_to_swap=26` 不变时，主线只保留“不增加显存峰值”的方向：

1. block swap 调度层优化。
2. block swap copy plan / slab 化。
3. LoKr fused kernel。

`lokr_project_chunk_bytes` 增大、关闭 full checkpoint / 改 selective checkpoint 可能提升速度，
但都会提高临时激活或常驻激活峰值；在 LoKr 低显存场景不作为当前主线。

### 1. 排除项：配置消融不作为主线

这些不需要改块数量，也不需要改算法结构，但都有显存或数值风险，当前只作为复核项：

- `block_swap_transfer_dtype = "fp8_e4m3"` 已有专项报告，不作为主线优化方向。
  结论见 `docs/findings/anima_fp8_blockswap_transfer_report.md`：H2D 带宽收益真实，
  但 frozen base 权重量化误差没有过门槛；默认仍保留 `bf16`，FP8 只作为实验开关。
- `lokr_project_chunk_bytes = 8388608 / 16777216`：当前 LoKr 自定义 autograd 按 row chunk
  循环，4MiB chunk 会制造更多 matmul/add/copy kernel。增大 chunk 可以减少 launch 数，
  代价是 LoKr 临时激活峰值上升；低显存路径不作为主线。
- `gradient_checkpointing = false` 或替换为更窄的 `selective_checkpoint`：当前 full
  checkpoint 会在 backward 重算 LoKr forward 路径，放大小 kernel 密度。保持 block swap
  数不变时，关闭/收窄 checkpoint 会增加激活峰值；低显存路径不作为主线。
- 正式吞吐跑关闭 `block_swap_profile_jsonl` / nsys GPU metrics：profile 用于归因，正式
  对比应少开 observer overhead。

因此当前主线不再安排 `fp8_e4m3`、增大 LoKr chunk、关闭/收窄 checkpoint 的实验矩阵。
FP8 可以保留为复核项，但不应排在主线前面；已知收益是 PCIe/H2D 层面的，不是 LoKr
低 MFU / OOM 余量的根因修复。

### 2. block swap 调度层优化

当前 offloader 已经是 CPU master + H2D-only，但还有明显 host/sync 成本：

- `_wait_blocks_move()` 原先会对 H2D end event 做 host synchronize。已改成生产路径使用
  `current_stream.wait_event(end_event)`，让依赖留在 CUDA stream 上，减少 host 阻塞；
  profiling 模式仍同步取精确 `h2d_ms`。
- `_swap_weight_devices_cached_cuda()` 每次 swap 都重新 `named_modules()`、构造
  `weight_swap_jobs`、新建 CUDA stream/events。可以在 CPU master 建好后预计算每个
  `(block_to_cpu, block_to_cuda)` 的 swap plan，并复用 copy stream / event 池。
- 方案 4 第一阶段已落地：`Offloader` 现在会缓存每个 block 的 `named_modules()` 映射、
  预热 forward 主路径 `(block_idx_to_cpu, block_idx_to_cuda)` 的 swap plan，并在后续 swap
  里直接复用；CUDA copy stream 也改成单实例复用，`ready_event / h2d_start / h2d_end`
  改成按 timing/marker 分池复用。
- 这轮改动不引入 GPU staging buffer、ring slot 或额外 copy slab，因此不增加显存峰值；
  目标只是压掉 host 侧 plan 构建、stream/event 构造和重复 `named_modules()` 遍历开销。
- 当前 profile window 有 `2364` 次 H2D memcpy。优先探索不增加 GPU 峰值的 slab 化：
  CPU 侧 pinned slab + 复用现有 GPU weight storage 的 view/copy plan，把每 block 多次小
  H2D 降到少量大 H2D。
- `2026-07-01` 追加了 block copy-plan 微基准，直接用 preview3 `net.blocks.0` 的真实
  20 个 frozen weight（合计约 `132.0MiB`）在 RTX 3080 上比较三条路径：
  逐 tensor `copy_`、`torch._foreach_copy_`、CPU slab 整体 H2D。结果基本打平：
  `loop_copy p50=13.31ms`、`foreach_copy p50=13.51ms`、`slab_copy p50=13.59ms`。
  这说明在当前尺寸上，单纯“减少小 H2D 次数”并不会显著降低纯传输时间；如果要继续做
  方案 4，重点应放在 copy 与 runtime 调度的耦合、或结合压缩/更强重叠，而不是只做
  memcpy 合并。
- GPU staging buffer / ring slot 虽然能更早预取，但会增加显存峰值；当前 LoKr 低显存
  主线暂不采用，只在显存预算明确时另开实验。

### 3. LoKr kernel 层优化

LoKr 当前热点来自 `LoKrAddGroupedDeltaFn` / grouped projection 的 Python loop：

- 对每个 row chunk、factor slice 执行多次 matmul、einsum、mul、add 和 dtype copy；
  这些函数还被 `torch.compiler.disable` 包住，compile 无法把它们融合。
- 短期不靠增大 chunk，而是在相同 chunk 峰值下重排计算：减少重复 `.float()` /
  `.to(dtype)`、复用 transposed/cast weight、把 scalar gate/mul/add 合并进同一写回路径。
- 中期把 `x_view @ w2.T` 做成 batched contraction，再用 `w1` 做 factor mixing，在不提高
  chunk 峰值的前提下减少 per-factor 小 matmul/elementwise。
- 长期写 Triton/CUDA fused kernel：forward 直接把 LoKr delta 加到 base output，
  backward fused 计算 `grad_x / grad_w1 / grad_w2`，避免 Python loop + 大量小 kernel。

### 4. 判据

每个方向都用同一套判据收敛：

- no-metrics run：`avg_step_sec`、MFU、峰值显存、loss。
- nsys stats：`cudaLaunchKernel*` 调用数、top elementwise/copy kernel 占比、H2D count/MB。
- GPU metrics：`GPU Active`、`SM Issue`、`Tensor Active`、DRAM read/write。
- block-swap JSONL：`h2d_ms`、`enqueue_ms`、`wait_ms`、`transfer_ms`。

### 5. 已落地变更

- `library/runtime/offloading.py`：非 profiling block-swap wait path 改为 CUDA
  `current_stream.wait_event(end_event)`，不再为了等待 H2D 完成而 host synchronize。
- profiling path 仍保留 `Event.synchronize()`，保证 `block_swap_profile_jsonl` 里的
  `h2d_ms / transfer_ms` 可读。
- `tests/test_block_swapping.py` 增加 fake CUDA stream/event 单测，防止生产路径退回
  host sync。
- `library/runtime/offloading.py`：新增 block module map 缓存、swap plan cache、copy
  stream 复用、event pool 复用，并在 `prepare_block_devices_before_forward()` 预热主路径
  swap plan。
- `tests/test_block_swapping.py` 补充了 swap plan cache、copy stream 复用、event pool
  复用单测，防止方案 4 退化。
- `scripts/experiments/blockswap_copy_plan_microbench.py`：新增真实 block 形状的 copy-plan
  微基准，对比逐 tensor `copy_`、`torch._foreach_copy_` 和 CPU slab 整体 H2D，作为
  方案 4 第二阶段是否值得接入 runtime 的先验证据。
- `scripts/experiments/blockswap_restore_path_microbench.py`：新增更贴近 offloader
  cached CUDA restore 路径的组件级微基准，把 `record_stream()`、weight 绑定和 host issue
  成本一起纳入，对比逐 tensor `copy_`、`_foreach_copy_` 与 slab restore 上限。
- `library/runtime/offloading.py`：在 cached CUDA block-swap restore 路径里新增
  `_try_foreach_h2d_copy()`，当同一次 swap 的目标 tensor 都是 CUDA、源 tensor 都是 CPU，
  且 dtype/numel 相容时，优先把逐 tensor `copy_` 合并成一次
  `torch._foreach_copy_(..., non_blocking=True)`；不满足条件或运行时报错时自动回退旧路径。
- `tests/test_block_swapping.py`：补充 `_foreach_copy_` 命中与 fallback 两个单测，覆盖
  “多 weight 命中 foreach” 和 “foreach 失败后仍逐 tensor copy” 两条路径。
- `library/runtime/offloading.py`：新增 `_SwapSlabPlan` / `_get_swap_slab_plan()` 元数据层，
  先把 slab 化需要的 offset / numel / shape / dtype 真相缓存起来，但默认 restore 路径暂不切换。
- `tests/test_block_swapping.py`：补充 slab plan offset 与总 numel 单测，确保后续若接 slot slab
  restore，不会在布局顺序上漂移。
- `library/runtime/offloading.py`：新增实验性 `restore_mode="slab"` 分支；默认仍是
  `foreach`，但设置环境变量 `ANIMA_BLOCK_SWAP_RESTORE_MODE=slab` 时，cached CUDA restore
  会优先走 GPU slab restore。
- `tests/test_block_swapping.py`：补充 slab restore 单测，验证 restore 路径会退化成一次 GPU slab
  `copy_`，且多个 weight 最终共享同一块 slab storage。

### 6. 方案 4 第二阶段：`torch._foreach_copy_` 小 H2D 合并

这一步只做一件事：在不增加显存峰值的前提下，减少 cached CUDA restore 路径里 host 侧
多次 `Tensor.copy_()` 的 Python / dispatcher 入队次数。

实现位置：

- `library/runtime/offloading.py:106`：`_try_foreach_h2d_copy()`
- `library/runtime/offloading.py:706`：`_swap_weight_devices_cached_cuda()`
- `tests/test_block_swapping.py:449`
- `tests/test_block_swapping.py:549`

验证命令：

```bash
timeout 60 .venv/bin/python -m py_compile library/runtime/offloading.py tests/test_block_swapping.py
timeout 60 .venv/bin/python -m pytest tests/test_block_swapping.py
timeout 60 .venv/bin/python -m pytest tests/test_compile_checkpoint_block_swap_hot.py -k "block_swap"
```

验证结果：

- `tests/test_block_swapping.py`：`39 passed`
- `tests/test_compile_checkpoint_block_swap_hot.py -k "block_swap"`：`12 passed`

组件层证据先看 block-swap JSONL 聚合均值：

| 口径 | wait_ms avg | h2d_ms avg | transfer_ms avg | enqueue_ms avg | submit_lag_ms avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2026-06-29` 原始基线 | `0.5012` | `18.2651` | `18.6990` | `12.0959` | `369.0220` |
| `2026-06-30` plan/event 复用 | `0.2574` | `16.8840` | `17.0662` | `9.0066` | `305.1385` |
| `2026-07-01` + `_foreach_copy_` | `0.2406` | `14.4137` | `14.5251` | `1.7429` | `343.5494` |

对 `2026-06-30` 的增量：

- `wait_ms` 再降约 `6.6%`
- `h2d_ms` 再降约 `14.6%`
- `transfer_ms` 再降约 `14.9%`
- `enqueue_ms` 再降约 `80.6%`
- 但 `submit_lag_ms` 反而回升约 `12.6%`

这说明 `_foreach_copy_` 至少在组件层确实压掉了 restore 路径的 host 侧 enqueue 成本，
而且 H2D event timing 也随之下降；它不是“完全没效果”的空改动。

但同口径热跑 `summary.json` 不能直接拿来判赢：

| 口径 | avg_step_sec | achieved_tflops | MFU | peak_reserved_gb |
| --- | ---: | ---: | ---: | ---: |
| `2026-06-29` 原始基线 | `6.23675` | `8.4580` | `7.10%` | `2.1484` |
| `2026-06-30` plan/event 复用 | `6.22025` | `8.4804` | `7.12%` | `2.1484` |
| `2026-07-01` + `_foreach_copy_` | `6.40800` | `8.2320` | `6.91%` | `2.1621` |

`2026-07-01` 这次热跑开始前，3080 已有外部进程占用：

- `nvidia-smi` 记录到 `memory_used=2843 MiB`、`utilization_gpu=30%`
- 同时存在外部进程
  `/home/scv/miniconda3/envs/tran/bin/python packaging/launch.py --requirements requirements_gpu.txt --ui qt`
  持续占用约 `2910 MiB`，`nvidia-smi pmon` 显示约 `14%` SM

因此这次 `avg_step_sec=6.408` 更像是共享 GPU 背景下的脏口径，不适合用来否定
`_foreach_copy_` 这条线本身。当前更稳妥的结论是：

1. “纯减少小 H2D 次数”在离线 DMA 微基准上仍然价值有限。
2. 但把逐 tensor `copy_` 收成 `_foreach_copy_` 后，真实训练里的 block-swap restore
   组件指标确实继续改善，尤其是 `enqueue_ms`。
3. 它是否能稳定转化成总步时收益，还需要一轮 **空闲 3080** 上的同口径复跑；在共享卡上，
   `summary.json` 不能作为最终胜负依据。

为避免只剩“整训练热跑被共享 GPU 污染”的争论，`2026-07-01` 还补了一份更贴近
offloader restore 路径的组件级微基准：

- `/tmp/anima-blockswap-copy-plan/restore_path_block0_30.json`

它直接用 preview3 `net.blocks.0` 的真实 20 个 frozen weight（约 `132.0MiB`），比较：

- `restore_loop`：逐 tensor `copy_`
- `restore_foreach`：`torch._foreach_copy_`
- `restore_slab`：假设 block slot 的 GPU weight storage 和 CPU master 都已 slab 化，
  每次 restore 只做一次 `gpu_slab.copy_(cpu_slab, non_blocking=True)`

而且把 `module_to_cpu.weight.data = source_master`、`record_stream(stream)`、收集 bindings、
以及 `module_to_cuda.weight.data = cuda_data_view` 一起纳入计时。结果如下：

| 指标 | restore_loop | restore_foreach | restore_slab |
| --- | ---: | ---: | ---: |
| host_issue_ms avg | `0.4235` | `0.3697` | `0.1478` |
| ready_ms avg | `14.0875` | `14.0942` | `13.6078` |
| gpu_copy_ms avg | `14.0122` | `14.0219` | `13.5370` |

这和前面的 copy-plan 微基准是相互印证的：

- `_foreach_copy_` **不会**显著改变纯 DMA / GPU ready 时间；
- 它的主要收益点就是压 host 侧 restore issue 开销；
- 真正想让 `ready_ms / gpu_copy_ms` 再往下走，只有 slab 化这种“把多次小 H2D 真压成极少数大 H2D”
  的路线才有明确上限收益。

按这组数据估算：

- `restore_foreach` 相比 `restore_loop`：主要只省 host issue，`ready_ms` 基本不动；
- `restore_slab` 相比 `restore_foreach`：`host_issue_ms` 还能再降约 `60.0%`，
  `ready_ms` 还能再降约 `3.4%`，`gpu_copy_ms` 还能再降约 `3.5%`。

这给了“减少小 H2D 次数”这一半更明确的收口判断：

1. `_foreach_copy_` 已经把“host 侧多次 `copy_` 调度”这层收益吃得差不多了。
2. 如果还想继续压真实 H2D 次数，就不能再靠 `foreach` 或 Python 侧收集；需要让 block slot
   的 GPU weight storage 具备 slab 化前提。
3. 这条路不是空想，但已经从“调度优化”进入“存储布局优化”层级，复杂度明显更高。

另外，本机批量 memcpy API 也验证过不可走：

- PyTorch `torch.cuda.cudart()` / `_cudart` 不暴露 `cudaMemcpyBatchAsync`
- 本机 `/lib/x86_64-linux-gnu/libcudart.so.12` 只查得到 `cudaMemcpyAsync`，没有
  `cudaMemcpyBatchAsync` / `cudaMemcpy3DBatchAsync`

因此在当前软件栈下，“不增加显存峰值、又想让 GPU 侧 H2D 次数显著下降”的现实主线只剩：

- 继续沿 `_foreach_copy_` 吃 host issue 收益
- 或者未来做 block slot slab 化，把 restore 路径改成极少数大 H2D

为避免这条路停在“想法”层，当前代码里已经把 slab 化所需的布局元数据真相层先接好了：

- `library/runtime/offloading.py` 现在会为每个 `(block_idx_to_cpu, block_idx_to_cuda)` 缓存
  `_SwapSlabPlan`
- plan 内记录每个 swappable weight 的 `offset / numel / shape / dtype`
- 对应测试已覆盖 offset 累加和总 `slab_numel`

这意味着如果后续要把 cached CUDA restore 真正切到 slot slab 路径，最难出错的“layout 真相”
已经不是空白状态；接下来主要剩：

1. 为 active slot 准备可复用的 GPU slab storage
2. 把 swappable weight 的 `weight.data` 绑定到 slab view
3. 用一次或极少数大 H2D 替换当前多 tensor restore

其中第 1-3 步现在已经有一个**实验性可跑通分支**：

- 默认：`restore_mode="foreach"`
- 实验开关：`ANIMA_BLOCK_SWAP_RESTORE_MODE=slab`

开启后，cached CUDA restore 会：

1. 复用目标 block 的 CPU master slab
2. 为 `(block_idx_to_cpu, block_idx_to_cuda)` slot 缓存一块 GPU slab
3. 用一次 `gpu_slab.copy_(cpu_slab, non_blocking=True)` 恢复整个 swappable frozen set
4. 再把 `module_to_cuda.weight.data` 绑定到各自 slab view

这个分支当前故意保持“实验开关 + 非默认”状态，原因有二：

- 它已经从调度优化进入 storage layout 优化，入侵性明显更高
- 还没在干净训练热跑上拿到总步时与稳定性结论

但从目标对齐角度看，它已经是当前仓库里**最贴近“真正减少小 H2D 次数”** 的实现路径。

`2026-07-01` 进一步在干净 3080 口径下重跑了真实训练热身：

- 命令：`ANIMA_BLOCK_SWAP_RESTORE_MODE=slab .venv/bin/python -m bench.mfu.run_training --suite baseline --steps 8 ...`
- 产物：`output/bench/hot_profile/hot_profile_20260701_lokr_blockswap_slab_v2/`

这次不是脏卡，也不是只过单测，而是完整跑通了 8-step baseline：

| 口径 | avg_step_sec | achieved_tflops | MFU | peak_reserved_gb |
| --- | ---: | ---: | ---: | ---: |
| `2026-06-29` 原始基线 | `6.23675` | `8.4580` | `7.10%` | `2.1484` |
| `2026-06-30` plan/event 复用 | `6.22025` | `8.4804` | `7.12%` | `2.1484` |
| `2026-07-01` `_foreach_copy_` | `6.40800` | `8.2320` | `6.91%` | `2.1621` |
| `2026-07-01` `slab_v2` | `5.96675` | `8.8407` | `7.42%` | `2.1719` |

按步时算：

- 相比 `2026-06-29` 原始基线：`+4.33%`
- 相比 `2026-06-30` plan/event 复用：`+4.08%`
- 相比 `_foreach_copy_` 版本：`+6.89%`

对应 block-swap JSONL 聚合均值：

| 口径 | wait_ms avg | h2d_ms avg | transfer_ms avg | enqueue_ms avg | submit_lag_ms avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2026-06-29` 原始基线 | `0.5012` | `18.2651` | `18.6990` | `12.0959` | `369.0220` |
| `2026-06-30` plan/event 复用 | `0.2574` | `16.8840` | `17.0662` | `9.0066` | `305.1385` |
| `2026-07-01` `_foreach_copy_` | `0.2406` | `14.4137` | `14.5251` | `1.7429` | `343.5494` |
| `2026-07-01` `slab_v2` | `0.3104` | `13.7307` | `13.9149` | `0.9393` | `299.1629` |

这组结果说明：

1. `_foreach_copy_` 把 host issue 收益吃到了，但对真实步时帮助有限。
2. 真把 restore 路径改成 slab 化后，`enqueue_ms` 还能再砍近一半，`h2d_ms / transfer_ms`
   继续下降，且这次已经真实转化成了训练步时收益。
3. `peak_reserved_gb` 只从 `2.1484` 上到 `2.1719`，增量约 `23.5 MiB`，不是“不可接受的显存峰值暴涨”。

另外，这条实验分支第一次真实热跑曾经因为 GPU slab cache 键选错而 OOM：

- 旧实现按 `(block_idx_to_cpu, block_idx_to_cuda)` 缓存 GPU slab
- 这会让不同 swap pair 各留一份 slab，白白吃掉几 GiB VRAM

修复后改成按 **physical slot** 复用 GPU slab：

- 对 `blocks_to_swap=26`、`num_blocks=28`，只保留 `num_blocks - blocks_to_swap = 2` 个 slot slab
- 这正是 `slab_v2` 能在真实训练里跑通且显存增量很小的关键

为了避免把 `slab_v2` 和一份脏卡 `foreach` 口径混比，随后又补了一轮同环境 clean 对照：

- `output/bench/hot_profile/hot_profile_20260701_lokr_blockswap_foreach_v2/`

这轮 clean `foreach_v2` 结果：

| 口径 | avg_step_sec | achieved_tflops | MFU | peak_reserved_gb |
| --- | ---: | ---: | ---: | ---: |
| `foreach_v2` | `6.29975` | `8.3734` | `7.03%` | `2.1484` |
| `slab_v2` | `5.96675` | `8.8407` | `7.42%` | `2.1719` |

这才是当前最可信的一组 apples-to-apples 结论：

- `slab_v2` 相比 clean `foreach_v2`：`avg_step_sec` 再降约 **`5.29%`**
- `achieved_tflops`：`8.3734 -> 8.8407`
- `mfu`：`7.03% -> 7.42%`
- `peak_reserved_gb` 只增加约 **`24 MiB`**

对应组件层指标也继续同方向改善：

| 口径 | h2d_ms avg | transfer_ms avg | enqueue_ms avg | submit_lag_ms avg |
| --- | ---: | ---: | ---: | ---: |
| `foreach_v2` | `13.9515` | `14.1250` | `1.7175` | `301.3730` |
| `slab_v2` | `13.7307` | `13.9149` | `0.9393` | `299.1629` |

所以在当前仓库与当前硬件口径下，可以把“减少小 H2D 次数”这半段的结论更新为：

1. `_foreach_copy_` 已经是这条线的低侵入收尾，但它主要只优化 host issue。
2. 真想把收益传导到训练步时，需要进一步走 **slot slab restore**。
3. 按 physical slot 复用 GPU slab 后，这条实验路径已经能在真实训练里稳定跑通，并带来
   约 **5.3%** 的步时收益，而显存代价目前只看到约 **24 MiB** 的 reserved 增量。

## 热测中修复

热测发现：

```text
configs/gui-methods/mfu_rokkotsu_cached.toml
configs/gui-methods/mfu_rokkotsu_plain_lora_ckpt.toml
```

都写了 `max_train_epochs = 6`。训练 bootstrap 中 `max_train_epochs` 会覆盖
`--max_train_steps`，导致：

```bash
.venv/bin/python -m bench.mfu.run_training --suite plain_lora --steps 20
```

实际仍计划跑 `1710` step。已移除这两个 MFU benchmark 配置里的
`max_train_epochs`，让 `bench.mfu.run_training --steps` 重新成为运行长度真相。

修复验证命令：

```bash
timeout -s INT 240 .venv/bin/python -m bench.mfu.run_training \
  --suite plain_lora \
  --steps 2 \
  --metric-step-window off \
  --peak-tflops 119.07072 \
  --allow-low-vram
```

验证结果：`plain_lora_ckpt_s42_2step OK elapsed=60.4s avg_step=1.212s tflops=43.523501 mfu=0.365526`。

## 解读

1. Plain LoRA 热窗口与此前归档结果一致，说明主训练链在 3080 上可以稳定跑到约
   `36% MFU`。
2. LoKr + 26 block swap 的显存占用非常低，但每 step 的 H2D 传输累计接近 `5.6s`，
   与 `6.24s/step` 的步时同量级；优化优先级应先看 block swap 策略，而不是 Rust。
3. nsys wrapper 参数兼容、`.qdstrm` 后处理和 `NSYS_PYTHON_SAMPLING=0` 开关已修复；
   当前开发机已能用新版 Nsight Systems 产出干净 LoKr `.nsys-rep`、`.sqlite` 和 stats 表。
4. clean nsys 证据把 LoKr 慢点拆成两层：block swap 带来约 `21.6GB` H2D / profile window，
   同时 LoKr 路径产生大量 elementwise/copy/launch 密集小 kernel；优化不该只盯单个大 GEMM。
5. GPU metrics 不影响基础 timeline、NVTX、CUDA API、kernel 和 memcpy 统计。本机已写入
   `/etc/modprobe.d/nvidia-perf-counter.conf` 并更新 initramfs；重启后运行态已变成
   `RmProfilingAdminOnly: 0`。
6. GPU metrics smoke 通过：
   `output/nsys/gpu_metrics_smoke_after_reboot_device1_pci.nsys-rep` /
   `.sqlite` 已生成，`GPU_METRICS` 表有 `124713` 行。由于本机还有一张不支持 metrics 的
   GTX 960，显式 `--gpu-metrics-devices=1 --gpu-metrics-set=ga10x` 最稳；带
   `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` 时，`cuda-visible` 也可正常工作。
7. LoKr GPU metrics rerun 显示 `GPU Active` 均值约 `50%`、`SM Issue` 均值约 `9%`、
   `Tensor Active` 均值约 `5%`，进一步支持“不是纯算力饱和，而是 offload + 小 kernel /
   launch 密度 + 同步排队”这个方向。
8. MFU runner 修复后，`plain_lora --steps 2` 已能自然截断并生成干净 summary.json；
   后续可以重新跑 `--steps 20/80` 得到正式对比表，不需要再手动中断。

## 下一步

- 在 **空闲 3080** 上重跑
  `output/bench/hot_profile/hot_profile_20260701_lokr_blockswap_foreach` 同口径 8-step 热测，
  要求起跑前 `nvidia-smi` 中 3080 没有额外 compute 进程、`memory.used` 接近桌面常驻值；
  只有这轮复跑才可用于判断 `_foreach_copy_` 是否把组件层收益转化成总步时收益。
- 重新跑修复后的 `bench.mfu.run_training --suite plain_lora --steps 20` 和
  `--suite baseline --steps 20`，生成正式干净 `summary.json` 对照。
- 对 LoKr 做 `blocks_to_swap` 小矩阵：例如 `0/8/16/26`，观察 MFU 与峰值显存曲线。
- 对 LoKr 小 kernel 密度做组件级分解：LoKr projection / custom autograd / block swap
  H2D / checkpoint recompute 分别关开，确认 launch 密度来自哪一层。
- 对 LoKr 的 GPU metrics sqlite 做 NVTX-window 过滤，区分 forward/backward 内的
  `GPU Active`、`SM Issue`、`Tensor Active` 和 DRAM bandwidth。
