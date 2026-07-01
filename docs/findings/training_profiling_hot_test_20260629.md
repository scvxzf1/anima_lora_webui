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
