状态：Nsight Systems 完成；Nsight Compute 已安装但硬件计数器被驱动权限阻断
日期：2026-08-12
原始摘要：`krea2_90hx_nsight_tiles.json`
探针：`scripts/krea2/probe_90hx_nsight_tiles.py`

# Krea-2 CMP 90HX Nsight tile/cache 验证

## 目标与边界

本阶段验证“RTX 30 系因输入不整除内核 tile，导致 cache 浪费和巨大性能
空泡”的强假设。使用本机空闲 `NVIDIA CMP 90HX` (`SM 8.6`, 50 SM, 9.65 GiB)，
不加载完整 DiT，不修改功耗、时钟、风扇或驱动配置。

实验分三层：

1. CUDA event 计时 Krea-2 真实 Linear/attention 形状。
2. Nsight Systems + NVTX 确认底层 kernel、grid/block 和内部 tile。
3. Nsight Compute 尝试读取 L2/DRAM/occupancy/warp-stall 计数器。

90HX 长时间空闲时会从 `P8` 冷态启动。首轮未充分预热的前三个 case 仅
`8-10 TFLOP/s`，随后恢复到 `73-77 TFLOP/s`。正式探针因此增加 3 秒代表性
GEMM 预热，Nsight capture 只从预热后开始。

## GEMM 整除与尾 tile

固定 `N=K=6144` BF16 forward，对 `M` 做扫描。两轮独立结果中，两组最有
判别力的对照为：

| 对照 | 第 1 轮 | 第 2 轮 | 尾块相对变化 |
| --- | ---: | ---: | ---: |
| `M=4096` 整除 vs `M=4107` 仅 11/128 有效尾 tile | 77.18 vs 76.25 TFLOP/s | 71.62 vs 71.82 TFLOP/s | -1.20% / +0.28% |
| `M=4608` 整除 vs `M=4607` 127/128 尾 tile | 80.68 vs 79.92 TFLOP/s | 70.85 vs 72.21 TFLOP/s | -0.94% / +1.92% |

Nsight Systems 显示这些形状全部走同一个 Ampere CUTLASS BF16 Tensor Core kernel：

```text
cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8
block = 256x1x1
M tile = 128 rows
```

grid 随 `ceil(M/128)` 增长：`4096 -> 256x3`、`4107 -> 264x3`、
`4480 -> 280x3`、`4544/4607/4608 -> 288x3`、`4672/4736 -> 296x3`、
`4864 -> 304x3`。`4107` 确实多发射了一组大部分无效的尾 tile，但没有
出现吞吐断崖；`4607/4608` 的相对快慢甚至在两轮中反转。

因此，“当前 Krea-2 输入因不整除 GEMM tile 而巨慢”判定为 **REJECT**。
这不表示 cache 层级对 GA102 没有影响，只是排除了“尾 tile 整除性是主因”。

## NF4 和 attention

Linear 训练口径是 frozen weight 的 forward + input-gradient backward：

| 形状 | BF16 | NF4 | NF4 额外开销 |
| --- | ---: | ---: | ---: |
| `M=4608,N=6144,K=6144` | 9.657 ms | 10.102 ms | +4.60% |
| `M=4608,N=16384,K=6144` | 30.172 ms | 31.229 ms | +3.50% |

Nsight 在 NF4 NVTX range 内确认了 `kDequantizeBlockwise` + 同类 cuBLASLt/CUTLASS
GEMM；`blocksize=64` 是 NF4 量化分组，不是 GEMM tile。该开销低于既有 3080
阶段 1 的 `13-25%` 代表性 Linear 开销，不支持 NF4 分块导致数量级减速。

Attention 使用当前 Krea 契约 `B=1,Hq=48,Hkv=12,L=4608,D=128`，forward +
backward：

| 路径 | 有效 token | 中位时间 | 相对 cuDNN |
| --- | ---: | ---: | ---: |
| cuDNN dense SDPA | 4107 | 43.631 ms | 基线 |
| FlashAttention varlen | 4107 | 27.733 ms | -36.44% |
| FlashAttention full | 4608 | 32.981 ms | -24.41% |

Flash varlen 中有效 token 打包相对 full-4608 Flash 再快 15.91%。Nsight 观测到：

- cuDNN forward/backward 主 tile：`64x64x128`，block 256。
- FA2 forward：`128x128`，block 128。
- FA2 backward 主核：`128x64`，block 256。

FA2 确实根据 head dim/长度使用内部 tile，但当前 `D=128`和 Krea token 形状
没有观测到尾块引发的异常空泡。

## Nsight Compute 权限边界

用户目录已安装 Nsight Compute CLI 2022.4.1，`~/.local/bin/ncu` 可用。
目标 NVTX 过滤 `bf16_fwd_proj_m4608/` 已正确命中，但驱动拒绝计数器：

```text
RmProfilingAdminOnly: 1
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

因此本阶段不宣称已证明 L2 hit rate、DRAM 带宽、occupancy 或 warp stall 的具体
机制。权限开放后的精确命令是：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
ncu --target-processes all \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section MemoryWorkloadAnalysis --section SchedulerStats \
  --nvtx --nvtx-include 'bf16_fwd_proj_m4608/' --launch-count 1 \
  --export output/nsight/krea2_90hx_m4608_ncu --force-overwrite \
  .venv/bin/python scripts/krea2/probe_90hx_nsight_tiles.py \
  --suite gemm-sweep --case bf16_fwd_proj_m4608 \
  --warmup 2 --repeats 1 --preheat-seconds 2
```

非 root 开放计数器的驱动配置与重启流程已记录在
`docs/optimizations/training_profiling.md`。本轮不擅自重载 NVIDIA 模块或重启主机。

## 阶段结论

1. **REJECT**：当前 Krea-2 的 GEMM 尾 tile 整除性不是 30 系巨慢主因。
2. **CONFIRMED**：项目没有显式 tile 参数，但 cuBLASLt/cuDNN/FA2 底层已按
   GA102 形状选择 `128/256` 级 tile，并用 predicated tail 处理不整除输入。
3. **CONFIRMED**：FA2 varlen 在 90HX attention 微基准中快 36.44%，与既有
   3080 `-34.3%` 结果一致。
4. **OPEN**：3080/90HX 相对 PG199 的巨大 GEMM 差距究竟有多少来自 L2/DRAM、
   tensor-pipe 利用率和 scoreboard stall，必须在开放 NCU 计数器后再定论。
