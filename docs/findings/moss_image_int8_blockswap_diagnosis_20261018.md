# MOSS-Image 块交换传输诊断：int8 的 0.94 s 与 3.5 GiB 归属

状态：实验 / 诊断结论
适用版本：当前 main（`dd0ea5fc07a95ff22e399deb14e8595e1f56f26e`）
入口命令：见下文「复现」（`diag_int8_restore.py` 微基准 + `execute.py` 训练臂）
相关代码：`library/runtime/offloading.py`、`library/runtime/block_swap_masters.py`、
`library/runtime/int8_linear.py`、`library/models/moss_image/block_swap.py`、
`library/training/moss_compat.py`
产物：`output/runs/moss-image-abc-precision-20261017-232000/`（本地产物，不随代码提交）

## 结论摘要

在同一份配置（fp16 常驻、`blocks_to_swap=20`、单张 256×256 / 22-token 缓存图、rank32、seed42、
10 步）下，只把 `block_swap_transfer_dtype` 从 `bf16` 换成 `int8`：

| 指标 | bf16 传输 | int8 传输 | 差值 |
| --- | ---: | ---: | ---: |
| 稳态步时（步 2–10 均值） | 2.9312 s | 2.3986 s | −0.533 s（1.222×） |
| 每步 H2D 字节（无 D2H） | 9.06 GB | 4.53 GB | −50% |
| 纯链路下限（3.09 GB/s） | 2.93 s | 1.47 s | — |
| 峰值 reserved | 5.93 GiB | 9.16 GiB | **+3.23 GiB** |
| 峰值 allocated | 5.84 GiB | 6.28 GiB | +0.35 GiB |
| 整卡观测峰值 | 6430 MiB | 9788 MiB | 占满 9.56 GiB |
| 更新期功耗峰 | 177.8 W | 170.7 W | — |

后续阶段又实现了融合反量化（`int8_restore_mode=fused`）、staging 环形与双流解耦，
并把 int8 与 `blocks_to_swap` 组合起来：**swap=7 时 0.8793 s/步（相对 bf16/swap20 3.33×）**、
整卡 9512 MiB；swap=20 单点最好为 1.9718 s（1.48×）。详见 3.5~3.7 节。

三个数字需要分开记：

1. **int8 实际只拿回 0.53 s，不是理论上的 1.46 s。** 步时 2.3986 s − 纯 H2D 下限 1.47 s =
   **0.93 s** 的非传输开销；其中反量化本身（隔离实测）只占 **0.16 s**，剩下 **0.77 s** 出现在
   「反量化 + 真实计算流 + 真实分配器」三者共存时，隔离环境复现不出来。
2. **+3.23 GiB 不是泄漏、也不是逐步增长。** 逐步探针显示 reserved 在第 0 步的首次
   forward/backward 内就从 5.64 涨到 9.17 GiB，此后 10 步**一直钉在 9.17 GiB 不动**；
   allocated 只在 5.63–5.76 GiB 之间波动。多出来的 3.4–3.5 GiB 是「缓存但空闲」的段。
3. **bf16 传输路径的开销精确为零。** 9.06 GB ÷ 3.09 GB/s = 2.93 s，实测 2.9312 s，
   残差 0.001 s；隔离微基准同样是 12.22 ms/张量恢复，等于纯 H2D 时间。bf16 恢复**不做任何
   GPU 侧计算、不分配任何临时张量**。

## 被测环境（先纠正两处前提）

- 卡是 **NVIDIA CMP 50HX**（TU102 挖矿卡，VBIOS 90.02.60.00.1A），**不是 RTX 3080**：
  56 个 SM、cc 7.5、10,240 MiB（9.56 GiB 可用）、**功耗上限 225 W**（`power.max_limit` 也是 225 W），
  PCIe 代数上限为 1（Gen1×16，约 3.1 GB/s 有效）。
- 因此「10 GB 显存 → 3080 → TDP 320 W」的推断链不成立；判断是否算力见底要看本卡 225 W 上限。
- 实测功耗：fp16 常驻臂在 171–190 W，而**混合精度 bf16 计算臂达到 232 W**（超过 225 W 上限采样值）。
  也就是说这张卡能被压到功耗墙，fp16 臂的低功耗是**DMA 主导工况的必然结果**，不等于存在可消除的停顿。
- 恢复次数：**240 次张量级恢复/步**（20 个换出 block × 6 个线性层 × forward+backward 两趟）成立，
  对应 **40 次 block 级恢复/步**。6 个线性层是 `proj.q/k/v/o/ffn_up_gate/ffn_down`，
  每 block 共 113,246,208 个参数。

## 代码级机制：三条恢复路径

三者的差别不在「传输多少」，而在**恢复过程中有没有 GPU 侧计算与临时分配**。

| 路径 | 每张量发生的事 | GPU 计算 | 临时分配 |
| --- | --- | --- | --- |
| bf16 传输 | 1 次 H2D + `copy_` 进常驻 slot | 无 | 无 |
| int8 + `copy`（默认） | H2D(int8) → `to(float32)` → `*scale` → `to(fp16)` → `copy_` 进 slot | 3 个 eager kernel + 1 次拷贝 | **3 个临时张量**（fp32×2、fp16×1） |
| int8 + `reuse_storage` | H2D(int8) → `to(fp16)` → `torch.mul(out=slot)` | 2 个 eager kernel | 1 个临时张量（fp16） |

代码依据：

- `library/runtime/block_swap_masters.py:91` `Int8BlockSwapCpuMaster.to_tensor()`：
  `quantized.to(device)` + `scale.to(device)` → `rows.to(torch.float32)` →
  `rows * scale[:, None]` → `.to(dtype=dtype)`。**没有任何融合反量化 kernel**，
  三步各产出一个新张量。
- `library/runtime/offloading.py:1550` 附近的 `copy` 分支：拿到 `restored` 后再
  `cuda_data_view.copy_(restored, non_blocking=True)`，即**第 4 次数据移动**。
- `reuse_storage` 分支走 `_restore_int8_cpu_master_into_tensor`（`block_swap_masters.py:387`），
  `chunk_rows<=0` 时单发：`torch.mul(rows, scale[:, None], out=dst_rows)` 直接写进常驻 slot。
- **反量化不在 default stream**：恢复整体在 `with torch.cuda.stream(self._get_copy_stream_for_slot(slot_id))`
  里（`offloading.py:1385-1386`），即 per-slot copy stream。它与同一流上的 H2D **串行**，
  计算流随后等待该流的 ready event —— 所以它不是「default stream 同步」，但**确实在关键路径上**，
  无法像 bf16 那样被 DMA 完全隐藏。
- 两个方向的传输：`module_to_cpu.weight.data = _parked_cpu_master_tensor(source_master)`
  只是把 CPU 模块重新指向 master，**int8 与 bf16 都没有 fp16 权重的 D2H 回传**
  （权重冻结，下次用 int8 master 重新反量化）。因此每步只有 H2D，字节数 = 2 × 20 × block 字节。

## 实测证据

### 1) 隔离微基准（`diag_int8_restore.py`，真实形状，无训练）

240 次张量级恢复 = 20 blocks × 6 linears × 2 passes：

| 路径 | 总耗时 | 每张量 | 峰值 reserved 增量 | 峰值 allocated |
| --- | ---: | ---: | ---: | ---: |
| bf16 | 2.9321 s | 12.22 ms | 0.11 GiB | 0.30 GiB |
| int8 `copy`（3 kernel + copy_） | 1.6346 s | 6.81 ms | **0.65 GiB** | 0.80 GiB |
| int8 `reuse`（分块写入 slot） | 1.6201 s | 6.75 ms | **0.00 GiB** | 0.42 GiB |

- int8 纯 H2D 上限：**2.93 GiB/s**（pinned）。
- 4.53 GB ÷ 2.93 GiB/s ≈ 1.47 s；隔离的 int8 恢复 1.63 s → **反量化+拷贝只值 0.16 s**。
- bf16 的 2.9321 s 与纯 H2D 完全一致 → bf16 路径无额外开销。

### 2) 真实训练臂（10 步，同步计时）

| 臂 | 传输 | 步时（均值/中位/最小/最大） | reserved | allocated | 整卡 |
| --- | --- | --- | ---: | ---: | ---: |
| `xfer-bf16` | bf16 | 2.9312 / 2.9258 / 2.9204 / 2.9831 | 5.93 GiB | 5.84 GiB | 6430 MiB |
| `xfer-int8` | int8 copy | 2.3986 / 2.3941 / 2.3524 / 2.4542 | 9.16 GiB | 6.28 GiB | 9788 MiB |
| `xfer-int8-reuse-clean` | int8 reuse | 2.6413 / 2.3870 / 1.9953 / 4.5692 | 9.17 GiB | 5.89 GiB | 9738 MiB |
| `xfer-int8-noexpand` | int8 copy，关 `expandable_segments` | 2.4696 / 2.4146 / 2.3989 / 2.8342 | 9.16 GiB | 6.28 GiB | 9734 MiB |

- **分配器配置对照（否定结果）**：把 `ANIMA_EXPANDABLE_SEGMENTS=0` 关掉后，步时从 2.3986 变成
  2.4696 s（**慢 3.0%**），峰值 reserved 完全不变（9.16 GiB）。因此 3.5 GiB 缓存段与 0.77 s 残差
  **不能归因于 `expandable_segments` 的段管理或碎片整理**。
- `xfer-int8` 臂的 `status=failed` 是验收脚本断言写错（它假定模型只有单一 dtype，而 int8 让模型
  同时存在 int8 与 fp16 张量），训练、保存与逐步数据均有效；该断言已修正，后续 `reuse-clean`
  与 `noexpand` 两臂均为 `completed`。
- `copy` 模式**极稳定**（极差 0.10 s）；`reuse_storage` 呈**双峰**（2.0 s 与 2.9–4.6 s 交替）。
  后者把反量化直接写进常驻 slot，引入了「slot 上一轮消费者是否结束」的复用依赖，
  会在部分步上暴露等待。**所以 reuse 不是「更快的 int8」，它只是更省临时显存、但时间更不稳。**

### 3) 分配器时间线（`memory_probe_jsonl`，int8 reuse 臂）

| 阶段 | allocated | reserved | 差值 |
| --- | ---: | ---: | ---: |
| step 0 首次 forward 前 | 5.371 GiB | 5.635 GiB | 0.26 GiB |
| step 0 首次 backward 中 | 5.452 GiB | **8.803 GiB** | 3.35 GiB |
| step 0 末 | 5.633 GiB | **9.168 GiB** | 3.54 GiB |
| step 1–9（每步三次采样） | 5.633 / 5.706 / 5.756 | **9.168 恒定** | 3.41–3.54 GiB |

reserved 在第一步内一次性定型后不再变化，也没有回落到驱动；**没有观察到逐步增长，也没有
观察到 reserved 反复涨落**（因此「频繁 cudaFree / 内存整理」缺少证据支持，见下文裁定）。

## 两个数字的归属

### 0.93 s 非传输开销

```
步时 2.3986 s
  − 纯 H2D 下限（4.53 GB / 3.09 GB/s）      1.47 s
  = 非传输开销                              0.93 s
     其中 反量化+copy_（隔离实测）           0.16 s
     剩余（仅训练时出现）                    0.77 s
```

- 0.16 s 的量级可以用 HBM 流量核对。`copy` 模式的真实流量是**每元素 23 B**
  （`int8→fp32` 读1写4、`×scale` 读4写4、`→fp16` 读4写2、`copy_` 读2写2），
  4.53e9 元素 ≈ **104 GB/步**；本机实测 HBM 拷贝带宽 **485.7 GB/s** → 约 **214 ms**，
  与隔离实测 0.16 s 同量级。**所以 0.77 s 不是反量化的算力/带宽成本**。
  （早期草稿曾按 7 B/元素 估算为 31.7 GB，该估算错误，已被 23 B/元素 与实测带宽取代。）
- 0.77 s 只在真实模型里出现（隔离复现不出 1.63 s 以上的部分），且 `copy` 模式步时极稳，
  说明它是**与常驻权重、优化器状态、autograd 分配共存时的分配器/流交互成本**，
  而不是每次恢复的固定代价。已排除的方向：反量化算力（45 ms 量级）、纯链路（已计入 1.47 s）。
  未分离的方向：分配器锁竞争、跨流事件等待（slot 复用依赖）。
  `expandable_segments` 段管理已通过开关对照排除（关闭后更慢、reserved 不变）。
  **本报告不把 0.77 s 归因为单一原因。**

### 3.4–3.5 GiB 缓存段

`copy` 模式下每次张量恢复要新建 3 个临时张量，按真实形状：

| 张量 | 元素数 | int8 staging | fp32 upcast | fp32 乘法结果 | fp16 结果 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `proj.q/k/v/o`（各） | 9,437,184 | 9 MB | 36 MB | 36 MB | 18 MB |
| `proj.ffn_up_gate` | 50,331,648 | 48 MB | 192 MB | 192 MB | 96 MB |
| `proj.ffn_down` | 25,165,824 | 24 MB | 96 MB | 96 MB | 48 MB |

单张量瞬时峰值 **528 MB**（`ffn_up_gate`），一个 block 若前后不释放可达 **约 1.2 GB**。
这些临时张量在 per-slot copy stream 上分配、用完即还给缓存，而常驻权重与激活同时占着池子，
分配器无法把它们合并回驱动，于是 reserved 比 allocated 长期高出 3.4–3.5 GiB。

对照：bf16 路径**不分配任何临时张量**，所以 reserved − allocated 只有 0.09 GiB。

这解释了「alloc 只涨 0.35 GiB、reserved 却涨 3.23 GiB」：
**allocated 是在步边界（临时量已释放）采样的活占用；reserved 是分配器保留的池子。**

## 第三阶段：队头阻塞验证与融合反量化

### 3.1 DMA 队头阻塞（0.77 s 的根因）

假设：`copy` 模式下，同一 copy stream 上「H2D → 3 个反量化 kernel → copy_」严格按序执行；
当反量化 kernel 因主计算流占满 SM 而排队时，**排在它后面的下一个张量 H2D 无法进入 DMA 引擎**，
PCIe 链路空转。

隔离验证（`diag_hol_bf16.py`）：用一条独立流跑 4096×4096 fp16 `matmul` 打满 56 个 SM，
同时跑 240 次张量恢复：

| 恢复路径 | SM 空闲 | SM 打满 | 变化 |
| --- | ---: | ---: | ---: |
| fp16 纯 DMA（`copy_`，即 bf16 臂的路径） | 2.873 s | 2.878 s | **+0.005 s** |
| int8 三 kernel（`copy`） | 1.655 s | 3.31–3.60 s | **+100%** |
| int8 融合单 kernel | 1.505 s | 2.38 s | +58% |

纯 DMA 路径对 SM 竞争**完全免疫**，带反量化 kernel 的路径随竞争线性恶化 → 队头阻塞成立。
真实训练处于部分饱和状态，因此 0.77 s 介于隔离的 0.16 s 与全饱和的 +1.65 s 之间。

### 3.2 融合反量化（新 `int8_restore_mode=fused`）

`torch.mul(int8_tensor, float32_scale, out=half_tensor)` 在 ATen TensorIterator 内
完成「int8→浮点、乘 scale、转 fp16、写入 slot」，**不产生 fp32/fp16 临时张量**。
实测：对全部 6 个线性层形状、fp16 与 bf16 两种输出，与 fp32 参考反量化**逐位相等**
（`max_abs_diff = 0.0`）。配合每 slot 预分配的静态 int8/scale staging，步循环内动态显存分配为 0。

实现（3 文件 +113 行，默认行为不变，需显式选择）：

- `library/runtime/block_swap_config.py`：`int8_restore_mode` 新增 `fused`（别名 `single_kernel`）。
- `library/runtime/block_swap_masters.py`：新增 `_restore_int8_cpu_master_fused()`。
- `library/runtime/offloading.py`：新增 `fused` 分支与 `_get_int8_fused_staging()`（每 slot 静态缓冲）。
- `tests/test_int8_fused_restore.py`：数值逐位等价、staging 零分配、形状校验、模式别名，5 项通过。

端到端（同配置 10 步）：

| 臂 | 稳态步时 | 中位 | 最小–最大 | reserved | allocated | 整卡峰值 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| bf16 传输 | 2.9312 | 2.9258 | 2.9204–2.9831 | 5.93 GiB | 5.84 GiB | 6430 MiB |
| int8 `copy` | 2.3986 | 2.3941 | 2.3524–2.4542 | 9.16 GiB | 6.28 GiB | 9788 MiB |
| int8 `reuse_storage` | 2.6413 | 2.3870 | 1.9953–4.5692 | 9.17 GiB | 5.89 GiB | 9738 MiB |
| **int8 `fused`** | **1.9740** | 1.9718 | 1.9659–1.9889 | **7.21 GiB** | 6.78 GiB | **7738 MiB** |

- 相对 bf16 **1.485×**，相对 int8 `copy` **1.215×**；步时极稳定（极差 0.023 s）。
- reserved 比 `copy` 低 **1.95 GiB**，整卡峰值低 **2050 MiB**，重新腾出余量。
- 残差从 0.93 s 降到 **0.50 s**（1.9740 − 1.47）。
- allocated 比 `copy` 高 0.50 GiB：这是每 slot 静态 staging（最大 48 MB/块 × slot 数）常驻所致，
  属于「用固定显存换零分配」，可接受。
- 逐 step loss 与 `copy` 臂最大绝对差 **7.4e-5**。该差异**已判定为运行间噪声**：
  用同一份 `fused` 代码重跑一次（`xfer-int8-fused-cfg`），两次 `fused` 之间最大差
  **6.78e-5**，与 copy-vs-fused 的 7.40e-5 同量级；且三臂第 1 步 loss 完全逐位相同
  （0.046997808）。结论：该配置下训练从第 2 步起本身不可逐位复现，**不能据此判定融合 kernel
  引入数值偏移**；算子级等价已由 3.2 的逐位测试确立。
- 配置路径已验证：`block_swap_int8_restore_mode = "fused"` 直接写进 TOML 即可生效
  （`xfer-int8-fused-cfg` 未使用任何 env 覆盖），reserved 7.21 GiB、整卡 7738 MiB、
  中位 1.9776 s，与 env 覆盖臂一致。

### 3.3 fp8_e4m3 在本卡的实测裁定

| 项目 | 实测 |
| --- | --- |
| `torch.float8_e4m3fn` 可用性 | 存在，转换不报错 |
| fp16 → fp8（编码） | **15.06 ms** / 1M 元素（软件模拟） |
| fp8 → fp16（解码） | 0.264 ms / 1M 元素（不慢） |
| int8 → fp16 | **0.175 ms** / 1M 元素（硬件 I2F） |
| e4m3 相对误差 | **2.25%** |
| int8 每行对称量化 | 约 0.39%（1/127 步长） |

结论修正：fp8 的问题**不是「解码指令缺失」**（解码很快），而是**编码在 SM75 上被软件模拟**
（比 int8 慢 86 倍）且 e4m3 只有 4 位尾数、误差 2.25% 远差于 int8。因此压缩传输的单字节格式
确实应当选 int8；`fp8_e4m3` 不再作为候选方向。


### 3.4 fused 臂多出的 +1.28 GiB 去哪了，以及 staging 策略三档实测

逐字节对账（bf16 臂 vs 每 slot 一份 staging 的 fused 臂）：

| 项 | bf16 | fused（每 slot） | 增量 |
| --- | ---: | ---: | ---: |
| 峰值 allocated | 5.84 GiB | 6.78 GiB | **+0.94 GiB** |
| 峰值 reserved | 5.93 GiB | 7.21 GiB | **+1.28 GiB** |
| reserved − allocated | 0.09 GiB | 0.44 GiB | +0.34 GiB |

- **+0.94 GiB 就是 staging 本身**：每 slot 一份，大小取该块最大张量 `proj.ffn_up_gate`
  的 50,331,648 B = 48.00 MiB，20 个换出块 × 48 MiB = **960 MiB = 0.9375 GiB**，
  与 6.78 − 5.84 = 0.94 GiB 逐字节吻合。
- **+0.34 GiB 的成因需要修正一个流行解释**：本实现首次调用即按该 slot 全部 6 层的最大
  payload 一次性分配 48 MiB，**不存在「先 9 MiB、后扩容到 48 MiB」的二次分配**，
  因此不是惰性扩容留下的僵尸段。更可能是 20 个 buffer 分别在不同 copy stream 上分配、
  块归属不同流导致的碎片；**未单独验证**。

staging 策略三档实测（同配置 10 步，`int8_restore_mode=fused`）：

| staging 策略 | staging 常驻 | allocated | reserved | 整卡峰值 | 稳态中位步时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 每 slot 一份（20 × 48 MiB，`RING=0`） | 960 MiB | 6.78 GiB | 7.21 GiB | 7738 MiB | **1.972 s** |
| 环形 2 份（`RING=2`，默认） | 96 MiB | 5.93 GiB | 6.16 GiB | 6658 MiB | 2.024 s |
| 全局一份（`RING=1`）+ event 门控 | 48 MiB | 5.89 GiB | 6.10 GiB | **6598 MiB** | 2.046 s |
| bf16 参照（无 staging） | 0 | 5.84 GiB | 5.93 GiB | 6430 MiB | 2.926 s |

读法：

- **共享 staging 能把显存压到与 bf16 几乎重合**（全局一份：allocated 仅高 0.05 GiB、
  reserved 高 0.17 GiB、整卡高 168 MiB），代价是 **+3.8% 步时**（1.972→2.046）。
- **环形 2 份没有找回速度**（2.024 与全局 2.046 同档）：只要 buffer 被多个 slot 共享，
  就必然引入「等上一使用者读完」的事件依赖，与份数多少关系不大。
- 因此这是**显存与速度的二选一**，不是可以同时拿到的优化：
  - 显存紧、准备下调 `blocks_to_swap` → `RING=1`（最省，6.10 GiB）；
  - 显存有余量、追求最快 → `RING=0`（每 slot 独立，1.972 s）。
  - 默认 `RING=2` 是折中档：比全局多 6 MiB staging，换一点分配余量。
- 即便 `RING=1`，reserved − allocated 仍有 0.21 GiB，未完全对齐 bf16 的 0.09 GiB。


### 3.5 组合实测：降 `blocks_to_swap` × int8 fused（本轮最大收益）

前四节把「单位传输成本」压到极限后，剩下最赚钱的杠杆是**减少传输总量**：每少换出 1 个 block，
每步少搬 2 × 113,246,208 B = 226.5 MB，常驻却只多 113,246,208 × 2 B = 216 MiB。

| 配置 | 每步 H2D | 稳态中位 | 相对 bf16/swap20 | allocated | reserved | 整卡峰值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 传输，swap20（原始基准） | 9.06 GB | 2.9258 s | 1.00× | 5.84 GiB | 5.93 GiB | 6430 MiB |
| int8 `copy`，swap20 | 4.53 GB | 2.3941 s | 1.22× | 6.28 GiB | 9.16 GiB | 9788 MiB |
| int8 `fused`，swap20 | 4.53 GB | 1.9718 s | 1.48× | 6.78 GiB | 7.21 GiB | 7738 MiB |
| int8 `fused`，swap12 | 2.72 GB | 1.2981 s | 2.25× | 7.62 GiB | 7.88 GiB | 8422 MiB |
| int8 `fused`，swap10 | 2.27 GB | 1.1204 s | 2.61× | 8.04 GiB | 8.31 GiB | 8858 MiB |
| int8 `fused` + `RING=0`，swap10 | 2.27 GB | 1.0818 s | 2.70× | 8.42 GiB | 8.78 GiB | 9338 MiB |
| int8 `fused`，swap9 | 2.04 GB | 1.0469 s | 2.79× | 8.25 GiB | 8.56 GiB | 9116 MiB |
| int8 `fused`，swap8 | 1.81 GB | 1.0265 s | 2.85× | 8.46 GiB | 8.75 GiB | 9314 MiB |
| **int8 `fused` + 双流，swap7** | 1.59 GB | **0.8793 s** | **3.33×** | 8.67 GiB | 8.95 GiB | 9512 MiB |
| **int8 `fused` + 双流，swap6** | 1.36 GB | **0.8656 s** | **3.38×** | 8.89 GiB | 9.16 GiB | 9732 MiB |
| bf16 传输，swap10（**未量化对照**） | 4.53 GB | 1.4810 s | 1.98× | 7.95 GiB | 8.04 GiB | 8590 MiB |

全部 10/10 次有效更新、零跳步、loss 有限、adapter 已保存。

读法与结论：

- **边际模型**：以 swap20→swap8 拟合，每 GB 边际耗时 **0.3475 s**（等效 2.88 GB/s），
  固定残差约 **0.398 s/步**。对照未量化臂在 swap10 的 4.53 GB → 1.4810 s（3.06 GB/s、截距近零）：
  int8 的边际速率略低、并叠加约 0.40 s 的固定串行成本（3.1 节的队头阻塞）。
- **同 swap 下 int8 的纯收益是 1.32×**（swap10：1.4810 → 1.1204），其余收益来自降 swap。
- **收益并未在 swap=8 进平台**：swap9→8 只省 0.020 s，但 swap8→7 突然省 **0.147 s**（1.0265 → 0.8793），
  说明 swap=8 附近存在阈值效应（换出块数少到某个量后，反量化与 GEMM 的 SM 争用显著缓解）。
- **推荐档位**：
  - **swap=7**：**0.8793 s / 3.33×**，整卡 9512 MiB（余量 276 MiB）——速度与余量的最佳平衡点。
  - swap=6：0.8656 s / 3.38×，但整卡 9732 MiB，距观测上限仅 **56 MiB**，只多赚 0.014 s，
    不建议作为默认（长训、分辨率变化或碎片都可能触顶）。
  - swap=8：1.0265 s / 2.85×，整卡 9314 MiB，最保守的加速档。
- **为什么 8→7 有跳变**：换出块从 8 降到 7 后每步 H2D 从 1.81 GB 降到 1.59 GB，
  但收益（0.147 s）远超带宽模型给出的 0.077 s/块，与 3.7 节「反量化抢 SM」的机制一致：
  SM 争用随换出块数下降而非线性缓解。
- **数值代价必须单独记账**：同 swap10 下 int8 相对未量化的逐 step loss 系统性偏高
  **0.3%~2.0%**（step1 −0.19%、step2 +2.05%、其余 +0.29%~+1.75%），远高于同码重跑的 7.4e-5 噪声。
  这是 10 个换出 block 的 int8 对称量化（每行 1/127 步长）造成的真实偏差，
  **与此前引用的 Anima ≤0.059% 不同量级**，正式启用前需要专门验收。
- 若不能接受该偏差，**未量化 + swap10** 仍有 **1.98×**，且零量化风险。

### 3.6 staging 分配器的逐份开销实测

对 3.4 节遗留的「每增一份 staging 多占多少」做隔离测量
（`diag_staging_alloc.py`，48 MiB int8 + 64 KiB scale = 一对 buffer）：

| buffers | 分配所在流 | allocated 增量 | reserved 增量 | 空隙 | 段数增量 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 独立 copy stream | 48.1 MiB | 50.0 MiB | **1.94 MiB** | 2 |
| 2 | 独立 copy stream | 96.1 MiB | 100.0 MiB | 3.9 MiB | 4 |
| 8 | 独立 copy stream | 384.5 MiB | 400.0 MiB | 15.5 MiB | 16 |
| 20 | 独立 copy stream | 961.2 MiB | 1000.0 MiB | **38.8 MiB** | 40 |
| 20 | **default stream** | 961.2 MiB | 962.0 MiB | **0.8 MiB** | 21 |

- 每份 buffer 实测只多占 **1.94 MiB**（大池 48 MiB 段 + 小池 2 MiB scale 段），
  **不是**「48 MiB 请求被抬到 60 MiB」推导出的 ~12 MiB；20 份合计 38.8 MiB。
- 因此 3.4 节里 RING=0 的 0.44 GiB 空隙中，staging 只能解释约 39 MiB，**其余未定位**。
- **把 staging 分配放在 default stream 上**可把这份空隙从 38.8 MiB 压到 0.8 MiB，方向有效但量级有限；
  默认 RING=2 只有 2 份（约 4 MiB），故本轮未采纳该修改。
- RING=2 相对 RING=0 的 0.05 s 差异**未能用事件依赖解释**：实现的 ring key 是
  `slot_id % ring`（`slot_id = block_idx % (num_blocks - blocks_to_swap)`），
  相邻 block 本来就用不同 buffer，不存在「按张量轮转导致跨 block 退化」的结构。


### 3.7 双流解耦（DMA 流 + priority=-1 反量化流）实测：否证，但拿到稳定性

假设：把 `copy_`（DMA）与 `torch.mul`（SM kernel）拆到两条流（1 条纯 DMA 流 + 1 条
`priority=-1` 反量化流，2 × 48 MiB 双缓冲 + 每缓冲 dma_ready/buffer_free 事件），
可以让 H2D 永不被前一张量的反量化阻塞，从而把 swap=20 下的 1.97 s 压到 ~1.5 s。

实现：`ANIMA_BLOCK_SWAP_INT8_DUAL_STREAM=1`（`_restore_int8_fused_dual_stream()`），
默认关闭；全 int8 块走双流，混合 dtype 块自动回退单流。

| 臂（swap=20, int8 fused） | staging 份数 | 中位 | 均值 | 极差 | allocated | reserved | 整卡 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 单流，每 slot 独立 | 20 | 1.9776 s | 2.0404 s | 1.963–2.532 s | 6.78 GiB | 7.21 GiB | 7738 MiB |
| 单流，环形 | 2 | 2.0243 s | 2.4230 s | 2.013–3.601 s | 5.93 GiB | 6.16 GiB | 6658 MiB |
| **双流，环形** | 2 | **2.0116 s** | **2.0108 s** | **2.002–2.017 s** | 5.93 GiB | 6.16 GiB | 6658 MiB |

对账口径：**1.05 GiB 的显存收益来自 staging 份数（20→2），不是双流本身**（双流与同份数单流的显存完全相同）。双流的真实收益是同等显存下中位略快 0.013 s，且极差收敛到 **0.015 s**；这份显存收益的价值在于让 3.5 节的低 swap 组合变得可行。

- **数值正确**：双流 vs 单流逐 step loss 最大差 5.76e-5，与同码重跑噪声 6.78e-5 同量级。
- **速度没有提升**（相对同份数单流中位快 0.013 s，等于噪声），1.52~1.60 s 的预期未成立。
- **实得收益是稳定性**：极差从 2 份单流的 1.59 s（含外部干扰尖峰）收到 **0.015 s**。
- **「份数收益值多少」的正确口径**：该收益来自 staging 份数而非双流，且其绝对量随 swap 数
  变化（每个换出块一份）：swap=20 时 20 份 ≈ 0.94 GiB，swap=10 时 10 份 ≈ 0.38 GiB，
  swap=6 时 6 份 ≈ 0.20 GiB。实测 swap=10 用 20 份（`RING=0`）为 9338 MiB，**仍然塞得下且更快
  3.5%**（1.0818 s vs 1.1204 s），故此处的份数开销在低 swap 下并不要命。
  真正靠它救命的是**激进档**：swap=6 用 2 份是 9732 MiB（距观测上限 9788 MiB 仅 56 MiB），
  换成每 slot 独立（6 份，多 ~200 MiB）将达约 9932 MiB，**超过卡上限而 OOM**。
  即：共享 staging 是 3.5 节 0.866 s 那一档的准入条件，而不是 swap=10 档的。

**为什么没有提升（机制）**：这 0.4~0.5 s 残差**不是队头阻塞，而是反量化 kernel 与训练
GEMM 抢 SM 的代价**。决定性证据来自 3.1 节的隔离实验：

- SM 空闲时 fused 240 次恢复 = **1.505 s**，而纯 H2D 下限为 4.53 GB / 3.09 GB/s ≈ 1.47 s
  → 反量化此时**完全被 DMA 掩盖**，成本≈0；
- SM 打满时同一路径 = **2.38 s**（+0.88 s）→ 这 0.88 s 全部来自 SM 争用。

双流能解开「DMA 与 SM 串行」，但解不开「SM 已被训练算满」。真实训练处于部分饱和，
因此落在 1.47 s（下限）与 2.38 s（全饱和）之间，实测 +0.5 s，与 0.88 s × 部分占用率一致。

**推论**：要动这 0.5 s，只能减少反量化的元素数或每元素成本（`fused` 已是
1 B 读 + 2 B 写 = 3 B/元素的下限），即**降低 `blocks_to_swap`**——这正是 3.5 节
swap=10 的收益来源。流结构不是杠杆。

### 3.8 其余三条提案的量级裁定

| 提案 | 实测/推算量级 | 裁定 |
| --- | --- | --- |
| `scale` 常驻 GPU（20 块共 2.42 MiB，省 240 次小包 H2D） | 小包 H2D 启动开销毫秒级；scale 总量 2.42 MiB（行数核算正确） | 收益远小于 0.4 s，未采纳 |
| staging 改 default stream 预分配 | 隔离实测空隙 38.8 → 0.8 MiB；但 RING=2 仅 2 份 ≈ 4 MiB | 方向正确、量级有限，未采纳 |
| 22 个常驻块改「显存内 INT8 驻留」 | 常驻权重 4.64 GiB → 2.32 GiB，省 2.1 GiB；全 42 块 int8 常驻 ≈ 4.5 GiB，**可支撑 swap=0（零 PCIe）** | **最有价值的下一步方向**，但需改 offloader 常驻块表示，未实现 |


### 3.9 两个探针：算力硬底 0.804 s 与 loss 偏差归因

在做「常驻块 INT8 驻留 → swap=0」这类架构改造前，先用两个零架构成本的探针把天花板测死。

#### 探针 1：swap=0 的真实算力硬底 = **0.8044 s/步**

方法（`ABC_TIE_TAIL_BLOCKS=12`）：模型加载到 host，只把前 30 个 block 与全部非 layer 模块搬上卡，
再把 Block 30..41 的 120 个参数 `.data` 绑定到 Block 0..11 的 GPU 张量（显存共享 / weight tying）。
于是 `blocks_to_swap=0`、**零 PCIe 传输、零反量化**，但 GPU 仍完整跑 42 个 block 的前向、重计算、
反向与 LoRA 更新。

| 指标 | 实测 |
| --- | ---: |
| 稳态中位步时 | **0.8044 s** |
| 均值 / 最小 / 最大 | 0.8077 / 0.7955 / 0.8243 s |
| 峰值 allocated / reserved | 7.53 / 7.61 GiB |
| 整卡峰值 / 功耗 | 8107 MiB / 162.7 W |

该臂 loss 为 NaN（权重共享导致数值发散），因此**只作计时探针，数值无效**；kernel 形状与数量不变，
计时有效。被复用的权重可能让 L2 命中率略优于真实 swap=0，故 0.8044 s 应视为**偏乐观的下界**。

#### 修正一个错误分解

早前把 swap=8 的 1.0265 s 拆成「0.4 s 反量化 + 0.63 s 传输」是错的：42 个 block 的
GEMM/反向/优化器不可能为 0。正确分解（`fused` 每元素 3 B，1.81e9 元素仅 5.43 GB HBM 流量 ≈ 11 ms）：

| 配置 | 每步 H2D | 纯传输 | 实测步时 | 未掩盖部分 | 掩盖率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| swap=20 | 4.53 GB | 1.47 s | 1.9718 s | 1.167 s | 21% |
| swap=12 | 2.72 GB | 0.88 s | 1.2981 s | 0.494 s | 44% |
| swap=10 | 2.27 GB | 0.73 s | 1.1204 s | 0.316 s | 57% |
| swap=8 | 1.81 GB | 0.59 s | 1.0265 s | 0.222 s | 62% |
| swap=7 | 1.59 GB | 0.51 s | 0.8793 s | 0.075 s | **85%** |
| swap=6 | 1.36 GB | 0.44 s | 0.8656 s | 0.061 s | **86%** |

（未掩盖部分 = 实测步时 − 0.8044 s 算力底。）**步时 = 算力底 + 未掩盖的传输**；
反量化本身的资源成本可忽略，低 swap 下传输已被计算掩盖 85% 以上。

#### 探针 1 的判决：**不做 swap=0 int8 驻留**

| 方案 | 步时 | 相对算力底 | 备注 |
| --- | ---: | ---: | --- |
| swap=0（算力底，零 PCIe） | 0.8044 s | 1.00× | 需 42 块全驻留，当前架构放不下 |
| swap=6 + int8 fused | 0.8656 s | +7.6% | 现有可达 |
| swap=7 + int8 fused | 0.8793 s | +9.3% | 现有可达，余量 276 MiB |

「常驻块 INT8 驻留」最多再赚 **0.06~0.075 s（7%~8.5%）**，代价是额外量化 20 个块
（精度损失约翻倍）与一次 offloader 常驻表示的架构改造。按「只有 <0.70 s 才立项」的判据，
**应当放弃该方向**；swap=6/7 已经贴着这张卡的算力硬底。

#### 探针 2：loss 偏差没有"敏感层"，只随量化字节比例缩放

swap=10 + `fused` 下用 `ABC_INT8_NAMES` 控制量化范围（参照为未量化的 bf16 传输臂）：

| 量化范围 | 占字节 | 中位步时 | 平均 \|Δloss\| | 最大 \|Δloss\| |
| --- | ---: | ---: | ---: | ---: |
| 未量化（参照） | 0% | 1.4810 s | 0% | 0% |
| 全量 6 层 | 100% | 1.1204 s | 1.088% | 2.049% |
| 仅 FFN 两层（up_gate + down） | 66.7% | 1.3991 s | **0.566%** | 1.735% |
| 5 层（跳过 `ffn_down`） | 83% | 1.3096 s | **1.066%** | 1.378% |

- **跳过 `ffn_down` 几乎不改善**（1.088% → 1.066%），"离群值敏感层"假设不成立。
- 偏差基本与量化字节比例成正比：66.7% → 0.57%，83% → 1.07%，100% → 1.09%；
  想压到 <0.1% 只能几乎不量化。
- int8 的取舍因此是**近似二元的**：要么接受约 1% 的逐 step loss 偏移换取 1.32×（同 swap）/
  3.33×（低 swap 组合）提速，要么走未量化路径。
- 仅量化 FFN 性价比很差：只快 5.9%（1.4810 → 1.3991），仍带 0.57% 偏差。

## 对既有假设的逐条裁定

| 假设 | 裁定 | 依据 |
| --- | --- | --- |
| 「10 GB 是 3080、TDP 320 W」 | **不成立** | 本卡为 CMP 50HX（TU102、56 SM），功耗上限 225 W |
| 「177 W 说明 SM 大量饥饿」 | **部分成立** | fp16 臂确实 SM 空闲，但这是 DMA 主导的必然结果；同一张卡的 bf16 计算臂实测 232 W，说明卡能被压满 |
| 「bf16 靠 DMA 引擎与计算 100% 并行」 | **成立** | bf16 恢复 12.22 ms/张量 = 纯 H2D 时间，无 GPU 计算、无临时分配 |
| 「int8 的反量化是 SM kernel」 | **成立** | `to(float32)`、`*scale`、`to(fp16)` 三步 eager kernel |
| 「反量化在 default stream，导致同步串行」 | **不成立（但结论相近）** | 它在 per-slot copy stream；与 H2D 串行、计算流等事件，仍在关键路径上 |
| 「反量化在预取流会争抢 SM」 | **部分成立** | 确实占用 SM，但算力成本仅约 45 ms/步，不足以解释 0.77 s |
| 「240 次权重恢复/步」 | **成立** | 20 blocks × 6 linears × 2 passes |
| 「record_stream 让分配器不敢复用，reserved 被撑爆」 | **部分成立** | 现象吻合（3.5 GiB 缓存段、第 0 步定型）；但本报告未逐条证明是 `record_stream` 单独所致 |
| 「触顶时频繁 cudaFree / 内存整理，贡献相当一部分 0.94 s」 | **缺少证据** | 逐步探针显示 reserved 自 step 1 起恒定，无涨落；隔离基准 `num_alloc_retries=0`；关 `expandable_segments` 后步时更慢、reserved 不变 |
| 「碎片/段管理是 3.5 GiB 与 0.77 s 的主因」 | **已排除** | `ANIMA_EXPANDABLE_SEGMENTS=0` 对照：步时 +3.0%，reserved 不变 |
| 「0.77 s 来自同流 H2D 与 SM 反量化交替造成的 DMA 队头阻塞」 | **成立（已隔离复现）** | SM 打满后纯 DMA 恢复 2.873→2.878 s（免疫），int8 三 kernel 1.655→3.31–3.60 s |
| 「copy 模式 HBM 流量是 23 B/元素、约 104 GB/步」 | **成立** | 本机 HBM 实测 485.7 GB/s，104 GB ≈ 214 ms，与隔离实测 0.16 s 同量级 |
| 「单 kernel 融合反量化（`mul(int8, fp32, out=fp16)`）可消掉临时量与两次 kernel」 | **成立** | 6 形状 × fp16/bf16 全部逐位等价；端到端 2.3986→1.9740 s，reserved 9.16→7.21 GiB |
| 「fp8 因 SM89 才有的转换指令而不可用」 | **部分成立** | 解码其实很快（0.264 ms），真正问题是编码被模拟（15.06 ms）与 e4m3 尾数不足（2.25% 误差） |
| 「int8 master 若漏 pin scale 会插入隐式同步」 | **不成立（已实现）** | 实测 6 个线性层的 `quantized` 与 `scale` 均为 pinned |
| 「fused 多出的 +0.94 GiB allocated = 20 slot × 48 MiB」 | **成立（逐字节吻合）** | 6.78 − 5.84 = 0.94 GiB vs 20 × 50,331,648 B = 0.9375 GiB |
| 「+0.34 GiB 来自 9 MiB→48 MiB 惰性扩容的僵尸段」 | **不成立** | 首次调用即按该 slot 最大 payload 一次性分配，无二次扩容 |
| 「全局单份 staging + event 门控可省 ~1.2 GiB」 | **成立** | reserved 7.21→6.10 GiB、整卡 7738→6598 MiB，但步时 +3.8% |
| 「RING=2 因按张量轮转 ring 而退化成 RING=1」 | **不成立** | ring key 为 `slot_id % ring`，相邻 block 天然不同 buffer |
| 「每份 staging 因 20 MiB 向上取整多占 ~12 MiB」 | **不成立** | 隔离实测每份 1.94 MiB，20 份共 38.8 MiB |
| 「staging 改在 default stream 分配可消除该空隙」 | **成立但量级小** | 38.8 → 0.8 MiB（0.44 GiB 空隙的其余部分未定位） |
| 「swap10 + int8 fused 可达 ~1.1~1.2 s/步、约 2.5×」 | **成立** | 实测中位 1.1204 s、相对 bf16/swap20 2.61× |
| 「DMA 流与反量化流双流解耦可拿回 0.47 s（→~1.52 s）」 | **不成立** | 双流 swap20 中位 2.0116 s vs 单流 1.9776 s；残差是 SM 争用而非队头阻塞 |
| 「双流可把 int8 传输还原成 100% 纯 DMA 连续流」 | **部分成立** | 传输确实解耦（极差 0.57→0.015 s），但反量化仍与 GEMM 抢 SM |
| 「22 个常驻块改显存内 INT8 可省 2.1 GiB 并支撑 swap=0」 | **机制成立但收益否决** | 算力底实测 0.8044 s，swap=6 已达 0.8656 s（+7.6%），swap=0 最多再赚 7%~8.5% |
| 「swap=8 的 1.03 s = 0.4 s 反量化 + 0.63 s 传输」 | **不成立（我的错误）** | 反量化 HBM 流量仅 11 ms；正解是算力底 0.804 s + 未掩盖传输 0.222 s |
| 「loss 偏差来自 ffn_down 等离群敏感层，跳过即可压到 <0.1%」 | **不成立** | 跳过 ffn_down 仅 1.088%→1.066%；偏差随量化字节比例缩放 |

## 可执行结论

1. **int8 传输在本机净赚 1.22×，可以留作实验项，但不建议默认开**：它把 reserved 抬到 9.16 GiB
   （整卡 9.56 GiB），几乎没有余量；任何激活峰值都可能直接 OOM。
2. **优先用 `copy` 模式，不要用 `reuse_storage` 提速**：后者省临时显存（隔离 0.65→0.00 GiB），
   但真实训练里步时双峰（2.0 / 4.6 s），不稳定。
3. **要压临时显存，用分块**：`ANIMA_BLOCK_SWAP_INT8_RESTORE_CHUNK_ROWS` 设为正数
   （默认 0 = 单发整张量），可避免 528 MB 级瞬时峰值，代价是更多小 kernel。
4. **降 `blocks_to_swap` 仍是第一优先级**：int8 的收益（0.53 s）小于把 swap 从 20 降到 10 的
   预期收益（约 1.4 s），而 int8 会吃掉降 swap 所需的显存余量，两者相互抵消。
5. **融合反量化已落地并验证**（`block_swap_int8_restore_mode=fused`）：单 kernel 直写 slot +
   每 slot 静态 staging，端到端 **1.9740 s/步（相对 bf16 1.485×）**，reserved 从 9.16 降到 7.21 GiB，
   整卡峰值从 9788 降到 7738 MiB。默认仍是 `copy`，需显式开启。
6. **剩余 0.50 s 仍是队头阻塞**：融合把每张量的 SM kernel 从 3 个降到 1 个，但没有消除
   「反量化 kernel 阻塞后续 H2D」的串行结构。下一步是两阶段入队（先发全部 H2D，再发全部
   `mul`），配合高优先级 copy stream（本机支持 `torch.cuda.Stream(priority=-1)`）。
7. **组合拳是本轮最大收益**：int8 `fused` + 低 `blocks_to_swap` 实测
   **swap=7 → 0.8793 s/步（3.33×）**、swap=6 → 0.8656 s（3.38×）、swap=8 → 1.0265 s（2.85×）、
   swap=10 → 1.1204 s（2.61×），基准为 bf16/swap20 的 2.9258 s。**推荐 swap=7**
   （整卡 9512 MiB，余量 276 MiB）；swap=6 只多赚 0.014 s 却把余量压到 56 MiB，不建议默认。
   代价是 int8 量化使逐 step loss 系统性偏高 0.3%~2.0%，需要单独验收；
   不能接受时可用未量化 swap10（1.4810 s，1.98×）。
8. **剩余 0.4~0.5 s 不是队列结构问题**：双流解耦（DMA 流 + `priority=-1` 反量化流）实测
   2.0116 s，与单流 1.9776 s 无实质差异，只换来极差从 0.57 s 收到 0.015 s。
   该残差是反量化 kernel 与训练 GEMM 抢 SM 的代价（SM 空闲 1.505 s ≈ 纯 H2D 下限 1.47 s；
   SM 打满 2.38 s），**只能靠降低 `blocks_to_swap` 减少反量化元素数来削**。
9. **天花板已经测到，剩下的是取舍而非优化**：weight-tying 探针给出 swap=0 算力硬底
   **0.8044 s/步**；swap=6 的 0.8656 s 仅高 7.6%，swap=7 的 0.8793 s 高 9.3%。
   因此「常驻块 INT8 驻留 → swap=0」最多再赚 7%~8.5%，**否决该架构改造**。
   同时 loss 偏差没有敏感层，只随量化字节比例缩放（66.7% 字节 → 0.57%，100% → 1.09%），
   所以 int8 的取舍是二元的：接受约 1% 偏移换 3.33×，或走未量化路径（swap10 = 1.4810 s，1.98×）。
10. **显存与速度在 staging 上是二选一**：`ANIMA_BLOCK_SWAP_INT8_STAGING_RING=1` 最省
   （reserved 6.10 GiB、整卡 6598 MiB，仅比 bf16 多 168 MiB）但慢 3.8%；`=0`（每 slot 独立）
   最快（1.972 s）但多占 1.05 GiB；默认 `=2` 折中。要同时拿"低显存 + 低 swap"，
   选 `RING=1` 并把 `blocks_to_swap` 从 20 下调。

## 1024 档长序列复测（2026-10-18 补充）

上文全部结论来自 256×256 / 22-token 缓存图（swap20 下 bf16 2.93 s/步）。本轮在**不改变其它任何
参数**的前提下，只替换 `block_swap_transfer_dtype=int8` + `block_swap_int8_restore_mode=fused`，
把同一 A/B 搬到 1024 档（896×1200 / 4200 图像 token）：

| 行 | 文本 cap | 传输 | 首步 s | 后 9 步均值 | min–max | alloc/reserved GiB | 整卡峰值 MiB | 温度 °C | 功耗 W | 退出 |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024-short | 4096（实测 35） | bf16 | 8.201 | **5.688** | 5.673–5.712 | 7.90 / 8.06 | 8972 | 47–60 | 224.78 | 0 |
| 1024-short-int8 | 4096（实测 35） | int8+fused | 7.857 | **5.822** | 5.785–5.855 | 7.99 / 8.38 | 8927 | 47–61 | 225.36 | 0 |
| 1024-cap2048-long | 2048 | bf16 | 11.313 | **9.396** | 9.338–9.469 | 8.99 / 9.20 | 9777 | 50–62 | 249.17 | 0 |
| 1024-cap2048-long-int8 | 2048 | int8+fused | 11.530 | – | – | 8.84 / 9.22 | 9788 | 46–56 | 238.10 | 1（第 2 步 OOM） |

短字幕配对（同配置、只改传输）：**int8 慢 0.134 s（−2.4%）**，reserved 反而 +0.31 GiB，整卡峰值
几乎不变（8972 → 8927 MiB）。长字幕配对下 int8 第 1 步 11.53 s 后即 OOM，`reserved 9.22 GiB`
与 bf16 的 9.20 GiB 相当，说明 int8 没有腾出可用余量。

### 与 256×256 结论的差异及其原因

| 维度 | 256×256（上文） | 1024 档（本节） | 解释 |
| --- | --- | --- | --- |
| int8 净收益 | 1.222× | **0.976×（略慢）** | 小分辨率步时以 H2D 为主，int8 减半传输直接见效；4200 token 时前反向算力占比升高，传输不再是瓶颈 |
| reserved 变化 | +3.23 GiB（copy 模式） | +0.31 GiB（fused 模式） | fused 已消掉 copy 模式的临时张量峰值，本节的对比基线正是 fused |
| 显存余量 | 少但可跑 | 长字幕下 9788 MiB ≈ 卡容量，第 2 步 OOM | 大 seq 的激活峰值吃掉全部余量 |

更直白的结论：**在 1024 档长序列上，int8 传输 + fused 还原既不提速，也不省显存。**
它仍可能在“传输受限”的硬件（更低 PCIe 代际、更小的 per-step 算力）上有效，但本机 50HX
在 4200 token 下不属于该区间。

### 复现

```bash
# 只改传输/还原，其余参数与 bf16 行完全一致
python tasks.py moss-tier-hot -- all \
  --rows 1024-short,1024-short-int8,1024-cap2048-long,1024-cap2048-long-int8 --steps 10
```

产物：`output/bench/moss_tier_hot/matrix-20261018e/`（本地产物，不提交）。日志中可见
`transfer_dtype=int8, int8_restore_mode=fused, int8_scope=all, staging_ring=2`，确认路径生效。

### 本节未验证

- 未测 `staging_ring=1`（文档第 10 条给出的最省显存档）在 4200 token 下是否能把长字幕装下；
- 未测 int8 + 更低 `blocks_to_swap` 的组合（小分辨率下收益最大的一档）；
- 未做数值验收：本轮只有速度与显存，`loss` 偏差 0.3%~2.0% 的既有结论未在 1024 档复核。

## 复现

```bash
cd output/runs/moss-image-abc-precision-20261017-232000
# 隔离微基准（真实形状，无训练）
CUDA_VISIBLE_DEVICES=GPU-fe9cb2a3-4a2b-d6cd-f4ca-219596a8dc44 \
  ../../../../.venv/bin/python -u diag_int8_restore.py
# 训练臂：ABC_ARMS 选择臂名；int8 需要研究级覆盖（契约门禁 + MOSS 名字映射）
ABC_ARMS=xfer-bf16,xfer-int8 ../../../../.venv/bin/python -u execute.py
ABC_ARMS=xfer-int8-reuse-clean ABC_INT8_RESTORE_MODE=reuse_storage ../../../../.venv/bin/python -u execute.py
# 融合反量化（新增 fused 模式）
ABC_ARMS=xfer-int8-fused ABC_INT8_RESTORE_MODE=fused ../../../../.venv/bin/python -u execute.py
# 队头阻塞隔离验证（SM 空闲 vs SM 打满）
CUDA_VISIBLE_DEVICES=GPU-fe9cb2a3-4a2b-d6cd-f4ca-219596a8dc44 ../../../../.venv/bin/python -u diag_hol_bf16.py
```

覆盖说明：`launch.py` 会打印 `ABC_TRANSFER_OVERRIDE` / `ABC_INT8_NAMES` /
`ABC_CONTRACT_BYPASS` / `ABC_INT8_RESTORE_MODE` 四行留证，并把 `transfer_overrides`
写进各臂 `environment.json`。**当前生产前置已接入**：`moss_compat.py` 允许 int8 但发出
实验性 WARNING；`int8_linear.py` 正式包含 MOSS 的 6 个 `proj.*` 候选名。默认仍不建议开启，
因为本轮 swap=10 的逐 step loss 偏移为 0.3%~2.0%。

## 未验证边界

- 单图 256×256、22-token、10 步、单卡；分辨率与文本长度变化会改变 compute/transfer 比例，
  0.77 s 的归属结论不能外推。
- 0.77 s 只做到「排除反量化算力与纯链路」，未逐项分离分配器锁、跨流事件、段管理。
- int8 的数值等价性未建立：逐 step loss 相对 bf16 传输臂偏差 +0.30%~+2.05%，明显大于
  Anima 记录过的 ≤0.059%，默认开启前需要专门验收。
- 未测 `direct_bind` 模式、未测分块 `CHUNK_ROWS` 的正数取值；`fp8_e4m3` 已按 3.3 节的
  实测数据排除（编码被模拟 + 尾数不足），不再作为候选。
- `fused` 模式与 `copy` 臂的 loss 差异（7.4e-5）与同代码重跑的噪声（6.78e-5）同量级，
  因此训练级"数值等价"只能按算子级逐位测试来论证；本配置下训练本身不可逐位复现，
  需要更严格的确定性开关才能做逐位验收。
- 两阶段入队（先发全部 H2D、再发全部 `mul`）未实现；3.9 节已证明剩余残差是算力底
  而非队列结构，故不再计划。
- 算力硬底 0.8044 s 来自 weight-tying 探针（loss 为 NaN、权重被 12 个块复用），
  是偏乐观的下界，不代表真实 swap=0 的精确值。
- staging 环形档位的步时存在偶发尖峰（2.66/3.41/3.60 s），与同期机器上的其他负载叠加，
  因此以中位数为准；单次运行不构成稳定吞吐结论。
- `RING=1` 下 reserved − allocated 仍有 0.21 GiB（bf16 为 0.09 GiB）；隔离实测每份 staging 只占
  1.94 MiB 空隙，其余部分未定位。
- swap=8/9 的整卡余量分别为 474/672 MiB，swap=7/6 为 276/56 MiB；均为 10 步单图短程验证，
  **不代表长训、分辨率变化或长时间碎片累积下安全**。swap=6 的 56 MiB 余量尤其不足以作为默认档。
- swap=7/6 两档只跑了「双流 + 2 份 staging」这一种组合，未做同档单流对照；
  由 swap=20 的同配置对照（中位差 0.013 s）推断为中性，但未单独验证。
- int8 的 loss 偏移（0.3%~2.0%）只做了 10 步单图观测，未做收敛/质量验收。
