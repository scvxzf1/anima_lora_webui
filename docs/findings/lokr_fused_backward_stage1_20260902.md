# LoKr fused backward stage 1-2：grad_w1 Triton 组件与训练验证

日期：2026-09-02

状态：backend ACCEPT；当前默认 `triton` + `triton_grad_w1_w2_grad_x`

上游研究：
[LyCORIS 4.0.0 fused kernel 宣传审计](lycoris_4_fused_kernel_audit_20260902.md)

## 结论

本阶段没有引入 `lycoris_lora`，也没有复制 LyCORIS kernel。实现只借鉴其
“tile 内重算 projection、避免大量小归约 launch”的算法方向，在本项目现有 LoKr
grouped-delta backward 上独立补充 `grad_w1` Triton reduction，并复用已经验证的
`triton_grad_w2_grad_x` 路径。

RTX 3080、BF16 activation、FP32 LoKr factor、真实热 shape 的 3-seed x 20-iteration
组件 A/B 结果：

| 指标 | `triton_grad_w2_grad_x` | `triton_grad_w1_w2_grad_x` | 变化 |
| --- | ---: | ---: | ---: |
| backward 平均 | `24.020 ms` | `19.958 ms` | `-16.91%` |
| forward + backward 平均 | `48.644 ms` | `44.615 ms` | `-8.28%` |
| backward launch | `583` | `133` | `-77.19%` |
| peak allocated | `0.584992 GiB` | `0.581286 GiB` | `-0.003706 GiB` |

三次 candidate backward 分别为 `19.931 / 19.994 / 19.950 ms`。stage 2 又在
CMP 170HX 上完成 3-seed x 50-step 真实训练：平均 step 从
`2.497463` 降到 `1.870028s`，降低 `25.11%`，三个 seed 最低收益
`23.99%`。checkpoint/resume、resident compile、gradient checkpointing 和 block swap
组合 gate 均通过。

这些证据支持将 `triton_grad_w1_w2_grad_x` 合入。单从研究门槛看，它仍缺长训质量
门槛、第二种 GPU 的端到端复核，且 Triton 已选中后的编译异常仍会显式抛出，因此原始
结论没有自行建议替换生产默认。后续项目策略明确将其提升为默认；这是默认策略变更，
不是新增的跨硬件或长训证据。

## 实现范围

### 新路径

- `networks/plugins/lokr/triton_backward.py`：新增 `grad_w1` tiled partial kernel。
- `networks/plugins/lokr/autograd.py`：新增显式 selector
  `triton_grad_w1_w2_grad_x`，组合：
  - 现有 Triton mixed buffer 计算 `grad_w2`；
  - 同一 mixed buffer 写回 `grad_x`；
  - 新 Triton kernel 重算 `x_slice @ w2.T` 并归约 `grad_w1`。
- `bench/lokr_grouped_delta/check_backward_parity.py`：记录逐梯度有限值、绝对误差、
  relative L2、cosine 与 norm。
- `bench/lokr_grouped_delta/run_microbench.py` 和 `run_ablation_matrix.py`：接受新 selector。

新 kernel 不生成完整 Kronecker weight，也不保存 forward projection。它按 row/out tile 生成
FP32 partial，随后用 PyTorch 对小 partial workspace 做最终求和。这个 stage 没有照搬
LyCORIS 的单 kernel 三梯度 + atomic 设计：本项目已有 `grad_w2/grad_x` 组件级实测路径，先补
剩余热点能以更小改动验证收益。

### 数学

令：

```text
X:  [N, factor, in_dim]
G:  [N, factor, out_dim]
W1: [factor, factor]
W2: [out_dim, in_dim]
g:  scalar gate
```

对每个 input factor `i`：

```text
P[n, i, :] = X[n, i, :] @ W2.T
grad_W1[o, i] = sum(n, d, g * G[n, o, d] * P[n, i, d])
```

kernel 的一个 program 负责一个 row tile、out-dim tile 和 input factor，并同时生成当前
output-factor group 的 partial。program 间没有写冲突，不使用 atomic；最终 reduction 在
partial workspace 上完成。

## 精度策略

生产热路径是 BF16 activation + FP32 factor。直接使用单次 TF32 dot 很快，但热 shape 的
`grad_w1` relative L2 为 `4.35e-4`，未采用。`tf32x3` 将误差降到 `6.29e-7`，但 backward
为 `25.99 ms`，相对既有 backend 没有净收益，也未采用。

最终路径利用 BF16/FP16 activation 可由 TF32 精确表示这一点，只把 FP32 `w2` 分解为：

```text
w2_high = truncate_to_tf32(w2)
w2_residual = w2 - w2_high
P ~= dot_tf32(x, w2_high.T) + dot_tf32(x, w2_residual.T)
```

这比通用 `tf32x3` 少一次 dot。FP32 activation 仍走 IEEE dot。

热 shape、seed 1234、candidate 对 eager backward 的误差：

| tensor | max abs | relative L2 | cosine |
| --- | ---: | ---: | ---: |
| `grad_x` | `9.77e-4` | `4.34e-5` | `1.0` |
| `grad_w1` | `1.91e-4` | `1.97e-6` | `1.0` |
| `grad_w2` | `1.89e-3` | `7.84e-7` | `1.0` |

`grad_x/grad_w2` 的差异来自既有 `triton_grad_w2_grad_x` 组合；本阶段新增的
`grad_w1` relative L2 为 `1.97e-6`。所有输出和梯度均 finite。

## 能力边界与 fallback

新 selector 复用 grouped-delta 的 fail-closed 条件：

- CUDA，compute capability `>= 7.5`，Triton 可导入；
- full `w2`、Linear、square factor layout；
- scalar timestep gate；
- `x/grad_out/w1/w2` 同设备且 contiguous；
- dtype 为 FP16、BF16 或 FP32；
- shape 必须满足 `factor * in_dim/out_dim` 契约。

条件不满足时，backward 回退现有 eager 公式。forward 的 dropout、非标量 T-LoRA gate、
decomposed `w2` 等限制仍由 `LoKrModule._can_use_fused_grouped_delta()` 负责，不在 kernel 内
放宽。

当前默认配置（也是本阶段热测组合）：

```toml
lokr_grouped_delta_backend = "triton"
lokr_grouped_delta_backward_backend = "triton_grad_w1_w2_grad_x"
```

Dragon 配置页现已在“常用 → LoKr 专用优化 → 融合计算后端”提供两个下拉框，支持
通过 WebUI 切换；“恢复默认”会选择上述 Triton 组合。TOML 和 `network_args` 仍然
保持兼容入口，显式配置 `eager` 可用于兼容性诊断和对照。

## 验证

### Correctness

- 新定向测试：`9 passed`。
- 覆盖 activation FP16/BF16/FP32、factor FP32/BF16。
- 覆盖 `factor=3, group_size=2` 尾 group。
- 覆盖 `rows=17, in_dim=11, out_dim=19` 非 tile 整除 shape。
- 覆盖 CPU fallback 与 non-contiguous CUDA 拒绝。
- 覆盖 non-reentrant checkpoint 对 grouped-delta custom autograd 的重算 smoke。
- 覆盖小函数 `torch.compile(..., backend="inductor")` 包围 custom op 的 CUDA smoke；输出与
  `x/w1/w2` 梯度均 finite。
- 完整 LoKr 定向测试：`47 passed`。
- registry/config 回归：`124 passed`。

组件环境：RTX 3080 10GB、sm86、Python 3.13、PyTorch `2.12.0+cu130`、Triton `3.7.0`。

### 性能协议

固定：

```text
outer_shape = [2, 1, 72, 56]
rows = 8064
factor = 8
in_features / in_dim = 2048 / 256
out_features / out_dim = 8192 / 1024
group_size = 8
activation = BF16
factor weights = FP32
warmup = 5
iterations = 20
seeds = 1234, 2027, 4099
```

原始结果位于：

- `output/bench/lokr_grouped_delta_microbench/20260902-0912-final_*`
- `output/bench/lokr_grouped_delta_microbench/20260902-0913-final_*`
- `output/bench/lokr_grouped_delta_backward_parity/20260902-0910-hot_bf16_compensated_tf32_seed1234_20260902/`

`output/` 是本机实验产物，不进入源码提交；本 finding 保存最终数字与协议。
这些结果在实现尚未提交的 dirty `dev` 工作树上生成，记录的基线 commit 为 `b40fa90f`；
可复现性应以本文 shape、dtype、seed、backend 和对应源码 diff 共同界定。

## Stage 2：CMP 170HX 端到端热测

### 环境与协议

```text
physical GPU index = 1
CUDA_DEVICE_ORDER = PCI_BUS_ID
CUDA_VISIBLE_DEVICES = 1
GPU = NVIDIA CMP 170HX, sm80, 65536 MiB
driver = 610.43.02
PyTorch = 2.12.0+cu130
Triton = 3.7.0
resolution = 896x1152
batch = 1
seeds = 42, 43, 44
steps = 50
metric intervals = 10..45
VAE / TE cache = 57 / 57
attention = Flash Attention 2.8.3
torch.compile = true, resident scope, dynamic seq
gradient_checkpointing = false
blocks_to_swap = 0
```

A/B 生成配置只相差 backward selector：

```text
baseline  = triton_grad_w2_grad_x
candidate = triton_grad_w1_w2_grad_x
```

### 训练速度与显存

| seed | baseline mean | candidate mean | step 时间降低 |
| ---: | ---: | ---: | ---: |
| 42 | `2.507361s` | `1.871889s` | `25.34%` |
| 43 | `2.524417s` | `1.867861s` | `26.01%` |
| 44 | `2.460611s` | `1.870333s` | `23.99%` |

三个 seed 平均：

- step：`2.497463 -> 1.870028s`，降低 `25.11%`；
- throughput：`24.03 -> 32.09 step/min`，提高 `33.55%`；
- median 降低 `24.84%`，p90 降低 `26.96%`；
- peak allocated：`14.5260 -> 14.5222 GiB`；
- peak reserved：`14.7500 -> 14.7480 GiB`；
- `6/6` run 均有 `run_end=ok` 且完成 50 steps。

### loss 与 checkpoint

`progress.jsonl` 存的是 cumulative `avr_loss`。本轮用
`raw[n] = n * avg[n] - (n - 1) * avg[n - 1]` 还原每步 raw loss，A/B 比较为：

| seed | raw-loss cosine | relative L2 |
| ---: | ---: | ---: |
| 42 | `0.99999581` | `0.002940` |
| 43 | `0.99999162` | `0.004246` |
| 44 | `0.99999499` | `0.003171` |

全部 loss finite。最终 checkpoint 每个都是 840 keys、`55,205,296` bytes，全部
tensor finite；三组 cosine 为 `0.99999353..0.99999421`，relative L2 为
`0.003402..0.003596`。这符合 BF16 训练中更换梯度归约顺序的预期数值差异，
不是 bitwise 一致声明。

### NVML 热态证据

NVML CSV 使用 checkpoint metadata 中的 `ss_training_started_at` 与 progress `ts`
对齐。因为 metric interval 10 是 step 9 到 step 10 的时间，采样窗口是 step 9
完成到 step 45 完成。三个 seed 的窗口平均：

| 指标 | baseline | candidate |
| --- | ---: | ---: |
| GPU utilization | `76.55%` | `99.38%` |
| SM clock | `1607 MHz` | `1566 MHz` |
| temperature | `73.53 C` | `80.70 C` |
| power | `215.28 W` | `247.16 W` |
| NVML memory used | `15486 MiB` | `15484 MiB` |

candidate 更热、平均时钟反而低约 `42 MHz`，因此 `25.11%` 提速不是冷卡或更高频率造成。
全部窗口样本的 UUID 均为目标 CMP 170HX。

### Nsight Systems 直接选路证据

candidate capture 中明确出现：

```text
_lokr_grad_w1_partials_kernel: 1176 launches, 832.587 ms
share of captured CUDA kernel time: 22.92%
```

同时存在 `_lokr_grouped_delta_forward_kernel` 与
`_lokr_grouped_delta_grad_w2_mix_kernel`。总 CUDA kernel launch 从 baseline `73,450`
降到 candidate `20,530`，降低 `72.05%`。这证明新 kernel 真实执行，不是请求
candidate 后静默走 eager fallback。

### 组合与 resume gate

candidate 另做 4-step 组合 smoke：Flash + resident compile + full gradient
checkpointing + `blocks_to_swap=12`。结果 `run_end=ok`，peak reserved
`4.0234 GiB`。该数字只证明兼容，不与无 swap 正式 A/B 比较速度。

resume gate 对比 uninterrupted 5-step 与 3-step save + resume 到 step 5：

- part 2 只出现 global step `4, 5`，`run_end=ok`；
- part 1 state 为 step 3，两侧最终 state 都是 step 5；
- optimizer、scheduler 和 RNG state 文件均存在；
- 最终 adapter 同为 840 keys、全部 finite，cosine `0.99999933`，
  relative L2 `0.001160`，max abs `0.001953125`。

这是近一致 resume PASS，不是 bitwise PASS；分段进程会重新建立 compiled execution。

机器可读总表位于：

`output/bench/lokr_cmp170hx_stage2_20260902-121737/stage2_summary.json`

`output/` 为本机实验产物，不进入源码提交。

## 默认启用后的残余风险

1. 更长训练的收敛、最终 adapter 和样图质量复核；50-step 只是热态性能与短程数值 gate。
2. PG199 或 RTX 3080 上使用新 full-gradient backend 的端到端 3-seed 交叉验证；
   RTX 3080 当前完成的是组件级验证。
3. Triton kernel 编译失败时的运行时降级策略；capability 不满足会 fallback，但已选中
   后的编译异常仍会向上抛出。

因此 `triton_grad_w1_w2_grad_x` 已从“组件 prototype”提升为通过 CMP 170HX 端到端
gate 的默认 backend。能力检查不满足时会回退 eager；已进入 Triton 编译后发生的异常
仍会向上抛出。对外可以精确声称本协议下 CMP 170HX 训练 step 降低 `25.11%`，不应
外推为其他 GPU、shape 或 LoRA family 的通用收益。
