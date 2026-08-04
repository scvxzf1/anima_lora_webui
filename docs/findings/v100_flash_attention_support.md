# V100 FlashAttention 支持现状与使用边界

状态：实验（目标仓库证据对照与当前仓库移植记录）

适用版本：当前 `main`

入口命令：`python tasks.py v100-flash-install` / `v100-flash-validate`

> 调研日期：2026-08-04
>
> 目标仓库基线：MonadForge `08c54137`
>
> 当前仓库基线：`fae04965`，V100 支持按当前拆分架构移植
>
> 验证边界：本次只运行 host-only 定向测试，未在 V100 上重跑 GPU 验收。
> 结论：**代码接入和 `torch.compile` 兼容已经完成，但严格验收仍为
> `validation_failed`。V100 生产训练应继续使用 Torch SDPA，而不是 Flash。**

## 1. 结论先行

目标仓库 MonadForge 实现了面向 Tesla V100 / Volta `sm_70` 的
`flash-attention-v100` 支持，而且不只是一个依赖声明：

- 有固定上游源码提交、源码构建器、wheel 安装探针和可追溯 manifest；
- 能识别 V100 fork，并把普通 FlashAttention 调用切换到专用兼容层；
- 兼容层用 `torch.library.custom_op` 包住原始 pybind CUDA 接口，使 dense、
  varlen、forward、backward 能进入 Dynamo/AOTAutograd fullgraph；
- 有 V100 专用 GPU 测试、失败样本重放、480-case tail matrix、真实 Anima
  eager/compiled 训练 smoke 和性能门禁；
- 验收通过后才会生成 `V100_flash` / `V100_sdpa` preset，失败时会撤销。

但这里的“支持”目前应理解为**可构建、可调用、可诊断、可严格验收**，而不是
“已经批准生产使用”。固定的上游 main 已消除旧版 dense-tail 的 NaN/Inf，仍在
十次完整 capture 重放中出现间歇性 FP16 精度超限。因此：

| 场景 | 当前建议 |
|---|---|
| V100 生产训练 | `attn_mode = "torch"` |
| V100 可选优化 | 先实机验证 `attn_mode = "mem_efficient"`，通过后再采用 |
| V100 Flash | 仅用于复现、诊断和继续验收，不作为生产默认 |
| 非 V100 GPU | 不受本报告结论直接约束，沿用各自已验证的 backend |

## 2. 支持是如何接入的

### 2.1 从配置到 CUDA kernel 的调用链

```text
TOML / --attn_mode / --v100_flash_stability
  -> library.training.model_loading::load_unet_lazily
  -> library.anima.weights::load_anima_model
  -> Anima AttentionParams
  -> Attention.forward
  -> networks.attention_dispatch::dispatch_attention
       |-- torch           -> torch.nn.functional.scaled_dot_product_attention
       |-- mem_efficient   -> SDPA EFFICIENT_ATTENTION backend
       `-- flash
            |-- 普通 provider -> flash_attn_func / flash_attn_varlen_func
            `-- V100 fork     -> networks.flash_attn_v100_compat
                                  -> torch.library.custom_op
                                  -> flash_attn_v100_cuda.{fwd,bwd,varlen_*}
```

关键实现点：

1. [`networks/attention_dispatch.py`](../../networks/attention_dispatch.py)
   通过 `flash_attn_func.__module__.startswith("flash_attn_v100.")` 识别 provider。
   命中后把 public dense/varlen 函数替换为当前仓库的兼容包装器。
2. [`networks/flash_attn_v100_compat.py`](../../networks/flash_attn_v100_compat.py)
   不修改第三方 kernel，而是把 `_v100_cuda.fwd/bwd/varlen_fwd/varlen_bwd`
   注册为 opaque custom ops，并提供 fake shape 实现。这样原始 pybind 调用不会
   造成 Dynamo graph break，也不会丢失 dynamic-sequence 标记。
3. dense wrapper 在进入扩展前把 BLHD 转成 BHLD，并把 head dimension 补齐到
   8 的倍数；varlen wrapper 通过 `cu_seqlens` / `max_seqlen` 进入相应 CUDA 接口。
4. [`library/anima/models.py`](../../library/anima/models.py) 在每个 attention 前按
   self/cross 类型专门化参数，完成可选 finite 检查后交给统一 dispatcher。

训练 CLI 本身的 `--attn_mode` 默认是 `None`，在
`library/training/model_loading.py` 中等价于 `torch`；
但仓库的 config-driven 默认值在 `configs/base.toml` 中是 `flash`，所以常规
`make lora` 最终仍会得到 Flash。V100 必须通过自定义 preset 或 CLI 显式覆盖。

### 2.2 V100 专用约束

- V100 路径只接受 FP16；dispatcher 会拒绝 V100 fork 的 BF16 输入。
- 自动 V100 判定严格检查 compute capability `(7, 0)`，不是笼统的“7.x GPU”。
- `flash4` 与本功能无关，当前 dispatcher 明确禁用该分支。
- 第三方 fork 只保证项目使用的 public dense/varlen facade；部分官方
  FlashAttention private wrapper 不存在，调用方必须把它们视为 optional。

### 2.3 三种稳定性模式

| 模式 | 实际行为 | 能否提升数值稳定性 |
|---|---|---|
| `off` | self-attention 和 cross-attention 都走 Flash | 否；正常 Flash 行为 |
| `hybrid` | self-attention 走 Flash，cross-attention 改走 Torch SDPA | 未解决已知问题；首个非有限值曾出现在 self-attention |
| `safe` | 继续全量 Flash，在 q/k/v、attention 输出、投影、残差、loss、梯度边界 fail-fast | 只能定位，不能修复 kernel |

配置入口是 `v100_flash_stability = "off|hybrid|safe"`，也可以临时设置
`ANIMA_V100_FLASH_STABILITY`。`ANIMA_DEBUG_FINITE=1` 会启用更广的有限性检查。
不要用 `nan_to_num` 掩盖问题，否则会丢失首个故障位置。

## 3. 构建与安装链

### 3.1 被固定的环境

[`scripts/v100_flash/install.py`](../../scripts/v100_flash/install.py) 的 preflight
不是宽松兼容检查，而是精确限定：

| 项目 | 要求 |
|---|---|
| OS / 架构 | Linux x86_64 |
| GPU | 名称包含 `V100` 且 capability 为 `sm_70` |
| Python | 3.13 |
| PyTorch | `2.10.0+cu129` |
| PyTorch CUDA | 12.9 |
| CUDA toolkit / nvcc | 12.9 |
| 默认 host compiler | GCC/G++ 14 |
| wheel tag | `cp313-cp313-linux_x86_64` |
| 上游仓库 | `ai-bond/flash-attention-v100` |
| 固定源码 | `c91cad40c0539805754819e6ea96c75184d816a6` |

构建时固定 `MAX_JOBS=2`、`NVCC_THREADS=2`，并清除可能改变 kernel 的
`MMA_NATIVE`、`MMA_884`、`ATTENTION_DEBUG` 环境变量。安装器还要求 upstream
`origin/main` 与固定 commit 完全一致；上游移动后会中止，要求维护者重新审查
并跑完整验收，而不是悄悄构建另一个版本。

### 3.2 安装入口

完整 V100 环境可从项目脚本建立：

```bash
./setup-v100-linux.sh
```

该脚本默认**不安装 Flash**。只有显式设置 `ANIMA_INSTALL_V100_FLASH=1` 才会
调用源码构建任务。已有 V100 环境可以直接运行：

```bash
make v100-flash-install ARGS="--cuda-home /usr/local/cuda-12.9"
```

安装器会构建固定的 cp313 wheel，安装后执行一个 FP16 dense probe，并把源码、
wheel、扩展、工具链哈希与完整日志写到 `output/v100-flash-install/`。此时状态只能
是 `installed_unvalidated`，不能据此启用生产 preset。

两个容易踩中的环境边界：

1. 普通 `flash-attn` 与 V100 fork 都提供顶层 `flash_attn` 包。安装器默认拒绝
   覆盖已有 provider；只有人工核对后才能使用 `--allow-reinstall`。
2. 常规 [`pyproject.toml`](../../pyproject.toml) 指向 Torch 2.12/CUDA 13.2 的
   普通 Flash wheel，而 V100 环境由 `requirements-v100.txt` 固定 Torch 2.10
   且故意不声明 Flash。V100 环境执行普通 `uv sync` 可能同时替换 Torch 和
   Flash provider；发生同步后必须重新执行 install + validate。

## 4. 严格验收门禁

```bash
make v100-flash-validate ARGS="\
  --capture /path/to/first_failure.pt \
  --dit /path/to/anima-base-v1.0.safetensors \
  --performance-baseline /path/to/tail-matrix.json"
```

capture 和 DiT 必须匹配仓库固定的 SHA-256；cross-attention sidecar 也有固定
哈希，performance baseline 的哈希会进入报告。验证开始前会撤销旧的受管
V100 preset。只有以下所有门禁通过后，才会重新生成
`configs/custom/presets/V100_flash.toml` 与 `V100_sdpa.toml`：

| 门禁 | 覆盖内容 |
|---|---|
| GPU pytest | provider 检测、installer/preset 策略、FP16/BF16 路由、dense/varlen fullgraph forward/backward、finite/LSE |
| capture replay | 原始、eager、compiled 路径；固定重复 10 次，防止偶然单次通过 |
| dense-tail matrix | 480 个 forward/backward case；head dim 16/32/64/128/256、Q/K 的 mod-16 尾长、causal/non-causal、dQ/dK/dV |
| Anima eager Flash | 真实 DiT/LoRA 5 个 optimizer step |
| Anima compiled Flash | dynamic sequence、8 个 swapped blocks、fullgraph 路径 |
| Anima compiled SDPA | 同一 production compile 顺序的对照组 |
| 性能比较 | 对齐长度与基线比较，并记录 compiled Flash/SDPA step 差异 |

GPU 测试默认不会意外占用显卡；只有设置 `ANIMA_TEST_GPU=1` 才会运行：

```bash
ANIMA_TEST_GPU=1 .venv/bin/python -m pytest -q \
  tests/test_v100_flash_install.py \
  tests/test_v100_flash_stability.py \
  tests/test_flash_attn_v100_compile.py \
  tests/test_web_config_service.py
```

## 5. 当前实测证据

权威的原始记录在
[`bench/v100_flash/README.md`](../../bench/v100_flash/README.md)。当前结果应拆成
两部分看：旧的非有限值问题已经改善，但生产验收仍未通过。

### 5.1 数值结果

| 验证项 | 固定 main `c91cad40` 的结果 |
|---|---|
| Host/GPU integration tests | 36 passed |
| 4112..4128 prefix replay | 17/17 finite |
| dense-tail matrix | 480/480 forward/backward passed |
| Real Anima eager Flash | 5/5 steps passed |
| Real Anima compiled Flash | 5/5 passed，无 graph break |
| Real Anima compiled SDPA | 5/5 passed |
| 4130-token full capture，10 次重复 | **失败** |

十次重放的每个输出都是 finite，失败原因已从旧版的 NaN/Inf 变为间歇性精度
超限：

- `compat_flash_eager` 第 6、8 次失败，最大 FP32-relative error
  `0.003173828125`；
- `compat_flash_compiled` 第 4 次失败，最大 error `0.0029296875`；
- 验收阈值为 `0.001963125`。

因此 manifest 保持 `validation_failed`，验证器撤销了生成的 Flash/SDPA preset。
此前三次重复曾偶然全部通过，这也是严格门禁把次数固定为 10 的原因。

### 5.2 性能结果

同机、同一 144-step、`target_res=[768]` 的训练记录为：

| Backend | 端到端时间 | 稳态 step median |
|---|---:|---:|
| Flash | 181 s | 0.5875 s |
| Torch SDPA | 165 s | 0.4850 s |
| Memory-efficient SDPA | 147 s | 0.4850 s |

该配置中 Flash 比 Torch 慢约 21%。这组数据只覆盖 2128/2196-token native
bucket，没有覆盖 1024 tier 的约 4096 tokens，因此不能外推成所有 V100 shape
上的普遍性能结论；但它至少没有为承担当前数值风险提供性能理由。

## 6. 当前生产配置

把硬件配置放在用户自有、gitignored 的
`configs/custom/presets/V100.toml`：

```toml
mixed_precision = "fp16"
save_precision = "fp16"
attn_mode = "torch"
torch_compile = true
gradient_checkpointing = true
unsloth_offload_checkpointing = false
blocks_to_swap = 8
```

然后运行：

```bash
PRESET=V100 make lora
```

注意：

- V100 没有原生 BF16；当前被验证的高效训练配置使用 FP16，不要套用 BF16 preset；
- `torch_compile=true` 与 `attn_mode="torch"` 已有真实 V100 证据，可以保留；
- 不要全局禁用 SDPA backend，text encoder、VAE 和第三方组件仍可能需要 fallback；
- `mem_efficient` 会只在 Anima attention 调用内强制 Efficient Attention。如果想采用，
  先运行实机 probe；不具备相应 CUDA kernel 时会显式失败，不会静默回退到 math；
- 当前没有获批的自动生成 V100 Flash preset。手工创建一个同名 Flash preset 不会
  改变其 `validation_failed` 状态。

可选的 memory-efficient 实机检查：

```bash
python -m bench.v100_flash.run_probe \
  --attn_mode mem_efficient --device cuda --steps 5 --label mem-efficient
```

## 7. Flash 诊断流程

以下命令只用于定位问题：

```bash
# 稳定对照组
python -m bench.v100_flash.run_probe --attn_mode torch --device cuda

# self-attn Flash + cross-attn Torch
python -m bench.v100_flash.run_probe \
  --attn_mode flash --stability hybrid --debug_finite --device cuda

# 全 Flash + fail-fast finite checks
python -m bench.v100_flash.run_probe \
  --attn_mode flash --stability safe --debug_finite --device cuda
```

单次 probe 结果写到 `bench/v100_flash/results/<timestamp>/result.json`；严格安装和
验收记录写到 `output/v100-flash-install/`。更窄的工具入口：

```bash
python -m bench.v100_flash.replay_capture --help
python -m bench.v100_flash.run_tail_matrix --help
python -m bench.v100_flash.run_anima_smoke --help
```

## 8. 已知缺口与风险

1. **没有官方 versioned fixed wheel。** 当前 pin 是上游 main 的一次快照，本地
   wheel 是诊断证据，不是发布制品。
2. **严格数值门禁失败。** finite 不等于数值正确；当前失败是可重复观察到的
   间歇性 tolerance breach。
3. **支持矩阵很窄。** 安装器只接受 Linux x86_64、Python 3.13、Torch
   2.10+cu129、CUDA toolkit 12.9 和 Tesla V100 `sm_70`。
4. **只支持 FP16。** V100 fork 的 BF16 dispatch 被明确禁用。
5. **常规依赖同步会破坏专用环境。** `pyproject.toml` 的普通 Torch/Flash
   依赖不是 V100 landing 的依赖源。
6. **Flash 推理未获得独立生产验收。** attention dispatcher 被训练和推理共用，
   但 `v100_flash_stability` 的标准配置入口是 training-only。标准 inference CLI
   和 `GenerationRequest` 已默认 `torch`；底层 programmatic harness 则默认
   `flash`，且不会自动注入诊断模式。直接使用 harness 的 V100 调用方必须显式传
   `attn_mode="torch"`，不能从训练 smoke 推导 Flash 推理已稳定。
7. **性能证据不完整。** 现有同机训练比较没有覆盖 1024 tier；是否在某些更长
   token shape 上存在 Flash 优势仍需实机数据，但任何性能结论都不能绕过数值门禁。

## 9. 变更历史与证据索引

- `2fcdd8dc`（`feat: land V100 FlashAttention validation and training evidence`）
  是主接入提交：新增兼容层、安装/验证器、bench 和测试。
- `08c54137`（当前基线）进一步收紧验证和清理：修复 zero-dropout
  `return_softmax` 传参，强制 machine-local 验收输入，保护用户 preset，并改进
  CUDA toolkit 探测。

主要证据入口：

| 文件 | 作用 |
|---|---|
| [`networks/attention_dispatch.py`](../../networks/attention_dispatch.py) | provider 检测、dtype gate、backend 路由、hybrid 行为 |
| [`networks/flash_attn_v100_compat.py`](../../networks/flash_attn_v100_compat.py) | Dynamo/AOTAutograd custom-op 兼容层 |
| [`library/training/model_loading.py`](../../library/training/model_loading.py) | V100 警告、稳定性解析和模型装载传参 |
| [`library/anima/training.py`](../../library/anima/training.py) | 训练 CLI 参数入口 |
| [`scripts/v100_flash/install.py`](../../scripts/v100_flash/install.py) | 固定源码构建、严格环境检查、wheel/extension manifest |
| [`scripts/v100_flash/validate.py`](../../scripts/v100_flash/validate.py) | 完整验收门禁与 preset 生命周期 |
| [`tests/test_flash_attn_v100_compile.py`](../../tests/test_flash_attn_v100_compile.py) | dense/varlen fullgraph GPU 回归测试 |
| [`tests/test_v100_flash_stability.py`](../../tests/test_v100_flash_stability.py) | provider、dtype、hybrid/safe 行为测试 |
| [`tests/test_v100_flash_install.py`](../../tests/test_v100_flash_install.py) | installer、wheel、重复次数和 preset 策略测试 |
| [`bench/v100_flash/README.md`](../../bench/v100_flash/README.md) | 原始 V100 数值、性能和 production recipe |
| [`requirements-v100.txt`](../../requirements-v100.txt) | V100 独立依赖约束与普通 Flash 排除说明 |

本报告是对仓库代码与已落地 V100 记录的静态审计；本次没有在另一台 V100 上重新
执行耗时的 capture/matrix/真实 DiT 验收。后续只有新的、完整的 10-repeat 验收
报告可以改变当前的生产结论。
