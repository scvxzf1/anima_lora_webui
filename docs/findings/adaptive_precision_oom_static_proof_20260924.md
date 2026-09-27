# FP16/FP32 与 OOM 重试的静态契约论证

状态：代码级契约、实现说明与 CPU 静态测试；没有启动训练、CUDA 热测或真实 OOM 注入。

更新：2026-09-26。本文解释当前实验实现的**控制流和数学不变量**，不把静态证明
升级为整网数值认证、训练质量认证或性能结论。

## 1. 目标与边界

当前实现分成两个正交问题：

1. **精度策略**：冻结 FP16/FP32 混合精度请求，安装到冻结的普通 `Linear` 基础层，
   让 LoRA 参数和外部残差流继续保持 FP32；同时冻结影响浮点执行的进程/设备环境。
2. **显存恢复**：遇到明确的 CUDA OOM 时，在新的 worker 进程中只调整授权的
   `MemoryPlan`，并确认不会偷偷改变精度策略。

这两个问题不能互相推断：

- OOM 只说明当前内存计划无法完成某个阶段，不说明某个层对 FP16 数值敏感；
- NaN/Inf 或梯度误差只说明数值候选失败，不说明增加 block swap 能解决它；
- 因此 OOM 重试绝不自动增加/删除 FP32 模块，精度校准也绝不自动改显存计划之外
  的训练语义。

当前仍明确关闭以下能力：

- 更新后 OOM 的生产续训；
- 数据 bucket、sampler、prefetch 游标的完整恢复；
- 自动敏感层选择进入正式训练；
- 复杂 adapter、NF4、compile、分布式和多进程混合精度实验。

## 2. 总体执行图

```text
merged args
    |
    | resolve_adaptive_precision() + require_training_contract()
    v
PrecisionRequest P=(request, environment) --canonical JSON--> H = SHA256(P)
    |
    | freeze Namespace(P, H)
    v
supervisor
    |
    +--> fresh worker(M_0, P, H)
    |       |
    |       +--> recompute H before model load
    |       +--> install FP16/FP32 islands
    |       +--> check realized D against installation and at commit/exit
    |       +--> train until ok / cuda_oom / error
    |       +--> structured result(H, D if installed, stage, optimizer_started)
    |
    +--> run_recovery()
            |
            +--> H matches and status=cuda_oom -> next_plan(M_k)
            +--> H mismatch -> stop, no next worker
            +--> non-CUDA failure -> stop, no retry
            +--> optimizer_started and no committed checkpoint -> stop
```

**设计原因：** 训练 worker 必须是新进程。CUDA allocator、offloader、梯度和部分
optimizer 更新都可能在异常后处于不可复用状态；在同一进程中清缓存再继续，无法证明
模型、参数和 optimizer 状态仍处于可解释边界。

## 3. 精度请求的定义与摘要

### 3.1 请求集合

一次实验训练的数值请求定义为：

```text
P = (mode, requested_mode, candidate, mixed_precision, model_family,
     adaptive_fp32_modules, adaptive_loss_scale, base_compute, attn_mode,
     environment)
H = SHA256(canonical_json(P))
```

实现位置：`library/training/adaptive_runtime/contract.py`。

`precision_request(args)` 收集以下字段：

| 字段 | 作用 | 为什么属于精度契约 |
| --- | --- | --- |
| `mode` | `off`、`bf16`、`fp16_fp32` 或 `fp32` 的解析结果 | `auto` 解析后必须在所有 worker 保持同一执行模式 |
| `requested_mode` | 用户原始请求 | 保留 UI/metadata 语义，避免 `auto` 与显式模式混淆 |
| `candidate` | 硬件策略选中的候选 | 记录 `auto` 的硬件决策来源 |
| `mixed_precision` | Accelerate 的全局 mixed precision 值 | 决定 scaler/autocast 的底层配置 |
| `model_family` | Anima、Krea-2 或 Z-Image | 不同 family 的 Linear、attention 和 swap 合约不同 |
| `adaptive_fp32_modules` | 显式 FP32 Linear glob 列表 | 改变 FP16/FP32 分区，直接改变数值路径 |
| `adaptive_loss_scale` | 初始 GradScaler scale | 改变 FP16 梯度溢出和更新行为 |
| `base_compute` | 基础权重计算策略 | NF4/BF16/其它基础计算不是同一数值路径 |
| `attn_mode` | attention backend | backend 可能改变累加顺序、dtype 和溢出边界 |
| `environment` | torch/CUDA/HIP、TF32、确定性、SDP backend、可见 GPU 身份 | 同一配置在不同运行时或设备上不是同一浮点实验 |

输出目录、日志路径、`blocks_to_swap`、重试次数和主机内存阈值被刻意排除。它们属于
资源调度，不属于数值策略；但设备身份和 backend 开关不属于“资源计划”，因此保留在
`environment` 中。这样才能证明“只换显存计划，不换精度计划”，同时不把换卡重试伪装成
同一个数值实验。

### 3.2 规范化和哈希

实现等价于：

```python
request = precision_request(args)
payload = json.dumps(
    request,
    sort_keys=True,          # 原因：字典插入顺序不能影响身份
    separators=(",", ":"),  # 原因：去掉无语义空白，避免不同序列化结果
    allow_nan=False,         # 原因：NaN/Inf 不是可复现的有效配置
).encode("utf-8")
contract_id = sha256(payload).hexdigest()
```

FP32 pattern 列表保留用户输入顺序。即使某些顺序产生相同最终 assignments，也不把
两个不同的显式请求伪装成同一个请求；这使审计记录能够回到原始配置。

`precision_environment()` 额外快照以下值：

- `torch.__version__`、CUDA/HIP 版本、`CUDA_VISIBLE_DEVICES`；
- 当前 GPU 的 index、数量、名称、compute capability、显存和 UUID（可探测时）；
- `torch.get_float32_matmul_precision()`、CUDA matmul 的 TF32/FP16/BF16 reduced-
  precision 开关；
- cuDNN 的 TF32、benchmark、deterministic，`torch` deterministic algorithms；
- `CUBLAS_WORKSPACE_CONFIG` 和 Flash/memory-efficient/math/cuDNN SDP backend flags。

CPU 静态测试没有 GPU 身份时记录 `available=false`。若 CUDA 报告可用但设备身份探测
失败，直接抛出 `precision_contract_mismatch`；不再用两个相同的 `unavailable` 值
伪装成可比较的设备身份。supervisor 在创建恢复目录前完成首次身份校验。
这里的严格含义是：相同规范化 JSON 必然得到相同摘要；对不同 JSON 只依赖摘要碰撞概率
足够低来防止**意外配置漂移**。这不是对抗性安全承诺，也不是模型权重指纹或数值安全证书。

## 4. FP16/FP32 训练的实际安装方式

### 4.1 安装顺序

`model_loading.py` 在每个支持的 model family 中调用 `_install_adaptive_precision()`。
其顺序约束是：

```text
load actual DiT weights
    -> install frozen precision islands
    -> apply/load LoRA adapter
    -> create offloader masters / block swap
    -> accelerator.prepare / optimizer
```

**原因：** 如果先 `model.to(weight_dtype)`、先创建 offloader master 或先 compile，
后续整体 cast/trace 可能覆盖或捕获错误的 dtype。精度 island 必须先成为模型结构的
一部分，再让 adapter、swap 和 optimizer 观察它。

### 4.2 精度分区

`training_precision.install_training_precision()` 执行以下策略：

1. 只接受普通、非量化、无 alias 的 floating `Parameter`；
2. 默认把所有普通 `torch.nn.Linear` 标为 FP16；
3. `adaptive_fp32_modules` 的 glob 匹配结果覆盖为 FP32；
4. 非 Linear 的浮点参数和 buffer 统一保留 FP32；
5. 冻结基础模型后再安装 island，LoRA 另由网络构建流程创建。

`islands.py::_island_forward()` 的边界是：

```python
args, kwargs = cast_floating_inputs_to_island_dtype(...)
with torch.autocast(device.type, enabled=False):
    result = original_linear(*args, **kwargs)
return cast_floating_outputs_to_fp32(result)
```

**原因：** Linear 内部可以使用 FP16，外部 residual stream 不因某个低精度层而被
隐式降到 FP16。返回 FP32 也让后续 norm、残差加法和未列入 island 的模块有明确的
输入 dtype。该函数只支持 plain Linear，是因为 RoPE、norm、softmax、自定义 kernel
和量化 Linear 的参数/输入契约不能由通用 `.half()` 自动推出。

这里是**替换 `forward`**，不是额外注册 `forward_pre_hook`。安装顺序为
`island -> LoRA`，所以 adapter 的包装发生在 island 外层：LoRA down 分支看到的是
FP32 输入，只有进入被冻结的 base `Linear` 时才复制为 FP16。这个顺序是为了避免
hook 包在 LoRA 合并 forward 外层、把 LoRA 输入误降成 FP16；对应测试应使用 live
adapter 检查 `lora_down.input.dtype == torch.float32`。

反向路径不能被“base weight 冻结”误解为全程 FP32：输出的 FP32 cast 会接收上游梯度，
梯度穿过 cast 后仍会在 FP16 `Linear` 的输入/权重计算路径中形成 activation gradient。
因此 `adaptive_loss_scale` 必须进入契约，GradScaler 的溢出检查也不能被关闭；本实现
只保证边界 dtype 和配置不漂移，不宣称 FP16 反向与 FP32 逐位相同。

每次 island forward 都可能产生一个 FP16 输入副本和一个 FP32 输出副本。主要节省的是
冻结权重的存储，activation 并不会按“全部变成 FP16”同比下降，甚至可能因边界副本略增。
所以 island 是数值/权重布局策略，不是已经被静态证明的 OOM 性能优化；显存收益必须另做
受控测量。

### 4.3 Accelerate 和可训练参数

FP16/FP32 模式使用：

- `AutocastKwargs(enabled=False)`：关闭全局 autocast，避免再次覆盖 island 的局部 dtype；
- `GradScalerKwargs(init_scale=adaptive_loss_scale)`：保留 Accelerate 的 FP16 scaler；
- plain DiT LoRA：可训练参数要求为 FP32；不允许 text encoder LoRA 或复杂 adapter。

关闭 autocast 不等于关闭 scaler。scaler 负责 loss scaling、unscale、gradient clipping
前的检查和跳步；island 负责冻结基础 Linear 的局部计算 dtype，两者解决的是不同问题。

## 5. worker 冻结与契约传递

### 5.1 supervisor 的冻结点

`supervisor.run_supervised()` 在启动前：

```python
resolve_adaptive_precision(args)
require_training_contract(args, world_size=...)
frozen_args = Namespace(**copy.deepcopy(vars(args)))
expected_precision_contract = precision_contract_id(frozen_args)
```

**原因：** `auto` 只能在父进程根据实际 compute capability 解析一次。worker 不能
重新读取 TOML 或重新探测硬件，否则同一个任务可能在不同尝试中得到不同 attention、
mixed precision 或 FP32 名单。

### 5.2 每次 worker 的参数白名单

`worker_arguments()` 从冻结配置深拷贝出子进程配置，并只覆盖：

| 覆盖字段 | 目的 | 不覆盖的字段 |
| --- | --- | --- |
| `blocks_to_swap` | 应用新的显存计划 | 所有精度字段保持冻结 |
| `adaptive_oom_retry=False` | 防止 worker 再递归启动 supervisor | 精度模式仍保持原值 |
| `resume` / `skip_until_initial_step` | 当前只允许空 resume | 真实 data cursor 仍未开放 |
| 保存周期和临时 worker 标记 | 防止 worker 与用户输出互相覆盖 | 底模、adapter、seed、loss scale 不变 |
| `_adaptive_precision_contract_id` | 让结果携带 `H` | 不可由 memory plan 改写 |

配置文件以 `O_EXCL`、`0600` 创建，读取后删除。这样既避免 worker 重新合并旧配置，
也避免把完整训练配置长期留下或暴露给同机其它用户。

### 5.3 结构化结果

`training_worker.execute()` 只把真实的 `torch.OutOfMemoryError` /
`torch.cuda.OutOfMemoryError` 分类为 `status="cuda_oom"`。包含 “CUDA out of memory”
文本的普通 `RuntimeError` 仍是 `status="error"`。

结果至少包含：

```text
status                 = ok | cuda_oom | error
stage                  = model_load | setup | forward | backward | optimizer | checkpoint
optimizer_started      = bool
precision_contract_id  = H
precision_manifest_id  = D（实际基础参数/buffer dtype manifest）
committed_checkpoint   = path or null
```

**原因：** 日志文本不可靠，必须用异常类型和显式生命周期状态决定是否可以重试。
这也防止把 host OOM、dtype mismatch、磁盘错误或 worker 崩溃误判成 GPU 显存不足。

`precision_contract_id` 不是 worker 可以自证的回显字段：它只是 supervisor 注入的
`expected_H`。worker 在加载模型**之前**、安装后的 `setup`、`commit` 以及完成/OOM
结果发布前，用自己的合并 `args` 重新调用 `precision_contract_id(args)` 得到 `H_local`，并要求
`H_local == expected_H`；不一致立即写入 `status=error`、
`failure_kind=precision_contract_mismatch`、`reason=precision_contract_mismatch`。
当用户请求是 `adaptive_precision="auto"` 时，还必须已经存在具体的
`adaptive_resolved_mode`/`adaptive_candidate`；worker 不得再次探测硬件，也不能把
未解析的 `auto` 当作执行模式。这一步切断“注入 H、原样回传 H、再拿回传 H 自证”的
循环，验证的是 worker 实际看到的配置。

### 5.4 实际 dtype manifest

请求摘要仍不能推出真正的 FP16/FP32 分区，因为中间还隔着 glob 匹配、模块遍历、LoRA
注入和 offloader master 创建。因此 `install_training_precision()` 在 island 安装完成
后记录所有基础 parameter/buffer 的 `(kind:name -> str(dtype))`，用 canonical JSON 得到
`D = realized_precision_manifest_id(model)`。`TrainingRuntime.validate_precision()` 在
`setup` 首次计算时必须先与**安装时**摘要比较，不能把 setup 时的漂移值建立为新基线。
每个 `commit` 在发布 `latest-state.json` 之前复核；worker 完成或捕获 CUDA OOM 后也
重新计算 `D`，漂移则改报 `status=error`，不进入下一次重试。`run_recovery()` 比较
所有**已安装模型**的 `ok`/`cuda_oom` 摘要；加载模型前发生的 `model_load` OOM 可无
`D`，但必须满足 `optimizer_started=False` 且 worker 已在加载前验证 H。其它缺失 `D`
仍然停止。这些检查覆盖**检查点时刻**的分区，不证明两次检查之间每个瞬间都未漂移。

这份 manifest 证明的是**实际 dtype 分区没有漂移**，覆盖 swap 搬运或 accelerator
准备可能造成的单一 `weight_dtype` 重建错误；它不证明参数仍在同一设备、kernel 选择
相同、LoRA 拓扑未变或数值误差为零。manifest 的名字集合在第一次安装时冻结，缺失或
新增 tensor 不在冻结的名字集合内，尚不能由该摘要识别；缺失原有 tensor 会被拒绝。

## 6. OOM 状态机与实现原因

### 6.1 `MemoryPlan` 和 `RetryLimits`

内存计划定义为：

```text
M = (s, g, b, a)
```

其中 `s=blocks_to_swap`、`g=gradient_checkpointing`、`b=micro_batch`、
`a=accumulation`，有效 batch 为 `B=b*a`。

`RetryLimits` 限制 `max_blocks`、`max_attempts` 和 `swap_increment`。构造时拒绝
负数、零尝试、零增量等无意义状态；每次尝试的 immutable `MemoryPlan` 放入 `seen`，
重复计划会结构化停止而不是无限循环。当前 supervisor 还固定从
`gradient_checkpointing=True` 开始，`RetryLimits.allow_checkpoint_change=False`；即使
调用方初始未启用 checkpoint，也必须显式授权才可把它作为下一步变量。

### 6.2 `next_plan()` 的决策顺序

当前决策顺序是一次只改变一个变量：

```python
if stage in {"model_load", "forward", "backward"}:
    increase blocks_to_swap up to max_blocks
if stage in {"forward", "backward"} and not checkpointing
   and limits.allow_checkpoint_change:
    enable gradient_checkpointing
if batch change was explicitly authorized:
    choose an exact divisor d and set (b, a) -> (b/d, a*d)
```

**为什么先增加 swap：** 这是对已验证训练数学影响最小的内存调度变化；它不改变
batch、optimizer、loss scale 或参数 dtype。

**为什么 optimizer 阶段不自动换计划：** optimizer 可能已经写入部分参数、moment
或 scaler 状态，单靠清缓存无法证明更新是原子的。没有完整提交检查点时只能停止。

**为什么 batch 调整需要显式授权：** 虽然 `B=b*a` 可以保持不变，但 batch 的分组、
梯度规约顺序、随机消费顺序和数值舍入仍可能变化，所以不能默认打开。

**为什么 checkpointing 不是 swap 的同级默认项：** checkpointing 会改变前向重算、RNG
保存/复放和部分 kernel 执行次数，属于训练数学/执行语义变更。当前训练契约要求
`gradient_checkpointing=True`，因此正常 supervisor 不会在 OOM 时再切换它；未来若开放，
必须单独授权、单独记录并单独验收，不能与增加 swap 合并在一次 `next_plan()` 调用中。

**主机内存耦合尚未自动解决：** 增加 `blocks_to_swap` 会增加 pinned host memory。当前
`next_plan()` 只受 `max_blocks` 限制，没有根据 host-memory reserve 反推可用上限；
`IsolatedRunner` 发现主机压力时返回 `host_limit`，而 `run_recovery()` 不把它升级成
GPU OOM 重试。这是刻意的 fail-closed 缺口，不应把 `max_blocks` 写成“必然可行”。

### 6.3 `run_recovery()` 的分支

伪代码如下，注释表示每个保护的原因：

```python
for attempt in range(max_attempts):
    if plan in seen:
        stop("repeated plan")              # 防止状态机循环

    result = fresh_worker(plan, resume)
    status = result.get("status")

    if status in {"ok", "cuda_oom"}:
        require(result["precision_contract_id"] == expected_H)
        D = result["precision_manifest_id"]
        if D is None:
            require(status == "cuda_oom" and stage == "model_load"
                    and result["optimizer_started"] is False)
            # 此时 DiT 尚未完成安装；worker 在加载前已独立验证 H。
        elif realized_manifest is None:
            realized_manifest = D
        else:
            require(D == realized_manifest)
        # 原因：资源调整不能顺便切换精度请求或实际 dtype 分区。

    if status == "ok":
        return success(plan)

    if status != "cuda_oom":
        return failed(result)                # timeout/error/host_limit/cancelled 不重试

    if not isinstance(result["optimizer_started"], bool):
        return failed("unknown_optimizer_progress")

    if result["optimizer_started"]:
        resume = result.get("committed_checkpoint")
        if not resume:
            return failed("no_committed_checkpoint")
            # 原因：不能从可能部分更新的进程继续训练。

    change = next_plan(plan, limits, stage=result["stage"])
    if change is None:
        return failed("no_authorized_adjustment")
    plan, reason = change
```

契约 ID 校验只对 `ok` 和 `cuda_oom` 做，因为超时、host limit、取消和进程崩溃可能
根本没有机会写 worker result；这些状态仍然立即失败，但不能被错误标为“精度契约不匹配”。

### 6.4 OOM 分类、步长与临时文件

worker 直接兼容 `torch.OutOfMemoryError` 和 `torch.cuda.OutOfMemoryError`，只有这两类
异常才产生 `status="cuda_oom"`。普通 `RuntimeError` 即使消息包含
`CUDA error: out of memory` 或 `CUBLAS_STATUS_ALLOC_FAILED`，也只标记
`failure_kind="possible_oom"` / `reason="possible_oom"` 并保持 `status="error"`，
用于观测 cuBLAS/cuDNN workspace 或 DataLoader 变形错误，但不自动重试；这样仍保持
fail-closed。未来可在有稳定错误分类和受控测试后单独授权扩大重试集合。

当前 `swap_increment` 是固定步长，尚未使用 `torch.cuda.memory_stats()` 或异常中的
requested bytes 做缺口估计，也未对 model-load 采用指数步长/二分；因此即使可行点在
预算内，也可能被粗步长或 `max_attempts` 提前截断。成功的 `MemoryPlan` 也尚未持久化
为下次同配置的搜索起点。

请求文件使用 `O_EXCL`、`0600`，worker `main()` 读完在 `finally` 删除，supervisor 的
runner 和 `IsolatedRunner` 也在 `finally` 兜底删除。worker 在读取前崩溃不会留下可复用
的明文配置；残留目录只保留结构化结果和日志供审计。

## 7. 提交边界与当前不续训原因

`TrainingRuntime` 用三个边界保护有效更新：

1. `phase("optimizer")` 在真正调用 `optimizer.step()` 之前把
   `optimizer_started=True` 设为 sticky；异常发生在 step 内也不会被当成“尚未更新”；
2. `after_optimizer()` 拒绝 GradScaler 跳步，跳步不推进 scheduler、不提交有效更新；
3. `commit()` 只在完整有效 step 后调用 `save_training_state()`，写入临时目录，检查
   adapter、optimizer、scheduler、RNG、scaler、训练步数和 `adaptive_precision.json`，
   最后原子发布并删除前一个 owned snapshot。

当前 `TrainingRuntime.result()` 固定返回：

```text
committed_checkpoint        = null
data_cursor_resume_supported = false
```

同时 `worker_arguments()` 拒绝非空 `resume`。这是有意的 fail-closed 设计：当前数据
集会原地 shuffle，Accelerate loader 存在 look-ahead/prefetch；只恢复 RNG、跳过若干
batch 或加载 optimizer 状态，不能证明下一批 latent、caption、bucket 和 noise 与
不中断轨迹相同。必须先完成数据游标协议，才可以把已发布 snapshot 接入更新后 OOM。

## 8. 数学不变量与证明

### 8.1 精度不变量

设第 `k` 次 worker 使用请求 `P_k`、摘要 `H_k`，并在安装后得到实际 dtype manifest
摘要 `D_k`。supervisor 在启动时冻结 `P_0` 和 `H_0`，每个新 worker 只允许覆盖
`MemoryPlan` 字段。

**基例：** `P_0` 由合并配置和一次硬件解析产生，`H_0=SHA256(P_0)`；worker 参数由
冻结 Namespace 复制，且在加载前独立重算 H。若模型尚未安装而 OOM，则 `D_0` 尚未
定义；不能把“缺席”当成一个与已安装 dtype 分区相等的摘要。

**归纳步：** 假设第 `k` 次结果满足 `H_k=H_0`，且所有已安装模型的尝试共享第一个
已定义的 manifest `D_*`。worker 在加载前、setup、commit 和完成/OOM 时重算 H；
已安装模型还将实际 dtype 与安装基线比较。若结果不是 `cuda_oom`，不会继续；否则
`next_plan()` 只改变 `M`。新 worker 仍从冻结 Namespace 生成并验证 `P_{k+1}`；
若 `D_{k+1}` 存在，必须等于自身安装基线和 `D_*`。只有加载前 OOM 可继续保持 `D`
未定义，不能由它推断实际分区。

所以在“摘要碰撞概率足够低、用于防意外漂移”的工程假设下：

```text
P_k = P_0 and H_k = H_0, for every allowed retry k
D_k = D_*, for every completed installation whose result participates in retry
```

其中 `D` 只证明**检查时刻**基础 tensor dtype 映射没有漂移，不证明设备驻留、kernel、LoRA 包装
顺序之外的全部运行时状态或低精度计算与 FP32 逐位相等。

### 8.2 内存计划有界性

在允许 swap 的阶段：

```text
s_{k+1} = min(max_swap, s_k + swap_increment)
```

若 `s_k < max_swap`，则 `s` 严格增加；达到上限后 `next_plan()` 返回 `None` 或进入
明确授权的下一种变化。再结合 `k < max_attempts` 和 `seen`，重试次数有限，不存在
由同一计划构成的无限循环。

### 8.3 有效 batch 守恒

只有显式允许 batch change 且 `d | b` 时，才允许：

```text
b' = b/d,  a' = a*d
B' = b' * a' = (b/d) * (a*d) = b*a = B
```

这是有效样本数的代数守恒，不等价于完整训练轨迹逐位一致；规约树、数据顺序和随机
消费顺序仍需独立验收。因此当前生产 supervisor 不开启该分支。

### 8.4 optimizer 原子性边界

若 `optimizer_started=False`，OOM 发生在模型加载、forward 或 backward，当前 worker
没有被允许提交 optimizer 更新，可以安全丢弃并用新进程尝试新的 `M`。

若 `optimizer_started=True`，可能存在部分参数或状态写入。只有 worker 同时提供已验证
的完整提交检查点，`run_recovery()` 才会把它作为 `resume` 传给下一次尝试；当前生产
worker 永远返回 `null`，所以更新后 OOM 必然停止。

这不是“恢复能力不足的偶然实现”，而是避免把部分更新当作合法状态的安全证明边界。

## 9. 数值边界与不能推出的结论

FP16/FP32 island 的局部校准可以比较输出、输入 VJP 和可训练参数梯度，但它只覆盖
给定权重、给定输入和给定算子。即使局部误差低，也不能推出：

- 整网、多 bucket、长时间训练的误差仍低；
- 生成质量、收敛速度或最终 LoRA 质量不变；
- 该精度策略一定比 BF16 或 FP32 更快；
- 任意 GPU、attention backend 或 adapter 都支持相同 dtype 组合。

因此本文证明对象是：

```text
OOM retry preserves the declared precision request and stays within bounds.
```

不是：

```text
FP16/FP32 mixed precision is numerically or operationally recommended everywhere.
```

## 10. 静态测试与源码对应关系

| 测试/源码 | 静态证明点 |
| --- | --- |
| `tests/test_adaptive_precision_contract.py` | 内存字段变化不改变 `H`；loss scale、FP32 名单、attention 改变会改变 `H`；规范化稳定；GPU 身份探测失败拒绝 |
| `tests/test_adaptive_training_supervisor.py::test_supervisor_retries_structured_startup_oom_for_each_dit_family` | 三个 DiT family 的每次 worker 都携带同一精度配置和契约 ID |
| `tests/test_adaptive_training_supervisor.py::test_retry_precision_contract_is_stable_when_memory_plan_changes` | swap 从 24 到 26 时 `H` 不变 |
| `tests/test_adaptive_training_supervisor.py::test_retry_precision_contract_mismatch_fails_closed_before_next_attempt` | 契约不匹配时不会启动第二次尝试 |
| `tests/test_adaptive_training_supervisor.py::test_worker_checks_contract_before_model_load` | worker 在训练器构建前用本地 args 校验 H，失败不加载模型 |
| `tests/test_adaptive_training_supervisor.py::test_worker_rechecks_contract_after_setup_oom` | setup 后配置漂移即使随后 OOM，也不能作为可重试结果 |
| `tests/test_adaptive_training_supervisor.py::test_unresolved_auto_worker_fails_precision_contract` | `adaptive_precision=auto` 没有父进程解析结果时 fail closed |
| `tests/test_adaptive_training_supervisor.py::test_realized_precision_manifest_mismatch_fails_closed` | 不同 worker 的实际 dtype manifest 漂移时停止 |
| `tests/test_adaptive_training_supervisor.py::test_missing_realized_precision_manifest_fails_closed` | worker 未报告实际 dtype manifest 时停止 |
| `tests/test_adaptive_training_supervisor.py::test_model_load_oom_without_manifest_retries_after_local_contract_check` | 加载前 H 已校验，未生成 D 的 model-load OOM 可增加 swap |
| `tests/test_adaptive_training_supervisor.py::test_worker_rechecks_realized_dtype_at_exit` | setup 后 dtype 漂移在完成/OOM 结果前被拒绝 |
| `tests/test_adaptive_training_supervisor.py::test_commit_rechecks_dtype_before_publishing_snapshot` | 每次提交快照前拒绝过期 dtype 摘要 |
| `tests/test_adaptive_training_supervisor.py::test_supervisor_identity_probe_failure_leaves_no_recovery_directory` | 父进程 GPU 身份无法确认时，不创建不可复用的恢复目录 |
| `tests/test_adaptive_training_supervisor.py::test_possible_cuda_oom_text_is_observable_but_not_retryable` | 普通 RuntimeError 的 CUDA OOM 文本只生成 `possible_oom` 可观测标签 |
| `tests/test_adaptive_training_supervisor.py::test_infrastructure_failure_keeps_its_reason_without_worker_contract` | timeout 等无 worker 结果的故障不会被错误重命名 |
| `tests/test_adaptive_runtime.py` | island 输出边界、校准失败、stage-specific plan、普通错误不重试 |
| `tests/test_adaptive_training_precision.py::test_realized_dtype_manifest_is_stable_and_detects_drift` | dtype manifest 在重复读取和显式 cast 后的稳定性/漂移检测 |
| `tests/test_adaptive_training_precision.py::test_first_runtime_check_compares_installed_manifest` | setup 首次验证对照安装时摘要，而非接纳漂移值 |
| `tests/test_adaptive_training_precision.py` | Accelerate handler、plain FP32 LoRA、checkpoint precision manifest 和 family contract |
| `tests/test_adaptive_training_loop.py` | forward/backward/optimizer/checkpoint 阶段标记、sticky optimizer flag、scaler 跳步不提交 |
| `library/training/adaptive_runtime/retry.py` | 有界计划、seen 防循环、checkpoint 缺失 fail-closed |
| `library/training/adaptive_runtime/supervisor.py` | 合并配置冻结、0600 私有 worker args、新进程入口 |
| `library/training/adaptive_runtime/training_runtime.py` | 有效更新提交边界和当前不续训标志 |

本轮静态门禁使用以下 CPU/纯 Python 五文件集合（显式排除真实 CUDA kernel 测试）：

```bash
CUDA_VISIBLE_DEVICES= timeout 120 .venv/bin/python -m pytest -q \
  tests/test_adaptive_precision_contract.py \
  tests/test_adaptive_runtime.py \
  tests/test_adaptive_training_precision.py \
  tests/test_adaptive_training_supervisor.py \
  tests/test_adaptive_training_loop.py \
  -k 'not real_accelerate_scaler_clip_and_resume'
```

结果为 `133 passed, 1 deselected`（另有非阻断 warning）。显式排除的测试仅用于
Accelerate GPU kernel 验证，不计入本轮静态证明。未执行 `bench/`、`train.py`、
CUDA kernel、真实 OOM、长训、性能或热稳态测试。

## 11. 后续开放条件

要开放“更新后 OOM 自动续训”，至少还需要：

1. 数据集 identity、bucket 顺序、sampler permutation、prefetch/look-ahead 和独立 RNG
   的可验证 snapshot；
2. 底模、adapter 结构、有效配置和精度 manifest 的身份校验；
3. 新进程恢复后下一批输入/target/timestep 与不中断对照的逐项比较；
4. optimizer、scheduler、scaler、有效更新计数和 checkpoint 发布顺序的故障注入测试；
5. 受控真实 CUDA OOM 与同配置不中断对照。

即使暂不开放中途续训，以下恢复增强也仍需单独完成静态契约和受控验证：

6. 在 `optimizer_started` 置位前，用数据集最坏分辨率/最长序列做合成
   forward+backward 预飞，把启动期可重试窗口覆盖到真实峰值 shape；这不是本轮热测，
   当前尚未实现。
7. 评估并显式记录 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 的部署策略，
   当前没有在代码或默认环境中擅自开启，避免把碎片缓解误写成已验证结论。
8. 让 `next_plan()` 根据 host-memory reserve 反推 `max_blocks`，并为 model-load 增加
   缺口估计的指数步长/二分；成功计划持久化后才能作为同配置下次搜索起点。
9. 增加 `test_no_unclassified_args`：每个新增配置字段必须明确归入 PRECISION、RESOURCE
   或 IGNORED，防止未来新增数值开关漏进 `P`。当前测试覆盖已知字段变化，但尚未提供
   机器可执行的“无遗漏”枚举约束。
10. 增加真实 offloader block swap out/in 的 N 轮静态夹具，逐轮比较 `D`；当前测试只对
    同一模型重复取 manifest 并显式 cast 制造漂移，尚未模拟 offloader 回搬路径。

在这些条件满足前，`committed_checkpoint=null` 和
`data_cursor_resume_supported=false` 必须继续保留，不能把静态契约证明写成生产恢复
能力或热测结论。
