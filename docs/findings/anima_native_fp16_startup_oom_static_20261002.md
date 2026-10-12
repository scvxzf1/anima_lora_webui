# Anima 原生 FP16 启动 OOM：静态实现验收

日期：2026-10-02。状态：静态实现、CPU 回归及独立检查完成；真实 GPU 启动 OOM 待验收。
范围：启动 OOM 的控制流、精度身份和资源上限；没有运行 GPU、底模训练或真实 OOM。
使用入口：[原生 FP16](../experimental/anima-native-fp16.md#启动-oom-回退)。

## 执行合同

父进程解析一次 `auto`，冻结 native 配置和本地基座/初始 adapter 来源 SHA256，
每次启动新 worker，唯一可改变的训练内存计划字段是 `blocks_to_swap`。
原生 AMP/scaler、FP16 基座、FP32 LoRA、残差/final 保护和完整 checkpoint 保持。
旧 islands v1 请求/manifest 和默认 `off` 不改；native 使用独立 schema。

仅结构化 `cuda_oom`、明确 CUDA 证据、阶段属于加载/前向/反向，且
`optimizer_started is False` 才可重试。未知或非布尔进度拒绝；typed
`torch.OutOfMemoryError` 与 CUDA OOM 类型可能相同，类型本身不足以判断设备。
CPU/主机 OOM、模糊 OOM、普通错误、进程崩溃、超时和取消均不重试。

首次 optimizer 开始前重验 dtype、AMP 和 guards，再设置 sticky 进度标志。
标志一旦开启永不撤销；随后 OOM 不重试，即使 supplied checkpoint 或本步
scaler skip 也不恢复。native runtime 只保留标量，不写完整模型/optimizer
诊断快照。实际尝试、有效更新和跳步在 optimizer 事件处计数，global step
另记录已提交边界，二者不混用。

正常周期保存、日志、训练后的采样/验证保留；输入 resume、自动续训、首次采样、
自动 swap、额外基座合并和梯度累积大于 1 明确拒绝。不开重试的普通
native FP16 手动保存/续训保留既有路径。

## 精度与文件身份

native 请求 H 冻结有效 network kwargs、数值配置、transfer/restore、设备环境
和本地文件内容身份。实际 D 使用 base 参数/buffer 的名称、shape、numel 与 dtype，
不含设备驻留位置或交换块数量；prepare 后再验真实 FP32 LoRA、AMP/scaler、
残差和完整 checkpoint。LoRA 实际 `fp32_compute` flag 冻结后不得漂移。

scaler 的 scale/tracker 是动态训练状态，不纳入静态策略；策略检查只调用
Python 配置 getter，不从每步 `state_dict()` 同步 CUDA scale/tracker。
来源内容在父进程冻结、worker 启动、模型/adapter 加载后核验；hash 读取前后
检查文件 stat，加载阶段另检查文件替换。没有每步 source hash 或 tensor clone。
这是防止普通配置/文件漂移的合同，不提供对抗恶意并发写入的文件系统证明。

## 上限与报告

沿用真实 block 结构 `N-2`、用户 cap 和 CPU 加载后的实际 weight 字节库存预算。
CPU masters 与尾部副本计费使用真实 dtype；`bf16` transport 表示原始 dtype
保留策略，foreach 恢复必须维持 FP16 数值。仅最早 model-load OOM 可尚无
manifest/host 测量；加载成功后的正 swap 尝试必须携带有效 host 预算。

上限只收紧，指数 swap 序列严格递增，最后机会可到有效上限；尝试数有界，
无 batch/checkpoint/dtype 降级。OOM 能否由更多 swap 解决不由静态数学保证。
父报告记录所有尝试、最终交换计划、停止理由和子报告路径；子报告提供真实 dtype
与有效更新，父进程不冒充加载了模型。`.adaptive-recovery` 不覆盖历史目录。

## 验证

新增 CPU 用例覆盖源文件内容/路径漂移、H/D 一致、实际 guards、scaler 动态状态、
optimizer 前精度拒绝、首步后停止、跳步计数、未知状态、host 上限、取消、超时、
保存配置保留和真实 CPU 子进程 PID 变化。

16 文件集成回归：**391 passed、1 skipped**，退出码 0，耗时 18.40 秒；
14 个 warnings 为既有 TorchScript 弃用提示。执行命令：

```bash
CUDA_VISIBLE_DEVICES='' timeout 60 .venv/bin/python -m pytest -q \
  tests/test_native_startup_runtime.py tests/test_native_startup_oom.py \
  tests/test_native_training_precision.py tests/test_adaptive_training_supervisor.py \
  tests/test_adaptive_precision_contract.py tests/test_adaptive_retry_policy.py \
  tests/test_adaptive_swap_capacity.py tests/test_adaptive_runtime_process.py \
  tests/test_training_precision_report.py tests/test_training_compat_matrix.py \
  tests/test_web_preflight_compat_matrix.py tests/test_adaptive_training_precision.py \
  tests/test_adaptive_training_loop.py tests/test_training_loop_runtime.py \
  tests/test_adaptive_training_replay.py tests/test_adaptive_precision_swap_cpu.py
```

核心新增/改文件 Ruff 通过；Pyright 为 0 errors / 0 warnings。
前端 `tsc -b --pretty false` 退出码 0，`trainingForm.test.ts` 20 passed。
独立 CPU 数学检查通过 675 组 cap/start/attempt/increment 组合和 20 个异常状态，
确认 swap 严格递增、尝试数有界、cap 不回升及其他内存计划不变。

独立集成验收通过实际 `IsolatedRunner` 启动两个 CPU 子进程。首进程人工抛出
带明确 CUDA 文本的 OOM，次进程执行真实 FP32 LoRA SGD 两步，确认参数改变。
两个 PID 为 479906 / 480060，swap 为 `0 → 4`，H 一致；首次尚未加载模型，
D 为 null，第二次得到有效 D。父报告记录 completed、2 次有效更新、0 次跳步，
保存和日志字段保留。该 fixture mock 了 checkpoint 元数据检查和 AMP/scaler，
没有执行 FP16 基座前向、真实 offloader、实际保存或 CUDA 训练。

独立检查发现失败尝试的子报告路径丢失，已将路径绑定放入报告会话的 finally，
并补定向回归。复跑验收脚本后，父报告同时引用失败与成功子报告；归档时再次
断言两条引用、状态、PID、swap、H 和有效更新计数均正确。
原始 JSON 保留 `/tmp` 路径与身份，不改写为归档路径；本机证据目录为
`output/tests/native-fp16-startup-oom-static-20261002-cpu-3m9o4xx9/`，
其中 `README.md` 说明 fixture 范围，`verification-summary.json` 提供副本映射与 SHA256。

全局文档完整性检查为 5 passed / 3 failed，退出码 1。失败仍是既有的旧文档
可达性、proposal 索引遗漏及部分旧文档缺状态标签；本次新增报告未出现在失败
清单中。没有为该任务改写这些无关历史文档。
报告收尾后，Markdown fence 与本地链接/锚点定向检查 2 passed，退出码 0；
归档 16 个原样副本的 SHA256、报告状态标签、findings 索引和空白检查通过。

保持 `production_ready=false`；本轮静态通过不构成 CUDA kernel、数值质量、
长期稳定性或性能认证。旧严格 0/12 失败与
[200步 A/B](anima_fp16_bf16_ab200_90hx_20261002.md) 范围不改。
